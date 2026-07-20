import asyncio
import os
import time
from typing import List, Optional

from PIL import Image
from fastapi import HTTPException
from fastapi.requests import Request

from manga_translator import Config
from server.instance import executor_instances
from server.sent_data_internal import NotifyType

class QueueElement:
    req: Request
    image: Image.Image | str
    config: Config

    def __init__(self, req: Request, image: Image.Image, config: Config, length):
        self.req = req
        if length > 10:
            #todo: store image in "upload-cache" folder
            self.image = image
        else:
            self.image = image
        self.config = config

    def get_image(self)-> Image:
        if isinstance(self.image, str):
            return Image.open(self.image)
        else:
            return self.image

    def __del__(self):
        if isinstance(self.image, str):
            os.remove(self.image)

    async def is_client_disconnected(self) -> bool:
        if await self.req.is_disconnected():
            return True
        return False


class BatchQueueElement:
    """Batch translation queue element"""
    req: Request
    images: List[Image.Image]
    config: Config
    batch_size: int

    def __init__(self, req: Request, images: List[Image.Image], config: Config, batch_size: int):
        self.req = req
        self.images = images
        self.config = config
        self.batch_size = batch_size

    async def is_client_disconnected(self) -> bool:
        if await self.req.is_disconnected():
            return True
        return False


class GalleryQueueElement:
    """Whole-gallery streaming translation queue element: a list of page images
    (kept as compressed bytes until the worker decodes them) translated in
    batch_size-page shared translation calls, streamed back page by page.
    Separate from the (broken) BatchQueueElement zip/JSON path."""
    req: Request
    images: List
    config: Config
    batch_size: int
    job_token: str
    cancelled: bool

    def __init__(self, req: Request, images: List, config: Config, batch_size: int = 0, job_token: str = "", parent=None):
        self.req = req
        self.images = images
        self.config = config
        self.batch_size = batch_size
        self.job_token = job_token       # client-issued id; lets /translate/gallery/cancel target this job
        self.cancelled = False           # set by an explicit cancel for a still-queued job
        self.parent = parent             # the scheduler's _SchedJob, when this chunk belongs to one

    async def is_client_disconnected(self) -> bool:
        # A gallery job is server-owned and re-attachable, so its liveness is NOT this one
        # creating connection — the original client can be long gone while a reconnected one
        # streams. Only an explicit /cancel_gallery or the liveness reaper (no attached client
        # for the grace window) sets `cancelled`; that, and only that, means "stop the GPU".
        # (Watching self.req here would abort a job the moment its first tab navigated away.)
        #
        # A job's chunks can run on several executors at once, so a cancel lands on the job
        # rather than on whichever chunk happens to be reachable — each chunk inherits it here.
        return self.cancelled or bool(self.parent is not None and getattr(self.parent, 'cancelled', False))


# token → the executor instances currently running chunks of that gallery job. A job can hold
# several at once (one chunk per executor), so the explicit cancel endpoint fans its
# token-scoped /cancel_gallery out to all of them.
#
# A list, not a set: ExecutorInstance is a pydantic model, and pydantic v2 sets __hash__ = None
# on non-frozen models, so a set cannot hold one. Membership here is identity anyway — two
# workers on the same ip:port are equal by value but are still two different workers.
running_galleries: dict = {}


class TaskQueue:
    def __init__(self):
        self.queue: List[QueueElement | BatchQueueElement | GalleryQueueElement] = []
        self.queue_event: asyncio.Event = asyncio.Event()

    def add_task(self, task: QueueElement | BatchQueueElement | GalleryQueueElement):
        self.queue.append(task)

    def get_pos(self, task: QueueElement | BatchQueueElement | GalleryQueueElement) -> Optional[int]:
        try:
            return self.queue.index(task)
        except ValueError:
            return None
    async def update_event(self):
        self.queue = [task for task in self.queue if not await task.is_client_disconnected()]
        self.queue_event.set()
        self.queue_event.clear()

    async def remove(self, task: QueueElement | BatchQueueElement):
        self.queue.remove(task)
        await self.update_event()

    async def wait_for_event(self, timeout: float | None = None):
        if timeout is None:
            await self.queue_event.wait()
            return
        # asyncio.wait rather than wait_for: wait_for can surface an external cancellation as
        # TimeoutError, and callers treat a timeout as "re-check and keep waiting" — which
        # would make a cancelled task immortal.
        waiter = asyncio.ensure_future(self.queue_event.wait())
        try:
            await asyncio.wait({waiter}, timeout=timeout)
        finally:
            waiter.cancel()

task_queue = TaskQueue()

# With a local worker there is always at least one executor, so a queued task could safely wait
# forever. A --delegate-only server has no local worker, so if every aux node disconnects the
# pool is empty indefinitely and a task would hang while the client polls a job that never
# moves. Fail the task once the pool has been empty this long instead; for a gallery chunk the
# scheduler turns that into a retry and, if capacity never returns, a clean job error.
NO_EXECUTOR_TIMEOUT_S = float(os.getenv('MT_NO_EXECUTOR_TIMEOUT_S', '180'))

async def wait_in_queue(task: QueueElement | BatchQueueElement, notify: NotifyType):
    """Will get task position report it. If its in the range of translators then it will try to aquire an instance(blockig) and sent a task to it. when done the item will be removed from the queue and result will be returned"""
    starved_since: Optional[float] = None
    while True:
        queue_pos = task_queue.get_pos(task)
        if queue_pos is None:
            if notify:
                return
            else:
                raise HTTPException(500, detail="User is no longer connected")  # just for the logs
        if notify:
            notify(3, str(queue_pos).encode('utf-8'))
        # Aux nodes take gallery chunks only, so a single-image/batch task must measure itself
        # against local capacity alone or it would queue forever behind executors that will
        # never be offered to it.
        gallery = isinstance(task, GalleryQueueElement)
        if queue_pos < executor_instances.free_executors(gallery):
            if await task.is_client_disconnected():
                await task_queue.update_event()
                if notify:
                    return
                else:
                    raise HTTPException(500, detail="User is no longer connected") #just for the logs

            instance = await executor_instances.find_executor(gallery)
            await task_queue.remove(task)
            if notify:
                notify(4, b"")

            try:
                # Process whole-gallery streaming task (always notify/streaming).
                # Watch for the client going away mid-gallery and tell the worker to
                # abort, so a closed tab doesn't keep the GPU busy for the whole rest
                # of the gallery while other clients wait.
                if isinstance(task, GalleryQueueElement):
                    if task.job_token:
                        running_galleries.setdefault(task.job_token, []).append(instance)
                    cancel_sent = False
                    try:
                        stream_task = asyncio.create_task(
                            instance.sent_gallery_stream(task.images, task.config, notify, task.batch_size, task.job_token))
                        while True:
                            done, _ = await asyncio.wait({stream_task}, timeout=2.0)
                            if done:
                                break
                            # An explicit cancel or the liveness reaper (no poll for the grace
                            # window) sets task.cancelled; forward a token-scoped abort so the GPU
                            # stops instead of finishing the gallery into nothing.
                            if not cancel_sent and await task.is_client_disconnected():
                                cancel_sent = True
                                await instance.cancel_gallery(task.job_token)
                        await stream_task
                    except BaseException:
                        # The main→worker stream broke (read timeout, connection reset, or the
                        # worker raising). The worker's pipeline runs independently of this stream,
                        # so unless we tell it to stop it keeps churning the GPU into a dead
                        # connection. Forward a token-scoped cancel so the worker halts, then let the
                        # error propagate to the handler below (which reports it to the client).
                        if not cancel_sent:
                            try:
                                await instance.cancel_gallery(task.job_token)
                            except Exception:
                                pass
                        raise
                    finally:
                        if task.job_token:
                            holders = running_galleries.get(task.job_token)
                            if holders is not None:
                                # Identity, not equality: two ExecutorInstances with the same
                                # ip:port compare equal, so a value-based remove could drop a
                                # different worker that is still running a chunk.
                                for k, held in enumerate(holders):
                                    if held is instance:
                                        holders.pop(k)
                                        break
                                if not holders:
                                    running_galleries.pop(task.job_token, None)
                # Process batch translation task
                elif isinstance(task, BatchQueueElement):
                    if notify:
                        await instance.sent_batch_stream(task.images, task.config, task.batch_size, notify)
                    else:
                        result = await instance.sent_batch(task.images, task.config, task.batch_size)
                else:
                    # Process single translation task
                    if notify:
                        await instance.sent_stream(task.image, task.config, notify)
                    else:
                        result = await instance.sent(task.image, task.config)

                await executor_instances.free_executor(instance)

                if notify:
                    return
                else:
                    return result

            except Exception as e:
                # 确保实例被释放
                await executor_instances.free_executor(instance)

                # 如果是连接错误，发送友好的错误消息
                if "Cannot connect to host" in str(e) or "Connection refused" in str(e):
                    error_msg = "Translation service is starting up, please wait a moment and try again."
                else:
                    error_msg = f"Translation failed: {str(e)}"

                if notify:
                    notify(2, error_msg.encode('utf-8'))
                    return
                else:
                    raise HTTPException(500, detail=error_msg)
        else:
            # Starvation guard: an EMPTY pool (not merely a busy one) can never serve this task,
            # so waiting on it is waiting for a node to connect. Give that a bounded window —
            # long enough to ride out an aux node restarting — then fail rather than hang.
            if executor_instances.capacity(gallery) == 0:
                if starved_since is None:
                    starved_since = time.monotonic()
                elif time.monotonic() - starved_since > NO_EXECUTOR_TIMEOUT_S:
                    try:
                        await task_queue.remove(task)
                    except ValueError:
                        pass
                    detail = ('no translation capacity: no worker is connected to this server'
                              if gallery else
                              'no local translator worker is running (this task cannot run on an aux node)')
                    if notify:
                        notify(2, detail.encode('utf-8'))
                        return
                    raise HTTPException(503, detail=detail)
            else:
                starved_since = None
            # Normally just a slow re-check. While starved, shorten the slice to whatever is
            # left of the window so the guard fires at its deadline instead of at the next
            # 5-second tick.
            wait = 5.0
            if starved_since is not None:
                wait = min(wait, max(0.05, NO_EXECUTOR_TIMEOUT_S - (time.monotonic() - starved_since)))
            await task_queue.wait_for_event(timeout=wait)