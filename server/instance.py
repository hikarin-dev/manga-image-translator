import asyncio
from asyncio import Event, Lock
from typing import List

from PIL import Image
from pydantic import BaseModel

from manga_translator import Config
from server.sent_data_internal import fetch_data_stream, NotifyType, fetch_data, fetch_gallery_stream, post_cancel

class ExecutorInstance(BaseModel):
    ip: str
    port: int
    busy: bool = False
    # Lower wins when several executors are free. The local worker sits at the default;
    # aux nodes register above it (numerically lower) so remote capacity is spent first
    # and this machine's GPU stays available as the fallback.
    priority: int = 100
    # An executor that can only take whole-gallery chunks. Aux nodes set this: the
    # single-image and batch paths return a pickled Context, which we will not
    # deserialize from a machine we don't own (see server.safe_pickle).
    gallery_only: bool = False
    # Held back unless it is the only thing left (--lazy sets this on the local worker). Unlike
    # priority, which only orders the free executors, a reserve executor is not offered work at
    # all while a non-reserve one exists — even a busy one. That is the difference between
    # "prefer the aux nodes" and "keep my GPU out of this unless everyone has gone home".
    reserve: bool = False
    # Gallery chunks that may run on this executor at once. A second one is offered only while every
    # chunk already on it has read all its pages (Executors.chunk_read), so its reading (detection,
    # OCR) fills the first one's translate/inpaint/render tail instead of the GPU idling through a
    # drain between chunks. 1 is one chunk at a time. `active`/`reading` are the chunks on it now.
    slots: int = 1
    active: int = 0
    reading: int = 0

    @property
    def label(self) -> str:
        return f"{self.ip}:{self.port}"

    def free_executor(self):
        self.busy = False

    async def sent(self, image: Image, config: Config):
        return await fetch_data("http://"+self.ip+":"+str(self.port)+"/simple_execute/translate", image, config)

    async def sent_stream(self, image: Image, config: Config, sender: NotifyType):
        await fetch_data_stream("http://"+self.ip+":"+str(self.port)+"/execute/translate", image, config, sender)

    async def sent_gallery_stream(self, images: List, config: Config, sender: NotifyType, batch_size: int = 0, job_token: str = "",
                                  pages: List = None, builds: str = None, context: List = None, capture: bool = True):
        await fetch_gallery_stream("http://"+self.ip+":"+str(self.port)+"/execute/translate_gallery_stream", images, config, sender, batch_size, job_token,
                                   pages=pages, builds=builds, context=context, capture=capture)

    async def cancel_gallery(self, job_token: str = ""):
        await post_cancel("http://"+self.ip+":"+str(self.port)+"/cancel_gallery", job_token)

    async def sent_batch(self, images: List[Image.Image], config: Config, batch_size: int):
        """发送批量翻译请求"""
        return await fetch_data("http://"+self.ip+":"+str(self.port)+"/simple_execute/translate_batch",
                               {"images": images, "config": config, "batch_size": batch_size})

    async def sent_batch_stream(self, images: List[Image.Image], config: Config, batch_size: int, sender: NotifyType):
        """发送批量翻译流式请求"""
        await fetch_data_stream("http://"+self.ip+":"+str(self.port)+"/execute/translate_batch",
                               {"images": images, "config": config, "batch_size": batch_size}, config, sender)

class Executors:
    def __init__(self):
        self.list: List[ExecutorInstance] = []
        self.lock: Lock = Lock()
        self.event = Event()

    def register(self, instance: ExecutorInstance):
        self.list.append(instance)

    def unregister(self, instance) -> None:
        """Drop an executor that went away (an aux node's socket closed). Any chunk it was
        running is failed by its own connection teardown; this only stops it being picked.

        Removes by identity: ExecutorInstance is a pydantic model, so two workers on the same
        ip:port compare equal and list.remove() could evict the wrong one."""
        for i, x in enumerate(self.list):
            if x is instance:
                self.list.pop(i)
                return

    def _eligible(self, gallery: bool) -> List[ExecutorInstance]:
        pool = [x for x in self.list if gallery or not getattr(x, 'gallery_only', False)]
        # A reserve executor stands in only when nothing else can serve this kind of task. Note
        # this is evaluated per task kind, so under --lazy the local worker is still the one
        # that runs single-image work (aux nodes are gallery-only) while sitting out galleries.
        active = [x for x in pool if not getattr(x, 'reserve', False)]
        return active if active else pool

    def capacity(self, gallery: bool = True) -> int:
        """How many executors could take a chunk if everything were free — what chunk sizing
        spreads a lone job over."""
        return len(self._eligible(gallery))

    def slot_capacity(self, gallery: bool = True) -> int:
        """How many chunks may be in flight at once, overlap slots included: the scheduler's cap."""
        return sum(getattr(x, 'slots', 1) for x in self._eligible(gallery))

    @staticmethod
    def _accepting(x, gallery: bool = True) -> bool:
        # Executors without slot accounting (aux nodes) are one chunk at a time, by their flag.
        if not hasattr(x, 'active'):
            return not x.busy
        if x.active == 0:
            return True
        # Only gallery chunks overlap: the worker serves anything else alone.
        return gallery and x.active < x.slots and x.reading == 0

    def free_executors(self, gallery: bool = True) -> int:
        return len([item for item in self._eligible(gallery) if self._accepting(item, gallery)])

    async def _find_instance(self, gallery: bool):
        while True:
            free = [x for x in self._eligible(gallery) if self._accepting(x, gallery)]
            if free:
                # Preferred capacity first; ties keep registration order so two equal aux
                # nodes still round-robin naturally as each is marked busy.
                free.sort(key=lambda x: getattr(x, 'priority', 100))
                return free[0]
            # Re-check on a timeout as well as on the event: free_executor() sets and
            # immediately clears the event, so a waiter that hasn't started awaiting yet
            # would otherwise miss the wake and hang. Harmless when idle, and with several
            # chunks in flight this race is otherwise easy to hit.
            #
            # asyncio.wait, not wait_for: wait_for can surface an external cancellation as
            # TimeoutError, which this loop would treat as "re-check" and swallow — and this
            # one waits while holding self.lock, so swallowing a cancel wedges the whole pool.
            waiter = asyncio.ensure_future(self.event.wait())
            try:
                await asyncio.wait({waiter}, timeout=1.0)
            finally:
                waiter.cancel()

    async def find_executor(self, gallery: bool = True) -> ExecutorInstance:
        async with self.lock:  # Using async with for lock management
            instance = await self._find_instance(gallery)
            if hasattr(instance, 'active'):
                instance.active += 1
                instance.reading += 1
            instance.busy = True
            return instance

    async def free_executor(self, instance: ExecutorInstance, read_done: bool = False):
        """A chunk (or task) on `instance` finished. `read_done`: it already reported reading
        all its pages (chunk_read), so it no longer counts as reading."""
        from server.myqueue import task_queue
        if hasattr(instance, 'active'):
            instance.active = max(0, instance.active - 1)
            if not read_done:
                instance.reading = max(0, instance.reading - 1)
            instance.busy = not self._accepting(instance)
        else:
            instance.free_executor()
        self.event.set()
        self.event.clear()
        await task_queue.update_event()

    def chunk_read(self, instance) -> None:
        """A gallery chunk on `instance` has read all its pages: the instance may take another."""
        if getattr(instance, 'reading', 0) <= 0:
            return
        instance.reading -= 1
        instance.busy = not self._accepting(instance)
        self.event.set()
        self.event.clear()
        from server.myqueue import task_queue
        asyncio.get_running_loop().create_task(task_queue.update_event())

executor_instances: Executors = Executors()
