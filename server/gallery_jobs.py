"""Server-owned gallery jobs.

A whole-gallery translation is a long job (minutes for a big gallery). Tying its life to one
client-held connection is fragile: a tab navigation, a service-worker suspension, or a flaky
socket severs it, and the GPU then either dies with it or keeps churning into nothing.

So the SERVER owns the job and the client only observes. The worker writes every finished-page
frame into the job's buffer (never straight to a socket); a client collects them with short
/translate/gallery/poll requests, advancing a cursor. A dropped connection is a non-event — the
next poll picks up where it left off. The only thing that stops a job is an explicit cancel or the
liveness reaper: a job with no poll for NO_CLIENT_GRACE_S is presumed abandoned and cancelled.

This is the standard async request/reply (job-as-a-resource) pattern, scoped to one GPU.
"""
import asyncio
import json
import logging
import math
import pickle
import re
import time

logger = logging.getLogger('gallery-jobs')

# Pipeline progress strings the worker emits as status-1 frames (longest alt first so 'tl-done'
# isn't shadowed by 'tl'): gallery-pre:k/n (reading), gallery-tl:b/m (translating a batch),
# gallery-tl-done:b/m (a batch finished).
_STATE_RE = re.compile(r'^gallery-(pre|tl-done|tl):(\d+)/(\d+)$')

# A running job is cancelled after this long with no poll. The service-worker poll loop polls every
# ~3s and re-arms itself before its ~4-min handoff, so under normal use this is never approached;
# the headroom is to outlast that re-arm handoff and brief service-worker respawns. Still short
# enough that a truly abandoned job (browser closed for good — which kills the SW and its loop)
# frees the GPU reasonably quickly.
NO_CLIENT_GRACE_S = 40.0
# How long a finished/cancelled/errored job's buffer lingers so a late poll can still collect the
# result before we free the memory.
DONE_RETENTION_S = 120.0
# Reaper cadence.
_REAP_EVERY_S = 5.0


class GalleryJob:
    """One whole-gallery translation. Buffers the frames the worker produces so a polling client
    can collect them past a cursor, and tracks per-stage progress for the client's status label."""

    def __init__(self, token: str):
        self.token = token
        self.task = None              # the GalleryQueueElement — lets the reaper/cancel forward a token-scoped abort
        self.durable: list[bytes] = []  # status-5 (page) / status-6 (study) frames, collected by cursor
        self.terminal: bytes | None = None  # status-0 summary or status-2 error, once produced
        self.last_poll = time.monotonic()  # updated on every /poll — the heartbeat the reaper watches
        self.last_state = ''          # latest status-1 progress string (gallery-pre:k/n …)
        self.emitted = 0              # finished pages produced so far (status-5 frames) — authoritative progress
        self.total = 0               # total pages in this job (set by start_gallery_job)
        # Furthest progress reached in each pipeline stage (the stages overlap, so we keep the max
        # of each rather than the single latest frame) — lets the client show an accurate label.
        self.pre = 0                  # pages whose text has been read (detect + OCR)
        self.tl_started = 0           # translation batches started
        self.tl_done = 0              # translation batches finished
        self.batches = 0              # total translation batches
        self.queue = 0                # queue position while waiting for a worker (status-3)
        self.dispatched = False       # a worker picked the job up (status-4)
        self.status = 'running'       # running | done | error | cancelled
        self.done_at: float | None = None

    @property
    def finished(self) -> bool:
        return self.terminal is not None

    # ── worker → job ──────────────────────────────────────────────────────────────────────
    # Used as the notify() sink: server.streaming.notify() calls put_nowait(encoded_frame).
    def put_nowait(self, frame: bytes) -> None:
        code = frame[0] if frame else 1
        if code == 5 or code == 6:
            self.durable.append(frame)        # finished page / its study layers — collected by cursor
            if code == 5:
                self.emitted += 1
        elif code == 0 or code == 2:
            self.terminal = frame             # the one terminal frame; everything after is ignored upstream
            self.status = 'done' if code == 0 else 'error'
            self.done_at = time.monotonic()
        elif code == 1:
            # A progress string. Fold each one into the per-stage furthest-progress counters the
            # poll meta exposes for the label (the worker emits these faster than we poll).
            try:
                s = frame[5:].decode('utf-8', 'ignore')
                self.last_state = s
                mm = _STATE_RE.match(s)
                if mm:
                    kind, a, b = mm.group(1), int(mm.group(2)), int(mm.group(3))
                    if kind == 'pre':
                        self.pre = max(self.pre, a)
                    elif kind == 'tl':
                        self.tl_started = max(self.tl_started, a); self.batches = max(self.batches, b)
                    elif kind == 'tl-done':
                        self.tl_done = max(self.tl_done, a); self.batches = max(self.batches, b)
            except Exception:
                pass
        elif code == 3:
            try:
                self.queue = int(frame[5:].decode('utf-8', 'ignore') or 0)
            except Exception:
                pass
        elif code == 4:
            self.dispatched = True
            self.queue = 0

    # ── polling client ↔ job ──────────────────────────────────────────────────────────────────
    def poll(self, since: int) -> bytes:
        """One short poll. Returns a response BODY: a status-7 metadata frame (JSON: cursor, status,
        state, done, total + per-stage counters) followed by the page/study frames produced past the
        client's cursor, plus the terminal frame once there is one. Everything is in the body — never
        headers — so it survives cross-origin fetches (app and translate server are different
        origins). Updating last_poll here is the polling heartbeat the reaper watches."""
        self.last_poll = time.monotonic()
        since = max(0, min(since, len(self.durable)))
        cursor = len(self.durable)
        meta = json.dumps({
            "cursor": cursor, "status": self.status, "state": self.last_state,
            "done": self.emitted, "total": self.total,
            "pre": self.pre, "tlStarted": self.tl_started, "tlDone": self.tl_done,
            "batches": self.batches, "queue": self.queue, "dispatched": self.dispatched,
        }, separators=(',', ':')).encode('utf-8')
        body = b'\x07' + len(meta).to_bytes(4, 'big') + meta
        body += b''.join(self.durable[since:])
        if self.terminal is not None:
            body += self.terminal
        return body

    @staticmethod
    def notfound_body(since: int) -> bytes:
        """A status-7 frame telling a poller the job is gone (reaped/evicted/restarted)."""
        meta = json.dumps({"cursor": since, "status": "notfound", "state": "", "done": 0, "total": 0},
                          separators=(',', ':')).encode('utf-8')
        return b'\x07' + len(meta).to_bytes(4, 'big') + meta


# ── registry + reaper ────────────────────────────────────────────────────────────────────
_jobs: dict[str, GalleryJob] = {}
_reaper_started = False


def get(token: str) -> GalleryJob | None:
    return _jobs.get(token) if token else None


def create(token: str) -> GalleryJob:
    job = GalleryJob(token)
    if token:
        _jobs[token] = job
    _ensure_reaper()
    return job


def _ensure_reaper() -> None:
    global _reaper_started
    if not _reaper_started:
        _reaper_started = True
        asyncio.create_task(_reap_loop())


async def _reap_loop() -> None:
    while True:
        await asyncio.sleep(_REAP_EVERY_S)
        now = time.monotonic()
        for token, job in list(_jobs.items()):
            if job.status == 'running' and not job.finished:
                # No poll for the grace window → presume abandoned, stop the GPU. The queue
                # watchdog reads task.cancelled and forwards a token-scoped /cancel_gallery.
                if (now - job.last_poll) > NO_CLIENT_GRACE_S and job.task is not None:
                    job.task.cancelled = True
                    job.status = 'cancelled'
                    job.done_at = now
                    logger.warning(
                        f'Gallery job {token[:8]}… reaped: no client poll for '
                        f'{now - job.last_poll:.0f}s (grace {NO_CLIENT_GRACE_S:.0f}s) — cancelling '
                        f'({job.emitted}/{job.total} pages emitted)')
            else:
                ref = job.done_at or job.last_poll
                if (now - ref) > DONE_RETENTION_S:
                    _jobs.pop(token, None)


# ── multi-tenant chunk scheduler ─────────────────────────────────────────────────────────
# One GPU, many clients. A whole gallery as one queue element means a big job blocks every
# later client for its full duration. Instead the scheduler owns every gallery job and feeds
# the executor queue ONE CHUNK of pages at a time, choosing whose chunk goes next:
#
#   • alone           — a solo job runs in large chunks (SOLO_CHUNK); near-zero overhead, and
#                       a newcomer waits at most one chunk before being serviced.
#   • 2..ACTIVE_MAX   — weighted round-robin over arrival order: the oldest job gets
#                       WEIGHT_OLDEST chunks per rotation, the others one each. First-come
#                       keeps priority, but every active client sees steady page progress.
#   • beyond          — a waiting list: no GPU time, queue position exposed via /poll.
#
# Chunk sizes are multiples of the job's LLM batch cap, so translation batches are composed
# of exactly the same page groups as an unchunked run. Cross-page-context translators
# (chatgpt) are never split — their context is request-local. Frames from each chunk are
# rewritten to job-absolute page indices / progress before landing in the job buffer, so
# clients see one continuous job.
ACTIVE_MAX = 3
SOLO_CHUNK = 48
SHARED_CHUNK = 16
WEIGHT_OLDEST = 2

_UNCHUNKABLE_TRANSLATORS = ('chatgpt', 'chatgpt_2stage')


class _SchedJob:
    """Scheduler-side state for one gallery job. Also serves as the job's cancel handle
    while no chunk is in flight (the reaper/cancel path sets `.cancelled` on job.task)."""

    def __init__(self, job, req, images, config, batch_size, transform):
        self.job = job
        self.req = req
        self.images = images
        self.config = config
        self.batch_size = int(batch_size) if batch_size else 0
        self.transform = transform
        self.total = len(images)
        self.next_page = 0
        self.failed: list[int] = []
        self.cancelled = False

    @property
    def finished(self) -> bool:
        return self.next_page >= self.total

    def chunk_size(self, solo: bool) -> int:
        remaining = self.total - self.next_page
        tr = str(getattr(self.config.translator, 'translator', ''))
        if tr in _UNCHUNKABLE_TRANSLATORS:
            return remaining
        cap = self.batch_size if self.batch_size > 0 else remaining
        base = SOLO_CHUNK if solo else SHARED_CHUNK
        pages = max(cap, (base // max(1, cap)) * max(1, cap))
        return min(pages, remaining)


_sched: dict[str, _SchedJob] = {}
_sched_order: list[str] = []
_sched_event = asyncio.Event()
_sched_started = False


def submit(job: GalleryJob, req, images, config, batch_size, transform) -> None:
    """Register a gallery job with the scheduler (replaces enqueueing it whole)."""
    global _sched_started
    sj = _SchedJob(job, req, images, config, batch_size, transform)
    _sched[job.token] = sj
    _sched_order.append(job.token)
    job.task = sj
    if not _sched_started:
        _sched_started = True
        asyncio.create_task(_sched_loop())
    _sched_event.set()


def cancel(token: str) -> bool:
    """Cancel a job wherever it is: waiting, between chunks, or mid-chunk (the in-flight
    element shares the token, so the executor-side cancel still reaches the worker)."""
    sj = _sched.get(token)
    if sj is not None:
        sj.cancelled = True
        _sched_event.set()
    job = _jobs.get(token)
    if job is not None and job.status == 'running' and not job.finished:
        if job.task is not None:
            job.task.cancelled = True
        job.status = 'cancelled'
        job.done_at = time.monotonic()
    return sj is not None or job is not None


_round_state: dict = {}   # token → chunks granted in the current rotation round


def _pick_next() -> "_SchedJob | None":
    live = [t for t in _sched_order if t in _sched]
    active = live[:ACTIVE_MAX]
    # Waiting list: expose how many jobs are ahead so clients show a queue position.
    for i, token in enumerate(live[ACTIVE_MAX:]):
        _sched[token].job.queue = i + 1
    if not active:
        return None
    for k, token in enumerate(active):
        quota = WEIGHT_OLDEST if k == 0 else 1
        if _round_state.get(token, 0) < quota:
            _round_state[token] = _round_state.get(token, 0) + 1
            return _sched[token]
    _round_state.clear()
    _round_state[active[0]] = 1
    return _sched[active[0]]


def _make_adapter(sj: _SchedJob, chunk_start: int, final: bool):
    """Per-chunk notify sink: rewrites chunk-local frames into job-absolute ones before
    they land in the job buffer, and merges intermediate summaries."""
    from server.streaming import notify

    def adapter(code: int, data: bytes) -> None:
        if code in (5, 6):
            # tokenLen(1) + token + idx(4 BE) + payload → add the chunk's page offset.
            tlen = data[0]
            b = 1 + tlen
            idx = int.from_bytes(data[b:b + 4], 'big') + chunk_start
            data = data[:b] + idx.to_bytes(4, 'big') + data[b + 4:]
            notify(code, data, sj.transform, sj.job)
        elif code == 1:
            s = data.decode('utf-8', 'ignore')
            mm = _STATE_RE.match(s)
            if mm:
                kind, a = mm.group(1), int(mm.group(2))
                if kind == 'pre':
                    s = f'gallery-pre:{chunk_start + a}/{sj.total}'
                elif sj.batch_size > 0:
                    base_b = chunk_start // sj.batch_size
                    m_total = math.ceil(sj.total / sj.batch_size)
                    s = f'gallery-{kind}:{base_b + a}/{m_total}'
            notify(1, s.encode('utf-8'), sj.transform, sj.job)
        elif code == 0:
            try:
                summary = pickle.loads(data)
            except Exception:
                summary = {}
            if isinstance(summary, dict):
                sj.failed.extend(chunk_start + int(i) for i in summary.get('failed', []))
            if final:
                notify(0, pickle.dumps({'count': sj.total, 'failed': sorted(set(sj.failed))}), sj.transform, sj.job)
            # non-final chunk summaries are folded, never emitted — the job is still running
        else:
            # 2 = error (terminal), 3/4 = queue position / dispatched — pass through
            notify(code, data, sj.transform, sj.job)

    return adapter


def _drop(token: str) -> None:
    _sched.pop(token, None)
    _round_state.pop(token, None)
    try:
        _sched_order.remove(token)
    except ValueError:
        pass


async def _sched_loop() -> None:
    from server.myqueue import task_queue, wait_in_queue, GalleryQueueElement
    while True:
        # Retire cancelled/errored jobs before picking (a cancel can land any time).
        for token in list(_sched_order):
            sj = _sched.get(token)
            if sj is None:
                _drop(token)
            elif sj.cancelled or sj.job.status in ('cancelled', 'error'):
                _drop(token)

        sj = _pick_next()
        if sj is None:
            _sched_event.clear()
            # Wake on new submissions; also tick so waiting-list positions stay fresh.
            try:
                await asyncio.wait_for(_sched_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            continue

        live_active = [t for t in _sched_order if t in _sched][:ACTIVE_MAX]
        start = sj.next_page
        count = sj.chunk_size(solo=(len(live_active) <= 1))
        end = start + count
        sj.next_page = end
        final = end >= sj.total

        element = GalleryQueueElement(sj.req, sj.images[start:end], sj.config, sj.batch_size, sj.job.token)
        element.cancelled = sj.cancelled
        sj.job.task = element          # reaper/cancel reach the in-flight chunk
        task_queue.add_task(element)
        if len(live_active) > 1 or start > 0:
            logger.info(
                f'Gallery job {sj.job.token[:8]}… chunk {start}-{end - 1}/{sj.total} '
                f'({len(live_active)} active, {max(0, len([t for t in _sched_order if t in _sched]) - ACTIVE_MAX)} waiting)')
        try:
            await wait_in_queue(element, _make_adapter(sj, start, final))
        except Exception as e:
            logger.error(f'Gallery job {sj.job.token[:8]}… chunk dispatch failed: {e}')
        finally:
            if element.cancelled:
                sj.cancelled = True
            sj.job.task = sj           # idle again — cancels land on the scheduler handle
        # Free consumed pages (the worker held its own copies).
        for j in range(start, end):
            sj.images[j] = None

        if sj.cancelled or sj.job.status in ('cancelled', 'error'):
            if sj.job.status == 'running':
                sj.job.status = 'cancelled'
                sj.job.done_at = time.monotonic()
            _drop(sj.job.token)
        elif final:
            _drop(sj.job.token)
