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
import pickle
import re
import time

from server import safe_pickle
from server.instance import executor_instances

logger = logging.getLogger('gallery-jobs')
# The main server (server/main.py) runs under uvicorn and never calls the worker's
# init_logging(), so the root logger stays at WARNING with no INFO console handler — our
# INFO lines (chunk dispatch, and the one folded gallery-level summary) would be dropped
# before printing, leaving only the worker's per-chunk logs visible. Give this module its
# own stdout handler at INFO so those always surface, without mutating global logging.
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(levelname)s: [gallery-jobs] %(message)s'))
    logger.addHandler(_h)
    logger.propagate = False

# Pipeline progress strings the worker emits as status-1 frames (longest alt first so 'tl-done'
# isn't shadowed by 'tl'). All three are in PAGE units: gallery-pre:k/n (pages read),
# gallery-tl:a/n (pages whose translation started), gallery-tl-done:a/n (pages translated).
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
        self.tl_started = 0           # pages whose translation request has started
        self.tl_done = 0              # pages whose translation has finished
        self.batches = 0              # denominator of the tl progress (total pages)
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


# ── multi-part uploads ───────────────────────────────────────────────────────────────────
# Cloudflare's free tier caps a request body at ~100MB, so the app uploads a big gallery as
# several /translate/gallery/start requests sharing one job token (part k of n). Parts
# buffer here and the job is created only when the last one lands; an upload that never
# completes is reaped after UPLOAD_TTL_S.
UPLOAD_TTL_S = 300.0
MAX_PENDING_UPLOADS = 6
MAX_UPLOAD_BUFFER_BYTES = 1024 * 1024 * 1024   # all pending parts together


class _PendingUpload:
    def __init__(self, token: str, parts: int, owner_ip: str):
        self.token = token
        self.parts = parts
        self.owner_ip = owner_ip
        self.got: dict[int, list] = {}
        self.created = time.monotonic()

    @property
    def pages(self) -> int:
        return sum(len(imgs) for imgs in self.got.values())

    @property
    def bytes(self) -> int:
        return sum(len(b) for imgs in self.got.values() for b in imgs)


_uploads: dict[str, _PendingUpload] = {}


def add_upload_part(token: str, part: int, parts: int, images: list, owner_ip: str = '',
                    max_pages: int = 0) -> tuple[str, list | None]:
    """Buffer one upload part; idempotent per (token, part). Returns (status, images):
    'exists' (job already created — a retry of an already-processed final part), 'busy'
    (buffer full), 'too_many_pages', 'pending' (more parts expected), or 'done' with the
    full gallery assembled in part order."""
    up = _uploads.get(token)
    if up is None:
        if get(token) is not None:
            return 'exists', None
        total_bytes = sum(u.bytes for u in _uploads.values())
        if len(_uploads) >= MAX_PENDING_UPLOADS or total_bytes >= MAX_UPLOAD_BUFFER_BYTES:
            return 'busy', None
        up = _uploads[token] = _PendingUpload(token, max(1, int(parts)), owner_ip)
        _ensure_reaper()
    up.got[int(part)] = images
    if max_pages and up.pages > max_pages:
        _uploads.pop(token, None)
        return 'too_many_pages', None
    if len(up.got) < up.parts:
        return 'pending', None
    _uploads.pop(token, None)
    ordered: list = []
    for k in sorted(up.got):
        ordered.extend(up.got[k])
    return 'done', ordered


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
        for token, up in list(_uploads.items()):
            if (now - up.created) > UPLOAD_TTL_S:
                _uploads.pop(token, None)
                logger.warning(
                    f'Gallery upload {token[:8]}… reaped: incomplete after {UPLOAD_TTL_S:.0f}s '
                    f'({len(up.got)}/{up.parts} parts, {up.pages} pages buffered)')


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
# How many times a job may have a chunk come back having delivered no pages at all before we
# stop retrying and fail it. Guards against a permanently broken executor spinning forever;
# any chunk that delivers even one page resets the count.
MAX_CHUNK_STALLS = 3

_UNCHUNKABLE_TRANSLATORS = ('chatgpt', 'chatgpt_2stage')


class _SchedJob:
    """Scheduler-side state for one gallery job. Also serves as the job's cancel handle
    while no chunk is in flight (the reaper/cancel path sets `.cancelled` on job.task)."""

    def __init__(self, job, req, images, config, batch_size, transform, source_url=''):
        self.job = job
        self.req = req
        try:
            self.owner_ip = str(getattr(req.state, 'client_ip', '') or '')  # set by server.edge
            # WHICH access key authenticated this job — the name, never the secret.
            self.owner_key = str(getattr(req.state, 'key', '') or '')
        except Exception:
            self.owner_ip = ''
            self.owner_key = ''
        # Client-supplied label for where this gallery came from. Opaque to the server: stored
        # and shown to a local operator, never parsed or acted on.
        self.source_url = str(source_url or '')[:2048]
        self.images = images
        self.config = config
        self.batch_size = int(batch_size) if batch_size else 0
        self.transform = transform
        self.total = len(images)
        self.next_page = 0
        self.failed: list[int] = []
        self.cancelled = False
        # Chunks of one job can run on several executors at once, so outstanding work is not a
        # single cursor. `retry` holds page ranges an executor dropped mid-chunk (re-dispatched
        # ahead of any fresh pages), `inflight` the chunks running right now, and `emitted` the
        # job-absolute pages already delivered. A job is complete only when all three are clear.
        self.retry: list[tuple[int, int]] = []
        self.inflight: set = set()
        self.emitted: set[int] = set()
        self.stalls = 0               # consecutive chunk failures that delivered no page at all
        # Gallery-level telemetry, folded from each chunk's summary so a multi-chunk
        # job reports ONE final result instead of e.g. 40/40 followed by 19/19.
        self.submitted_at = time.monotonic()
        self.chunks_done = 0
        self.tel_wall = 0.0                    # summed chunk walls (compute time, excludes queue gaps)
        self.tel_emitted = 0
        self.tel_study_meta = 0
        self.tel_study_image = 0
        self.tel_bubbles = 0
        self.tel_cancelled = False
        self.tel_stages: dict[str, float] = {}
        self.tel_waits: dict[str, float] = {}
        self.tel_gpu: list[tuple[float, float]] = []   # (avg, chunk wall) → wall-weighted average
        self.tel_cpu: list[tuple[float, float]] = []
        self.tel_gpu_max = 0.0
        self.tel_cpu_max = 0.0
        self.tel_vram_max = 0.0
        self.tel_vram_total = 0.0
        # LLM request/token accounting (summed across chunks; cost is additive, max_wall is a max).
        self.tel_llm_requests = 0
        self.tel_llm_in = 0
        self.tel_llm_out = 0
        self.tel_llm_cache_hit = 0
        self.tel_llm_cache_miss = 0
        self.tel_llm_cost = 0.0
        self.tel_llm_max_wall = 0.0

    def fold_telemetry(self, tel: dict) -> None:
        if not isinstance(tel, dict):
            return
        self.chunks_done += 1
        llm = tel.get('llm') or {}
        if llm:
            self.tel_llm_requests += int(llm.get('requests', 0))
            self.tel_llm_in += int(llm.get('in', 0))
            self.tel_llm_out += int(llm.get('out', 0))
            self.tel_llm_cache_hit += int(llm.get('cache_hit', 0))
            self.tel_llm_cache_miss += int(llm.get('cache_miss', 0))
            self.tel_llm_cost += float(llm.get('cost', 0.0))
            self.tel_llm_max_wall = max(self.tel_llm_max_wall, float(llm.get('max_wall', 0.0)))
        wall = float(tel.get('wall') or 0.0)
        self.tel_wall += wall
        self.tel_emitted += int(tel.get('emitted') or 0)
        self.tel_study_meta += int(tel.get('study_meta') or 0)
        self.tel_study_image += int(tel.get('study_image') or 0)
        self.tel_bubbles += int(tel.get('bubbles') or 0)
        self.tel_cancelled = self.tel_cancelled or bool(tel.get('cancelled'))
        for k, v in (tel.get('stage_times') or {}).items():
            self.tel_stages[k] = self.tel_stages.get(k, 0.0) + float(v)
        for k, v in (tel.get('queue_wait') or {}).items():
            self.tel_waits[k] = self.tel_waits.get(k, 0.0) + float(v)
        if tel.get('gpu_avg') or tel.get('gpu_max'):
            self.tel_gpu.append((float(tel.get('gpu_avg') or 0.0), wall))
            self.tel_gpu_max = max(self.tel_gpu_max, float(tel.get('gpu_max') or 0.0))
        if tel.get('cpu_avg') or tel.get('cpu_max'):
            self.tel_cpu.append((float(tel.get('cpu_avg') or 0.0), wall))
            self.tel_cpu_max = max(self.tel_cpu_max, float(tel.get('cpu_max') or 0.0))
        self.tel_vram_max = max(self.tel_vram_max, float(tel.get('vram_max') or 0.0))
        self.tel_vram_total = max(self.tel_vram_total, float(tel.get('vram_total') or 0.0))

    def summary_line(self) -> str:
        """The ONE gallery-level completion summary, folded across every chunk of this
        job_token. Reported as whole-gallery totals (59/59, not 40/40 then 19/19)."""
        def wavg(pairs):
            tw = sum(w for _, w in pairs)
            return (sum(a * w for a, w in pairs) / tw) if tw else 0.0
        since_submit = time.monotonic() - self.submitted_at
        failed = len(set(self.failed))
        summed = sum(self.tel_stages.values())
        # since_submit is real end-to-end (includes any scheduler queue waits between chunks);
        # tel_wall is summed on-GPU compute. Throughput uses the real wall; overlap uses compute.
        wall = since_submit or self.tel_wall
        stages = ', '.join(f'{k}={v:.1f}' for k, v in sorted(self.tel_stages.items(), key=lambda kv: -kv[1])) or 'n/a'
        waits = ', '.join(f'{k}={v:.1f}' for k, v in sorted(self.tel_waits.items(), key=lambda kv: -kv[1])) or 'n/a'
        gpu = (f'GPU avg={wavg(self.tel_gpu):.0f}% max={self.tel_gpu_max:.0f}%' if self.tel_gpu else 'GPU n/a')
        vram = (f'VRAM max={self.tel_vram_max:.0f}MB' + (f'/{self.tel_vram_total:.0f}MB' if self.tel_vram_total else '')
                if self.tel_vram_max else 'VRAM n/a')
        cpu = (f'CPU avg={wavg(self.tel_cpu):.0f}% max={self.tel_cpu_max:.0f}%' if self.tel_cpu else 'CPU n/a')
        llm_line = ''
        if self.tel_llm_requests:
            in_tok, out_tok = self.tel_llm_in, self.tel_llm_out
            hit = self.tel_llm_cache_hit
            hit_pct = (100.0 * hit / in_tok) if in_tok else 0.0
            llm_line = (
                f'\n  LLM: {self.tel_llm_requests} requests, in={in_tok} (cache hit {hit_pct:.0f}%) '
                f'out={out_tok} tok, slowest request={self.tel_llm_max_wall:.1f}s, ~${self.tel_llm_cost:.4f} est')
        return (
            f'Gallery job {self.job.token[:8]}… summary: {self.tel_emitted}/{self.total} pages emitted, '
            f'{failed} failed, {self.chunks_done} chunk(s)'
            + (' — CANCELLED' if self.tel_cancelled else '') + '\n'
            f'  compute={self.tel_wall:.1f}s wall={since_submit:.1f}s '
            f'overlap={(summed / self.tel_wall if self.tel_wall else 1):.2f}x '
            f'pages/sec={(self.tel_emitted / wall if wall else 0):.2f} '
            f'per_page={(wall / self.total if self.total else 0):.2f}s '
            f'study_pages(meta={self.tel_study_meta}, image={self.tel_study_image}) bubbles={self.tel_bubbles}'
            + llm_line + '\n'
            f'  stages(s): {stages}\n'
            f'  queue_wait(s): {waits}\n'
            f'  {gpu} | {vram} | {cpu}')

    @property
    def has_work(self) -> bool:
        """Pages still to hand out — either a dropped range to redo or fresh pages."""
        return bool(self.retry) or self.next_page < self.total

    @property
    def complete(self) -> bool:
        return not self.has_work and not self.inflight

    def take_range(self, solo: bool, split: int = 1) -> tuple[int, int]:
        """Claim the next page range to dispatch. Dropped ranges go first so a failed chunk is
        redone promptly rather than after the whole rest of the gallery."""
        if self.retry:
            return self.retry.pop(0)
        start = self.next_page
        end = start + self.chunk_size(solo, split)
        self.next_page = end
        return start, end

    def chunk_size(self, solo: bool, split: int = 1) -> int:
        remaining = self.total - self.next_page
        tr = str(getattr(self.config.translator, 'translator', ''))
        if tr in _UNCHUNKABLE_TRANSLATORS:
            return remaining
        cap = self.batch_size if self.batch_size > 0 else remaining
        base = SOLO_CHUNK if solo else SHARED_CHUNK
        pages = max(cap, (base // max(1, cap)) * max(1, cap))
        if split > 1:
            # Several executors are free and this job is the only claimant. A typical gallery
            # is smaller than one chunk, so without this it would be handed to a single machine
            # whole and the rest of the pool would idle through it. Still whole multiples of the
            # batch cap, so translation batches group exactly as in an unchunked run. (With
            # batch_size 0 the cap IS the remainder, so this leaves that case untouched.)
            share = -(-remaining // split)
            share = max(cap, (share // max(1, cap)) * max(1, cap))
            pages = min(pages, share)
        return min(pages, remaining)


_sched: dict[str, _SchedJob] = {}
_sched_order: list[str] = []
_sched_event = asyncio.Event()
_sched_started = False


def live_job_count() -> int:
    """Jobs the scheduler owns (running + waiting) — the admission-control depth."""
    return len(_sched)


def live_jobs_for_ip(ip: str) -> int:
    """Live jobs plus in-progress multi-part uploads owned by one external client."""
    return (sum(1 for sj in _sched.values() if getattr(sj, 'owner_ip', '') == ip)
            + sum(1 for up in _uploads.values() if up.owner_ip == ip))


def queue_snapshot() -> dict:
    live = [t for t in _sched_order if t in _sched]
    return {'live_jobs': len(live), 'active': min(len(live), ACTIVE_MAX),
            'waiting': max(0, len(live) - ACTIVE_MAX), 'uploads_pending': len(_uploads)}


def _record(sj: _SchedJob, cancelled: bool = False) -> None:
    """Fold a retired job into the metrics exactly once (a cancel can race the final
    chunk's summary — whichever path retires the job first wins)."""
    if getattr(sj, '_recorded', False):
        return
    sj._recorded = True
    try:
        from server import stats
        stats.record_job(sj, cancelled=cancelled)
    except Exception:
        pass


def submit(job: GalleryJob, req, images, config, batch_size, transform, source_url='') -> None:
    """Register a gallery job with the scheduler (replaces enqueueing it whole)."""
    global _sched_started
    sj = _SchedJob(job, req, images, config, batch_size, transform, source_url)
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
    # A job whose pages are all handed out is still active (its chunks are running) but has
    # nothing left to give — skip it so it can't consume a rotation slot or an executor.
    active = [t for t in active if _sched[t].has_work]
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


def _make_adapter(sj: _SchedJob, chunk_start: int, state: dict):
    """Per-chunk notify sink: rewrites chunk-local frames into job-absolute ones before
    they land in the job buffer, and merges intermediate summaries. Never emits the job's
    terminal frame — with chunks in flight on several executors, only _maybe_finish knows
    when the last page has actually landed."""
    from server.streaming import notify

    def adapter(code: int, data: bytes) -> None:
        if code in (5, 6):
            # tokenLen(1) + token + idx(4 BE) + payload → add the chunk's page offset.
            tlen = data[0]
            b = 1 + tlen
            idx = int.from_bytes(data[b:b + 4], 'big') + chunk_start
            data = data[:b] + idx.to_bytes(4, 'big') + data[b + 4:]
            if code == 5:
                # Delivered pages: what a failed chunk must NOT redo, and what tells the
                # scheduler the job is done.
                sj.emitted.add(idx)
            notify(code, data, sj.transform, sj.job)
        elif code == 1:
            s = data.decode('utf-8', 'ignore')
            mm = _STATE_RE.match(s)
            if mm:
                # All three progress kinds are page-based (a/n within the chunk) —
                # offset into job-absolute pages uniformly.
                kind, a = mm.group(1), int(mm.group(2))
                s = f'gallery-{kind}:{chunk_start + a}/{sj.total}'
            notify(1, s.encode('utf-8'), sj.transform, sj.job)
        elif code == 0:
            try:
                # An executor may be an aux node on someone else's machine — never hand its
                # bytes to bare pickle.loads (see server.safe_pickle).
                summary = safe_pickle.loads(data)
            except Exception as e:
                logger.warning(f'Gallery job {sj.job.token[:8]}… discarding unreadable chunk summary: {e}')
                summary = {}
            if isinstance(summary, dict):
                sj.failed.extend(chunk_start + int(i) for i in summary.get('failed', []))
                sj.fold_telemetry(summary.get('telemetry'))
            # Chunk summaries are only ever folded. The job's own terminal frame is emitted by
            # _maybe_finish once every chunk — including any that had to be redone — is in.
        elif code == 2:
            # An executor failing is NOT the job's terminal state — these pages can still go to
            # another one. wait_in_queue reports a dead executor by calling us with a status-2
            # frame, and forwarding that would mark the whole gallery errored and finished
            # before failover ever ran. Record it for _run_chunk; only _fail() ends the job.
            state['error'] = data.decode('utf-8', 'ignore') or 'executor failed'
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


_inflight_chunks = 0    # chunks running across the whole pool, the concurrency accounting


def _maybe_finish(sj: _SchedJob) -> None:
    """Emit the job's one terminal frame, once every page has actually landed. With chunks in
    flight on several executors the last chunk to return is not necessarily the one carrying
    the last page, so completion is decided here rather than at dispatch time."""
    if sj.job.status != 'running' or not sj.complete:
        return
    from server.streaming import notify
    logger.info(sj.summary_line())
    _record(sj)
    notify(0, pickle.dumps({'count': sj.total, 'failed': sorted(set(sj.failed))}), sj.transform, sj.job)
    _drop(sj.job.token)


def _fail(sj: _SchedJob, detail: str) -> None:
    """Give up on a job whose pages can't be delivered. Cancels its siblings first so no chunk
    keeps running for a job the client has already been told failed."""
    from server.streaming import notify
    sj.cancelled = True
    logger.error(f'Gallery job {sj.job.token[:8]}… giving up: {detail}')
    notify(2, f'Translation failed: {detail}'.encode('utf-8'), sj.transform, sj.job)
    _record(sj)
    _drop(sj.job.token)


def _dispatch(sj: _SchedJob, start: int, end: int):
    """Register a chunk and queue it. The accounting happens here, synchronously, so a chunk
    counts as in flight the moment it is scheduled rather than when its task first runs — a
    sibling finishing in that gap would otherwise see an empty inflight set and declare the
    whole job complete."""
    from server.myqueue import task_queue, GalleryQueueElement
    global _inflight_chunks
    element = GalleryQueueElement(sj.req, sj.images[start:end], sj.config, sj.batch_size,
                                  sj.job.token, parent=sj)
    element.cancelled = sj.cancelled
    sj.inflight.add(element)
    _inflight_chunks += 1
    task_queue.add_task(element)
    return element


async def _run_chunk(sj: _SchedJob, element, start: int, end: int) -> None:
    """Own one chunk end to end: await it, then either retire its pages or hand the
    un-delivered remainder back to the scheduler for another executor."""
    from server.myqueue import wait_in_queue
    global _inflight_chunks

    state: dict = {}
    try:
        await wait_in_queue(element, _make_adapter(sj, start, state))
    except Exception as e:
        state['error'] = str(e) or e.__class__.__name__
    finally:
        sj.inflight.discard(element)
        _inflight_chunks -= 1
        if element.cancelled:
            sj.cancelled = True

    # A page is ACCOUNTED FOR if it was delivered or if the worker reported it as failed. The
    # distinction matters: a page the pipeline cannot translate is never emitted as a frame,
    # it only appears in the chunk summary's `failed` list. Counting solely delivered pages
    # would read that as "the executor died here", re-translate every good page after it, and
    # burn the stall budget until the job errored — losing a whole gallery over one bad page.
    # Failed pages are reported to the client in the job's terminal summary instead.
    failed = set(sj.failed)
    done_upto = start
    while done_upto < end and (done_upto in sj.emitted or done_upto in failed):
        done_upto += 1
    for j in range(start, done_upto):
        sj.images[j] = None

    if not (sj.cancelled or sj.job.status in ('cancelled', 'error')):
        if done_upto < end:
            # A chunk that accounted for nothing counts as a stall; any progress clears it.
            sj.stalls = 0 if done_upto > start else sj.stalls + 1
            if sj.stalls > MAX_CHUNK_STALLS:
                _fail(sj, state.get('error') or 'executor made no progress')
                _sched_event.set()
                return
            sj.retry.insert(0, (done_upto, end))
            got = len([i for i in range(start, end) if i in sj.emitted])
            bad = len([i for i in range(start, end) if i in failed])
            logger.warning(
                f'Gallery job {sj.job.token[:8]}… pages {done_upto}-{end - 1} not delivered '
                f'({state.get("error") or "chunk ended early"}; {got} delivered, {bad} reported '
                f'failed of {end - start}) — re-queueing for another executor')
        else:
            sj.stalls = 0
            _maybe_finish(sj)

    _sched_event.set()


async def _sched_loop() -> None:
    global _inflight_chunks
    while True:
        _sched_event.clear()

        # Retire cancelled/errored jobs. One with chunks still unwinding stays until they
        # return, so its executors are released before the job is forgotten.
        for token in list(_sched_order):
            sj = _sched.get(token)
            if sj is None:
                _drop(token)
            elif sj.cancelled or sj.job.status in ('cancelled', 'error'):
                if sj.job.status == 'running':
                    sj.job.status = 'cancelled'
                    sj.job.done_at = time.monotonic()
                if not sj.inflight:
                    _record(sj, cancelled=(sj.job.status != 'error'))
                    _drop(token)

        # Fill the pool: keep dispatching while an executor could still pick work up. Without
        # this the whole pool is worth one executor, since a second one would never be handed
        # a chunk until the first had finished its own.
        while _inflight_chunks < max(1, executor_instances.capacity()):
            sj = _pick_next()
            if sj is None:
                break
            live = [t for t in _sched_order if t in _sched]
            claimants = [t for t in live[:ACTIVE_MAX] if _sched[t].has_work]
            solo = len(claimants) <= 1
            # When one job has the pool to itself, size its chunks to spread over the whole
            # pool; when jobs are already competing they spread across executors on their own.
            split = max(1, executor_instances.capacity()) if solo else 1
            start, end = sj.take_range(solo, split)
            end = min(end, sj.total)
            if end <= start:
                continue
            element = _dispatch(sj, start, end)
            if len(live) > 1 or start > 0 or _inflight_chunks > 1:
                logger.info(
                    f'Gallery job {sj.job.token[:8]}… chunk {start}-{end - 1}/{sj.total} '
                    f'({len(live)} live, {_inflight_chunks} chunk(s) in flight, '
                    f'{max(0, len(live) - ACTIVE_MAX)} waiting)')
            asyncio.create_task(_run_chunk(sj, element, start, end))

        # Wake on a submission or a finished chunk; also tick so waiting-list positions and
        # newly-joined aux capacity are picked up.
        #
        # asyncio.wait rather than wait_for: wait_for can report an EXTERNAL cancellation as
        # TimeoutError, which the `pass` below would swallow — leaving this loop immortal and
        # hanging server shutdown. asyncio.wait does no such conversion, so a real cancel still
        # propagates and a timeout just returns.
        waiter = asyncio.ensure_future(_sched_event.wait())
        try:
            await asyncio.wait({waiter}, timeout=5.0)
        finally:
            waiter.cancel()
