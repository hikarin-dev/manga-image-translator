"""Tests for the auxiliary worker pool: joining, preference, concurrency, and failover.

Everything here runs against fake executors — no GPU, no models, no sockets to a real node —
so the whole dispatch path (scheduler → queue → executor → frame adapter → job buffer) is
exercised in-process.

Covered:
  * join handshake — bad token / wrong protocol / version skew are refused; a good one
    registers the node as an executor and unregisters it when the socket closes.
  * aux nodes are preferred over the local worker, and are never offered single-image work.
  * a lone gallery is split across the pool instead of being handed to one machine whole.
  * chunks really do run concurrently (the thing a second executor is worth nothing without).
  * an executor dying mid-chunk re-queues only the pages it never delivered, and a permanently
    broken executor fails the job instead of retrying forever.
"""
import asyncio
import json
import pickle
import types

import pytest

import server.aux_pool as aux_mod
import server.gallery_jobs as gj
from server.instance import ExecutorInstance, executor_instances


# ── harness ──────────────────────────────────────────────────────────────────────────────

def _page_frame(idx: int, token: str = 'tok') -> bytes:
    """The worker's status-5 payload: tokenLen(1) + token + idx(4 BE) + image bytes."""
    t = token.encode()
    return bytes([len(t)]) + t + idx.to_bytes(4, 'big') + b'IMG'


def _summary(failed=()) -> bytes:
    return pickle.dumps({'count': 0, 'failed': list(failed), 'telemetry': {'wall': 0.1, 'emitted': 0}})


class FakeExecutor:
    """Duck-types an executor. `plan` decides what it does with a chunk: emit every page,
    emit some then die, or die immediately."""

    gallery_only = False

    def __init__(self, name, priority=100, emit='all', gate=None):
        self.name = name
        self.priority = priority
        self.busy = False
        self.emit = emit
        self.gate = gate                 # optional asyncio.Event to hold the chunk open
        self.chunks = []                 # (start_idx, page_count) per chunk received
        self.concurrent = 0
        self.max_concurrent = 0

    @property
    def label(self):
        return self.name

    def free_executor(self):
        self.busy = False

    async def sent_gallery_stream(self, images, config, sender, batch_size=0, job_token=""):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.chunks.append(len(images))
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self.emit == 'die':
                raise RuntimeError(f'{self.name} exploded')
            # Never emit past the chunk it was actually given — a real worker cannot, and
            # doing so here would forge page indices belonging to another executor's chunk.
            n = len(images) if self.emit == 'all' else min(self.emit, len(images))
            for i in range(n):
                sender(5, _page_frame(i, job_token))
                await asyncio.sleep(0)
            if self.emit != 'all':
                raise RuntimeError(f'{self.name} died after {n} page(s)')
            sender(0, _summary())
        finally:
            self.concurrent -= 1

    async def cancel_gallery(self, job_token=""):
        pass


@pytest.fixture
def pool():
    """Reset the scheduler, queue and executor registry between tests.

    All three are module-level singletons holding asyncio primitives. A production process has
    exactly one event loop, but each test gets a fresh one from asyncio.run, and an Event binds
    itself to the first loop that touches it — so they have to be rebuilt or the second test to
    run trips 'bound to a different event loop'."""
    from server.myqueue import task_queue, running_galleries
    gj._sched.clear()
    gj._sched_order.clear()
    gj._round_state.clear()
    gj._jobs.clear()
    gj._inflight_chunks = 0
    gj._sched_started = False
    gj._reaper_started = True            # keep the reaper out of these tests
    gj._sched_event = asyncio.Event()
    task_queue.queue.clear()
    task_queue.queue_event = asyncio.Event()
    running_galleries.clear()
    executor_instances.list.clear()
    executor_instances.lock = asyncio.Lock()
    executor_instances.event = asyncio.Event()
    yield executor_instances
    executor_instances.list.clear()
    gj._sched.clear()
    gj._sched_order.clear()


def _submit(pages: int, batch_size: int = 8, token: str = 'tok'):
    """Register a gallery job with the scheduler, as start_gallery_job would."""
    job = gj.create(token)
    job.total = pages
    config = types.SimpleNamespace(translator=types.SimpleNamespace(translator='none'))
    gj.submit(job, types.SimpleNamespace(state=types.SimpleNamespace(client_ip='')),
              [f'page{i}'.encode() for i in range(pages)], config, batch_size,
              lambda summary: json.dumps(summary if isinstance(summary, dict) else {}).encode())
    return job


async def _drain(job, timeout=5.0):
    """Run the scheduler until the job produces its terminal frame."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while job.terminal is None and loop.time() < deadline:
        await asyncio.sleep(0.01)
    return job.terminal


# ── join handshake ───────────────────────────────────────────────────────────────────────

class _FakeWS:
    """Enough of a Starlette WebSocket for handle_join: a scripted inbound queue and a
    record of what was sent back."""

    def __init__(self, inbound, hold=None):
        self.inbound = list(inbound)
        self.sent = []
        self.accepted = False
        self.hold = hold        # when set, stay connected until it fires (else close at EOF)

    async def accept(self):
        self.accepted = True

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def send_bytes(self, data):
        self.sent.append(data)

    async def receive_text(self):
        if not self.inbound:
            raise RuntimeError('closed')
        return self.inbound.pop(0)

    async def receive(self):
        if not self.inbound:
            if self.hold is not None:
                await self.hold.wait()
            return {'type': 'websocket.disconnect'}
        item = self.inbound.pop(0)
        return {'type': 'websocket.receive', 'bytes': item} if isinstance(item, bytes) \
            else {'type': 'websocket.receive', 'text': item}


def _hello(**over):
    base = {'protocol': aux_mod.AUX_PROTOCOL, 'token': 'sekrit', 'name': 'node-a',
            'version': 'v1', 'caps': {'gpu': True}}
    base.update(over)
    return json.dumps(base)


@pytest.mark.parametrize('hello,reason', [
    (_hello(token='wrong'), 'bad join token'),
    (_hello(protocol=999), 'protocol mismatch'),
    (_hello(version='v2'), 'version mismatch'),
])
def test_join_refused(pool, monkeypatch, hello, reason):
    """A node that fails any check is told why and never enters the pool."""
    monkeypatch.setattr(aux_mod, 'JOIN_TOKEN', 'sekrit')
    monkeypatch.setattr(aux_mod, 'code_version', lambda: 'v1')
    ws = _FakeWS([hello])
    asyncio.run(aux_mod.handle_join(ws))
    assert ws.sent[0]['ok'] is False
    assert reason in ws.sent[0]['error']
    assert executor_instances.list == [], 'a refused node must not be registered'


def test_join_refused_when_token_unset(pool, monkeypatch):
    """Aux joining is off unless the operator sets a token — an empty secret must never
    compare equal to an empty offered token."""
    monkeypatch.setattr(aux_mod, 'JOIN_TOKEN', '')
    ws = _FakeWS([_hello(token='')])
    asyncio.run(aux_mod.handle_join(ws))
    assert ws.sent[0]['ok'] is False and 'not accepting' in ws.sent[0]['error']
    assert executor_instances.list == []


def test_join_accepted_then_unregistered_on_close(pool, monkeypatch):
    """A good handshake registers the node; closing the socket takes it back out."""
    monkeypatch.setattr(aux_mod, 'JOIN_TOKEN', 'sekrit')
    monkeypatch.setattr(aux_mod, 'code_version', lambda: 'v1')
    seen = {}

    async def scenario():
        hold = asyncio.Event()
        ws = _FakeWS([_hello()], hold=hold)
        task = asyncio.create_task(aux_mod.handle_join(ws))
        await asyncio.sleep(0.05)
        seen['during'] = list(executor_instances.list)   # node is in the pool while connected
        hold.set()                                       # now drop the socket
        await task
        return ws

    ws = asyncio.run(scenario())
    assert ws.sent[0]['ok'] is True and ws.sent[0]['aux_id'].startswith('aux-')
    assert len(seen['during']) == 1 and seen['during'][0].name == 'node-a'
    assert seen['during'][0].gallery_only is True
    assert executor_instances.list == [], 'a departed node must leave the pool'


def test_version_skew_override(pool, monkeypatch):
    """Skew is refusable by default but must be overridable for deliberate rolling upgrades."""
    monkeypatch.setattr(aux_mod, 'JOIN_TOKEN', 'sekrit')
    monkeypatch.setattr(aux_mod, 'code_version', lambda: 'v1')
    monkeypatch.setattr(aux_mod, 'ALLOW_VERSION_SKEW', True)
    assert aux_mod._validate(json.loads(_hello(version='v2'))) is None


def test_unknown_version_does_not_block(pool, monkeypatch):
    """An export with no git metadata reports 'unknown' — that must warn, not lock the node out."""
    monkeypatch.setattr(aux_mod, 'JOIN_TOKEN', 'sekrit')
    monkeypatch.setattr(aux_mod, 'code_version', lambda: 'unknown')
    assert aux_mod._validate(json.loads(_hello(version='v2'))) is None


# ── selection ────────────────────────────────────────────────────────────────────────────

def test_aux_is_preferred_over_local(pool):
    """Remote capacity is spent before this machine's GPU, whatever the registration order."""
    local = FakeExecutor('local', priority=100)
    remote = FakeExecutor('aux', priority=10)
    pool.register(local)
    pool.register(remote)

    picked = asyncio.run(pool.find_executor(gallery=True))
    assert picked is remote

    # With the preferred one busy, work still flows — it falls back rather than waiting.
    assert asyncio.run(pool.find_executor(gallery=True)) is local


def test_single_image_work_skips_aux_nodes(pool):
    """Aux nodes are gallery-only: the single-image path returns a pickled Context, which we
    won't reconstruct from a machine we don't own."""
    remote = FakeExecutor('aux', priority=10)
    remote.gallery_only = True
    local = FakeExecutor('local', priority=100)
    pool.register(remote)
    pool.register(local)

    assert pool.capacity(gallery=True) == 2
    assert pool.capacity(gallery=False) == 1
    assert asyncio.run(pool.find_executor(gallery=False)) is local


def test_local_executor_defaults_below_aux():
    """The registered local worker must sort after aux without anyone setting it explicitly."""
    assert ExecutorInstance(ip='127.0.0.1', port=5004).priority > aux_mod.AUX_PRIORITY


class RealModelExecutor(ExecutorInstance):
    """Subclasses the REAL pydantic model instead of duck-typing it.

    FakeExecutor above is a plain class and therefore hashable; ExecutorInstance is NOT
    (pydantic v2 sets __hash__ = None on non-frozen models). That difference is exactly what
    hid a crash on every local-worker chunk, so at least one test has to use the real type."""

    async def sent_gallery_stream(self, images, config, sender, batch_size=0, job_token=""):
        for i in range(len(images)):
            sender(5, _page_frame(i, job_token))
            await asyncio.sleep(0)
        sender(0, _summary())

    async def cancel_gallery(self, job_token=""):
        pass


def test_pydantic_executor_is_unhashable():
    """Pins the property that broke things, so a future 'just use a set' change fails loudly
    here rather than at runtime on every gallery."""
    with pytest.raises(TypeError):
        {ExecutorInstance(ip='127.0.0.1', port=5004)}


def test_local_worker_completes_a_gallery(pool):
    """Regression: running_galleries was a set, which cannot hold a pydantic ExecutorInstance.
    Every chunk dispatched to the local worker died with 'unhashable type: ExecutorInstance',
    and the retry path then burned the stall budget and failed the whole job."""
    pool.register(RealModelExecutor(ip='127.0.0.1', port=5004))

    async def scenario():
        job = _submit(pages=16, batch_size=8)
        assert await _drain(job) is not None, 'local worker must be able to finish a gallery'
        return job

    job = asyncio.run(scenario())
    assert job.status == 'done'
    assert job.emitted == 16


def test_mixed_pool_of_aux_and_local(pool):
    """The actual deployment: an aux node alongside the pydantic local worker."""
    from server.myqueue import running_galleries
    local = RealModelExecutor(ip='127.0.0.1', port=5004)
    remote = FakeExecutor('aux', priority=10)
    pool.register(remote)
    pool.register(local)

    async def scenario():
        job = _submit(pages=32, batch_size=8)
        assert await _drain(job) is not None
        return job

    job = asyncio.run(scenario())
    assert job.status == 'done' and job.emitted == 32
    assert remote.chunks, 'the preferred aux node should have taken work'
    assert running_galleries == {}, 'holders must be cleaned up when chunks finish'


def test_unregister_removes_by_identity(pool):
    """Two workers on the same ip:port are equal by value but are different workers —
    unregistering one must not evict the other."""
    a = ExecutorInstance(ip='127.0.0.1', port=5004)
    b = ExecutorInstance(ip='127.0.0.1', port=5004)
    assert a == b, 'precondition: pydantic compares these equal'
    pool.register(a)
    pool.register(b)
    pool.unregister(b)
    assert len(pool.list) == 1 and pool.list[0] is a


# ── concurrency ──────────────────────────────────────────────────────────────────────────

def test_one_gallery_is_split_across_the_pool(pool):
    """A gallery smaller than one chunk must still be spread over the free executors —
    otherwise a second machine adds nothing to the common case."""
    a, b = FakeExecutor('a', priority=10), FakeExecutor('b', priority=20)
    pool.register(a)
    pool.register(b)

    async def scenario():
        job = _submit(pages=32, batch_size=8)
        assert await _drain(job) is not None
        return job

    job = asyncio.run(scenario())
    assert a.chunks and b.chunks, f'both executors should get work, got a={a.chunks} b={b.chunks}'
    assert sum(a.chunks) + sum(b.chunks) == 32, 'every page dispatched exactly once'
    assert job.emitted == 32


def test_chunks_actually_run_in_parallel(pool):
    """The pool has to have more than one chunk in flight at a time. Serially-dispatched
    chunks would still pass the split test above while delivering no speedup at all."""
    gate = asyncio.Event()
    peak = {'n': 0}

    class Counting(FakeExecutor):
        async def sent_gallery_stream(self, images, config, sender, batch_size=0, job_token=""):
            peak['n'] = max(peak['n'], gj._inflight_chunks)
            await super().sent_gallery_stream(images, config, sender, batch_size, job_token)

    a, b = Counting('a', priority=10, gate=gate), Counting('b', priority=20, gate=gate)
    pool.register(a)
    pool.register(b)

    async def scenario():
        job = _submit(pages=32, batch_size=8)
        await asyncio.sleep(0.15)        # let both chunks be dispatched and block on the gate
        inflight_while_blocked = gj._inflight_chunks
        gate.set()
        await _drain(job)
        return inflight_while_blocked

    inflight = asyncio.run(scenario())
    assert inflight >= 2, f'expected concurrent chunks, only {inflight} in flight'
    assert peak['n'] >= 2


# ── failover ─────────────────────────────────────────────────────────────────────────────

def test_dead_executor_requeues_only_undelivered_pages(pool):
    """A node that delivers some pages then drops must not cost those pages, and must not
    make another executor redo them."""
    flaky = FakeExecutor('flaky', priority=10, emit=3)   # 3 pages, then dies
    good = FakeExecutor('good', priority=100)
    pool.register(flaky)
    pool.register(good)

    async def scenario():
        job = _submit(pages=16, batch_size=8)
        assert await _drain(job) is not None, 'job must still finish despite the failure'
        return job

    job = asyncio.run(scenario())
    assert job.status == 'done'
    assert job.emitted == 16, f'every page delivered exactly once, got {job.emitted}'
    assert sum(good.chunks) < 16 + 8, 'the surviving node redid only the undelivered remainder'


def test_permanently_broken_executor_fails_the_job(pool):
    """A node that never delivers anything must not spin the job forever — it fails with an
    error frame after the stall budget."""
    pool.register(FakeExecutor('broken', priority=10, emit='die'))

    async def scenario():
        job = _submit(pages=16, batch_size=8)
        assert await _drain(job, timeout=8.0) is not None
        return job

    job = asyncio.run(scenario())
    assert job.status == 'error'
    assert job.terminal[0] == 2, 'client must be told the job failed'


def test_all_pages_delivered_once_under_failure(pool):
    """The whole point of the retry path: no page silently lost, none delivered twice."""
    pool.register(FakeExecutor('flaky', priority=10, emit=2))
    pool.register(FakeExecutor('good', priority=100))

    async def scenario():
        job = _submit(pages=24, batch_size=8)
        await _drain(job)
        return job

    job = asyncio.run(scenario())
    seen = []
    for frame in job.durable:
        size = int.from_bytes(frame[1:5], 'big')
        data = frame[5:5 + size]
        tlen = data[0]
        seen.append(int.from_bytes(data[1 + tlen:1 + tlen + 4], 'big'))
    assert sorted(seen) == list(range(24)), f'pages delivered: {sorted(seen)}'


# ── AuxInstance frame routing ────────────────────────────────────────────────────────────

def test_frames_route_to_the_right_chunk(pool):
    """Two chunks on one node must not cross-talk, and a dropped socket must fail whatever
    that node still held so the scheduler can re-dispatch it."""
    async def scenario():
        inst = aux_mod.AuxInstance(_FakeWS([]), 'n', 'v1', {})
        got_a, got_b = [], []
        inst._pending[1] = aux_mod._Chunk(lambda s, d: got_a.append((s, d)), 'job-a')
        inst._pending[2] = aux_mod._Chunk(lambda s, d: got_b.append((s, d)), 'job-b')

        inst.on_frame(1, 5, b'first')
        inst.on_frame(2, 5, b'second')
        inst.on_frame(99, 5, b'stale')        # chunk we already gave up on — must be dropped
        inst.on_end(1, None)

        pending_b = inst._pending[2]
        inst.fail_all('socket closed')
        return got_a, got_b, pending_b

    got_a, got_b, pending_b = asyncio.run(scenario())
    assert got_a == [(5, b'first')] and got_b == [(5, b'second')]
    assert pending_b.done.done() and isinstance(pending_b.done.exception(), RuntimeError)


def test_cancel_is_token_scoped(pool):
    """A cancel for one gallery must not abort another running on the same node."""
    async def scenario():
        ws = _FakeWS([])
        inst = aux_mod.AuxInstance(ws, 'n', 'v1', {})
        inst._pending[1] = aux_mod._Chunk(lambda s, d: None, 'job-a')
        inst._pending[2] = aux_mod._Chunk(lambda s, d: None, 'job-b')
        await inst.cancel_gallery('job-b')
        return ws.sent

    sent = asyncio.run(scenario())
    assert sent == [{'type': 'cancel', 'chunk': 2}], f'only job-b should be cancelled, got {sent}'
