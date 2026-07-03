"""Regression tests for the gallery translation pipeline's cross-gallery safety, cancel, and polling.

Covered:
  * progress-queue isolation — a dead/aborted job's undrained frames can NEVER be
    delivered into the next job's stream (the cross-gallery image mix-up).
  * status-5 page frames carry the job token (the second, client-checked guard).
  * /cancel_gallery is token-scoped — a stale cancel can't abort a different gallery.
  * a queued gallery flagged cancelled is dropped before it reaches the GPU.
  * GalleryJob.poll(since) — the server-owned job buffers frames and hands them back past a cursor
    with an authoritative metadata frame.

These exercise share.py / myqueue.py / gallery_jobs.py logic directly with a stub MangaTranslator,
so no GPU, models, or running server are needed.
"""
import asyncio

import pytest
from PIL import Image

import manga_translator.mode.share as share_mod


class _StubManga:
    """Minimal stand-in for MangaTranslator: just enough for MangaShare to wire its hooks
    and for the cancel route to flip flags. No models, no GPU."""
    def __init__(self, params=None):
        self._gallery_cancel = False
        self._gallery_job_token = ''
        self._is_streaming_mode = False
    def add_progress_hook(self, hook):
        self._progress_hook = hook
    def add_page_result_hook(self, hook):
        self._page_result_hook = hook
    def add_page_bubbles_hook(self, hook):
        self._page_bubbles_hook = hook


@pytest.fixture
def share(monkeypatch):
    monkeypatch.setattr(share_mod, 'MangaTranslator', _StubManga)
    # nonce 'None' → check_nonce() becomes a no-op, so the TestClient needs no header.
    return share_mod.MangaShare({'nonce': 'None'})


def _parse_page_frame(frame: bytes):
    assert frame[0] == 5, 'not a status-5 page frame'
    size = int.from_bytes(frame[1:5], 'big')
    data = frame[5:5 + size]
    tlen = data[0]
    token = data[1:1 + tlen].decode('utf-8')
    idx = int.from_bytes(data[1 + tlen:1 + tlen + 4], 'big')
    png = data[1 + tlen + 4:]
    return token, idx, png


def test_progress_queue_isolation(share):
    """Job A renders a page but its consumer dies (frame left undrained). Job B then runs
    on its own fresh queue. B's stream must yield ONLY B's frames — A's leaked frame stays
    orphaned on A's queue. (Under the old single shared queue, B would replay A's page.)"""
    async def scenario():
        prh = share.manga._page_result_hook

        # Job A: render page 5, consumer dies → frame sits in qA, never drained.
        qA = asyncio.Queue()
        share.progress_queue = qA
        share.manga._gallery_job_token = 'gallery-A'
        await prh(5, Image.new('RGB', (2, 2), 'red'))

        # Job B starts with its OWN fresh queue (what /execute now assigns per job).
        qB = asyncio.Queue()
        share.progress_queue = qB
        share.manga._gallery_job_token = 'gallery-B'
        await prh(2, Image.new('RGB', (2, 2), 'blue'))
        qB.put_nowait(b'\x00' + (0).to_bytes(4, 'big'))  # B's terminal summary frame

        frames = [f async for f in share.progress_stream(qB)]
        return qA, frames

    qA, frames = asyncio.run(scenario())

    page_frames = [f for f in frames if f[0] == 5]
    assert len(page_frames) == 1, 'B should see exactly its own one page'
    token, idx, _ = _parse_page_frame(page_frames[0])
    assert token == 'gallery-B' and idx == 2, "B's stream must carry only B's page"
    assert qA.qsize() == 1, "A's leaked frame must stay orphaned, never crossing into B"


def test_page_frame_carries_token(share):
    """Every status-5 page frame is tokenLen + token + index(4 BE) + image bytes (WebP, or PNG
    where the Pillow build lacks WebP)."""
    async def scenario():
        share.progress_queue = asyncio.Queue()
        share.manga._gallery_job_token = 'tok-123'
        await share.manga._page_result_hook(7, Image.new('RGB', (3, 3), 'green'))
        return share.progress_queue.get_nowait()

    frame = asyncio.run(scenario())
    token, idx, img = _parse_page_frame(frame)
    assert token == 'tok-123'
    assert idx == 7
    assert img[:4] == b'RIFF' or img[:8] == b'\x89PNG\r\n\x1a\n', 'payload should be a WebP or PNG image'


def test_worker_cancel_is_token_scoped(share):
    """/cancel_gallery only aborts when the token matches the running job (or is omitted)."""
    from fastapi.testclient import TestClient
    client = TestClient(share.build_app())
    share.manga._gallery_job_token = 'RUNNING'

    # Wrong token → must NOT cancel (this is the bug that mixed galleries up).
    share.manga._gallery_cancel = False
    r = client.post('/cancel_gallery', data={'job_token': 'SOMEONE-ELSE'})
    assert r.json()['cancelling'] is False
    assert share.manga._gallery_cancel is False

    # Matching token → cancels.
    r = client.post('/cancel_gallery', data={'job_token': 'RUNNING'})
    assert share.manga._gallery_cancel is True

    # Omitted token (the cancel backstop) → legacy "cancel whatever is running".
    share.manga._gallery_cancel = False
    r = client.post('/cancel_gallery', data={'job_token': ''})
    assert share.manga._gallery_cancel is True


def test_queued_gallery_cancel_drops_before_dispatch():
    """A queued GalleryQueueElement flagged cancelled reports as 'disconnected', so wait_in_queue
    drops it before spending any GPU time. Crucially a gallery's liveness is NOT the creating
    connection — a still-connected req must NOT read as disconnected (the job is server-owned and
    driven by polls); only an explicit cancel or the reaper sets `cancelled`."""
    from server.myqueue import GalleryQueueElement

    class _FakeReq:
        async def is_disconnected(self):
            return True   # creating client gone — must be ignored; only `cancelled` counts

    async def scenario():
        task = GalleryQueueElement(_FakeReq(), [], None, 0, 'tok')
        before = await task.is_client_disconnected()   # req gone but not cancelled → still live
        task.cancelled = True
        after = await task.is_client_disconnected()
        return before, after

    before, after = asyncio.run(scenario())
    assert before is False, 'a gallery must not die just because its creating connection dropped'
    assert after is True


def test_gallery_job_poll_cursor():
    """Polling model: poll(since) returns a status-7 metadata frame (cursor/status/state/done/total —
    all in the BODY so cross-origin reads work) + the page frames past the cursor + the terminal once
    present. done is the server's authoritative emitted-page count, so the client never tallies it."""
    import json
    import server.gallery_jobs as gj

    def frame(code, body=b''):
        return bytes([code]) + len(body).to_bytes(4, 'big') + body

    def page(idx, tok=b'tok', png=b'PNG'):
        data = bytes([len(tok)]) + tok + idx.to_bytes(4, 'big') + png
        return b'\x05' + len(data).to_bytes(4, 'big') + data

    def parse(body):
        off, meta, pages, terminal = 0, None, 0, False
        while len(body) - off >= 5:
            st = body[off]
            size = int.from_bytes(body[off + 1:off + 5], 'big')
            data = body[off + 5:off + 5 + size]
            off += 5 + size
            if st == 7: meta = json.loads(data)
            elif st == 5: pages += 1
            elif st in (0, 2): terminal = True
        return meta, pages, terminal

    job = gj.GalleryJob('tok'); job.total = 5
    job.put_nowait(frame(1, b'gallery-pre:2/5'))     # progress → last_state (not buffered)
    job.put_nowait(page(0)); job.put_nowait(page(1))  # 2 durable pages

    meta, pages, term = parse(job.poll(0))
    assert meta['cursor'] == 2 and meta['status'] == 'running' and meta['state'] == 'gallery-pre:2/5'
    assert meta['done'] == 2 and meta['total'] == 5 and pages == 2 and not term

    meta2, pages2, _ = parse(job.poll(meta['cursor']))
    assert meta2['cursor'] == 2 and pages2 == 0, 'poll at the cursor returns no new pages'

    job.put_nowait(page(2)); job.put_nowait(frame(0, b'{"count":3,"failed":[]}'))
    meta3, pages3, term3 = parse(job.poll(2))
    assert meta3['cursor'] == 3 and meta3['status'] == 'done' and meta3['done'] == 3
    assert pages3 == 1 and term3, 'poll returns the new page + the terminal summary'


def test_gallery_job_poll_tracks_pipeline_stages():
    """The poll metadata exposes per-stage progress folded from the worker's status-1/3/4 frames,
    so the client can show 'Reading text k/n' → 'Translating batch b/m' → 'Rendering' accurately."""
    import json
    import server.gallery_jobs as gj

    def frame(code, body=b''):
        return bytes([code]) + len(body).to_bytes(4, 'big') + body

    def meta(job):
        body = job.poll(0)
        size = int.from_bytes(body[1:5], 'big')
        return json.loads(body[5:5 + size])

    job = gj.GalleryJob('tok'); job.total = 36
    job.put_nowait(frame(4))                         # dispatched
    job.put_nowait(frame(1, b'gallery-pre:5/36'))
    assert meta(job)['pre'] == 5 and meta(job)['dispatched'] is True
    job.put_nowait(frame(1, b'gallery-pre:36/36'))
    job.put_nowait(frame(1, b'gallery-tl:1/4'))
    job.put_nowait(frame(1, b'gallery-tl-done:1/4'))
    m = meta(job)
    assert m['pre'] == 36 and m['tlStarted'] == 1 and m['tlDone'] == 1 and m['batches'] == 4
