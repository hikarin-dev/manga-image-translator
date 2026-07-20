"""Auxiliary worker node — the `--aux` side.

Run this on any spare machine:

    python server/main.py --aux --join https://translate.example.com --aux-token <secret>

It starts a normal translator worker bound to this machine's loopback, dials the main
server, and relays gallery chunks between the two. Nothing listens on a public interface,
so the node needs no tunnel, no port forward, and no inbound firewall rule.

The node is a thin relay by design: chunks arrive already pickled in exactly the shape the
worker's /execute/translate_gallery_stream expects, so they are forwarded verbatim and the
frames that come back are forwarded verbatim too. No translation logic lives here, which is
what keeps an aux node from drifting away from the main server's behaviour.
"""
import asyncio
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from urllib.parse import urlparse, urlunparse

import aiohttp

logger = logging.getLogger('aux-agent')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(levelname)s: [aux] %(message)s'))
    logger.addHandler(_h)
    logger.propagate = False

# A whole chunk (up to SOLO_CHUNK pages of image bytes) arrives as one message, so the
# client's 1MB default would reject it outright. Bounded rather than unlimited so a bug on
# either side can't drive this process out of memory.
MAX_MESSAGE_BYTES = 256 * 1024 * 1024

RECONNECT_MIN_S = 2.0
RECONNECT_MAX_S = 30.0
WORKER_READY_TIMEOUT_S = 900.0


def join_url(base: str) -> str:
    """http(s)://host → ws(s)://host/aux/join. Accepts a ws(s) URL unchanged."""
    u = urlparse(base if '://' in base else 'https://' + base)
    scheme = {'http': 'ws', 'https': 'wss'}.get(u.scheme, u.scheme)
    path = u.path.rstrip('/')
    if not path.endswith('/aux/join'):
        path += '/aux/join'
    return urlunparse((scheme, u.netloc, path, '', '', ''))


def default_name() -> str:
    """What this node calls itself in the main server's log and roster."""
    try:
        return os.uname().nodename          # POSIX
    except AttributeError:
        return os.getenv('COMPUTERNAME') or 'aux'


def capabilities() -> dict:
    """Best-effort hardware report, so the operator can see what joined without logging in."""
    caps = {'gpu': False, 'vram_mb': 0}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            caps.update(gpu=True, vram_mb=int(props.total_memory / (1024 * 1024)), device=props.name)
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            caps.update(gpu=True, device='mps')
    except Exception:
        pass
    return caps


def spawn_worker(port: int, args) -> subprocess.Popen:
    """Start the translator worker on loopback. Same invocation the main server uses for its
    own worker, minus the executor registration — here the agent is the only caller.

    An aux node is somebody's spare desktop, not an operator console, so it stays quiet:

      * `--verbose` is never forwarded. On the worker that flag writes every intermediate
        pipeline image plus final.png into result/ for each page — gigabytes of someone
        else's manga accumulating on a machine that is only lending its GPU.
      * the worker's own stdout/stderr (model loading, per-page OCR, translation chatter)
        goes to logs/aux-worker.log instead of the console, which leaves only this agent's
        one line per chunk visible. Pass --verbose to the AGENT to watch it live instead;
        that still doesn't turn on the worker's result/ dumping.
    """
    cmds = [sys.executable, '-m', 'manga_translator', 'shared',
            '--host', '127.0.0.1', '--port', str(port), '--nonce', 'None']
    for flag, attr in (('--use-gpu', 'use_gpu'), ('--use-gpu-limited', 'use_gpu_limited'),
                       ('--ignore-errors', 'ignore_errors')):
        if getattr(args, attr, False):
            cmds.append(flag)
    if getattr(args, 'models_ttl', 0):
        cmds.append('--models-ttl=%s' % args.models_ttl)
    if getattr(args, 'context_size', 0):
        cmds.append('--context-size=%s' % args.context_size)
    if getattr(args, 'pre_dict', None):
        cmds.extend(['--pre-dict', args.pre_dict])
    if getattr(args, 'post_dict', None):
        cmds.extend(['--post-dict', args.post_dict])

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sink = None
    if not getattr(args, 'verbose', False):
        log_dir = os.path.join(root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, 'aux-worker.log')
        # Kept, not discarded: when a node won't start, this file is the only diagnosis.
        sink = open(path, 'ab', buffering=0)
        logger.info(f'worker output -> {path}')
    logger.info(f'starting local worker on 127.0.0.1:{port}')
    proc = subprocess.Popen(cmds, cwd=root, stdout=sink,
                            stderr=subprocess.STDOUT if sink is not None else None)
    proc._aux_log = sink            # keep the handle alive for the process's lifetime
    return proc


async def wait_for_worker(worker_url: str, proc: subprocess.Popen) -> bool:
    """Block until the worker answers. First launch downloads models, so this can legitimately
    take many minutes — we must not join the pool and accept a chunk before we can serve it."""
    deadline = time.monotonic() + WORKER_READY_TIMEOUT_S
    async with aiohttp.ClientSession() as s:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                logger.error(f'local worker exited with code {proc.returncode} before becoming ready')
                return False
            try:
                async with s.get(worker_url + '/is_locked', timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2.0)
    logger.error('local worker did not become ready in time')
    return False


class _Relay:
    """One connected session: forwards chunks to the local worker and frames back up."""

    def __init__(self, ws, worker_url: str):
        self.ws = ws
        self.worker_url = worker_url
        self.jobs: dict[int, str] = {}      # chunk id → job token, for token-scoped cancels
        self.tasks: dict[int, asyncio.Task] = {}

    async def handle(self, message) -> None:
        if isinstance(message, bytes):
            if len(message) < 4:
                return
            cid = int.from_bytes(message[:4], 'big')
            self.tasks[cid] = asyncio.create_task(self._run(cid, message[4:]))
            return
        try:
            ev = json.loads(message)
        except Exception:
            return
        if ev.get('type') == 'cancel':
            await self._cancel(int(ev.get('chunk') or 0))

    async def _cancel(self, cid: int) -> None:
        token = self.jobs.get(cid)
        if token is None:
            return
        logger.info(f'chunk {cid}: cancel requested')
        from server.sent_data_internal import post_cancel
        await post_cancel(self.worker_url + '/cancel_gallery', token)

    async def _run(self, cid: int, payload: bytes) -> None:
        from server.sent_data_internal import fetch_gallery_stream
        error = None
        frames: asyncio.Queue = asyncio.Queue()
        pump = asyncio.create_task(self._pump(cid, frames))
        try:
            attrs = pickle.loads(payload)
            self.jobs[cid] = attrs.get('job_token', '')
            pages = len(attrs.get('images') or [])
            logger.info(f'chunk {cid}: {pages} page(s) received, dispatching to local worker')
            started = time.monotonic()
            await fetch_gallery_stream(
                self.worker_url + '/execute/translate_gallery_stream',
                attrs['images'], attrs['config'],
                lambda status, data: frames.put_nowait((status, data)),
                attrs.get('batch_size', 0), attrs.get('job_token', ''))
            logger.info(f'chunk {cid}: done in {time.monotonic() - started:.1f}s')
        except Exception as e:
            error = str(e) or e.__class__.__name__
            logger.error(f'chunk {cid}: failed — {error}')
        finally:
            await frames.put(None)        # drain sentinel: every frame is sent before 'end'
            try:
                await pump
            except Exception:
                pass
            self.jobs.pop(cid, None)
            self.tasks.pop(cid, None)
        try:
            await self.ws.send(json.dumps({"type": "end", "chunk": cid, "error": error}))
        except Exception:
            pass

    async def _pump(self, cid: int, frames: asyncio.Queue) -> None:
        head = cid.to_bytes(4, 'big')
        while True:
            item = await frames.get()
            if item is None:
                return
            status, data = item
            await self.ws.send(head + bytes([status]) + (data or b''))

    def abort(self) -> None:
        for task in list(self.tasks.values()):
            task.cancel()


async def session(url: str, hello: dict, worker_url: str) -> bool:
    """One connection attempt. Returns True if the rejection was fatal (retrying cannot fix
    it — wrong token, wrong protocol, wrong version), False to reconnect."""
    import websockets
    async with websockets.connect(url, max_size=MAX_MESSAGE_BYTES,
                                  ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(hello))
        reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if not reply.get('ok'):
            logger.error(f'join refused: {reply.get("error")}')
            return True
        logger.info(f'joined {url} as {reply.get("aux_id")} — waiting for work')
        relay = _Relay(ws, worker_url)
        try:
            async for message in ws:
                await relay.handle(message)
        finally:
            relay.abort()
    return False


async def run(args) -> int:
    token = (getattr(args, 'aux_token', '') or os.getenv('MT_AUX_TOKEN') or '').strip()
    if not args.join:
        logger.error('--aux needs --join <main server url>')
        return 2
    if not token:
        logger.error('--aux needs --aux-token (or MT_AUX_TOKEN in .env)')
        return 2

    url = join_url(args.join)
    worker_port = args.port + 1
    worker_url = f'http://127.0.0.1:{worker_port}'
    proc = spawn_worker(worker_port, args)

    try:
        if not await wait_for_worker(worker_url, proc):
            return 1
        from server.aux_pool import AUX_PROTOCOL, code_version
        hello = {'protocol': AUX_PROTOCOL, 'token': token,
                 'name': args.aux_name or default_name(),
                 'version': code_version(), 'caps': capabilities()}
        logger.info(f'local worker ready; joining {url} as "{hello["name"]}" '
                    f'(version {hello["version"]}, caps {hello["caps"]})')

        backoff = RECONNECT_MIN_S
        while True:
            if proc.poll() is not None:
                logger.error(f'local worker died (code {proc.returncode}) — stopping')
                return 1
            try:
                if await session(url, hello, worker_url):
                    return 1
                logger.warning('disconnected from main server')
                backoff = RECONNECT_MIN_S
            except Exception as e:
                logger.warning(f'connection failed ({e}); retrying in {backoff:.0f}s')
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)
    finally:
        logger.info('shutting down local worker')
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        if getattr(proc, '_aux_log', None) is not None:
            proc._aux_log.close()
