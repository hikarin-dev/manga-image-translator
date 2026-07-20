"""Auxiliary worker pool — main-server side.

(Named aux_pool, not aux: AUX is a reserved DOS device name, so a file called aux.py is
unopenable by git and many tools on Windows even though Python imports it happily.)

An aux node is another machine running this same repo with `--aux`. It dials IN to us over
a WebSocket and is then handed gallery chunks exactly as if it were a local executor. The
node never accepts a connection, so it needs no tunnel, no port forward, and no inbound
firewall rule — the standard agent-pull shape (kubeadm join / swarm join / CI runners).

Why pull rather than us dialing the node: adding capacity should cost one command on the
new machine, not a hostname + certificate + inbound exposure per node.

Wire protocol (AUX_PROTOCOL), after the socket is accepted:

  aux → us   text    {"protocol":1,"token":…,"name":…,"version":…,"caps":{…}}   (once)
  us  → aux  text    {"ok":true,"aux_id":…}  |  {"ok":false,"error":…} then close
  us  → aux  binary  chunk_id(4 BE) + pickle({images, config, batch_size, job_token})
  us  → aux  text    {"type":"cancel","chunk":id}
  aux → us   binary  chunk_id(4 BE) + status(1) + frame payload
  aux → us   text    {"type":"end","chunk":id,"error":null|"…"}

The frame payload is byte-identical to what a local worker streams back, so everything
downstream (page-index rewriting, telemetry folding, the poll buffer) is unchanged and
cannot tell the difference. Frames are deserialized through server.safe_pickle — an aux
node is not a trust boundary we control.
"""
import asyncio
import hmac
import itertools
import json
import logging
import os
import pickle
import subprocess
import time

from server.instance import executor_instances

logger = logging.getLogger('aux')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(levelname)s: [aux] %(message)s'))
    logger.addHandler(_h)
    logger.propagate = False

# Bump when the wire format above changes incompatibly. A mismatch is always fatal —
# there is no negotiation, because a half-understood chunk is worse than no aux node.
AUX_PROTOCOL = 1

JOIN_TOKEN = (os.getenv('MT_AUX_TOKEN') or '').strip()
# Aux nodes are preferred over the local worker (lower sorts first in Executors), so remote
# capacity is spent before this machine's GPU. Tunable in case a node is slower than local.
AUX_PRIORITY = int(os.getenv('MT_AUX_PRIORITY', '10'))
ALLOW_VERSION_SKEW = (os.getenv('MT_AUX_ALLOW_VERSION_SKEW') or '').strip().lower() in ('1', 'true', 'yes')

_ids = itertools.count(1)


def code_version() -> str:
    """Identity of the code+models this process is running, so a node on a different commit
    is refused instead of silently producing differently-rendered pages inside the SAME
    gallery as the local worker. Best-effort: an export with no git metadata reports
    'unknown', which downgrades the check to a warning rather than blocking the join."""
    env = (os.getenv('MT_AUX_VERSION') or '').strip()
    if env:
        return env
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return 'unknown'


class _Chunk:
    """One in-flight chunk on one aux node."""

    def __init__(self, sender, job_token: str):
        self.sender = sender
        self.job_token = job_token
        self.done: asyncio.Future = asyncio.get_event_loop().create_future()


class AuxInstance:
    """Duck-types ExecutorInstance, but work travels out over the node's own socket rather
    than us dialing it. Gallery-only: the single-image and batch paths hand back a pickled
    Context, which we will not reconstruct from a machine we don't own."""

    gallery_only = True

    def __init__(self, ws, name: str, version: str, caps: dict, priority: int = AUX_PRIORITY):
        self.ws = ws
        self.name = name
        self.version = version
        self.caps = caps or {}
        self.priority = priority
        self.busy = False
        self.aux_id = f'aux-{next(_ids)}'
        self.joined_at = time.monotonic()
        self.chunks_done = 0
        self._pending: dict[int, _Chunk] = {}
        self._chunk_ids = itertools.count(1)

    @property
    def label(self) -> str:
        return f'{self.name}({self.aux_id})'

    def free_executor(self):
        self.busy = False

    # ── dispatch ─────────────────────────────────────────────────────────────────────────
    async def sent_gallery_stream(self, images: list, config, sender, batch_size: int = 0, job_token: str = ""):
        cid = next(self._chunk_ids)
        chunk = _Chunk(sender, job_token)
        self._pending[cid] = chunk
        payload = pickle.dumps({"images": images, "config": config,
                                "batch_size": batch_size, "job_token": job_token})
        try:
            await self.ws.send_bytes(cid.to_bytes(4, 'big') + payload)
        except Exception as e:
            self._pending.pop(cid, None)
            raise RuntimeError(f'aux {self.label} unreachable: {e}') from e
        try:
            await chunk.done
        finally:
            self._pending.pop(cid, None)
        self.chunks_done += 1

    async def cancel_gallery(self, job_token: str = ""):
        """Best-effort, matching the local worker's fire-and-forget cancel. Token-scoped so a
        late cancel cannot abort a different gallery that started on this node meanwhile."""
        for cid, chunk in list(self._pending.items()):
            if job_token and chunk.job_token != job_token:
                continue
            try:
                await self.ws.send_text(json.dumps({"type": "cancel", "chunk": cid}))
            except Exception:
                pass

    async def sent(self, *a, **k):
        raise RuntimeError('aux nodes take gallery chunks only')

    sent_stream = sent_batch = sent_batch_stream = sent

    # ── socket → chunk routing ───────────────────────────────────────────────────────────
    def on_frame(self, cid: int, status: int, data: bytes) -> None:
        chunk = self._pending.get(cid)
        if chunk is None:
            return   # a frame for a chunk we already gave up on — drop it
        try:
            chunk.sender(status, data)
        except Exception:
            logger.exception(f'{self.label}: frame handler raised')

    def on_end(self, cid: int, error: str | None) -> None:
        chunk = self._pending.get(cid)
        if chunk is None or chunk.done.done():
            return
        if error:
            chunk.done.set_exception(RuntimeError(f'aux {self.label}: {error}'))
        else:
            chunk.done.set_result(None)

    def fail_all(self, reason: str) -> None:
        """The socket died. Every chunk this node held has to fail so the scheduler can
        re-dispatch the un-emitted remainder elsewhere."""
        for chunk in list(self._pending.values()):
            if not chunk.done.done():
                chunk.done.set_exception(RuntimeError(f'aux {self.label} disconnected: {reason}'))
        self._pending.clear()


def _reject(reason: str) -> dict:
    return {"ok": False, "error": reason}


def _validate(hello: dict) -> str | None:
    """Returns a rejection reason, or None to admit."""
    if not isinstance(hello, dict):
        return 'malformed handshake'
    if hello.get('protocol') != AUX_PROTOCOL:
        return f'protocol mismatch (node speaks {hello.get("protocol")!r}, this server speaks {AUX_PROTOCOL})'
    if not JOIN_TOKEN:
        return 'this server is not accepting aux nodes (MT_AUX_TOKEN is unset)'
    if not hmac.compare_digest(str(hello.get('token') or ''), JOIN_TOKEN):
        return 'bad join token'
    ours, theirs = code_version(), str(hello.get('version') or 'unknown')
    if not ALLOW_VERSION_SKEW and 'unknown' not in (ours, theirs) and ours != theirs:
        return (f'version mismatch (node {theirs}, server {ours}) — pages would render '
                f'inconsistently within one gallery; set MT_AUX_ALLOW_VERSION_SKEW=1 to override')
    return None


async def handle_join(ws) -> None:
    """Serve one aux node for the life of its socket. Registers it as an executor on a
    successful handshake and unregisters it on any exit path."""
    await ws.accept()
    inst = None
    try:
        try:
            hello = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=30))
        except Exception:
            await ws.send_text(json.dumps(_reject('expected a JSON handshake within 30s')))
            return

        bad = _validate(hello)
        if bad:
            logger.warning(f'aux join refused ({hello.get("name", "?")}): {bad}')
            await ws.send_text(json.dumps(_reject(bad)))
            return

        inst = AuxInstance(ws, str(hello.get('name') or 'aux'),
                           str(hello.get('version') or 'unknown'), hello.get('caps') or {})
        await ws.send_text(json.dumps({"ok": True, "aux_id": inst.aux_id,
                                       "server_version": code_version()}))
        executor_instances.register(inst)
        caps = inst.caps
        logger.info(f'aux node joined: {inst.label} version={inst.version} '
                    f'gpu={caps.get("gpu")} vram={caps.get("vram_mb")}MB priority={inst.priority} '
                    f'(pool now {executor_instances.capacity()} executors)')

        while True:
            msg = await ws.receive()
            if msg.get('type') == 'websocket.disconnect':
                break
            raw = msg.get('bytes')
            if raw is not None:
                if len(raw) < 5:
                    continue
                cid = int.from_bytes(raw[:4], 'big')
                inst.on_frame(cid, raw[4], raw[5:])
                continue
            text = msg.get('text')
            if text:
                try:
                    ev = json.loads(text)
                except Exception:
                    continue
                if ev.get('type') == 'end':
                    inst.on_end(int(ev.get('chunk') or 0), ev.get('error'))
    except Exception as e:
        logger.info(f'aux socket closed: {e}')
    finally:
        if inst is not None:
            executor_instances.unregister(inst)
            inst.fail_all('socket closed')
            logger.info(f'aux node left: {inst.label} after {inst.chunks_done} chunk(s), '
                        f'{time.monotonic() - inst.joined_at:.0f}s connected '
                        f'(pool now {executor_instances.capacity()} executors)')


def nodes() -> list[dict]:
    """Operator view of the pool. Local-only — an aux roster is not public information."""
    out = []
    for x in executor_instances.list:
        if isinstance(x, AuxInstance):
            out.append({"id": x.aux_id, "name": x.name, "version": x.version,
                        "busy": x.busy, "priority": x.priority, "caps": x.caps,
                        "chunks_done": x.chunks_done,
                        "connected_s": round(time.monotonic() - x.joined_at)})
        else:
            out.append({"id": "local", "name": getattr(x, 'label', 'local'),
                        "busy": x.busy, "priority": getattr(x, 'priority', 100)})
    return out
