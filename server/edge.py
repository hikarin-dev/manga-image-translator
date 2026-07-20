"""Edge hardening for remote exposure (Cloudflare Tunnel / any reverse proxy).

This API was written for a trusted localhost caller; exposing it through a tunnel makes
every route reachable from the internet, including internal ones (/register accepts a
worker whose responses the server unpickles — that is remote code execution). This module
gates everything that isn't the public translate surface.

A request is EXTERNAL when it carries a Cloudflare tunnel header (cloudflared connects
from 127.0.0.1, so the peer address alone can't tell tunnel traffic from local traffic)
or when its peer address isn't loopback (direct LAN / port-forward access). External
requests:
  • may only use the allowlisted public routes — everything else 404s;
  • must present the shared access token when MT_ACCESS_TOKEN is set (constant-time
    compare; the open "/" reachability ping is exempt);
  • have a request-body size cap and, on job creation, page-count / page-size /
    image-type / per-IP rate limits (see main.py's /translate/gallery/start).
Local requests are untouched — the local app keeps the full API and no limits.

Config comes from the environment (the repo's gitignored .env is loaded on import):
  MT_ACCESS_TOKEN        shared secret required from external clients (empty = off)
  MT_ALLOWED_ORIGINS     comma-separated CORS origins for browser clients
  MT_MAX_BODY_MB         per-request body cap for external requests   (default 120)
  MT_MAX_PAGE_MB         per-page byte cap for external uploads       (default 8)
  MT_MAX_PAGES_PER_JOB   page cap per external translation job        (default 150)
  MT_MAX_JOBS_PER_IP     concurrent jobs per external IP              (default 2)
  MT_MAX_STARTS_PER_HOUR job creations per external IP per hour       (default 12)
  MT_MAX_LIVE_JOBS       global live-job cap before "at capacity"     (default 8)
"""
import hmac
import ipaddress
import json
import logging
import os
import time
from collections import deque

logger = logging.getLogger('edge')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(levelname)s: [edge] %(message)s'))
    logger.addHandler(_h)
    logger.propagate = False

from dotenv import load_dotenv
load_dotenv()

ACCESS_TOKEN = (os.getenv('MT_ACCESS_TOKEN') or '').strip()

# ── access keys ────────────────────────────────────────────────────────────────────────
# MT_ACCESS_TOKEN is the unnamed default key. MT_ACCESS_KEYS issues additional NAMED keys:
#
#   MT_ACCESS_KEYS=alice:<secret>,bob:<secret>
#
# The name is what gets recorded against a job and shown on the dashboard — never the secret,
# which is not stored, logged, or returned anywhere. Issuing one key per person is what makes
# the record useful: with a single shared secret, "which token" has only one possible answer.
#
# Names are the seam for per-key policy later (rate limits, permitted translators): resolution
# already happens once per request, so a limit lookup has somewhere to hang. Nothing is
# enforced per-key yet — every key currently gets the same global limits.
def _parse_keys(raw: str) -> dict:
    keys = {}
    for entry in (raw or '').split(','):
        entry = entry.strip()
        if not entry or ':' not in entry:
            continue
        name, _, secret = entry.partition(':')
        name, secret = name.strip(), secret.strip()
        if name and secret:
            keys[secret] = name
    return keys


ACCESS_KEYS = dict(_parse_keys(os.getenv('MT_ACCESS_KEYS', '')))
if ACCESS_TOKEN:
    ACCESS_KEYS.setdefault(ACCESS_TOKEN, 'default')


def resolve_key(presented: str) -> str | None:
    """Name of the key this request authenticated with, or None if it matches nothing.
    Compared in constant time against every key so a wrong guess leaks no timing signal."""
    matched = None
    for secret, name in ACCESS_KEYS.items():
        if hmac.compare_digest(presented, secret):
            matched = name
    return matched


ALLOWED_ORIGINS = [o.strip().rstrip('/') for o in (
    os.getenv('MT_ALLOWED_ORIGINS')
    or 'https://hikarin-dev.github.io,http://localhost:5500,http://127.0.0.1:5500'
).split(',') if o.strip()]

MAX_BODY_BYTES = int(float(os.getenv('MT_MAX_BODY_MB', '120')) * 1024 * 1024)
MAX_PAGE_BYTES = int(float(os.getenv('MT_MAX_PAGE_MB', '8')) * 1024 * 1024)
MAX_PAGES_PER_JOB = int(os.getenv('MT_MAX_PAGES_PER_JOB', '150'))
MAX_JOBS_PER_IP = int(os.getenv('MT_MAX_JOBS_PER_IP', '2'))
MAX_STARTS_PER_HOUR = int(os.getenv('MT_MAX_STARTS_PER_HOUR', '12'))
MAX_LIVE_JOBS = int(os.getenv('MT_MAX_LIVE_JOBS', '8'))

# Routes an external client may reach. Everything else — /register, /simple_execute/*,
# /results*, /result static files, /queue-size, /reset-context, /docs, the legacy
# translate endpoints — stays local-only.
PUBLIC_PATHS = {
    '/translate/gallery/start',
    '/translate/gallery/poll',
    '/translate/gallery/cancel',
    '/stats',
}
_LOOPBACK = {'127.0.0.1', '::1', 'localhost'}

# ── operator dashboard ─────────────────────────────────────────────────────────────────
# Reachable through the tunnel, but only from an allowlisted address — it exposes the aux
# roster and a job history carrying client IPs, so it is not merely token-worthy.
#
# An IP allowlist is trustworthy HERE specifically because the tunnel is the only way in:
# there is no forwarded port, so every external request arrives via cloudflared and
# cf-connecting-ip is set by Cloudflare rather than by the caller. Do not reuse this
# reasoning if the origin is ever exposed directly.
#
# Allowed = MT_DASHBOARD_IPS (exact addresses or CIDR) plus, automatically, the address of
# every aux node currently connected — a node dials in from the machine its operator is
# sitting at, so the people lending GPUs can watch the pool without you curating a list.
DASHBOARD_PATHS = {'/dashboard', '/dashboard/data'}


def _parse_nets(raw: str) -> list:
    nets = []
    for entry in (raw or '').split(','):
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))   # a bare IP becomes a /32
        except ValueError:
            pass
    return nets


DASHBOARD_NETS = _parse_nets(os.getenv('MT_DASHBOARD_IPS', ''))


def _in_nets(ip: str, nets: list) -> bool:
    if not ip or not nets:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in nets)


def dashboard_allowed(ip: str) -> bool:
    if _in_nets(ip, DASHBOARD_NETS):
        return True
    # Imported lazily: aux_pool pulls in the executor registry, and edge is imported first.
    try:
        from server.aux_pool import connected_ips
        return ip in connected_ips()
    except Exception:
        return False


class EdgeGate:
    """Pure ASGI middleware (BaseHTTPMiddleware buffers streaming responses; this doesn't).
    Must sit INSIDE CORSMiddleware so its rejections still get CORS headers — i.e. add it
    to the app BEFORE adding CORSMiddleware (Starlette wraps in reverse add order)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = {k.decode('latin-1').lower(): v.decode('latin-1')
                   for k, v in (scope.get('headers') or [])}
        client = (scope.get('client') or ('', 0))[0] or ''
        cf_ip = headers.get('cf-connecting-ip', '')
        external = bool(cf_ip) or client not in _LOOPBACK
        state = scope.setdefault('state', {})
        state['external'] = external
        state['client_ip'] = cf_ip or client
        # Overwritten below once an external caller's key is resolved. Loopback needs no key.
        state['key'] = '' if external else 'local'
        if external:
            path = scope.get('path') or '/'
            if path == '/':
                # Keep the app's reachability ping working, but don't serve the local
                # web UI (whose endpoints are blocked out here anyway) to the internet.
                return await self._respond(send, 200, b'Shiori translation server is running.\n',
                                           content_type=b'text/plain; charset=utf-8')
            if path in DASHBOARD_PATHS:
                # 404, not 403: a caller who isn't allowlisted learns nothing about the page
                # existing. Allowlisted callers skip the access token — a browser cannot send
                # the X-Access-Token header on a plain navigation, so the address IS the
                # credential here, and anyone sharing that public address (the rest of the
                # household, the same CGNAT pool) is inside it.
                if not dashboard_allowed(state['client_ip']):
                    # Say which address was refused. Without this the operator has no way to
                    # learn what to allowlist — the response deliberately reveals nothing, and
                    # a home address is rarely what a "what is my IP" site reports. Bounded
                    # noise: only dashboard paths reach here.
                    logger.warning(
                        f'dashboard refused for {state["client_ip"] or "unknown"} - add it to '
                        f'MT_DASHBOARD_IPS in .env and restart to allow it')
                    return await self._reject(send, 404, 'Not found')
                return await self.app(scope, receive, send)
            if path not in PUBLIC_PATHS:
                return await self._reject(send, 404, 'Not found')
            if ACCESS_KEYS:
                key = resolve_key(headers.get('x-access-token', ''))
                if key is None:
                    return await self._reject(send, 401, 'access token missing or wrong')
                state['key'] = key
            try:
                if int(headers.get('content-length') or 0) > MAX_BODY_BYTES:
                    return await self._reject(
                        send, 413, f'request too large (max {MAX_BODY_BYTES // (1024 * 1024)}MB per request)')
            except ValueError:
                pass
        await self.app(scope, receive, send)

    @staticmethod
    async def _respond(send, status: int, body: bytes, content_type: bytes = b'application/json'):
        await send({'type': 'http.response.start', 'status': status,
                    'headers': [(b'content-type', content_type),
                                (b'content-length', str(len(body)).encode())]})
        await send({'type': 'http.response.body', 'body': body})

    @classmethod
    async def _reject(cls, send, status: int, detail: str):
        # Same {"detail": ...} shape as FastAPI's HTTPException, so clients parse one format.
        await cls._respond(send, status, json.dumps({'detail': detail}).encode('utf-8'))


# ── upload validation (external requests only) ─────────────────────────────────────────

def looks_like_image(b: bytes) -> bool:
    if len(b) >= 12 and b[:4] == b'RIFF' and b[8:12] == b'WEBP':
        return True
    if len(b) >= 12 and b[4:8] == b'ftyp' and b[8:12] in (b'avif', b'avis'):
        return True   # AVIF — in the wild it's often served under other extensions; Pillow ≥11 decodes it natively
    return b.startswith((b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff', b'GIF87a', b'GIF89a', b'BM'))


def validate_pages(images: list) -> str | None:
    """Reject oversized or non-image uploads early, before they reach the queue/worker."""
    for i, b in enumerate(images):
        if len(b) > MAX_PAGE_BYTES:
            return f'page {i + 1} is too large (max {MAX_PAGE_BYTES // (1024 * 1024)}MB per page)'
        if not looks_like_image(b):
            return f'page {i + 1} is not a supported image'
    return None


# ── per-IP admission control for job creation ──────────────────────────────────────────

_starts: dict[str, deque] = {}


def check_admission(ip: str, pages: int) -> tuple[int, str] | None:
    """Gate creating a NEW job for an external client. Returns (status, detail) to reject,
    None to admit. Counting the start happens here, so only admitted jobs consume quota."""
    from server import gallery_jobs
    if pages > MAX_PAGES_PER_JOB:
        return 413, f'too many pages (max {MAX_PAGES_PER_JOB} per translation)'
    live = gallery_jobs.live_job_count()
    if live >= MAX_LIVE_JOBS:
        return 503, f'server is at capacity ({live} jobs queued) — try again in ~{live * 2} min'
    if gallery_jobs.live_jobs_for_ip(ip) >= MAX_JOBS_PER_IP:
        return 429, 'you already have a translation running — wait for it to finish'
    q = _starts.setdefault(ip, deque())
    now = time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= MAX_STARTS_PER_HOUR:
        return 429, 'hourly translation limit reached — try again later'
    q.append(now)
    return None
