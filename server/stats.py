"""Lightweight service metrics: in-memory day counters + one JSONL line per finished job.

Answers the Phase-1 operator questions (is the box keeping up? how much was it used
today? what did the last jobs look like?) without a metrics stack. /stats serves the
live snapshot; logs/jobs.jsonl is the durable per-job record (one JSON object per line,
survives restarts — the in-memory day counters don't, jobs.jsonl is the source of truth).
"""
import datetime
import json
import os
import subprocess
import time
from collections import deque

_BOOT = time.time()
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
JOBS_LOG = os.path.join(LOG_DIR, 'jobs.jsonl')

_day: str | None = None
_counters: dict = {}
_recent: deque = deque(maxlen=20)


def _roll() -> None:
    global _day, _counters
    d = datetime.date.today().isoformat()
    if d != _day:
        _day = d
        _counters = {'jobs': 0, 'jobs_cancelled': 0, 'pages': 0, 'pages_failed': 0,
                     'compute_s': 0.0, 'llm_cost_usd': 0.0}


def record_job(sj, cancelled: bool = False) -> None:
    """Fold one finished (or cancelled) gallery job into the day counters and jobs.jsonl.
    `sj` is a gallery_jobs._SchedJob — its tel_* fields are already summed across chunks."""
    _roll()
    wall = round(time.monotonic() - sj.submitted_at, 1)
    entry = {
        'ts': datetime.datetime.now().isoformat(timespec='seconds'),
        'token': (sj.job.token or '')[:8],
        'ip': getattr(sj, 'owner_ip', ''),
        'key': getattr(sj, 'owner_key', ''),      # access-key NAME, never the secret
        # Whatever the client said this gallery came from. Free-form and never interpreted
        # here — the server only stores and displays it.
        'source_url': getattr(sj, 'source_url', ''),
        'pages': sj.total,
        'emitted': sj.tel_emitted,
        'failed': len(set(sj.failed)),
        'cancelled': bool(cancelled or sj.tel_cancelled),
        'wall_s': wall,
        'sec_per_page': round(wall / sj.total, 2) if sj.total else 0.0,
        'compute_s': round(sj.tel_wall, 1),
        'chunks': sj.chunks_done,
        'stages_s': {k: round(v, 1) for k, v in sj.tel_stages.items()},
        'waits_s': {k: round(v, 1) for k, v in sj.tel_waits.items()},
        'gpu_max_pct': round(sj.tel_gpu_max),
        'vram_max_mb': round(sj.tel_vram_max),
        'llm_cost_usd': round(sj.tel_llm_cost, 4),
        'llm_requests': sj.tel_llm_requests,
        'llm_in': sj.tel_llm_in,
        'llm_out': sj.tel_llm_out,
    }
    _counters['jobs'] += 1
    if entry['cancelled']:
        _counters['jobs_cancelled'] += 1
    _counters['pages'] += entry['emitted']
    _counters['pages_failed'] += entry['failed']
    _counters['compute_s'] = round(_counters['compute_s'] + entry['compute_s'], 1)
    _counters['llm_cost_usd'] = round(_counters['llm_cost_usd'] + entry['llm_cost_usd'], 4)
    _recent.append(entry)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(JOBS_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, separators=(',', ':')) + '\n')
    except Exception:
        pass


_gpu_cache: tuple[float, dict | None] = (0.0, None)


def gpu_snapshot() -> dict | None:
    """Instantaneous GPU state via nvidia-smi, cached ~5s (per-job telemetry already has
    the during-job numbers; this is the "right now" gauge)."""
    global _gpu_cache
    ts, val = _gpu_cache
    if time.time() - ts < 5:
        return val
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3)
        u, mu, mt = [s.strip() for s in out.stdout.strip().splitlines()[0].split(',')]
        val = {'util_pct': int(u), 'vram_used_mb': int(mu), 'vram_total_mb': int(mt)}
    except Exception:
        val = None
    _gpu_cache = (time.time(), val)
    return val


# Per-job fields that identify WHO asked for WHAT. Everything here is stripped for any caller
# that is not on loopback, so an aux node or an allowlisted remote address sees that a job ran
# and how long it took, but never who submitted it or what they were reading.
_PRIVILEGED_JOB_FIELDS = ('ip', 'key', 'token', 'source_url', 'stages_s', 'waits_s',
                          'llm_cost_usd', 'llm_requests', 'llm_in', 'llm_out')


def _redact(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in _PRIVILEGED_JOB_FIELDS}


def snapshot(gpu: dict | None, full: bool = False) -> dict:
    """Assemble the /stats payload. Runs on the event loop (it reads scheduler state);
    the caller fetches `gpu` off-loop via gpu_snapshot() since nvidia-smi blocks.

    `full` is for loopback callers only. Redaction happens HERE rather than in the page, so a
    reduced caller never receives the sensitive fields at all — hiding them client-side would
    leave them one devtools tab away."""
    _roll()
    from server import gallery_jobs
    from server.instance import executor_instances
    today = {'date': _day, **_counters}
    if not full:
        today.pop('llm_cost_usd', None)
    return {
        'full': full,
        'uptime_s': int(time.time() - _BOOT),
        'queue': gallery_jobs.queue_snapshot(),
        'workers': {
            'registered': len(executor_instances.list),
            'busy': len([i for i in executor_instances.list if i.busy]),
        },
        'today': today,
        'gpu': gpu,
        'recent_jobs': [dict(e) for e in _recent] if full else [_redact(e) for e in _recent],
    }
