"""Lightweight pipeline profiler for whole-gallery translation.

The per-stage `await`-duration timing in the pipeline conflates real compute with
time spent waiting on a shared GPU thread / queues, so it can't tell us whether the
stages actually overlap or where the bottleneck is. This profiler adds the missing
signals the redesign needs:

  • a background sampler thread for GPU utilization (nvidia-smi), VRAM (torch + smi)
    and CPU utilization (psutil) — averaged and peaked over the run;
  • queue-wait accumulators (how long each stage sits blocked waiting for upstream
    work — a starved stage means the bottleneck is elsewhere);
  • pages/sec and an overlap factor (summed stage compute ÷ wall).

Everything degrades gracefully: no nvidia-smi → no GPU-util numbers; no psutil →
no CPU numbers. Sampling runs on its own daemon thread so it never blocks the loop.
"""
import contextvars
import os
import shutil
import subprocess
import sys
import threading
import time

try:
    import psutil
except Exception:
    psutil = None

# torch is used only as a VRAM fallback when nvidia-smi is missing, and only if this process has
# loaded it already: importing it here would load torch/CUDA into every process-pool worker that
# imports mask refinement (~2.6 GB of commit each).
def _torch():
    return sys.modules.get('torch')


# ── per-run accounting ────────────────────────────────────────────────────────
# Sub-stage splits (the hot models' CPU-pool pre/post vs GPU-lane forward), model loads and LLM
# usage (request count, tokens, DeepSeek's cache split, the slowest request) are reported from
# whichever thread did the work. Each gallery run binds its own record in a context variable, and
# utils.executors carries the context into CPU-pool and GPU-lane work, so every report lands in the
# run that caused it even with several runs in flight on one worker. Outside a run, reports go to a
# process-wide record.
class _RunTelemetry:
    def __init__(self):
        self.substage: dict[str, float] = {}
        self.model_loads: list[dict] = []
        self.llm: dict[str, float] = {}


_GLOBAL_TELEMETRY = _RunTelemetry()
_RUN_TELEMETRY: contextvars.ContextVar = contextvars.ContextVar('run_telemetry', default=None)
_substage_lock = threading.Lock()
_llm_lock = threading.Lock()


def _tel() -> _RunTelemetry:
    return _RUN_TELEMETRY.get() or _GLOBAL_TELEMETRY


def bind_run_telemetry() -> None:
    """Give the calling task, and everything it starts, its own telemetry record."""
    _RUN_TELEMETRY.set(_RunTelemetry())


def add_substage(key: str, dt: float) -> None:
    t = _tel()
    with _substage_lock:
        t.substage[key] = t.substage.get(key, 0.0) + dt


def snapshot_substages() -> dict[str, float]:
    t = _tel()
    with _substage_lock:
        return dict(t.substage)


# Loading is reported separately from stage awaits; those awaits can already contain this time, so
# clients must not add the two together.
def record_model_load(model: str, start: float, seconds: float) -> None:
    t = _tel()
    with _substage_lock:
        t.model_loads.append({'model': model, 'start': start, 'seconds': seconds})


def reset_model_loads() -> None:
    t = _tel()
    with _substage_lock:
        t.model_loads.clear()


def snapshot_model_loads(origin: float) -> list[dict]:
    t = _tel()
    with _substage_lock:
        return [{'model': e['model'], 'at_s': round(e['start'] - origin, 4),
                 'seconds': round(e['seconds'], 4)} for e in t.model_loads]


def add_llm_usage(requests: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0,
                  cache_hit: int = 0, cache_miss: int = 0, wall: float = 0.0) -> None:
    t = _tel()
    with _llm_lock:
        u = t.llm
        u['requests'] = u.get('requests', 0) + requests
        u['prompt_tokens'] = u.get('prompt_tokens', 0) + prompt_tokens
        u['completion_tokens'] = u.get('completion_tokens', 0) + completion_tokens
        u['cache_hit'] = u.get('cache_hit', 0) + cache_hit
        u['cache_miss'] = u.get('cache_miss', 0) + cache_miss
        u['sum_wall'] = u.get('sum_wall', 0.0) + wall
        u['max_wall'] = max(u.get('max_wall', 0.0), wall)


def reset_llm_usage() -> None:
    t = _tel()
    with _llm_lock:
        t.llm.clear()


def snapshot_llm_usage() -> dict[str, float]:
    t = _tel()
    with _llm_lock:
        return dict(t.llm)


class Profiler:
    def __init__(self, interval: float = 1.0, enabled: bool = True):
        self.interval = interval
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread = None
        self._smi = shutil.which('nvidia-smi')
        self.cpu: list[float] = []
        self.gpu: list[float] = []
        self.vram_used: list[float] = []   # MB
        self.vram_total: float = 0.0       # MB (from smi, if available)
        # This process's committed memory and RSS, its children's (the process pool) commit, and
        # its thread count. Commit is what runs Windows out of virtual memory, so it is the number
        # that decides whether a long job survives.
        self.mem: dict[str, list[float]] = {'commit': [], 'rss': [], 'children': [], 'threads': []}
        self.mem_start: float | None = None
        self._kids: list = []
        self.queue_wait: dict[str, float] = {}
        self.t0 = None

    # ── counters ────────────────────────────────────────────────────────────
    def add_wait(self, key: str, dt: float) -> None:
        self.queue_wait[key] = self.queue_wait.get(key, 0.0) + dt

    # ── sampling ────────────────────────────────────────────────────────────
    @staticmethod
    def _commit(p) -> float:
        info = p.memory_info()
        return getattr(info, 'private', info.vms) / 2**30

    def _sample_mem(self) -> None:
        p = psutil.Process()
        self.mem['commit'].append(self._commit(p))
        self.mem['rss'].append(p.memory_info().rss / 2**30)
        self.mem['threads'].append(float(p.num_threads()))
        # Listing children walks every process on the machine while holding the GIL, so the pool
        # (long-lived, spawned once) is looked up only every 30 samples.
        if not self._kids or len(self.mem['commit']) % 30 == 1:
            self._kids = p.children(recursive=True)
        kids = 0.0
        for c in self._kids:
            try:
                kids += self._commit(c)
            except psutil.Error:
                pass
        self.mem['children'].append(kids)

    def mem_summary(self) -> dict:
        """GB (threads as a count); empty when psutil is missing."""
        if not psutil or not self.mem['commit']:
            return {}
        end = self._commit(psutil.Process())
        out = {
            'pid': os.getpid(),
            'commit_start': round(self.mem_start or 0.0, 2), 'commit_end': round(end, 2),
            'commit_max': round(max(self.mem['commit'] + [end]), 2),
            'rss_max': round(max(self.mem['rss']), 2),
            'children_max': round(max(self.mem['children']), 2),
            'threads_max': int(max(self.mem['threads'])),
        }
        # On Windows every byte of VRAM PyTorch holds is charged to this process's commit as well,
        # so what the allocator keeps (reserved) versus what the models need (peak allocated) is the
        # other half of the memory picture.
        torch = _torch()
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    out['cuda_reserved'] = round(torch.cuda.memory_reserved() / 2**30, 2)
                    out['cuda_reserved_max'] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
                    out['cuda_allocated_max'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            except Exception:
                pass
        return out

    def _sample_loop(self) -> None:
        if psutil:
            try:
                psutil.cpu_percent(None)  # prime the delta baseline
            except Exception:
                pass
        while not self._stop.wait(self.interval):
            if psutil:
                try:
                    self.cpu.append(psutil.cpu_percent(None))
                    self._sample_mem()
                except Exception:
                    pass
            used_mb = None
            if self._smi:
                try:
                    out = subprocess.run(
                        [self._smi, '--query-gpu=utilization.gpu,memory.used,memory.total',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=2)
                    u, mu, mt = (x.strip() for x in out.stdout.strip().splitlines()[0].split(','))
                    self.gpu.append(float(u))
                    used_mb = float(mu)
                    self.vram_total = float(mt)
                except Exception:
                    pass
            if used_mb is None and _torch() is not None:
                try:
                    used_mb = _torch().cuda.memory_allocated() / 1e6
                except Exception:
                    used_mb = None
            if used_mb is not None:
                self.vram_used.append(used_mb)

    def start(self) -> None:
        self.t0 = time.perf_counter()
        if psutil:
            try:
                self.mem_start = self._commit(psutil.Process())
            except Exception:
                pass
        if self.enabled and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample_loop, name='mit-profiler', daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # ── reporting ───────────────────────────────────────────────────────────
    def summary(self, stage_times: dict, pages: int, emitted: int) -> str:
        wall = (time.perf_counter() - self.t0) if self.t0 else 0.0
        busy = sum(stage_times.values())

        def avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        def mx(xs):
            return max(xs) if xs else 0.0

        stages = ', '.join(f'{k}={v:.1f}' for k, v in sorted(stage_times.items(), key=lambda kv: -kv[1]))
        waits = ', '.join(f'{k}={v:.1f}' for k, v in sorted(self.queue_wait.items(), key=lambda kv: -kv[1])) or 'n/a'
        gpu_line = (f'GPU util avg={avg(self.gpu):.0f}% max={mx(self.gpu):.0f}%'
                    if self.gpu else 'GPU util n/a (no nvidia-smi)')
        vram_line = (f'VRAM used avg={avg(self.vram_used):.0f}MB max={mx(self.vram_used):.0f}MB'
                     + (f' / {self.vram_total:.0f}MB' if self.vram_total else ''))
        cpu_line = f'CPU avg={avg(self.cpu):.0f}% max={mx(self.cpu):.0f}%' if self.cpu else 'CPU n/a'
        return (
            f'stages(s): {stages} | summed={busy:.1f} wall={wall:.1f} '
            f'overlap={(busy / wall if wall else 1):.2f}x pages={pages} '
            f'pages/sec={(emitted / wall if wall else 0):.2f} per_page={(wall / pages if pages else 0):.2f}s\n'
            f'  queue_wait(s): {waits}\n'
            f'  {gpu_line} | {vram_line} | {cpu_line}'
        )
