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
import shutil
import subprocess
import threading
import time

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


# ── cross-thread sub-stage accumulator ─────────────────────────────────────────
# The hot models (detection/OCR/inpainting) split their _infer into CPU-pool pre/post
# and a GPU-thread forward; those parts run on different threads with no access to the
# translator instance, so they report here. The gallery pipeline snapshots this dict
# around a run and folds the delta into its stage summary — giving the host-vs-kernel
# split the wall numbers alone can't show.
_substage: dict[str, float] = {}
_substage_lock = threading.Lock()


def add_substage(key: str, dt: float) -> None:
    with _substage_lock:
        _substage[key] = _substage.get(key, 0.0) + dt


def snapshot_substages() -> dict[str, float]:
    with _substage_lock:
        return dict(_substage)


# ── LLM request/token accounting ───────────────────────────────────────────────
# GPT translators (deepseek, gemini, …) report every API call here: request count,
# input/output tokens, DeepSeek's cache hit/miss split, and per-request wall time. The
# gallery pipeline resets this at the start of each chunk and reads it at the end, so the
# summary can show how many requests were made, the token cost, and — crucially for the
# "long silent wait" symptom — the wall of the SLOWEST single request. Reset (not
# snapshot-diff) is safe because the worker runs one gallery chunk at a time.
_llm_usage: dict[str, float] = {}
_llm_lock = threading.Lock()


def add_llm_usage(requests: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0,
                  cache_hit: int = 0, cache_miss: int = 0, wall: float = 0.0) -> None:
    with _llm_lock:
        _llm_usage['requests'] = _llm_usage.get('requests', 0) + requests
        _llm_usage['prompt_tokens'] = _llm_usage.get('prompt_tokens', 0) + prompt_tokens
        _llm_usage['completion_tokens'] = _llm_usage.get('completion_tokens', 0) + completion_tokens
        _llm_usage['cache_hit'] = _llm_usage.get('cache_hit', 0) + cache_hit
        _llm_usage['cache_miss'] = _llm_usage.get('cache_miss', 0) + cache_miss
        _llm_usage['sum_wall'] = _llm_usage.get('sum_wall', 0.0) + wall
        _llm_usage['max_wall'] = max(_llm_usage.get('max_wall', 0.0), wall)


def reset_llm_usage() -> None:
    with _llm_lock:
        _llm_usage.clear()


def snapshot_llm_usage() -> dict[str, float]:
    with _llm_lock:
        return dict(_llm_usage)


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
        self.queue_wait: dict[str, float] = {}
        self.t0 = None

    # ── counters ────────────────────────────────────────────────────────────
    def add_wait(self, key: str, dt: float) -> None:
        self.queue_wait[key] = self.queue_wait.get(key, 0.0) + dt

    # ── sampling ────────────────────────────────────────────────────────────
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
            if used_mb is None and torch is not None:
                try:
                    used_mb = torch.cuda.memory_allocated() / 1e6
                except Exception:
                    used_mb = None
            if used_mb is not None:
                self.vram_used.append(used_mb)

    def start(self) -> None:
        self.t0 = time.perf_counter()
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
