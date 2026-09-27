"""Process-wide worker pools that let the translation pipeline overlap stages.

The pipeline is otherwise one asyncio event loop running synchronous torch/PIL
calls, so no two compute stages ever run at the same time — GPU detection/OCR of
one page and CPU rendering of another are forced to take turns. These helpers
split the heavy work onto dedicated threads so the main loop only orchestrates:

  * submit_gpu(coro) — runs a coroutine on ONE dedicated GPU worker thread (its
    own event loop). Keeping every CUDA call on a single thread means a single
    CUDA context and kernels that serialize naturally (a GPU runs one kernel at a
    time anyway), but now off the main loop so the loop is free to drive CPU
    rendering and the async LLM network call concurrently.

  * run_cpu(fn, ...) — runs a blocking CPU function (rendering, numpy diffs,
    PNG/WebP encode, image decode) on a shared thread pool. PIL/numpy/torch
    release the GIL during their C/CUDA sections, so these overlap with GPU work
    and with each other.

Both pools are created lazily on first use and shared by every mode (local CLI,
ws, shared server). The GPU thread is a daemon so it never blocks interpreter
exit.
"""
import asyncio
import contextvars
import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Two GPU lanes (worker threads sharing the one CUDA context). The OCR beam search is a
# Python-driven loop whose host gaps would otherwise stall every other model's kernels;
# giving detection/inpainting their own lane lets their kernels fill those gaps. Each
# MODEL stays pinned to one lane, so per-model ops keep their exact order — outputs are
# unchanged, only the interleaving between different models changes.
_gpu_loops: dict = {}
_gpu_lock = threading.Lock()

_cpu_pool: ThreadPoolExecutor = None
_cpu_lock = threading.Lock()


def _run_loop_forever(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_gpu_loop(lane: int = 0) -> asyncio.AbstractEventLoop:
    loop = _gpu_loops.get(lane)
    if loop is not None:
        return loop
    with _gpu_lock:
        loop = _gpu_loops.get(lane)
        if loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=_run_loop_forever, args=(loop,),
                                      name=f'mit-gpu-worker-{lane}', daemon=True)
            thread.start()
            _gpu_loops[lane] = loop
    return loop


def submit_gpu(coro, lane: int = 0):
    """Schedule `coro` on the given GPU worker lane; return an awaitable bound
    to the CALLER's running loop. If called from within that lane's loop itself
    (re-entrant model calls), the coroutine is returned as-is to run inline and
    avoid a self-deadlock."""
    loop = _ensure_gpu_loop(lane)
    try:
        if asyncio.get_running_loop() is loop:
            return coro
    except RuntimeError:
        pass
    return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_in_context(contextvars.copy_context(), coro), loop))


async def _in_context(context, coro):
    # A lane runs its own loop, so its task would not see the caller's context variables (which
    # gallery run this work belongs to, where its telemetry goes). Re-bind them in the lane's task.
    for var, value in context.items():
        var.set(value)
    return await coro


async def _call(fn, args, kwargs):
    return fn(*args, **kwargs)


def run_gpu(fn, *args, lane: int = 0, **kwargs):
    """Run a blocking function on a GPU lane (its model calls then stay on that lane's thread),
    awaited from the caller's loop. For models that keep their CPU work elsewhere."""
    return submit_gpu(_call(fn, args, kwargs), lane)


def _ensure_cpu_pool() -> ThreadPoolExecutor:
    global _cpu_pool
    if _cpu_pool is not None:
        return _cpu_pool
    with _cpu_lock:
        if _cpu_pool is None:
            workers = max(2, min(8, (os.cpu_count() or 4)))
            _cpu_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='mit-cpu')
    return _cpu_pool


async def run_cpu(fn, *args, **kwargs):
    """Run a blocking CPU function on the shared CPU pool, awaited from the
    current event loop so it overlaps with GPU work and other CPU tasks."""
    loop = asyncio.get_running_loop()
    if kwargs:
        fn = functools.partial(fn, **kwargs)
    # In the caller's context, like asyncio.to_thread: the pool thread's work belongs to its run.
    return await loop.run_in_executor(_ensure_cpu_pool(), functools.partial(contextvars.copy_context().run, fn), *args)


# ── process pool for GIL-heavy work ────────────────────────────────────────────
# Mask refinement (~1s/page) and study-layer building (per-bubble encodes, ~40% of a dense
# gallery's wall) are Python-heavy; on threads they serialize on the GIL with the OCR beam loop
# and every other Python section, capping the whole pipeline (cores sit idle). A process pool
# runs the exact same functions out-of-process — identical outputs, no GIL contention. Both
# stages overlap in the pipeline (mask in inpaint_stage, study in the render workers) and now
# share this pool, so it's sized to the machine rather than a flat 2. Workers hold no torch
# models (just numpy/PIL), so they're cheap; spawned lazily (or via prewarm) and reused.
_proc_pool = None
_proc_lock = threading.Lock()


def _ensure_proc_pool():
    global _proc_pool
    if _proc_pool is not None:
        return _proc_pool
    with _proc_lock:
        if _proc_pool is None:
            from concurrent.futures import ProcessPoolExecutor
            workers = max(2, min(4, (os.cpu_count() or 4) // 2))
            _proc_pool = ProcessPoolExecutor(max_workers=workers)
    return _proc_pool


def _noop():
    return None


def prewarm_proc_pool():
    """Fire-and-forget: spawn the pool workers now (imports are the slow part on
    Windows) so the first page doesn't pay for it."""
    try:
        pool = _ensure_proc_pool()
        pool.submit(_noop)
    except Exception:
        pass


async def run_proc(fn, *args):
    """Run a picklable function on the process pool (GIL-free), awaited from the
    current event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ensure_proc_pool(), fn, *args)


# ── memory hygiene between chunks ─────────────────────────────────────────────────────────────
async def _empty_cuda_cache():
    import sys
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _heapmin():
    import ctypes
    import sys
    if sys.platform == 'win32':
        try:
            ctypes.CDLL('ucrtbase')._heapmin()
        except Exception:
            pass


TRIM_EVERY_PAGES = 40
_pages_since_trim = 0


async def trim_memory(pages: int) -> bool:
    """Give back what finished chunks left cached: PyTorch's cached VRAM (on Windows every reserved
    byte is also charged to this process's commit, and the cache is never returned on its own), the
    C heap's free pages (freed numpy/torch temporaries keep whole heap segments committed), and
    unreachable reference cycles. At most every TRIM_EVERY_PAGES pages, never per page: each release
    costs the next pages some re-allocation, and a chunk running alongside a moment's stall.
    True when it trimmed."""
    global _pages_since_trim
    _pages_since_trim += pages
    if _pages_since_trim < TRIM_EVERY_PAGES:
        return False
    _pages_since_trim = 0
    import gc
    gc.collect()
    if _gpu_loops:
        await submit_gpu(_empty_cuda_cache(), next(iter(_gpu_loops)))
    await run_cpu(_heapmin)
    return True
