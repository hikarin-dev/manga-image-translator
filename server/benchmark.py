"""Benchmark-only, token-scoped metrics. Never exposes another client's job history."""
import time

API_VERSION = 1


def job_metrics(sj):
    def average(pairs):
        weight = sum(w for _, w in pairs)
        return round(sum(v * w for v, w in pairs) / weight, 3) if weight else None

    wall = time.monotonic() - sj.submitted_at
    loads = [dict(event, chunk=i) for i, chunk in enumerate(sj.benchmark_chunks)
             for event in chunk.get('model_loads', [])]
    loads_complete = bool(sj.benchmark_chunks) and all('model_loads' in c for c in sj.benchmark_chunks)
    return {
        'api_version': API_VERSION,
        'wall_s': round(wall, 4), 'compute_s': round(sj.tel_wall, 4),
        'pages': sj.total, 'emitted': len(sj.emitted), 'failed': len(set(sj.failed)),
        'chunks': sj.chunks_done,
        'stages_s': {k: round(v, 4) for k, v in sj.tel_stages.items()},
        'waits_s': {k: round(v, 4) for k, v in sj.tel_waits.items()},
        'gpu_avg_pct': average(sj.tel_gpu), 'gpu_max_pct': sj.tel_gpu_max if sj.tel_gpu else None,
        'cpu_avg_pct': average(sj.tel_cpu), 'cpu_max_pct': sj.tel_cpu_max if sj.tel_cpu else None,
        'vram_max_mb': sj.tel_vram_max or None, 'vram_total_mb': sj.tel_vram_total or None,
        'llm_requests': sj.tel_llm_requests, 'llm_in': sj.tel_llm_in, 'llm_out': sj.tel_llm_out,
        'llm_cache_hit': sj.tel_llm_cache_hit, 'llm_cache_miss': sj.tel_llm_cache_miss,
        'llm_cost_usd': round(sj.tel_llm_cost, 6), 'llm_max_wall_s': sj.tel_llm_max_wall,
        'reuse': sj.tel_reuse, 'model_loads': loads,
        'model_load_s': round(sum(e['seconds'] for e in loads), 4) if loads_complete else None,
        'model_load_coverage': 'ModelWrapper.load; GPU-lane queue included; downloads and lazy renderer/session initialization are not isolated.',
        'chunk_metrics': sj.benchmark_chunks,
        'timing_notes': 'Stage awaits and model loads overlap. Compute is summed chunk wall, not GPU kernel time. CPU/GPU samples are system-wide.',
    }
