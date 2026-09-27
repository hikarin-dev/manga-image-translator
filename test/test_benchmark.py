"""Benchmark metrics stay job scoped; normal translation responses remain compact."""
import asyncio
import json
import pickle
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from manga_translator.config import Config
from manga_translator.utils.profiling import record_model_load, reset_model_loads, snapshot_model_loads
from server import gallery_jobs, edge
from server.benchmark import job_metrics


def scheduled(benchmark=True):
    job = gallery_jobs.GalleryJob('a-random-token')
    job.benchmark = benchmark
    req = SimpleNamespace(state=SimpleNamespace(client_ip='private-ip', key='private-key'))
    return gallery_jobs._SchedJob(job, req, [b'x'], Config(), 1, lambda x: x)


def test_chunk_aggregation_keeps_loads_separate_and_missing_samples_unknown():
    sj = scheduled()
    chunk = {'wall': 2, 'stage_times': {'ocr': 4}, 'queue_wait': {'render_q': 3},
             'model_loads': [{'model': 'ocr', 'at_s': 0, 'seconds': 1}],
             'llm': {'requests': 1, 'cost': .02, 'in': 20, 'out': 10}, 'reuse': {'ocr': {'ran': 1}}}
    sj.fold_telemetry(chunk)
    sj.fold_telemetry({**chunk, 'model_loads': [], 'sampling': {'gpu_samples': 1}, 'gpu_avg': 0, 'gpu_max': 0})
    sj.emitted.add(0)
    result = job_metrics(sj)
    assert result['stages_s']['ocr'] == 8 and result['compute_s'] == 4
    assert result['model_load_s'] == 1 and len(result['model_loads']) == 1
    assert result['llm_requests'] == 2 and result['llm_cost_usd'] == .04
    assert result['gpu_avg_pct'] == 0 and result['cpu_avg_pct'] is None
    assert 'private-ip' not in json.dumps(result) and 'private-key' not in json.dumps(result)


@pytest.mark.parametrize('benchmark', [False, True])
def test_only_benchmark_terminal_frames_include_metrics(monkeypatch, benchmark):
    sj = scheduled(benchmark)
    sj.next_page = 1
    sj.emitted.add(0)
    sj.fold_telemetry({'wall': 1, 'model_loads': []})
    results = []
    import server.streaming
    monkeypatch.setattr(server.streaming, 'notify', lambda code, data, *_: results.append((code, pickle.loads(data))))
    monkeypatch.setattr(gallery_jobs, '_record', lambda *_: None)
    monkeypatch.setattr(gallery_jobs, '_drop', lambda *_: None)
    gallery_jobs._maybe_finish(sj)
    assert results[0][0] == 0
    assert ('benchmark' in results[0][1]) == benchmark


def test_benchmark_admission_is_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(gallery_jobs, '_sched', {'a': scheduled(False)})
    assert gallery_jobs.benchmark_conflict(True)
    assert not gallery_jobs.benchmark_conflict(False)
    monkeypatch.setattr(gallery_jobs, '_sched', {'a': scheduled(True)})
    assert gallery_jobs.benchmark_conflict(False)
    monkeypatch.setattr(gallery_jobs, '_sched', {})
    assert not gallery_jobs.benchmark_conflict(True)


def test_load_telemetry_resets_between_chunks():
    reset_model_loads()
    record_model_load('OCR', 10, 2)
    assert snapshot_model_loads(9) == [{'model': 'OCR', 'at_s': 1, 'seconds': 2}]
    reset_model_loads()
    assert snapshot_model_loads(0) == []


def test_benchmark_info_exposes_policy_without_job_history(monkeypatch):
    from server import main
    async def snapshot(_req):
        return {'queue': {}, 'workers': {}, 'gpu': None, 'uptime_s': 10,
                'recent_jobs': [{'ip': 'private-ip', 'source_url': 'private-source'}]}
    monkeypatch.setattr(main, 'service_stats', snapshot)
    with TestClient(main.app, client=('127.0.0.1', 5555)) as client:
        result = client.get('/benchmark/info').json()
    assert result['api_version'] == 1
    assert result['limits']['starts_per_hour'] is None
    assert 'recent_jobs' not in result
    assert '/benchmark/gallery/start' in edge.PUBLIC_PATHS
    assert {r.path for r in main.app.routes} >= {'/benchmark/gallery/start', '/benchmark/gallery/poll', '/benchmark/gallery/cancel'}
