import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import feedback
from server import edge


def asset(data, assets, media_type='image/png'):
    sha = feedback.digest(data)
    path = 'assets/' + sha
    assets[path] = data
    return {'path': path, 'sha256': sha, 'bytes': len(data), 'media_type': media_type}


@pytest.fixture
def evidence(tmp_path):
    assets = {}
    original = asset(b'original', assets)
    output = asset(b'output', assets)
    background = asset(b'background', assets)
    raw = asset(b'raw mask png', assets)
    text = asset(b'text mask png', assets)
    # The page's own pipeline data, as the app keeps it, with the config and builds behind it.
    pipeline = {'job': 'mfz3k2ab', 'lines': [{'pts': [[0, 0], [9, 0], [9, 9], [0, 9]], 'score': 0.9, 'text': 'a'},
                                             {'pts': [[20, 0], [29, 0], [29, 9], [20, 9]], 'score': 0.9, 'text': 'b'}],
                'read': [0, 1], 'regions': [{'lines': [0], 'tr': 'Hello'}, {'lines': [1], 'tr': 'world'}],
                'masks': {'raw': raw, 'text': text}}
    report = {'schema': 'typesetting-feedback', 'version': 1, 'report_id': 'b' * 32,
              'created_at': '2026-09-20T00:00:00Z', 'issues': ['grouping'], 'note': 'Join the two regions',
              'selection': {'primary': 0, 'related': [1]}, 'display': {'view': 'study', 'readerMode': 'strip'},
              'page': {'original': original, 'translated': output, 'study_background': background,
                       'bubbles': [{'id': 0, 'tr': 'Hello'}, {'id': 1, 'tr': 'world'}]},
              'pipeline': pipeline, 'translation': {'at': 1, 'config': {'render': {'renderer': 'shiori_v2'}},
                                                    'builds': {'render': 'a' * 12}},
              'missing': [], 'fidelity': 'complete'}
    return report, assets


def test_report_is_self_contained_and_retries_are_idempotent(evidence, tmp_path):
    report, assets = evidence
    data = feedback.encode(report, assets)
    ref = feedback.save_archive(data, tmp_path / 'feedback')
    repeated = feedback.save_archive(data, tmp_path / 'feedback')
    assert repeated == ref
    blob = feedback.export_archive(ref['report_id'], ref['archive_sha256'], tmp_path / 'feedback')
    manifest, saved_assets = feedback.decode(blob)
    assert manifest['note'] == report['note']
    assert manifest['selection']['related'] == [1]
    assert manifest['fidelity'] == 'complete'
    for path, original in assets.items():
        assert saved_assets[path] == original
    # The page's pipeline data and masks travel inside the report; nothing is looked up.
    assert manifest['pipeline']['regions'] == report['pipeline']['regions']
    assert saved_assets[manifest['pipeline']['masks']['text']['path']] == b'text mask png'
    assert manifest['translation']['builds'] == {'render': 'a' * 12}
    report['note'] = 'different note'
    with pytest.raises(ValueError, match='immutable'):
        feedback.save_archive(feedback.encode(report, assets), tmp_path / 'feedback')


def test_concurrent_retries_acknowledge_one_immutable_archive(evidence, tmp_path):
    data = feedback.encode(*evidence)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: feedback.save_archive(data, tmp_path / 'feedback'), range(2)))
    assert replies[0] == replies[1]


@pytest.mark.parametrize('change', ['mismatch', 'region', 'related', 'id_type'])
def test_mixed_runs_and_invalid_region_selections_are_rejected(evidence, tmp_path, change):
    report, assets = evidence
    if change == 'mismatch':
        report['missing'] = ['matching_run']
    elif change == 'region':
        report['page']['bubbles'][0]['id'] = report['selection']['primary'] = 7
    elif change == 'related':
        report['selection']['related'] = [9]
    else:
        report['page']['bubbles'][0]['id'] = report['selection']['primary'] = 'r0000'
    with pytest.raises(ValueError):
        feedback.save_archive(feedback.encode(report, assets), tmp_path / 'feedback')
    assert not (tmp_path / 'feedback').exists()


def test_historical_and_text_only_reports_are_explicitly_incomplete(evidence, tmp_path):
    report, assets = evidence
    for mask in report['pipeline']['masks'].values():
        del assets[mask['path']]
    report['pipeline'] = None
    del assets[report['page']['study_background']['path']]
    report['page']['study_background'] = None
    ref = feedback.save_archive(feedback.encode(report, assets), tmp_path / 'feedback')
    assert ref['fidelity'] == 'incomplete'
    assert set(ref['missing']) == {'pipeline', 'study_background'}


def test_archive_names_hashes_and_export_identity_are_validated(evidence, tmp_path):
    report, assets = evidence
    with pytest.raises(ValueError, match='unexpected'):
        feedback.save_archive(feedback.encode(report, assets | {'../outside': b'bad'}), tmp_path / 'feedback')
    name = next(iter(assets))
    with pytest.raises(ValueError, match='hash'):
        feedback.save_archive(feedback.encode(report, assets | {name: b'wrong'}), tmp_path / 'feedback')
    with pytest.raises(FileNotFoundError):
        feedback.export_archive('../outside', 'a' * 64, tmp_path / 'feedback')


def test_http_save_ack_export_and_disk_failure(evidence, tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, 'ROOT', tmp_path / 'feedback')
    app = FastAPI(); app.include_router(feedback.router)
    client = TestClient(app)
    data = feedback.encode(*evidence)
    response = client.post('/feedback/save', files={'archive': ('report.zip', data, 'application/zip')})
    assert response.status_code == 200
    receipt = response.json()
    assert (tmp_path / 'feedback' / (receipt['report_id'] + '.zip')).is_file()
    response = client.post('/feedback/export', data={'report_id': receipt['report_id'], 'archive_sha256': receipt['archive_sha256']})
    assert feedback.digest(response.content) == receipt['archive_sha256']
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(feedback, 'save_archive', fail)
    response = client.post('/feedback/save', files={'archive': ('report.zip', data, 'application/zip')})
    assert response.status_code == 507
    assert 'report_id' not in response.json()


def test_feedback_routes_use_existing_remote_access_controls(evidence, tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, 'ROOT', tmp_path / 'feedback')
    monkeypatch.setattr(edge, 'ACCESS_KEYS', {'test-token': 'test'})
    app = FastAPI(); app.include_router(feedback.router); app.add_middleware(edge.EdgeGate)
    client = TestClient(app)
    data = feedback.encode(*evidence)
    upload = {'archive': ('report.zip', data, 'application/zip')}
    assert client.post('/feedback/save', files=upload).status_code == 401
    assert client.post('/feedback/export', data={'report_id': 'a' * 32, 'archive_sha256': 'b' * 64}).status_code == 401
    response = client.post('/feedback/save', files=upload, headers={'X-Access-Token': 'test-token'})
    assert response.status_code == 200
