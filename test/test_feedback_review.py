import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from server import edge, feedback, feedback_review


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, 'ROOT', tmp_path / 'feedback')
    monkeypatch.setattr(edge, 'DASHBOARD_NETS', edge._parse_nets('203.0.113.7'))
    monkeypatch.setattr(edge, 'ACCESS_KEYS', {'user-token': 'user'})
    raw = io.BytesIO(); Image.new('RGB', (20, 30), 'white').save(raw, 'PNG')
    image = raw.getvalue(); sha = feedback.digest(image)
    asset = {'path': 'assets/' + sha, 'sha256': sha, 'bytes': len(image), 'media_type': 'image/png'}
    report = {'schema': 'typesetting-feedback', 'version': 1, 'report_id': 'd' * 32,
              'created_at': '2026-09-20T00:00:00Z', 'issues': ['ocr'], 'note': '<script>untrusted note</script>',
              'selection': {'primary': 0, 'related': [], 'surface': 'ocr'},
              'page': {'original': asset, 'translated': None, 'study_background': None,
                       'bubbles': [{'id': 0, 'src': 'read me', 'box': {'x': 0, 'y': 0, 'w': 1, 'h': 1}}]},
              'pipeline': None, 'display': {'surface': 'ocr'}, 'fidelity': 'incomplete', 'missing': ['pipeline']}
    ref = feedback.save_archive(feedback.encode(report, {asset['path']: image}))
    app = FastAPI(); app.include_router(feedback_review.router); app.add_middleware(edge.EdgeGate)
    # TestClient connections are non-loopback; use the configured trusted tunnel header.
    operator = TestClient(app, headers={'CF-Connecting-IP': '203.0.113.7'})
    return report, ref, image, operator, app


def test_operator_lists_reads_previews_exports_and_updates_without_mutating_evidence(reports):
    report, ref, image, client, _ = reports
    base = '/dashboard/feedback'
    listing = client.get(base + '/list').json()
    assert listing['counts']['new'] == 1
    assert listing['reports'][0]['surface'] == 'ocr'
    detail = client.get(base + '/' + ref['report_id'])
    assert detail.headers['cache-control'] == 'no-store'
    assert detail.json()['manifest']['note'] == report['note']
    assert client.get(base + '/' + ref['report_id'] + '/asset/' + report['page']['original']['sha256']).content == image
    before = client.get(base + '/' + ref['report_id'] + '/export').content
    result = client.post(base + '/' + ref['report_id'] + '/review', json={'status': 'resolved', 'note': 'OCR bounds corrected'},
                         headers={'X-Feedback-Review': '1'})
    assert result.status_code == 200
    assert client.get(base + '/' + ref['report_id']).json()['review']['note'] == 'OCR bounds corrected'
    assert client.get(base + '/list?status=new').json()['total'] == 0
    assert client.get(base + '/list?status=resolved&q=bounds').json()['total'] == 1
    assert client.get(base + '/' + ref['report_id'] + '/export').content == before
    assert feedback.digest(before) == ref['archive_sha256']


def test_public_users_and_worker_peers_cannot_discover_feedback_management(reports, monkeypatch):
    report, ref, _, _, app = reports
    monkeypatch.setattr(edge, 'dashboard_allowed', lambda ip: True)  # A worker peer may see pool stats.
    client = TestClient(app, headers={'CF-Connecting-IP': '203.0.113.55', 'X-Access-Token': 'user-token'})
    base = '/dashboard/feedback'
    for path in ['', '/list', '/review.js', '/review.css', '/' + ref['report_id'], '/' + ref['report_id'] + '/export',
                 '/' + ref['report_id'] + '/asset/' + report['page']['original']['sha256']]:
        assert client.get(base + path).status_code == 404
    assert client.post(base + '/' + ref['report_id'] + '/review', json={'status': 'archived'}).status_code == 404


def test_review_changes_reject_cross_origin_and_invalid_actions(reports):
    _, ref, _, client, _ = reports
    url = '/dashboard/feedback/' + ref['report_id'] + '/review'
    assert client.post(url, json={'status': 'resolved'}).status_code == 403
    assert client.post(url, json={'status': 'resolved'}, headers={'X-Feedback-Review': '1', 'Origin': 'https://elsewhere.invalid'}).status_code == 404
    assert client.get('/dashboard/feedback/list', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 404
    assert client.post(url, json={'status': 'unknown'}, headers={'X-Feedback-Review': '1'}).status_code == 409
    assert client.get('/dashboard/feedback/' + ref['report_id']).json()['review']['status'] == 'new'


def test_unrelated_assets_and_invalid_identifiers_are_not_served(reports):
    _, ref, _, client, _ = reports
    assert client.get('/dashboard/feedback/' + ref['report_id'] + '/asset/' + 'a' * 64).status_code == 404
    assert client.get('/dashboard/feedback/not-an-id').status_code == 404
    response = client.get('/dashboard/feedback')
    assert "script-src 'self'" in response.headers['content-security-policy']
    assert 'untrusted note' not in response.text


def test_reports_without_pipeline_data_keep_source_feedback(reports):
    _, ref, _, client, _ = reports
    manifest = client.get('/dashboard/feedback/' + ref['report_id']).json()['manifest']
    assert manifest['selection']['surface'] == 'ocr'
    assert manifest['fidelity'] == 'incomplete'
    assert set(manifest['missing']) == {'pipeline', 'translated', 'study_background'}
