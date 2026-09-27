"""GET /capabilities, the resolve endpoint and the start endpoint's page-data fields."""
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from manga_translator.config import Config
from manga_translator import page_data, stages
from server import capabilities, edge


@pytest.fixture
def client():
    from server.main import app
    return TestClient(app, client=('127.0.0.1', 5555))


def _config_with(path, value):
    doc = {}
    target = doc
    *parents, leaf = path.split('.')
    for part in parents:
        target = target.setdefault(part, {})
    target[leaf] = value
    return Config.parse_raw(json.dumps(doc))


def test_document_offers_only_valid_choices():
    doc = capabilities.document()
    assert doc['api_version'] == 2 and doc['features'] == {'pipeline_data': True}
    assert [s['id'] for s in doc['stages']] == ['detect', 'ocr', 'translate', 'inpaint', 'render']
    keys = set()
    for stage in doc['stages']:
        assert sum(i['default'] for i in stage['implementations']) == 1
        for impl in stage['implementations']:
            _config_with(stage['implementation_param'], impl['id'])    # a real Config value
            assert impl['version'] and ('build' in impl)
        for param in stage['params']:
            keys.add(param['key'])
            _config_with(param['key'], param['default'] if param['type'] != 'language' else 'ENG')
            if param['type'] == 'enum':
                assert param['default'] in [c['value'] for c in param['choices']]
    for preset in doc['presets']:
        assert set(preset['values']) <= keys
    assert {l['id'] for l in doc['languages']} >= {'ENG', 'JPN'} and all(l['bcp47'] for l in doc['languages'])
    assert capabilities.document()['etag'] == doc['etag']


def test_document_reveals_no_paths_or_secrets(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-very-secret')
    text = json.dumps(capabilities.document())
    assert 'sk-very-secret' not in text
    assert 'manga_translator' not in text and '\\\\' not in text and 'Users' not in text


def test_missing_api_key_marks_translator_unavailable(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    translate = next(s for s in capabilities.document()['stages'] if s['id'] == 'translate')
    deepseek = next(i for i in translate['implementations'] if i['id'] == 'deepseek')
    assert deepseek['available'] is False and deepseek['unavailable_reason']
    assert deepseek['batching'] == {'mode': 'adaptive', 'default': 10, 'user_tunable': False}
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'x')
    translate = next(s for s in capabilities.document()['stages'] if s['id'] == 'translate')
    assert next(i for i in translate['implementations'] if i['id'] == 'deepseek')['available'] is True


def test_removing_a_model_from_the_registry_removes_it(monkeypatch):
    trimmed = dict(capabilities.IMPLEMENTATIONS)
    trimmed['ocr'] = [row for row in trimmed['ocr'] if row[0] != 'hayai']
    monkeypatch.setattr(capabilities, 'IMPLEMENTATIONS', trimmed)
    ocr = next(s for s in capabilities.document()['stages'] if s['id'] == 'ocr')
    assert 'hayai' not in [i['id'] for i in ocr['implementations']]


def test_capabilities_endpoint_supports_etag(client):
    first = client.get('/capabilities')
    assert first.status_code == 200
    etag = first.headers['etag']
    assert etag.strip('"') == first.json()['etag']
    assert client.get('/capabilities', headers={'If-None-Match': etag}).status_code == 304


def test_new_routes_are_public_behind_the_token(monkeypatch):
    from server.main import app
    monkeypatch.setattr(edge, 'ACCESS_KEYS', {'k': 'default'})
    external = TestClient(app, client=('203.0.113.9', 5555))
    assert external.get('/capabilities').status_code == 401
    assert external.get('/capabilities', headers={'X-Access-Token': 'k'}).status_code == 200
    assert external.post('/translate/gallery/resolve', data={'config': '{}'},
                         headers={'X-Access-Token': 'k'}).status_code == 200


def _png():
    buf = io.BytesIO()
    Image.new('RGB', (8, 8), 'white').save(buf, 'PNG')
    return buf.getvalue()


@pytest.mark.parametrize('fields, files, status, detail', [
    ({'config': '{"detector": {"detection_size": "huge"}}'}, [], 400, 'invalid config'),
    ({}, [('stage', ('s', b'', 'application/octet-stream'))] * 2, 400, 'one to one'),
    ({'builds': '0' * 16}, [], 409, 'updated'),
    ({'context': '[{"src": ["a"]}]'}, [], 400, 'invalid context'),
])
def test_start_rejects_bad_fields_readably(client, fields, files, status, detail):
    response = client.post('/translate/gallery/start', data={'job_token': 'x', **fields},
                           files=[('image', ('p.png', _png(), 'image/png'))] + files)
    assert response.status_code == status and detail in response.json()['detail']


def test_start_passes_on_whether_the_client_wants_page_data(client, monkeypatch):
    from server import main
    seen = {}

    async def start(*args, **kwargs):
        seen.update(kwargs)
        return {'token': 'x', 'started': True}
    monkeypatch.setattr(main, 'start_gallery_job', start)
    monkeypatch.setattr(main.executor_instances, 'capacity', lambda gallery=False: 1)
    for fields, capture in (({}, True), ({'capture': '0'}, False)):
        response = client.post('/translate/gallery/start', data={'job_token': 'x', **fields},
                               files=[('image', ('p.png', _png(), 'image/png'))])
        assert response.status_code == 200 and seen['capture'] is capture


def test_resolve_endpoint(client):
    body = client.post('/translate/gallery/resolve', data={'config': json.dumps({'render': {'renderer': 'manga2eng'}})}).json()
    assert body['config']['render']['renderer'] == 'manga2eng'
    assert set(body['builds']) == set(stages.STAGES), 'manga2eng segments balloons, so every stage applies'
    assert 'render.renderer' in body['fields']['render'] and body['fields']['prepare'] == [
        f for f, owners in stages.FIELDS.items() if 'prepare' in owners]
    # The start endpoint accepts exactly these builds.
    from server import main
    assert body["signature"] == stages.signature(body["builds"]) == stages.signature(
        stages.stage_builds(Config.parse_raw('{"render": {"renderer": "manga2eng"}}'), main.stage_runtime))
