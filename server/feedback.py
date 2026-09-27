"""Self-contained feedback archives: everything a report needs travels in it (page images,
study layers, and the page's pipeline data with the config and builds that produced it), so a
saved report never depends on the reporter's library or on anything else kept on this server."""
import asyncio
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parents[1] / 'feedback'
HEX = re.compile(r'^[0-9a-f]{64}$')
RUN = re.compile(r'^[0-9a-f]{32}$')
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ENTRIES = 20000
ISSUES = {'placement', 'readability', 'line_breaks', 'overflow', 'grouping', 'ocr', 'translation', 'other'}
router = APIRouter()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def references(value):
    """Every asset reference ({path, sha256, …}) anywhere in a manifest."""
    if isinstance(value, dict):
        if 'sha256' in value and 'path' in value:
            yield value
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def encode(manifest, assets):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, separators=(',', ':')).encode())
        for name, data in assets.items():
            archive.writestr(name, data)
    return buf.getvalue()


def decode(data):
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError('Feedback archive too large')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        names = [e.filename for e in entries]
        if (len(entries) > MAX_ENTRIES or len(names) != len(set(names))
                or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES
                or 'manifest.json' not in names
                or archive.getinfo('manifest.json').file_size > 16 * 1024 * 1024):
            raise ValueError('Invalid feedback archive size or entries')
        manifest = json.loads(archive.read('manifest.json'))
        if (not isinstance(manifest, dict) or manifest.get('schema') != 'typesetting-feedback' or manifest.get('version') != 1
                or not RUN.fullmatch(manifest.get('report_id', ''))):
            raise ValueError('Unsupported feedback manifest')
        refs = list(references(manifest))
        expected = {'manifest.json'}
        for ref in refs:
            if (not HEX.fullmatch(ref['sha256'])
                    or ref['path'] != 'assets/' + ref['sha256']):
                raise ValueError('Invalid feedback asset name')
            expected.add(ref['path'])
        if expected != set(names):
            raise ValueError('Missing or unexpected feedback assets')
        assets = {name: archive.read(name) for name in expected - {'manifest.json'}}
        for ref in refs:
            asset = assets[ref['path']]
            if len(asset) != ref['bytes'] or digest(asset) != ref['sha256']:
                raise ValueError('Feedback asset hash mismatch')
    return manifest, assets


def save_archive(data, root=None):
    manifest, assets = decode(data)
    root = Path(root) if root is not None else ROOT
    path = root / (manifest['report_id'] + '.zip')
    request_hash = digest(data)
    if path.exists():
        previous_data = path.read_bytes()
        previous, _ = decode(previous_data)
        if previous.get('request_sha256') != request_hash:
            raise ValueError('Feedback report is immutable')
        return receipt(previous, previous_data)
    issues = manifest.get('issues')
    selection = manifest.get('selection', {})
    page = manifest.get('page', {})
    if not isinstance(selection, dict) or not isinstance(page, dict):
        raise ValueError('Invalid feedback page')
    if selection.get('surface', 'translation') not in ('translation', 'ocr', 'original'):
        raise ValueError('Invalid feedback surface')
    bubbles = page.get('bubbles', [])
    ids = [b.get('id') for b in bubbles]
    selected = [selection.get('primary')] + selection.get('related', [])
    if (not isinstance(issues, list) or not issues or any(i not in ISSUES for i in issues)
            or not isinstance(manifest.get('note'), str) or len(manifest['note']) > 10000
            or not ids or len(set(ids)) != len(ids) or any(i not in ids for i in selected)
            or len(set(selected)) != len(selected) or not isinstance(manifest.get('display'), dict)):
        raise ValueError('Invalid feedback selection or description')
    missing = set(manifest.get('missing', []))
    if 'matching_run' in missing:
        raise ValueError('Displayed page and its pipeline data do not belong to the same translation')
    for field in ('original', 'translated', 'study_background'):
        if not page.get(field):
            missing.add(field)
    pipeline = manifest.get('pipeline')
    if pipeline is not None:
        regions = pipeline.get('regions') if isinstance(pipeline, dict) else None
        if not isinstance(regions, list) or any(not isinstance(i, int) or not 0 <= i < len(regions) for i in ids):
            raise ValueError('Feedback page does not match its pipeline data')
    else:
        missing.add('pipeline')
    manifest['missing'] = sorted(missing)
    manifest['fidelity'] = 'incomplete' if missing else 'complete'
    manifest['request_sha256'] = request_hash
    output = encode(manifest, assets)
    if len(output) > MAX_ARCHIVE_BYTES or sum(len(a) for a in assets.values()) > MAX_ARCHIVE_BYTES:
        raise ValueError('Feedback archive too large')
    # Publish without replacing another writer's immutable report (including retry races).
    root.mkdir(parents=True, exist_ok=True)
    fd, pending = tempfile.mkstemp(prefix='.pending-', dir=root)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(pending, path)
        except FileExistsError:
            return save_archive(data, root)
    finally:
        os.unlink(pending)
    return receipt(manifest, output)


def receipt(manifest, output):
    return {'report_id': manifest['report_id'], 'archive_sha256': digest(output),
            'fidelity': manifest['fidelity'], 'missing': manifest['missing'], 'storage': 'translation-server'}


def export_archive(report_id, archive_sha256, root=None):
    if not RUN.fullmatch(report_id) or not HEX.fullmatch(archive_sha256):
        raise FileNotFoundError('Feedback report not found')
    root = Path(root) if root is not None else ROOT
    data = (root / (report_id + '.zip')).read_bytes()
    if digest(data) != archive_sha256:
        raise ValueError('Feedback archive hash mismatch')
    return data


@router.post('/feedback/save', tags=['api'])
async def save_feedback(archive: UploadFile = File(...)):
    try:
        data = await archive.read(MAX_ARCHIVE_BYTES + 1)
        return await asyncio.to_thread(save_archive, data)
    except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise HTTPException(409, detail=str(exc))
    except OSError:
        raise HTTPException(507, detail='Feedback could not be saved to disk')
    finally:
        await archive.close()


@router.post('/feedback/export', tags=['api'])
async def export_feedback(report_id: str = Form(...), archive_sha256: str = Form(...)):
    try:
        data = await asyncio.to_thread(export_archive, report_id, archive_sha256)
    except FileNotFoundError:
        raise HTTPException(404, detail='Feedback report not found')
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc))
    return Response(data, media_type='application/zip', headers={
        'Content-Disposition': f'attachment; filename="feedback-{report_id}.zip"',
    })
