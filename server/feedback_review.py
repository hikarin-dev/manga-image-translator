"""Private feedback inbox on the existing server, with review state separate from evidence."""
import asyncio
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from server import edge, feedback

STATES = ('new', 'reviewed', 'resolved', 'archived')
PRIVATE_HEADERS = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                   'Referrer-Policy': 'no-referrer'}


def require_operator(request: Request):
    if not edge.feedback_operator(request):
        raise HTTPException(404, detail='Not found')
    # Other web pages cannot use an operator's browser to read or change the inbox.
    origin = request.headers.get('origin')
    if ((origin and origin.rstrip('/') != str(request.base_url).rstrip('/'))
            or request.headers.get('sec-fetch-site') in ('cross-site', 'same-site')):
        raise HTTPException(404, detail='Not found')
    if request.method not in ('GET', 'HEAD') and request.headers.get('x-feedback-review') != '1':
        raise HTTPException(403, detail='Review action header required')


router = APIRouter(prefix='/dashboard/feedback', dependencies=[Depends(require_operator)], include_in_schema=False)


def report_path(report_id):
    if not feedback.RUN.fullmatch(report_id):
        raise FileNotFoundError('Report not found')
    return feedback.ROOT / (report_id + '.zip')


def read_manifest(report_id):
    with zipfile.ZipFile(report_path(report_id)) as archive:
        if archive.getinfo('manifest.json').file_size > 16 * 1024 * 1024:
            raise ValueError('Invalid manifest size')
        manifest = json.loads(archive.read('manifest.json'))
    if manifest.get('schema') != 'typesetting-feedback' or manifest.get('report_id') != report_id:
        raise ValueError('Invalid report manifest')
    return manifest


def read_review(report_id):
    path = feedback.ROOT / 'reviews' / (report_id + '.json')
    return json.loads(path.read_bytes()) if path.exists() else {'status': 'new', 'note': '', 'updated_at': None}


def inbox(status='', query='', offset=0, limit=40):
    counts = dict.fromkeys(STATES, 0)
    rows, unreadable = [], 0
    paths = sorted((p for p in feedback.ROOT.glob('*.zip') if feedback.RUN.fullmatch(p.stem)),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            manifest, review = read_manifest(path.stem), read_review(path.stem)
            counts[review['status']] += 1
            if status and review['status'] != status:
                continue
            if query and query.casefold() not in ' '.join([path.stem, manifest['note'], *manifest['issues'],
                                                         review['note']]).casefold():
                continue
            rows.append({'report_id': path.stem, 'received_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                         'issues': manifest['issues'], 'surface': manifest['selection'].get('surface', 'translation'),
                         'note': manifest['note'][:240], 'fidelity': manifest['fidelity'], 'status': review['status']})
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            unreadable += 1
    return {'reports': rows[offset:offset + limit], 'total': len(rows), 'counts': counts, 'unreadable': unreadable}


def image_asset(report_id, asset_hash):
    if not feedback.HEX.fullmatch(asset_hash):
        raise FileNotFoundError('Image not found')
    manifest = read_manifest(report_id)
    ref = next((r for r in feedback.references(manifest['page']) if r['sha256'] == asset_hash), None)
    if not ref:
        raise FileNotFoundError('Image not found')
    media = ref.get('media_type', '')
    if media not in ('image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/avif', 'image/bmp'):
        raise ValueError('Unsupported preview image')
    with zipfile.ZipFile(report_path(report_id)) as archive:
        if ref['path'] != 'assets/' + asset_hash or archive.getinfo(ref['path']).file_size > feedback.MAX_ARCHIVE_BYTES:
            raise ValueError('Invalid image reference')
        data = archive.read(ref['path'])
    if feedback.digest(data) != asset_hash:
        raise ValueError('Image hash mismatch')
    return data, media


def update_review(report_id, status, note):
    read_manifest(report_id)  # Do not create orphaned state for a nonexistent report.
    if status not in STATES or len(note) > 10000:
        raise ValueError('Invalid review')
    result = {'status': status, 'note': note, 'updated_at': datetime.now(timezone.utc).isoformat()}
    feedback.atomic_write(feedback.ROOT / 'reviews' / (report_id + '.json'), json.dumps(result, ensure_ascii=False).encode())
    return result


async def checked(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except FileNotFoundError:
        raise HTTPException(404, detail='Report or asset not found')
    except (ValueError, KeyError, TypeError, zipfile.BadZipFile):
        raise HTTPException(409, detail='Feedback evidence is invalid or unavailable')
    except OSError:
        raise HTTPException(507, detail='Feedback storage is unavailable')


@router.get('')
async def review_page():
    return FileResponse(Path(__file__).with_name('feedback_review.html'), headers={**PRIVATE_HEADERS,
        'Content-Security-Policy': "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})


@router.get('/review.js')
async def review_script():
    return FileResponse(Path(__file__).with_name('feedback_review.js'), media_type='text/javascript', headers=PRIVATE_HEADERS)


@router.get('/review.css')
async def review_styles():
    return FileResponse(Path(__file__).with_name('feedback_review.css'), media_type='text/css', headers=PRIVATE_HEADERS)


@router.get('/list')
async def list_reports(response: Response, status: str = '', q: str = Query('', max_length=200),
                       offset: int = Query(0, ge=0), limit: int = Query(40, ge=1, le=100)):
    if status and status not in STATES:
        raise HTTPException(422, detail='Invalid status')
    response.headers.update(PRIVATE_HEADERS)
    return await checked(inbox, status, q, offset, limit)


@router.get('/{report_id}')
async def report_detail(report_id: str, response: Response):
    response.headers.update(PRIVATE_HEADERS)
    return {'manifest': await checked(read_manifest, report_id), 'review': await checked(read_review, report_id)}


@router.get('/{report_id}/asset/{asset_hash}')
async def preview_image(report_id: str, asset_hash: str):
    data, media = await checked(image_asset, report_id, asset_hash)
    return Response(data, media_type=media, headers={**PRIVATE_HEADERS, 'Content-Security-Policy': "default-src 'none'; sandbox"})


@router.get('/{report_id}/export')
async def download_report(report_id: str):
    await checked(read_manifest, report_id)
    return FileResponse(report_path(report_id), media_type='application/zip',
                        filename=f'feedback-{report_id}.zip', headers=PRIVATE_HEADERS)


class ReviewUpdate(BaseModel):
    status: str
    note: str = Field('', max_length=10000)


@router.post('/{report_id}/review')
async def save_review(report_id: str, review: ReviewUpdate, response: Response):
    response.headers.update(PRIVATE_HEADERS)
    return await checked(update_review, report_id, review.status, review.note)
