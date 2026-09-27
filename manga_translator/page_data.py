"""A page's pipeline data: what a client keeps so a later run of the same page can skip stages.

The translation server keeps nothing between jobs. It returns each page's data with the page's
result and accepts it back with a later request. The record is compact JSON:

    lines    detected text lines, in detection order: {pts, score} plus OCR output
             {text, prob, fg, bg, dir} for the lines OCR recognized
    read     OCR's order of the recognized lines (indices into lines)
    regions  merged regions: {lines: [line indices], size, prob, fg, bg, lang, src} plus `panel`
             (panel-aware sorting only), `angle` (non-zero only), `text` (only when it differs from its lines joined, e.g.
             after the pre-dictionary), `tr` (translation) and `keep: false` (filtered out)
    bubbles  balloon masks as bit-packed crops: {box: [x, y, w, h], bits}
    end      only for a page that stopped early: the stage that found no text

Masks travel as lossless WebP blobs (PNG past WebP's size limit): `raw` (the detector's text
mask) and `text` (the refined mask inpainting erases). Values restore with the exact types the pipeline produces, so a
restored stage feeds later stages exactly what running it would have.

Container (request `stage` files and status-9 result frames):
    [u32 big-endian JSON length][JSON, UTF-8][blobs, in BLOBS order]
where the JSON carries `blobs: {name: size}`.
"""
import base64
import json
import zlib

import cv2
import numpy as np

from .utils.generic import Quadrilateral
from .utils.textblock import TextBlock

STAGES = ('prepare', 'detect', 'ocr', 'merge', 'translate', 'mask', 'inpaint', 'bubbles', 'render')
ORDER = {stage: index for index, stage in enumerate(STAGES)}
BLOBS = ('raw', 'text')
# Stages whose earlier output the client may offer for reuse when their inputs come out equal.
KEEPABLE = ('translate', 'mask', 'bubbles')
MAX_JSON = 4 * 1024 * 1024


# ── Container ────────────────────────────────────────────────────────────────────────────────
def pack(record, blobs=None):
    blobs = {name: blobs[name] for name in BLOBS if blobs and blobs.get(name)}
    doc = dict(record)
    if blobs:
        doc['blobs'] = {name: len(data) for name, data in blobs.items()}
    head = json.dumps(doc, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return len(head).to_bytes(4, 'big') + head + b''.join(blobs.values())


def unpack(data):
    """(record, blobs) from a container; ValueError when it is malformed."""
    if len(data) < 4:
        raise ValueError('page data too short')
    size = int.from_bytes(data[:4], 'big')
    if size > MAX_JSON or 4 + size > len(data):
        raise ValueError('page data JSON length out of range')
    record = json.loads(data[4:4 + size])
    if not isinstance(record, dict):
        raise ValueError('page data must be a JSON object')
    sizes = record.pop('blobs', None) or {}
    if not isinstance(sizes, dict) or any(name not in BLOBS or not isinstance(n, int) or n < 0 for name, n in sizes.items()):
        raise ValueError('invalid page data blobs')
    blobs, offset = {}, 4 + size
    for name in BLOBS:
        if name in sizes:
            blobs[name] = data[offset:offset + sizes[name]]
            offset += sizes[name]
    if offset != len(data) or any(len(blob) != sizes[name] for name, blob in blobs.items()):
        raise ValueError('page data blobs do not match their sizes')
    return record, blobs


# ── Masks ────────────────────────────────────────────────────────────────────────────────────
WEBP_MAX_SIDE = 16383


def encode_mask(mask):
    """Lossless: WebP (about half the size of PNG for these masks), or PNG for a page too long for WebP."""
    mask = np.ascontiguousarray(mask)
    if max(mask.shape[:2]) <= WEBP_MAX_SIDE:
        ok, data = cv2.imencode('.webp', mask, [cv2.IMWRITE_WEBP_QUALITY, 101])
    else:
        ok, data = cv2.imencode('.png', mask, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError('mask could not be encoded')
    return data.tobytes()


def decode_mask(data):
    mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError('unreadable mask')
    return mask


def encode_bubbles(masks):
    """Full-page uint8 balloon masks (255 inside) as bit-packed crops of their non-zero area."""
    items = []
    for mask in masks or []:
        mask = np.asarray(mask)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            items.append({'box': [0, 0, 0, 0]})
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) + 1 - x, int(ys.max()) + 1 - y
        crop = mask[y:y + h, x:x + w]
        item = {'box': [x, y, w, h]}
        values = np.unique(crop)
        if mask.dtype == np.uint8 and set(values.tolist()) <= {0, 255}:
            item['bits'] = base64.b64encode(zlib.compress(np.packbits(crop > 0).tobytes(), 6)).decode('ascii')
        else:
            item['raw'] = base64.b64encode(zlib.compress(np.ascontiguousarray(crop, dtype=np.uint8).tobytes(), 6)).decode('ascii')
        items.append(item)
    return items


def decode_bubbles(items, shape):
    h, w = shape[:2]
    masks = []
    for item in items:
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = item['box']
        if bw and bh:
            if 'bits' in item:
                bits = np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(item['bits'])), np.uint8))[:bw * bh]
                mask[y:y + bh, x:x + bw] = bits.reshape(bh, bw) * np.uint8(255)
            else:
                mask[y:y + bh, x:x + bw] = np.frombuffer(zlib.decompress(base64.b64decode(item['raw'])),
                                                         np.uint8).reshape(bh, bw)
        masks.append(mask)
    return masks


# ── Text lines ───────────────────────────────────────────────────────────────────────────────
def _points(pts):
    return np.asarray(pts, dtype=np.int32).tolist()


def _score(value):
    # The shortest decimal that reads back as the same float32.
    return float(np.format_float_positional(np.float32(value), unique=True, trim='-'))


def detected(textlines):
    """What detection produced, captured before OCR fills in (and reorders) the same objects."""
    return [(np.array(q.pts, dtype=np.int32), q.prob, id(q)) for q in textlines]


def encode_lines(detection, recognized):
    """(lines, read) from the detection capture and OCR's output. OCR lines are matched to the
    detected line they came from (same object, or the same points); others are appended."""
    lines = [{'pts': _points(pts), 'score': _score(score)} for pts, score, _ in detection]
    by_object = {key: index for index, (_, _, key) in enumerate(detection)}
    by_points = {}
    for index, (pts, _, _) in enumerate(detection):
        by_points.setdefault(pts.tobytes(), []).append(index)
    used, read = set(), []
    for q in recognized:
        pts = np.asarray(q.pts, dtype=np.int32)
        index = by_object.get(id(q))
        if index is None or index in used or not np.array_equal(pts, detection[index][0]):
            index = next((i for i in by_points.get(pts.tobytes(), ()) if i not in used), None)
        if index is None:
            index = len(lines)
            lines.append({'pts': _points(pts)})
        used.add(index)
        lines[index].update({'text': q.text, 'prob': float(q.prob), 'fg': [int(q.fg_r), int(q.fg_g), int(q.fg_b)],
                             'bg': [int(q.bg_r), int(q.bg_g), int(q.bg_b)], 'dir': q.assigned_direction})
        q._line_id = index
        read.append(index)
    return lines, read


def detect_lines(record):
    """Detection's output: fresh line objects in detection order."""
    return [Quadrilateral(np.array(line['pts'], dtype=np.int32), '', np.float32(line['score']))
            for line in record.get('lines') or [] if 'score' in line]


def ocr_lines(record):
    """OCR's output: the recognized lines in OCR's order."""
    out = []
    for index in record.get('read') or []:
        line = record['lines'][index]
        q = Quadrilateral(np.array(line['pts'], dtype=np.int32), line['text'], line['prob'], *line['fg'], *line['bg'])
        q.assigned_direction = line['dir']
        q._line_id = index
        out.append(q)
    return out


# ── Regions ──────────────────────────────────────────────────────────────────────────────────
def member_lines(region, recognized):
    """Indices of the recognized lines a merged region is made of (merging copies their points)."""
    free = {}
    for q in recognized:
        free.setdefault(np.asarray(q.pts, dtype=np.int32).tobytes(), []).append(q._line_id)
    members = []
    for pts in np.asarray(region.lines, dtype=np.int32):
        candidates = free.get(pts.tobytes())
        members.append(candidates.pop(0) if candidates else None)
    return members


def encode_regions(regions, recognized):
    out = []
    for region in regions:
        members = member_lines(region, recognized)
        region._line_ids = members
        item = {'lines': members, 'size': int(region.font_size)}
        if region.angle:
            item['angle'] = float(region.angle)
        joined = TextBlock(np.zeros((len(region.texts), 4, 2), np.int32), list(region.texts)).text if region.texts else ''
        if region.text != joined:
            item['text'] = region.text
        item.update({'prob': float(region.prob), 'fg': [int(c) for c in region.fg_colors],
                     'bg': [int(c) for c in region.bg_colors], 'lang': region._source_lang,
                     'src': region.source_direction})
        if hasattr(region, 'panel_index'):   # set by panel-aware sorting only
            item['panel'] = int(region.panel_index)
        out.append(item)
    return out


def regions(record):
    """Merge's output (before translation): fresh regions in reading order."""
    lines = record['lines']
    out = []
    for item in record.get('regions') or []:
        members = [lines[i] for i in item['lines']]
        region = TextBlock([m['pts'] for m in members], [m['text'] for m in members], font_size=item['size'],
                           angle=np.float64(item['angle']) if 'angle' in item else 0,
                           prob=np.float64(item['prob']), fg_color=tuple(item['fg']), bg_color=tuple(item['bg']),
                           source_lang=item['lang'])
        if 'text' in item:
            region.text = item['text']
        region.source_direction = item['src']
        if 'panel' in item:
            region.panel_index = item['panel']
        region._line_ids = list(item['lines'])
        out.append(region)
    return out


def encode_translations(items, regions_, kept):
    """Record each region's translation and whether it survived the filter."""
    kept_ids = {id(r) for r in kept}
    for item, region in zip(items, regions_):
        item['tr'] = region.translation
        if id(region) not in kept_ids:
            item['keep'] = False
        else:
            item.pop('keep', None)


def apply_translations(record, regions_, config):
    """Restore translation's effect on the regions; returns the survivors."""
    kept = []
    for item, region in zip(record['regions'], regions_):
        region.translation = item.get('tr', '')
        region.target_lang = config.translator.target_lang
        region._alignment = config.render.alignment
        region._direction = config.render.direction
        if item.get('keep', True):
            kept.append(region)
    return kept


def translated(record):
    return bool(record.get('regions')) and all('tr' in item for item in record['regions'])


def survivors(record):
    """Point sets of the regions that survived translation (what mask refinement reads)."""
    lines = record['lines']
    return [[lines[i]['pts'] for i in item['lines']] for item in record.get('regions') or [] if item.get('keep', True)]


def region_points(regions_):
    return [np.asarray(r.lines, dtype=np.int32).tolist() for r in regions_]


# ── Context ──────────────────────────────────────────────────────────────────────────────────
# A job that starts mid-gallery (one page translated on its own, say) gets the pages before it
# from the client, so a context-aware translator still sees what came earlier.
MAX_CONTEXT_PAGES = 16
MAX_CONTEXT_BYTES = 64 * 1024


def context_pages(text):
    """[(sources, translations)] per earlier page, oldest first, from the JSON
    [{"src": [...], "tr": [...]}]; ValueError when malformed."""
    if len(text) > MAX_CONTEXT_BYTES:
        raise ValueError('context too large')
    pages = json.loads(text)
    if not isinstance(pages, list) or len(pages) > MAX_CONTEXT_PAGES:
        raise ValueError(f'context must be a list of at most {MAX_CONTEXT_PAGES} pages')
    out = []
    for page in pages:
        src, tr = (page.get('src'), page.get('tr')) if isinstance(page, dict) else (None, None)
        if not (isinstance(src, list) and isinstance(tr, list) and len(src) == len(tr)
                and all(isinstance(s, str) for s in src + tr)):
            raise ValueError('each context page needs src and tr lists of equal length')
        out.append((src, tr))
    return out
