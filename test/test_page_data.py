"""The page-data record restores every stage output exactly: real pages (14, three OCR models,
rotated regions, mixed languages; texts scrubbed) go through capture → JSON → restore and must
come back with identical attributes and types."""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from manga_translator import page_data
from manga_translator.config import Config
from manga_translator.utils.generic import BBox, Quadrilateral
from manga_translator.utils.textblock import TextBlock

FIXTURE = Path(__file__).parent / 'fixtures' / 'pipeline_pages.json.gz'
# Recomputed from the points on first use; never stored.
CACHED = {'area', 'aabb', 'angle', 'cosangle', 'font_size', 'structure', 'aspect_ratio', 'is_approximate_axis_aligned',
          'center', 'xyxy'}
IDS = {'_snapshot_line_id', '_snapshot_region_id', '_snapshot_line_ids', '_line_id', '_line_ids'}


def _old_decode(value):
    """The fixture holds exact object states in the previous lossless encoding."""
    classes = {'Q': Quadrilateral, 'T': TextBlock, 'B': BBox}
    if isinstance(value, list):
        return [_old_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    (tag, payload), = value.items()
    if tag == '__f':
        return float(payload)
    if tag == '__a':
        dtype, shape, data = payload
        return np.asarray(_old_decode(data), dtype=np.dtype(dtype)).reshape(shape)
    if tag == '__n':
        dtype, item = payload
        return np.dtype(dtype).type(float(item) if isinstance(item, str) else item)
    if tag == '__t':
        return tuple(_old_decode(v) for v in payload)
    if tag == '__d':
        return {k: _old_decode(v) for k, v in payload.items()}
    if tag == '__p':
        return {_old_decode(k): _old_decode(v) for k, v in payload}
    if tag == '__o':
        cls_tag, state = payload
        obj = classes[cls_tag].__new__(classes[cls_tag])
        obj.__dict__.update({k: _old_decode(v) for k, v in state.items()})
        return obj
    raise ValueError(tag)


def _pages():
    doc = json.loads(gzip.decompress(FIXTURE.read_bytes()))
    for page in doc['pages']:
        yield (page['ocr'], _old_decode(page['detect_lines']), _old_decode(page['ocr_lines']),
               _old_decode(page['merge_regions']), page['translate_items'])


PAGES = list(_pages())


def _same(a, b, path):
    assert type(a) is type(b), f'{path}: {type(a).__name__} != {type(b).__name__}'
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), path
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f'{path}[{i}]')
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b), path
    else:
        assert a == b, f'{path}: {a!r} != {b!r}'


def _state(obj):
    return {k: v for k, v in vars(obj).items() if k not in CACHED | IDS}


def _assert_restored(originals, restored, what):
    assert len(originals) == len(restored), what
    for i, (a, b) in enumerate(zip(originals, restored)):
        sa, sb = _state(a), _state(b)
        assert set(sa) == set(sb), f'{what}[{i}]: {set(sa) ^ set(sb)}'
        for key in sa:
            _same(sa[key], sb[key], f'{what}[{i}].{key}')


def _capture(detect, ocr, regions, items):
    record = {}
    record['lines'], record['read'] = page_data.encode_lines(page_data.detected(detect), ocr)
    for region, item in zip(regions, items):
        region.translation = item['t']
    kept = [r for r, item in zip(regions, items) if item['k']]
    record['regions'] = page_data.encode_regions(regions, ocr)
    page_data.encode_translations(record['regions'], regions, kept)
    for region in regions:
        region.translation = ''   # the merge-stage state the fixture holds
    return record, kept


@pytest.mark.parametrize('ocr_model,detect,ocr,regions,items', PAGES, ids=[f'{p[0]}-{i}' for i, p in enumerate(PAGES)])
def test_real_pages_round_trip_exactly(ocr_model, detect, ocr, regions, items):
    record, kept = _capture(detect, ocr, regions, items)
    restored, blobs = page_data.unpack(page_data.pack(record, {}))
    assert blobs == {}
    _assert_restored(detect, page_data.detect_lines(restored), 'detect')
    _assert_restored(ocr, page_data.ocr_lines(restored), 'ocr')
    again = page_data.regions(restored)
    _assert_restored(regions, again, 'regions')
    assert [r._line_ids for r in again] == [r['lines'] for r in restored['regions']]

    config = Config()
    survivors = page_data.apply_translations(restored, again, config)
    assert [r.translation for r in again] == [item['t'] for item in items]
    assert page_data.region_points(survivors) == page_data.region_points(kept) == page_data.survivors(restored)
    assert all(r.target_lang == config.translator.target_lang for r in again)


def test_record_is_compact_and_readable():
    _, detect, ocr, regions, items = PAGES[0]
    record, _ = _capture(detect, ocr, regions, items)
    text = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
    region = record['regions'][0]
    assert set(region) <= {'lines', 'size', 'angle', 'text', 'prob', 'fg', 'bg', 'lang', 'src', 'panel', 'tr', 'keep'}
    assert set(record['lines'][0]) <= {'pts', 'score', 'text', 'prob', 'fg', 'bg', 'dir'}
    # Geometry and source text live once, on the lines.
    assert 'pts' not in region and 'texts' not in region
    sizes = [len(json.dumps(page_data.pack(*_capture(d, o, r, i)[:1], {})[4:].decode())) for _, d, o, r, i in PAGES]
    assert max(sizes) < 16 * 1024 and len(text) < 8 * 1024


def test_container_carries_masks_losslessly():
    raw = (np.random.default_rng(1).random((64, 48)) * 255).astype(np.uint8)
    text = np.zeros((64, 48), np.uint8)
    text[10:20, 5:30] = 255
    data = page_data.pack({'lines': []}, {'raw': page_data.encode_mask(raw), 'text': page_data.encode_mask(text)})
    record, blobs = page_data.unpack(data)
    assert record == {'lines': []}
    assert blobs['raw'][:4] == b'RIFF' and blobs['raw'][8:12] == b'WEBP'
    assert np.array_equal(page_data.decode_mask(blobs['raw']), raw)
    assert np.array_equal(page_data.decode_mask(blobs['text']), text)


def test_masks_past_webp_limits_use_png_and_png_masks_still_decode():
    import io
    from PIL import Image
    strip = (np.random.default_rng(2).random((page_data.WEBP_MAX_SIDE + 1, 4)) * 255).astype(np.uint8)
    data = page_data.encode_mask(strip)
    assert data[:4] == b'\x89PNG' and np.array_equal(page_data.decode_mask(data), strip)
    small = strip[:40]
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format='PNG')
    assert np.array_equal(page_data.decode_mask(buf.getvalue()), small)
    with pytest.raises(ValueError):
        page_data.decode_mask(b'not an image')


@pytest.mark.parametrize('data', [b'', b'\x00\x00\x00\x05{}', b'\x00\x00\x00\x02[]',
                                  b'\x00\x00\x00\x0c{"blobs":{}}x', b'\x00\x00\x00\x13{"blobs":{"raw":5}}ab'])
def test_malformed_containers_are_rejected(data):
    with pytest.raises(ValueError):
        page_data.unpack(data)


def test_balloon_masks_round_trip():
    a = np.zeros((40, 30), np.uint8)
    a[5:20, 3:18] = 255
    b = np.zeros((40, 30), np.uint8)
    empty = np.zeros((40, 30), np.uint8)
    b[30:39, 20:29] = 255
    items = page_data.encode_bubbles([a, b, empty])
    assert set(items[0]) == {'box', 'bits'} and items[0]['box'] == [3, 5, 15, 15]
    for original, restored in zip([a, b, empty], page_data.decode_bubbles(json.loads(json.dumps(items)), a.shape)):
        assert restored.dtype == np.uint8 and np.array_equal(original, restored)


def test_context_pages_are_validated():
    assert page_data.context_pages('[{"src": ["あ"], "tr": ["Ah"]}, {"src": [], "tr": []}]') == [(['あ'], ['Ah']), ([], [])]
    too_many = json.dumps([{'src': [], 'tr': []}] * (page_data.MAX_CONTEXT_PAGES + 1))
    for bad in ['{}', 'nope', '[{"src": ["a"]}]', '[{"src": ["a"], "tr": []}]', '[{"src": [1], "tr": ["x"]}]',
                too_many, '"' + 'x' * page_data.MAX_CONTEXT_BYTES + '"']:
        with pytest.raises(ValueError):
            page_data.context_pages(bad)
