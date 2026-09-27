"""Page data end to end with fake models: capture, restore, conditional keeps, fallbacks.

Real merging, rendering and page-data capture run; detection, OCR, translation, mask
refinement, inpainting and balloon segmentation are counted fakes. The server holds nothing:
each run returns every page's data, and a later run is sent that data back the way the app
would (a Python mirror of its rule decides `from` and `keep`).
"""
import asyncio
import io
import json
import uuid
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import manga_translator.manga_translator as pipeline
from manga_translator import page_data, stages
from manga_translator.config import Config, Renderer, Translator
from manga_translator.detection import detector_cache
from manga_translator.manga_translator import MangaTranslator
from manga_translator.ocr import ocr_cache
from manga_translator.rendering import text_render_eng
from manga_translator.utils import Quadrilateral
from manga_translator.utils.profiling import Profiler

W, H = 320, 240
ORDER = page_data.ORDER


def page_bytes(seed):
    pixels = np.full((H, W, 3), 250, np.uint8)
    pixels[80 + seed:100 + seed, 90:220] = 30 + seed      # a dark "line of text"
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, 'PNG')
    return buf.getvalue()


def blank_page():
    buf = io.BytesIO()
    Image.new('RGB', (W, H), 'white').save(buf, 'PNG')
    return buf.getvalue()


def config_for(**changes):
    config = Config()
    config.render.renderer = Renderer.manga2Eng
    config.translator.translator = Translator.sugoi
    config.translator.no_text_lang_skip = True
    config.force_simple_sort = True
    config.study_mode_generation = 'disabled'
    for path, value in changes.items():
        target = config
        *parents, leaf = path.split('.')
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, value)
    return config


def resolve(config, context_size=0):
    return stages.resolve(config, stages.runtime_values({'kernel_size': 3, 'context_size': context_size}))


def _value(doc, path):
    for part in path.split('.'):
        doc = doc[part]
    return doc


def _changed(stage, old, new):
    return (old['builds'].get(stage) != new['builds'].get(stage)
            or any(_value(old['config'], f) != _value(new['config'], f) for f in new['fields'][stage]))


def plan(data, old, new, force=None):
    """The app's rule for one page: None when current, else the container to send."""
    record, blobs = page_data.unpack(data)
    # Stages the new config doesn't run (balloons for a renderer without them) never count.
    start = next((s for s in stages.STAGES[1:] if s in new['builds'] and _changed(s, old, new)), None)
    if force and (start is None or ORDER[force] < ORDER[start]):
        start = force
    if record.get('end') and (start is None or ORDER[start] > ORDER[record['end']]):
        return None
    if start is None:
        return None
    keep = [s for s in page_data.KEEPABLE if ORDER[s] >= ORDER[start] and not _changed(s, old, new)
            and not (force and ORDER[s] >= ORDER[force])]
    return page_data.pack({**record, 'from': start, 'keep': keep}, blobs)


def _layer(url):
    import base64
    return np.array(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGBA'))


def _rebuild(payload):
    page = _layer(payload['bg'])[..., :3]
    for bubble in payload['bubbles']:
        if 'text' not in bubble:
            continue   # text-only: the client typesets it
        layer = _layer(bubble['text'])
        page[layer[..., 3] == 255] = layer[..., :3][layer[..., 3] == 255]
    return page


class Harness:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.calls = Counter()
        self.context_seen = []
        monkeypatch.setattr(MangaTranslator, '_setup_log_file', lambda self: None)
        monkeypatch.setattr(pipeline, 'prewarm_proc_pool', lambda: None)
        monkeypatch.setattr(pipeline, 'Profiler', lambda **kw: Profiler(enabled=False))

        async def run_proc(fn, *args):
            return fn(*args)
        monkeypatch.setattr(pipeline, 'run_proc', run_proc)
        monkeypatch.setattr(text_render_eng, 'detect_bubbles', lambda _: pytest.fail('renderer segmented balloons itself'))

        def bubbles(img):
            self.calls['bubbles'] += 1
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[50:190, 40:280] = 255
            return [mask]
        monkeypatch.setattr(pipeline, 'detect_bubbles', bubbles)
        for key in ('default', 'ctd'):
            monkeypatch.setitem(detector_cache, key, SimpleNamespace())
        for key in ('48px', 'mocr', 'mocr_fast'):
            monkeypatch.setitem(ocr_cache, key, SimpleNamespace())

        async def detect(key, img, size, *args, **kwargs):
            self.calls['detect'] += 1
            dark = (img.mean(axis=2) < 128)
            mask_raw = (dark * 255).astype(np.uint8)
            ys, xs = np.nonzero(dark)
            if xs.size == 0:
                return [], mask_raw, None
            y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            line = Quadrilateral(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), '', np.float32(0.9))
            return [line], mask_raw, None

        self.ocr_text = lambda rgb, config: '原文' + str(int(rgb.mean()))

        async def ocr(key, rgb, lines, config, *args, **kwargs):
            self.calls['ocr'] += 1
            for line in lines:
                _ = line.aabb   # real OCR direction sorting caches this
                line.text = self.ocr_text(rgb, config)
                line.prob = 0.93
            return lines

        async def translate(pairs, batch_size, page_done=None):
            for ctx, cfg in pairs:
                if ctx.text_regions:
                    self.calls['translate'] += 1
                    self.context_seen.append([dict(p) for p in self.mt.all_page_translations])
                for r in ctx.text_regions or []:
                    r.translation = f'{cfg.translator.translator} reads {r.text}.'
                    r.target_lang = cfg.translator.target_lang
                    r._alignment = cfg.render.alignment
                    r._direction = cfg.render.direction
            return pairs

        async def mask(config, ctx):
            self.calls['mask'] += 1
            out = np.zeros(ctx.img_rgb.shape[:2], np.uint8)
            for region in ctx.text_regions:
                x1, y1, x2, y2 = region.xyxy
                out[y1:y2, x1:x2] = 255
            return out

        async def inpaint(ctx, config):
            self.calls['inpaint'] += 1
            if ctx.mask is None:
                ctx.mask = await mask(config, ctx)
            ctx.img_inpainted = ctx.img_rgb.copy()
            ctx.img_inpainted[ctx.mask > 0] = 255 - (config.inpainter.inpainting_size % 7)
            ctx.gimp_mask = None
            return ctx

        self._fakes = (translate, mask, inpaint)
        monkeypatch.setattr(pipeline, 'dispatch_detection', detect)
        monkeypatch.setattr(pipeline, 'dispatch_ocr', ocr)

    def run(self, config, raws, pages=None, builds=None, context_size=0, context=None, capture=True):
        translate, mask, inpaint = self._fakes
        mt = self.mt = MangaTranslator({'models_ttl': 60, 'kernel_size': 3, 'context_size': context_size})
        self.monkeypatch.setattr(mt, '_batch_translate_contexts', translate)
        self.monkeypatch.setattr(mt, '_run_mask_refinement', mask)
        self.monkeypatch.setattr(mt, '_inpaint_stage', inpaint)
        self.calls.clear()
        self.context_seen.clear()
        outputs, data, order = {}, {}, []

        async def on_data(index, container):
            order.append(('data', index))
            data[index] = container

        layered, study = set(), {}

        async def on_result(index, image):
            order.append(('page', index))
            if image == pipeline.STUDY_LAYERS:
                layered.add(index)   # the page is its study layers, which follow
            else:
                outputs[index] = np.array(image.convert('RGB'))

        async def on_study(index, payload):
            study[index] = payload
        mt.add_page_data_hook(on_data)
        mt.add_page_result_hook(on_result)
        mt.add_page_bubbles_hook(on_study)

        async def go():
            try:
                return await mt.translate_gallery_stream(list(raws), config, batch_size=1, job_token='t',
                                                         pages=pages, builds=builds, context=context,
                                                         capture=capture)
            finally:
                if mt._detector_cleanup_task:
                    mt._detector_cleanup_task.cancel()
        summary = asyncio.run(go())
        assert summary['failed'] == [], summary
        for index in data:
            assert order.index(('data', index)) < order.index(('page', index)), 'data precedes its page'
        for index in layered:   # rebuilt the way the app shows it: the bg with every text layer over it
            outputs[index] = _rebuild(study[index])
        return SimpleNamespace(summary=summary, data=data, outputs=outputs, calls=dict(self.calls),
                               reuse=summary['telemetry']['reuse'], layered=layered, study=study)

    def rerun(self, before, old_config, new_config, raws, force=None, context_size=0):
        old, new = resolve(old_config, context_size), resolve(new_config, context_size)
        pages = [plan(before.data[i], old, new, force) for i in range(len(raws))]
        return self.run(new_config, raws, pages, context_size=context_size), pages


@pytest.fixture
def harness(monkeypatch):
    return Harness(monkeypatch)


RAWS = [page_bytes(0), page_bytes(12)]


def record(run, index=0):
    return page_data.unpack(run.data[index])


def test_fresh_run_returns_every_page_s_data(harness):
    fresh = harness.run(config_for(), RAWS)
    assert fresh.calls == {'detect': 2, 'ocr': 2, 'translate': 2, 'mask': 2, 'inpaint': 2, 'bubbles': 2}
    data, blobs = record(fresh)
    assert set(data) == {'lines', 'read', 'regions', 'bubbles'}
    assert data['regions'][0]['tr'].startswith('sugoi reads ')
    assert data['lines'][0]['score'] == pytest.approx(0.9) and data['read'] == [0]
    assert set(blobs) == {'raw', 'text'}
    assert np.array_equal(page_data.decode_mask(blobs['raw']) > 0, np.array(Image.open(io.BytesIO(RAWS[0]))).mean(axis=2) < 128)


def test_renderer_change_only_inpaints_and_renders(harness):
    fresh = harness.run(config_for(), RAWS)
    changed = config_for(**{'render.renderer': Renderer.manga2EngPillow})
    resumed, pages = harness.rerun(fresh, config_for(), changed, RAWS)
    assert [page_data.unpack(p)[0]['from'] for p in pages] == ['render', 'render']
    assert resumed.calls == {'inpaint': 2}, 'no detection, OCR, translation or mask refinement'
    reference = harness.run(changed, RAWS)
    for i in range(2):
        np.testing.assert_array_equal(resumed.outputs[i], reference.outputs[i])
    assert record(resumed)[0]['regions'] == record(fresh)[0]['regions']


def test_study_change_restores_balloons_too(harness):
    text_only = config_for(study_mode_generation='text_only')
    fresh = harness.run(text_only, RAWS)
    changed = config_for(study_mode_generation='text_and_image')
    resumed, _ = harness.rerun(fresh, text_only, changed, RAWS)
    assert resumed.calls == {'inpaint': 2}
    assert resumed.reuse['bubbles'] == {'reused': 2, 'ran': 0}
    assert record(resumed)[1] == record(fresh)[1], 'reused masks pass through unchanged'
    assert resumed.layered == {0, 1}, 'each page comes back as its study layers alone'
    assert all(p['bg'].startswith('data:image/webp') and all('text' in b for b in p['bubbles']) for p in resumed.study.values())
    assert fresh.layered == {0, 1} and all('bg' in p and not any('text' in b for b in p['bubbles']) for p in fresh.study.values()),         'text-only study sends each page as its bg and text, no text layers'


def test_translator_change_keeps_detection_and_the_refined_mask(harness):
    fresh = harness.run(config_for(), RAWS)
    changed = config_for(**{'translator.translator': Translator.jparacrawl})
    resumed, pages = harness.rerun(fresh, config_for(), changed, RAWS)
    head = page_data.unpack(pages[0])[0]
    assert head['from'] == 'translate' and head['keep'] == ['mask', 'bubbles']
    assert resumed.calls == {'translate': 2, 'inpaint': 2}
    reference = harness.run(changed, RAWS)
    for i in range(2):
        np.testing.assert_array_equal(resumed.outputs[i], reference.outputs[i])
    assert record(resumed)[0]['regions'][0]['tr'].startswith('jparacrawl reads ')


def test_inpainting_change_keeps_text(harness):
    fresh = harness.run(config_for(), RAWS)
    changed = config_for(**{'inpainter.inpainting_size': 1024})
    resumed, _ = harness.rerun(fresh, config_for(), changed, RAWS)
    assert resumed.calls == {'inpaint': 2}
    reference = harness.run(changed, RAWS)
    np.testing.assert_array_equal(resumed.outputs[0], reference.outputs[0])


def test_ocr_change_with_identical_text_keeps_translation(harness):
    fresh = harness.run(config_for(), RAWS)
    changed = config_for(**{'ocr.ocr': 'mocr'})
    resumed, pages = harness.rerun(fresh, config_for(), changed, RAWS)
    assert page_data.unpack(pages[0])[0]['keep'] == ['translate', 'mask', 'bubbles']
    assert resumed.calls == {'ocr': 2, 'inpaint': 2}, 'same text: translation and mask are kept'
    assert resumed.reuse['translate'] == {'reused': 2, 'ran': 0}


def test_ocr_change_with_new_text_translates_again(harness):
    fresh = harness.run(config_for(), RAWS)
    harness.ocr_text = lambda rgb, config: '別の文' + str(int(rgb.mean()))
    resumed, _ = harness.rerun(fresh, config_for(), config_for(**{'ocr.ocr': 'mocr'}), RAWS)
    assert resumed.calls == {'ocr': 2, 'translate': 2, 'inpaint': 2}, 'survivors unchanged: the mask is still kept'


def test_force_translate_runs_everything_from_there(harness):
    fresh = harness.run(config_for(), RAWS)
    resumed, _ = harness.rerun(fresh, config_for(), config_for(), RAWS, force='translate')
    assert resumed.calls == {'translate': 2, 'mask': 2, 'inpaint': 2, 'bubbles': 2}


def test_unchanged_pages_are_not_sent(harness):
    fresh = harness.run(config_for(), RAWS)
    assert [plan(fresh.data[i], resolve(config_for()), resolve(config_for())) for i in range(2)] == [None, None]


@pytest.mark.parametrize('data', [b'garbage', page_data.pack({'from': 'render'}),
                                  page_data.pack({'from': 'merge', 'lines': [], 'read': [3]}),
                                  page_data.pack({'from': 'prepare', 'lines': []}),
                                  page_data.pack({'from': 'translate', 'keep': ['everything'], 'lines': []})])
def test_invalid_page_data_runs_the_page_in_full(harness, data):
    resumed = harness.run(config_for(), RAWS[:1], [data])
    assert resumed.calls['detect'] == 1 and resumed.calls['translate'] == 1


def test_text_less_page_ends_early_and_stays_current(harness):
    fresh = harness.run(config_for(), [blank_page()])
    data, blobs = record(fresh)
    assert data == {'end': 'detect', 'lines': [], 'read': []} and blobs == {}
    changed = config_for(**{'render.renderer': Renderer.manga2EngPillow})
    assert plan(fresh.data[0], resolve(config_for()), resolve(changed)) is None
    # Sent anyway (its output was reverted, say): restored, nothing runs.
    again = harness.run(config_for(), [blank_page()], [page_data.pack({**data, 'from': 'render', 'keep': []})])
    assert again.calls == {} and record(again)[0] == data


def test_a_worker_with_other_builds_runs_in_full_and_returns_no_data(harness):
    fresh = harness.run(config_for(), RAWS[:1])
    current = resolve(config_for())
    pages = [plan(fresh.data[0], current, current, force='render')]
    other = harness.run(config_for(), RAWS[:1], pages, builds='0' * 16)
    assert other.calls['detect'] == 1 and other.data == {}
    right = harness.run(config_for(), RAWS[:1], pages, builds=stages.signature(current['builds']))
    assert 'detect' not in right.calls and 0 in right.data


def test_a_client_that_keeps_no_page_data_gets_none_and_the_same_pages(harness, monkeypatch):
    fresh = harness.run(config_for(), RAWS)
    encoded = []
    encode = page_data.encode_mask
    monkeypatch.setattr(page_data, 'encode_mask', lambda mask: encoded.append(1) or encode(mask))
    bare = harness.run(config_for(), RAWS, capture=False)
    assert bare.data == {} and encoded == [], 'nothing is encoded or sent'
    assert bare.calls == fresh.calls
    assert fresh.outputs.keys() == bare.outputs.keys()
    assert all(np.array_equal(fresh.outputs[i], bare.outputs[i]) for i in fresh.outputs)
    # Data the client still holds is used all the same.
    current = resolve(config_for())
    pages = [plan(fresh.data[i], current, current, force='render') for i in range(len(RAWS))]
    reused = harness.run(config_for(), RAWS, pages, capture=False)
    assert 'detect' not in reused.calls and reused.data == {}


def test_context_mode_sees_reused_neighbours_in_order(harness):
    raws = [page_bytes(0), page_bytes(12), page_bytes(24)]
    base = config_for(**{'translator.translator': Translator.chatgpt})
    fresh = harness.run(base, raws, context_size=4)
    old = resolve(base, 4)
    pages = [plan(fresh.data[i], old, old, force='render') for i in range(3)]
    pages[1] = None   # this page runs in full
    resumed = harness.run(base, raws, pages, context_size=4)
    assert resumed.calls['translate'] == 1
    assert len(harness.context_seen) == 1 and len(harness.context_seen[0]) == 1, \
        'the re-translated page sees its reused predecessor as context'
    assert list(harness.context_seen[0][0].values()) == [record(fresh, 0)[0]['regions'][0]['tr']]


def test_a_job_starting_mid_gallery_gets_the_earlier_pages_as_context(harness):
    raws = [page_bytes(0), page_bytes(12), page_bytes(24)]
    base = config_for(**{'translator.translator': Translator.chatgpt})
    fresh = harness.run(base, raws, context_size=4)
    earlier = [record(fresh, i)[0]['regions'][0]['tr'] for i in range(2)]
    context = page_data.context_pages(json.dumps([{'src': ['a'], 'tr': [earlier[0]]}, {'src': ['b'], 'tr': [earlier[1]]}]))
    harness.run(base, raws[2:], context_size=4, context=context)
    assert [list(page.values()) for page in harness.context_seen[0]] == [[earlier[0]], [earlier[1]]]


def test_adaptive_batching_counts_only_pages_that_need_translation(harness):
    raws = [page_bytes(0), page_bytes(12), page_bytes(24)]
    base = config_for(**{'translator.translator': Translator.deepseek})
    fresh = harness.run(base, raws)
    old = resolve(base)
    pages = [plan(fresh.data[i], old, old, force='render') for i in range(3)]
    pages[1] = None
    resumed = harness.run(base, raws, pages)
    assert resumed.calls['translate'] == 1 and resumed.calls['detect'] == 1
    assert resumed.reuse['translate'] == {'reused': 2, 'ran': 1}


def test_chunks_and_retries_keep_page_data_aligned(monkeypatch):
    from server import gallery_jobs
    monkeypatch.setattr(gallery_jobs, '_inflight_chunks', 0)
    monkeypatch.setattr('server.myqueue.task_queue.add_task', lambda element: None)
    pages = [b'data-%d' % i for i in range(10)]
    job = gallery_jobs.GalleryJob('tok')
    sj = gallery_jobs._SchedJob(job, SimpleNamespace(state=SimpleNamespace()), [b'p%d' % i for i in range(10)],
                                config_for(), 2, lambda x: b'', pages=pages, builds='abc', capture=False)
    element = gallery_jobs._dispatch(sj, 4, 8)
    assert element.pages == pages[4:8] and element.images == sj.images[4:8] and element.builds == 'abc'
    assert element.capture is False
    sj.retry.insert(0, (6, 8))                                   # an executor dropped the tail
    retry = gallery_jobs._dispatch(sj, *sj.take_range(True))
    assert retry.pages == pages[6:8] and retry.images == sj.images[6:8]


def test_multipart_upload_assembles_page_data_in_order():
    from server import gallery_jobs
    token = uuid.uuid4().hex

    async def upload():   # the part buffer's reaper needs a running loop
        assert gallery_jobs.add_upload_part(token, 1, 2, [b'two'], pages=[b'second']) [0] == 'pending'
        return gallery_jobs.add_upload_part(token, 0, 2, [b'one'], pages=None)
    status, images, pages = asyncio.run(upload())
    assert status == 'done' and images == [b'one', b'two'] and pages == [None, b'second']


def test_page_data_frames_are_buffered_for_the_client():
    from server import gallery_jobs
    job = gallery_jobs.GalleryJob('tok')
    payload = bytes([3]) + b'tok' + (0).to_bytes(4, 'big') + page_data.pack({'lines': []})
    frame = b'\x09' + len(payload).to_bytes(4, 'big') + payload
    job.put_nowait(frame)
    assert job.durable == [frame] and job.emitted == 0
