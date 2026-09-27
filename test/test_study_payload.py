"""Study payload style hints and the worker stream reader (no model downloads)."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from manga_translator.config import Config, Renderer
from manga_translator.manga_translator import MangaTranslator
from manga_translator.textline_merge import dispatch as merge
from manga_translator.utils import Context, Quadrilateral


def context(renderer=Renderer.manga2Eng):
    image = Image.new('RGB', (320, 240), 'white')
    config = Config()
    config.render.renderer = renderer
    line = Quadrilateral(np.array([[70, 70], [240, 70], [240, 170], [70, 170]]), 'original', 0.92)
    regions = asyncio.run(merge([line], 320, 240))
    region = regions[0]
    region.translation = 'A complete sentence remains intact.'
    region.target_lang = 'ENG'
    region.font_size = 24
    rgb = np.array(image)
    return config, Context(input=image, img_rgb=rgb, img_inpainted=rgb.copy(), text_regions=regions)


@pytest.fixture
def translator(monkeypatch):
    import manga_translator.manga_translator as pipeline

    async def run_proc(fn, *args):
        return fn(*args)
    monkeypatch.setattr(pipeline, 'run_proc', run_proc)
    monkeypatch.setattr(MangaTranslator, '_setup_log_file', lambda self: None)
    return MangaTranslator({'models_ttl': 0, 'kernel_size': 3})


@pytest.mark.parametrize('stroke_width', [0.0, 1.5, None])
def test_study_payload_preserves_native_outline_width(translator, stroke_width):
    config, ctx = context()
    ctx.img_rendered = ctx.img_inpainted.copy()
    region = ctx.text_regions[0]
    region._drawn_fg, region._drawn_bg = [0, 0, 0], [0, 0, 0]
    if stroke_width is not None:
        region._drawn_stroke_width = stroke_width
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))
    style = payload['bubbles'][0]['style']
    if stroke_width is None:
        assert 'strokeWidth' not in style, 'do not invent a width for renderers without this hint'
    else:
        assert style['strokeWidth'] == stroke_width
    config.render.disable_font_border = True
    config.render.font_color = '000000:1E64B4'
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))
    style = payload['bubbles'][0]['style']
    assert style['borderDisabled'] is True
    assert style['strokeColorExplicit'] is True


def test_hybrid_study_payload_preserves_paint_without_manga2eng_caps(translator):
    config, ctx = context(Renderer.shioriV2)
    ctx.img_rendered = ctx.img_inpainted.copy()
    r = ctx.text_regions[0]
    r._drawn_fg, r._drawn_bg = [0, 0, 0], [30, 100, 180]
    r._drawn_stroke_width, r._drawn_paint_policy = 1, 'manga2eng'
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))
    style = payload['bubbles'][0]['style']
    assert style['paintPolicy'] == 'manga2eng'
    assert style['strokeWidth'] == 1
    assert style['bg'] == [30, 100, 180]
    assert not style.get('caps')


def test_study_bubbles_carry_record_ids(translator):
    """Feedback identifies regions and lines by their index in the page's pipeline data;
    region 0 is a real id."""
    config, ctx = context()
    ctx.img_rendered = ctx.img_inpainted.copy()
    ctx.text_regions[0]._region_id = 0
    ctx.text_regions[0]._line_ids = [0]
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))
    bubble = payload['bubbles'][0]
    assert bubble['id'] == 0 and bubble['line_ids'] == [0]


def test_study_bubbles_carry_the_renderer_shape_as_page_fractions(translator):
    config, ctx = context()
    ctx.img_rendered = ctx.img_inpainted.copy()
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))
    assert 'shape' not in payload['bubbles'][0], 'renderers that report no shape add none'
    ctx.text_regions[0]._drawn_shape = [(32, 24), (288, 24.5), (160, 216)]
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))
    assert payload['bubbles'][0]['shape'] == [[0.1, 0.1], [0.9, 0.1021], [0.5, 0.9]]


def _decode(url):
    import base64, io
    return np.array(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGBA'))


def _drawn(ctx):
    """Text drawn over the inpaint, as a renderer would, and the page output made from it."""
    rendered = ctx.img_inpainted.copy()
    rendered[100:140, 90:220] = (20, 20, 20)
    rendered[99, 90:220] = (128, 128, 128)   # an antialiased edge
    ctx.img_rendered = rendered
    ctx.result = Image.fromarray(rendered).convert('RGBA')
    return rendered


def test_text_only_study_keeps_the_background_and_leaves_out_the_text_layers(translator):
    config, ctx = context()
    _drawn(ctx)
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))
    assert payload['bg'].startswith('data:image/webp;base64,')
    assert np.array_equal(_decode(payload['bg'])[..., :3].shape, ctx.img_inpainted.shape)
    assert all('text' not in b for b in payload['bubbles'])
    assert payload['rebuilds'] is True, 'the page is shown as its text over the bg, so no image is sent'
    ctx.img_alpha = Image.new('L', (320, 240), 200)
    assert 'rebuilds' not in asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_only'))


def test_study_layers_rebuild_the_page_when_its_render_is_the_output(translator):
    config, ctx = context()
    rendered = _drawn(ctx)
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_and_image'))
    assert payload['rebuilds'] is True
    rebuilt = _decode(payload['bg'])[..., :3]
    for b in payload['bubbles']:
        layer = _decode(b['text'])
        on = layer[..., 3] == 255
        rebuilt[on] = layer[..., :3][on]
    changed = (rendered != ctx.img_inpainted).any(axis=2)
    assert np.array_equal(rebuilt[changed], rendered[changed]), 'the text is exact'
    assert np.abs(rebuilt.astype(int) - rendered.astype(int)).max() <= 8, 'the q95 background is close everywhere else'


@pytest.mark.parametrize('change', ['alpha', 'resized'])
def test_study_layers_do_not_stand_in_when_the_output_differs_from_the_render(translator, change):
    config, ctx = context()
    _drawn(ctx)
    if change == 'alpha':
        ctx.img_alpha = Image.new('L', (320, 240), 200)
    else:
        ctx.result = ctx.result.resize((160, 120))
    payload = asyncio.run(translator._build_bubble_overlays(ctx, config, mode='text_and_image'))
    assert 'rebuilds' not in payload and payload['bubbles']


def test_stream_rejects_truncated_payload():
    from server.sent_data_internal import process_stream

    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(b'\x05\x00\x00\x00\x08short')
        reader.feed_eof()
        seen = []
        with pytest.raises(asyncio.IncompleteReadError):
            await process_stream(SimpleNamespace(content=reader), lambda *args: seen.append(args))
        assert not seen
    asyncio.run(run())
