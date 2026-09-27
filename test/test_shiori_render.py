"""Pipeline-boundary regressions for the pinned scene renderer (no model downloads)."""
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from PIL import Image

from manga_translator.config import Config, RenderConfig
from manga_translator.rendering import shiori_render as render
from manga_translator.rendering import text_render, text_render_eng
from manga_translator.utils import TextBlock


def region(x=10, y=10, width=20, height=20, **kw):
    return TextBlock([[[x, y], [x+width, y], [x+width, y+height], [x, y+height]]],
                     texts=['source'], translation='Hello world!', target_lang='ENG', **kw)


def test_shared_bubble_chooses_smallest_containing_mask():
    page = (100, 100)
    large = np.ones(page, np.uint8)
    small = np.zeros(page, np.uint8)
    small[5:70, 5:70] = 1
    bubbles = [({'id': 400}, large, int(large.sum())), ({'id': 3}, small, int(small.sum()))]
    assert render._bubble_for_region(region(), bubbles, page) == 3
    assert render._bubble_for_region(region(40, 40), bubbles, page) == 3
    assert render._bubble_for_region(region(75, 75), bubbles, page) == 400
    # A nearby bubble that contains less than 90% must not capture this text.
    assert render._bubble_for_region(region(60, 60), [bubbles[1]], page) is None


def test_bubble_ids_are_not_limited_to_a_byte():
    native = SimpleNamespace(bubble_geometry=lambda *args: '[{"x":0,"y":0},{"x":5,"y":0},{"x":0,"y":5}]')
    masks = [np.ones((6, 6), np.uint8)] * 257
    bubbles = render._page_bubbles(native, np.zeros((6, 6, 3), np.uint8), masks)
    assert [bubble['id'] for bubble, _, _ in bubbles] == list(range(257))


def test_source_direction_and_explicit_overrides():
    r = region(width=10, height=50, bold=True, italic=True)
    assert render._block_input(r, 0, None)['sourceDirection'] == 'vertical'
    config = RenderConfig(direction='horizontal', alignment='right', font_size=31, disable_font_border=True)
    inferred = dict(angleDegrees=8, writingMode='Vertical', color=[12, 80, 150],
                    strokeColor=[255, 255, 255], strokeWidth=3)
    block = render._block_input(r, 0, None, config, inferred)
    assert block['translation'] == r.translation
    assert block['sourceText'] == 'source'
    assert block['sourceDirection'] == 'vertical'
    assert block['writingMode'] == 'horizontal'
    assert block['transform']['rotationDeg'] == 8
    assert block['fontSize'] == 31
    assert block['alignment'] == 'End'
    assert block['fontWeight'] == 700 and block['fontStyle'] == 'italic'
    assert block['strokeWidth'] == 0
    assert block['color'] == [12, 80, 150, 255]


def test_detector_mask_is_restricted_to_source_quadrilaterals():
    r = region()
    detector = np.zeros((50, 50), np.uint8)
    detector[12:18, 12:18] = 1
    detector[40:45, 40:45] = 1
    x, y, mask = render._region_mask(r, detector.shape, detector)
    assert (x, y) == (10, 10)
    assert mask.sum() == 36


def test_native_font_selection_changes_with_the_font_file(monkeypatch):
    import sys
    instances = []
    class Native:
        def __init__(self, families):
            instances.append(self)
        def register_font(self, path):
            return path, path
    monkeypatch.setitem(sys.modules, 'shiori_renderer', SimpleNamespace(
        UPSTREAM_REVISION=render.UPSTREAM_REVISION, PageRenderer=Native))
    monkeypatch.setattr(render, '_renderer', None)
    monkeypatch.setattr(render, '_document_font', None)
    monkeypatch.setattr(render, '_document_font_path', None)
    first, _ = render._get_renderer('font-one.ttf')
    assert render._get_renderer('font-one.ttf')[0] is first
    assert render._get_renderer('font-two.ttf')[0] is not first
    assert len(instances) == 2


def test_native_joined_bubbles_through_the_python_adapter(monkeypatch):
    native = pytest.importorskip('shiori_renderer')
    if getattr(native, 'UPSTREAM_REVISION', None) != render.UPSTREAM_REVISION:
        pytest.skip('requires the current rebuilt native renderer')
    import cv2
    canvas = np.full((360, 480, 3), 255, np.uint8)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.ellipse(mask, (145, 165), (125, 145), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (335, 200), (120, 140), 0, 0, 360, 1, -1)
    regions = [region(105, 90, 65, 95), region(315, 150, 55, 100)]
    monkeypatch.setattr(render, '_renderer', None)
    output = asyncio.run(render.dispatch_shiori_render(canvas, canvas, regions, bubbles=[mask]))
    assert output.shape == canvas.shape and np.any(output != canvas)
    assert all(r._drawn_font_size > 24 for r in regions)
    assert all(r._drawn_stroke_width == render.BLACK_OUTLINE_WIDTH for r in regions)
    assert all(r._drawn_bg == [255, 255, 255] for r in regions)
    assert regions[0]._drawn_rect[2] < regions[1]._drawn_rect[0]
    assert all(''.join(r._drawn_lines).replace(' ', '') == 'Helloworld!' for r in regions)


def test_native_shapes_split_a_joined_balloon_at_its_neck(monkeypatch):
    native = pytest.importorskip('shiori_renderer')
    if getattr(native, 'UPSTREAM_REVISION', None) != render.UPSTREAM_REVISION:
        pytest.skip('requires the current rebuilt native renderer')
    import cv2
    canvas = np.full((360, 480, 3), 255, np.uint8)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.ellipse(mask, (145, 165), (125, 145), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (335, 200), (120, 140), 0, 0, 360, 1, -1)
    joined = [region(105, 90, 65, 95), region(315, 150, 55, 100)]
    free = region(420, 10, 40, 20)
    monkeypatch.setattr(render, '_renderer', None)
    asyncio.run(render.dispatch_shiori_render(canvas, canvas, joined + [free], bubbles=[mask]))
    # Each text of the joined balloon gets its own lobe: together they cover the balloon, and
    # they meet only along the cut between the two necks.
    left, right = (set(r._drawn_shape) for r in joined)
    xs, ys = zip(*(left | right))
    assert (min(xs), min(ys), max(xs), max(ys)) == (20, 20, 455, 340)
    necks = sorted(left & right)
    assert len(necks) == 2 and all(145 < x < 335 for x, _ in necks)
    assert max(x for x, _ in left) == max(x for x, _ in necks)
    assert min(x for x, _ in right) == min(x for x, _ in necks)
    assert free._drawn_shape == [(420, 10), (460, 10), (460, 30), (420, 30)]


def test_native_typography_from_detector_mask(monkeypatch):
    native = pytest.importorskip('shiori_renderer')
    if getattr(native, 'UPSTREAM_REVISION', None) != render.UPSTREAM_REVISION:
        pytest.skip('requires the current rebuilt native renderer')
    original = np.full((96, 96, 3), [16, 24, 88], np.uint8)
    original[38:58, 20:76] = 255
    original[41:55, 23:73] = 0
    detector_mask = np.zeros((96, 96), np.uint8)
    detector_mask[38:58, 20:76] = 255
    r = region(20, 38, 56, 20)
    monkeypatch.setattr(render, '_renderer', None)
    output = asyncio.run(render.dispatch_shiori_render(
        np.full_like(original, 180), original, [r], bubbles=[], text_mask=detector_mask))
    assert output.shape == original.shape
    assert r._drawn_fg == [0, 0, 0]
    assert r._drawn_bg == [255, 255, 255]
    assert r._drawn_stroke_width == 3


@pytest.mark.parametrize('fg', [[0, 0, 0], [16, 20, 24], [32, 32, 32]])
def test_black_text_always_defaults_to_white_outline(fg):
    for width in [None, 0, 0.5, 3]:
        inferred = dict(angleDegrees=0, writingMode='Horizontal', color=fg,
                        strokeColor=[30, 100, 180], strokeWidth=width)
        block = render._block_input(region(), 0, None, inferred=inferred)
        assert block['color'][:3] == fg
        assert block['strokeColor'] == [255, 255, 255, 255]
        assert block['strokeWidth'] == max(render.BLACK_OUTLINE_WIDTH, width or 0)


@pytest.mark.parametrize('fg', [[255, 255, 255], [30, 100, 180], [33, 33, 33]])
def test_other_fills_keep_the_inferred_outline(fg):
    inferred = dict(angleDegrees=0, writingMode='Horizontal', color=fg,
                    strokeColor=[0, 0, 0], strokeWidth=2)
    block = render._block_input(region(), 0, None, inferred=inferred)
    assert block['strokeColor'] == [0, 0, 0, 255]
    assert block['strokeWidth'] == 2


def test_outline_policy_respects_explicit_settings():
    inferred = dict(angleDegrees=0, writingMode='Horizontal', color=[30, 100, 180],
                    strokeColor=None, strokeWidth=None)
    block = render._block_input(region(), 0, None, RenderConfig(font_color='000000'), inferred)
    assert block['strokeColor'] == [255, 255, 255, 255]
    assert block['strokeWidth'] == render.BLACK_OUTLINE_WIDTH
    custom = RenderConfig(font_color='000000:1E64B4')
    block = render._block_input(region(), 0, None, custom, inferred)
    assert block['strokeColor'] == [30, 100, 180, 255]
    assert block['strokeWidth'] == render.BLACK_OUTLINE_WIDTH
    custom.disable_font_border = True
    assert render._block_input(region(), 0, None, custom, inferred)['strokeWidth'] == 0


def test_native_white_halo_adds_visible_pixels_without_changing_line_layout():
    native = pytest.importorskip('shiori_renderer')
    if getattr(native, 'UPSTREAM_REVISION', None) != render.UPSTREAM_REVISION:
        pytest.skip('requires the current rebuilt native renderer')
    canvas = np.full((180, 280, 3), 128, np.uint8)
    outlined, plain = region(30, 30, 220, 120), region(30, 30, 220, 120)
    output = asyncio.run(render.dispatch_shiori_render(
        canvas, canvas, [outlined], bubbles=[], render_config=RenderConfig(font_size=28)))
    baseline = asyncio.run(render.dispatch_shiori_render(
        canvas, canvas, [plain], bubbles=[], render_config=RenderConfig(font_size=28, disable_font_border=True)))
    assert outlined._drawn_lines == plain._drawn_lines
    assert outlined._drawn_font_size == plain._drawn_font_size == 28
    assert outlined._drawn_stroke_width == render.BLACK_OUTLINE_WIDTH
    assert plain._drawn_stroke_width == 0
    assert baseline.max() == 128
    assert np.count_nonzero(np.all(output > 240, axis=2)) > 20, 'visible white halo on a gray background'
    assert np.count_nonzero(np.all(output < 16, axis=2)) > 20, 'black fill remains visible'


@pytest.mark.parametrize('fg,bg,bubble_bg', [
    ([0, 0, 0], [0, 0, 0], [255, 255, 255]),
    ([255, 255, 255], [255, 255, 255], [0, 0, 0]),
    ([30, 80, 150], [250, 210, 90], [128, 128, 128]),
    ([0, 0, 0], [40, 100, 180], [128, 128, 128]),
])
def test_hybrid_uses_manga2eng_colors_but_shiori_geometry(fg, bg, bubble_bg):
    r = region(fg_color=fg, bg_color=bg)
    r._bubble_bg = bubble_bg
    expected_fg, expected_bg = r.get_font_colors()
    inferred = dict(angleDegrees=12, writingMode='Vertical', color=[200, 20, 30],
                    strokeColor=[40, 200, 70], strokeWidth=8)
    block = render._block_input(r, 0, None, inferred=inferred, manga2eng_paint=True)
    assert block['color'] == list(expected_fg) + [255]
    assert block['strokeColor'] == list(expected_bg) + [255]
    assert block['transform']['rotationDeg'] == 12
    assert block['sourceDirection'] == 'vertical'


@pytest.mark.parametrize('size,expected', [(9, 0), (10, 1), (24, 1), (28.9, 1), (29, 2), (43, 3), (100, 7)])
def test_hybrid_radius_matches_manga2eng_freetype_stroker(monkeypatch, size, expected):
    radii = []
    real_stroker = text_render.freetype.Stroker

    class RecordingStroker(real_stroker):
        def set(self, radius, *args):
            radii.append(radius / 64)
            return super().set(radius, *args)

    monkeypatch.setattr(text_render.freetype, 'Stroker', RecordingStroker)
    text_render.set_font(str(render.BASE_PATH + '/fonts/ccvictoryspeech.ttf'))
    fill, border = np.zeros((220, 220), np.uint8), np.zeros((220, 220), np.uint8)
    text_render.put_char_horizontal(int(size), 'A', (50, 120), fill, border,
                                    int(int(size) * text_render_eng.MANGA2ENG_STROKE_RATIO))
    assert radii == ([expected] if expected else [])
    assert render.manga2eng_stroke_width(size) == expected


def _record_native_calls(monkeypatch):
    """Record what the native renderer is given and returns for each rendered page."""
    calls = []
    real = render._get_renderer

    class Recorder:
        def __init__(self, renderer):
            self._renderer = renderer

        def __getattr__(self, name):
            return getattr(self._renderer, name)

        def render_page(self, rgba, w, h, blocks, options):
            out = self._renderer.render_page(rgba, w, h, blocks, options)
            calls.append({'blocks': json.loads(blocks), 'options': json.loads(options),
                          'engine_layout': json.loads(out[1])})
            return out

    monkeypatch.setattr(render, '_get_renderer', lambda font_path: (Recorder(real(font_path)[0]), real(font_path)[1]))
    return calls


@pytest.mark.parametrize('case', ['joined', 'free', 'vertical', 'rotated', 'rtl'])
def test_hybrid_preserves_native_layout_and_records_final_paint(case, monkeypatch):
    native = pytest.importorskip('shiori_renderer')
    assert hasattr(native.PageRenderer, 'layout_page'), 'rebuild the native renderer for hybrid'
    import cv2
    canvas = np.full((360, 480, 3), 128, np.uint8)
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.ellipse(mask, (145, 165), (125, 145), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (335, 200), (120, 140), 0, 0, 360, 1, -1)
    regions = [region(105, 90, 65, 95), region(315, 150, 55, 100)]
    bubbles = [mask] if case == 'joined' else []
    config = RenderConfig()
    if case == 'vertical':
        config.direction = 'vertical'
        for r in regions:
            r.target_lang, r.translation = 'JPN', '本当ですか？'
    elif case == 'rtl':
        for r in regions:
            r.target_lang, r.translation = 'ARA', 'مرحبا بالعالم'
    elif case == 'rotated':
        config.font_size, config.alignment = 28, 'right'
        center = np.array([130, 130])
        # Exact integer rectangle after rotation: avoid turning it into a skewed
        # quadrilateral through pixel rounding (upstream renders those unrotated).
        matrix = np.array([[0.8, 0.6], [-0.6, 0.8]])
        for r in regions:
            r.lines = np.rint((r.lines - center) @ matrix.T + center).astype(np.int32)
            r.min_rect = r.lines.copy()
    hybrid_regions = deepcopy(regions)
    captures = _record_native_calls(monkeypatch)
    outputs = []
    for dispatcher, items in [(render.dispatch_shiori_render, regions),
                              (render.dispatch_shiori_render_v2, hybrid_regions)]:
        outputs.append(asyncio.run(dispatcher(canvas, canvas, items, bubbles=bubbles, render_config=config)))
    shiori, hybrid = captures
    for plain, painted, block, r in zip(shiori['engine_layout'], hybrid['engine_layout'], hybrid['blocks'], hybrid_regions):
        for key in ('fontSize', 'lines', 'geometry', 'rotationDeg', 'renderedDirection'):
            assert painted[key] == plain[key], key
        if case == 'rotated':
            assert abs(painted['rotationDeg']) > 10
        for origin, extent in [('x', 'width'), ('y', 'height')]:
            assert painted[origin] + painted[extent] / 2 == pytest.approx(plain[origin] + plain[extent] / 2, abs=0.0002)
        assert painted['strokeWidth'] == render.manga2eng_stroke_width(plain['fontSize'])
        assert block['strokeWidth'] == painted['strokeWidth']
        assert painted['textColor'] == list(r.get_font_colors()[0])
        assert painted['strokeColor'] == list(r.get_font_colors()[1])
        assert r._drawn_paint_policy == 'manga2eng'
        assert not getattr(r, '_typeset_eng', False)
    if case == 'joined':
        assert all(r._drawn_font_size > 24 for r in hybrid_regions)
        assert hybrid_regions[0]._drawn_rect[2] < hybrid_regions[1]._drawn_rect[0]
    renderer, _ = render._get_renderer(str(render.BASE_PATH + '/fonts/ccvictoryspeech.ttf'))
    renderer = renderer._renderer
    rgba = np.dstack([canvas, np.full(canvas.shape[:2], 255, np.uint8)])
    replay, metadata = renderer.render_page(rgba.tobytes(), 480, 360,
                                             json.dumps(hybrid['blocks']), json.dumps(hybrid['options']))
    replay_pixels = np.frombuffer(replay, np.uint8).reshape(360, 480, 4)[:, :, :3]
    # Pristine upstream also varies in the faint antialiased outline fringe over
    # gray on this GPU. Solid fill/outline pixels and layout must replay exactly.
    for color in (0, 255):
        np.testing.assert_array_equal(np.all(replay_pixels == color, axis=2), np.all(outputs[1] == color, axis=2))
    # The dark fill, its antialiasing, and untouched background are stable too.
    np.testing.assert_array_equal(np.minimum(replay_pixels, 128), np.minimum(outputs[1], 128))
    # Diagnostics include ephemeral scene entity IDs; the actual layout is stable.
    assert [{k: v for k, v in item.items() if k != 'diagnostics'} for item in json.loads(metadata)] == [
        {k: v for k, v in item.items() if k != 'diagnostics'} for item in hybrid['engine_layout']]


@pytest.mark.parametrize('disabled', [False, True])
def test_hybrid_explicit_paint_and_border_settings(disabled):
    pytest.importorskip('shiori_renderer')
    canvas = np.full((180, 280, 3), 128, np.uint8)
    r = region(30, 30, 220, 120)
    config = RenderConfig(font_size=43, font_color='000000:1E64B4', disable_font_border=disabled)
    output = asyncio.run(render.dispatch_shiori_render_v2(canvas, canvas, [r], bubbles=[], render_config=config))
    assert r._drawn_font_size == 43
    assert r._drawn_fg == [0, 0, 0]
    assert r._drawn_stroke_width == (0 if disabled else 3)
    if not disabled:
        assert r._drawn_bg == [30, 100, 180]
        assert np.count_nonzero(np.all(output == [30, 100, 180], axis=2)) > 20
    else:
        assert output.max() == 128
