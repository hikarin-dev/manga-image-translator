import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

import manga_translator.manga_translator as pipeline
from manga_translator.config import Config, Renderer
from manga_translator.rendering import text_render, text_render_eng
from manga_translator.utils import Context, TextBlock


@pytest.fixture(autouse=True)
def real_font_without_bubble_model(monkeypatch):
    font = Path(__file__).resolve().parents[1] / 'fonts' / 'ccvictoryspeech.ttf'
    text_render.set_font(str(font))
    monkeypatch.setattr(text_render_eng, 'detect_bubbles', lambda image: None)


def make_region(box, translation, font_size=32, angle=0):
    x1, y1, x2, y2 = box
    return TextBlock(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
        texts=['source'],
        translation=translation,
        font_size=font_size,
        angle=angle,
        target_lang='ENG',
        fg_color=(0, 0, 0),
        bg_color=(255, 255, 255),
    )


def render(regions, page_bubbles=None, width=320, height=240, page=None):
    if page is None:
        page = np.full((height, width, 3), 255, dtype=np.uint8)
    height, width = page.shape[:2]
    output = text_render_eng.render_textblock_list_eng(
        page.copy(), regions, original_img=page, page_bubbles=page_bubbles,
        downscale_constraint=0.8, safe_layout=True, verbose=True,
    )
    assert output.shape == page.shape
    assert output.dtype == page.dtype
    changed = np.any(output != page, axis=2)
    recorded = np.zeros((height, width), dtype=bool)
    for region in regions:
        if not region.translation.strip():
            continue
        x1, y1, x2, y2 = region._drawn_rect
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        assert region._drawn_font_size > 0
        recorded[y1:y2, x1:x2] = True
        remaining = ''.join(' '.join(text_render_eng.seg_eng(region.translation)).split())
        for line in region._drawn_lines:
            chunk = ''.join(line.split())
            if remaining.startswith(chunk):
                remaining = remaining[len(chunk):]
            else:
                # Emergency wrapping may insert a hyphen at a line boundary.
                assert chunk.endswith('-') and remaining.startswith(chunk[:-1])
                remaining = remaining[len(chunk) - 1:]
        assert remaining == ''
        assert changed[y1:y2, x1:x2].any()
    assert not (changed & ~recorded).any()
    return output, changed


def assert_within_layout(region, width=320, height=240):
    x1, y1, x2, y2 = region._drawn_rect
    bx1, by1, bx2, by2 = region.enlarged_xyxy
    assert max(0, bx1) <= x1 < x2 <= min(width, bx2)
    assert max(0, by1) <= y1 < y2 <= min(height, by2)


def assert_disjoint(regions):
    for index, left in enumerate(regions):
        lx1, ly1, lx2, ly2 = left._drawn_rect
        for right in regions[index + 1:]:
            rx1, ry1, rx2, ry2 = right._drawn_rect
            assert lx2 <= rx1 or rx2 <= lx1 or ly2 <= ry1 or ry2 <= ly1


def test_dense_untrusted_text_uses_nearby_clean_space_before_shrinking(monkeypatch):
    def unexpected_contour(*args, **kwargs):
        pytest.fail('Untrusted layout must not use the legacy contour heuristic')

    monkeypatch.setattr(text_render_eng, 'extract_ballon_region', unexpected_contour)
    region = make_region(
        (190, 135, 285, 215),
        'Every word of this much longer translation must remain visible inside '
        'the clear space even when there is no reliable speech bubble. '
        'The renderer must use the available room without covering nearby '
        'artwork or dropping the final sentence.',
        font_size=40,
    )

    render([region], width=480, height=360)

    assert region._drawn_font_size >= 20
    assert_within_layout(region, width=480, height=360)


@pytest.mark.parametrize('boxes', [
    [(25, 80, 130, 160), (138, 80, 245, 160)],
    [(35, 65, 180, 155), (130, 85, 275, 175)],
    [(80, 60, 240, 180), (80, 60, 240, 180)],
    [(80, 60, 240, 180), (80, 60, 240, 180), (80, 60, 240, 180)],
    [(80, 60, 240, 180), (80, 60, 238, 178), (80, 60, 236, 176)],
])
def test_neighboring_untrusted_regions_do_not_overlap(boxes):
    translations = [
        'Please keep all of this first translation readable.',
        'This second translation also needs enough room to fit.',
        'The third translation must have its own space too.',
    ]
    regions = [
        make_region(box, translation, 50)
        for box, translation in zip(boxes, translations)
    ]

    render(regions)

    assert_disjoint(regions)
    for region in regions:
        assert_within_layout(region)


@pytest.mark.parametrize(('box', 'angle'), [
    ((-12, -8, 105, 65), 38),
    ((225, -10, 332, 65), -34),
    ((-10, 172, 110, 250), 75),
    ((225, 175, 332, 248), -65),
])
def test_rotated_text_at_page_edges_keeps_all_words(box, angle):
    region = make_region(
        box, 'All these words must remain on the page after rotation.',
        font_size=55, angle=angle,
    )

    render([region])

    assert_within_layout(region)
    assert region.angle == angle


@pytest.mark.parametrize('font_size', [500, 2000])
def test_unreliable_source_font_size_is_bounded_by_text_geometry(font_size):
    region = make_region((100, 85, 220, 155), 'Wait!', font_size)

    render([region])

    assert region._drawn_font_size <= min(region.xywh[2:]) * 1.4
    assert_within_layout(region)


@pytest.mark.parametrize('shape', ['rectangle', 'oval'])
def test_trusted_bubble_still_fits_readable_text(shape):
    bubble = np.zeros((240, 320), dtype=np.uint8)
    if shape == 'rectangle':
        bubble[40:200, 50:270] = 255
    else:
        cv2.ellipse(bubble, (160, 120), (110, 80), 0, 0, 360, 255, -1)
    region = make_region(
        (120, 80, 200, 160),
        'There is plenty of room for this translation inside the speech bubble.',
        font_size=24,
    )

    _, changed = render([region], page_bubbles=[bubble])

    assert not (changed & (bubble == 0)).any()
    assert region._drawn_font_size >= region.font_size * 0.65


@pytest.mark.parametrize('shape', ['rectangle', 'oval'])
def test_shared_bubble_keeps_english_in_the_original_ocr_lobes(shape):
    bubble = np.zeros((400, 520), dtype=np.uint8)
    if shape == 'rectangle':
        bubble[40:360, 50:470] = 255
    else:
        cv2.ellipse(bubble, (260, 200), (210, 160), 0, 0, 360, 255, -1)
    right = make_region(
        (310, 100, 355, 300),
        'Please listen carefully because there is something important I need to tell you.',
        font_size=30,
    )
    left = make_region(
        (180, 100, 225, 300),
        'We have plenty of room to make both of these sentences comfortable to read.',
        font_size=30,
    )
    # Deliberately reverse input order: OCR positions control the allocation.
    regions = [left, right]

    _, changed = render(regions, page_bubbles=[bubble], width=520, height=400)

    assert not (changed & (bubble == 0)).any()
    assert_disjoint(regions)
    assert left._drawn_rect[2] <= right._drawn_rect[0]
    for region in regions:
        assert region._bubble_source == 'segmented'
        assert region._drawn_font_size >= 20
        assert_within_layout(region, width=520, height=400)


def test_shapes_are_a_balloon_its_share_or_the_text_space():
    bubble = np.zeros((400, 520), dtype=np.uint8)
    bubble[40:360, 50:470] = 255
    alone = np.zeros((400, 520), dtype=np.uint8)
    alone[5:35, 400:510] = 255
    left = make_region((180, 100, 225, 300), 'We have plenty of room for both sentences.', font_size=30)
    right = make_region((310, 100, 355, 300), 'Please listen carefully to me.', font_size=30)
    single = make_region((420, 10, 490, 30), 'Hi.', font_size=16)
    free = make_region((20, 370, 120, 395), 'Outside.', font_size=16)

    render([left, right, single, free], page_bubbles=[bubble, alone], width=520, height=400)

    # A shared balloon is outlined as the halves the layout divides it into, between the texts.
    assert left._drawn_shape == [(50, 40), (50, 359), (267, 359), (267, 40)]
    assert right._drawn_shape == [(268, 40), (268, 359), (469, 359), (469, 40)]
    assert single._drawn_shape == [(400, 5), (400, 34), (509, 34), (509, 5)]
    xs, ys = zip(*free._drawn_shape)
    assert min(xs) <= 20 and min(ys) <= 370 and max(xs) >= 119 and max(ys) >= 394


def test_untrusted_vertical_text_expands_without_covering_artwork_or_neighbor():
    page = np.full((400, 520, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (60, 40), (460, 360), (0, 0, 0), 5)
    cv2.line(page, (340, 45), (340, 355), (0, 0, 0), 5)
    region = make_region(
        (195, 95, 235, 305),
        'There is enough clear space around this narrow source column to keep '
        'the English translation readable.',
        font_size=30,
    )
    neighbor = make_region((370, 130, 415, 270), 'Wait for me!', font_size=30)

    output, changed = render([region, neighbor], page_bubbles=[], page=page)

    assert region._drawn_font_size >= 20
    assert region.enlarged_xyxy[2] - region.enlarged_xyxy[0] > 40
    assert 63 <= region._drawn_rect[0] < region._drawn_rect[2] <= 337
    assert 43 <= region._drawn_rect[1] < region._drawn_rect[3] <= 357
    assert not (changed & np.any(page < 255, axis=2)).any()
    np.testing.assert_array_equal(output[:, 337:344], page[:, 337:344])
    assert_disjoint([region, neighbor])
    assert_within_layout(region, width=520, height=400)
    assert_within_layout(neighbor, width=520, height=400)


def test_untrusted_text_allows_only_a_small_margin_over_textured_background():
    rng = np.random.default_rng(7)
    page = rng.integers(20, 180, (240, 320, 3), dtype=np.uint8)
    region = make_region(
        (100, 70, 190, 170),
        'This translation must stay inside its limited text area.',
        font_size=30,
    )

    render([region], page_bubbles=[], page=page)

    assert np.max(np.abs(region.enlarged_xyxy - region.xyxy)) <= 12
    assert_within_layout(region)


def test_syllabic_break_is_a_last_resort_for_a_narrow_column(monkeypatch):
    from manga_translator.rendering import eng_hyphenation

    # Deterministic dictionary boundary: extra-ordinary. No network/cache needed.
    monkeypatch.setattr(eng_hyphenation, 'syllable_break_positions', lambda word: (5,))
    page = np.zeros((420, 300, 3), dtype=np.uint8)
    page[40:380, 115:185] = 255
    region = make_region((115, 40, 185, 380), 'extraordinary', 30)

    render([region], page=page)

    assert region._drawn_font_size >= 18
    assert region._drawn_lines == ['EXTRA-', 'ORDINARY']
    assert_within_layout(region, width=300, height=420)


def test_whole_word_margin_is_tried_before_dictionary_breaks(monkeypatch):
    from manga_translator.rendering import eng_hyphenation

    def unexpected_hyphenation(word):
        pytest.fail('A whole word fits at a readable size using the small margin')

    monkeypatch.setattr(eng_hyphenation, 'syllable_break_positions', unexpected_hyphenation)
    page = np.zeros((240, 320, 3), dtype=np.uint8)
    page[40:200, 100:175] = 255
    region = make_region((100, 40, 175, 200), 'properly', 24)

    render([region], page=page)

    assert region._drawn_font_size >= 18
    assert region._drawn_lines == ['PROPERLY']
    assert_within_layout(region)


@pytest.mark.parametrize('word', ['WHAT', 'WORST', 'PEOPLE', 'MERCY', 'Himesaki', 'Midorimine'])
def test_short_words_and_names_remain_whole(word):
    page = np.zeros((320, 240, 3), dtype=np.uint8)
    page[30:290, 90:150] = 255
    region = make_region((90, 30, 150, 290), word, 28)

    render([region], page=page)

    assert region._drawn_lines == [word.upper()]
    assert_within_layout(region, width=240, height=320)


def test_wrap_never_repeatedly_cuts_a_word_or_hyphenates_successive_lines():
    def width(word):
        return sum(text_render_eng.get_char_glyph(c, 20, 0).metrics.horiAdvance >> 6 for c in word)

    word = 'EXTRAORDINARY'
    cuts = {word: (5, 7, 9)}
    wrapped = text_render_eng._wrap_long_words(
        [word], [width(word)], width(' '), [width('EXTRA-'), width('ORDINARY')], 20, ' ', cuts)
    assert [line[0] for line in wrapped] == ['EXTRA-', 'ORDINARY']
    # The remainder cannot be forced into a series of tiny fragments.
    assert text_render_eng._wrap_long_words(
        [word], [width(word)], width(' '), [width('EXTRA-')] * 6, 20, ' ', cuts) is None
    # A second long word requiring another inserted break rejects the layout.
    assert text_render_eng._wrap_long_words(
        [word, word], [width(word)] * 2, width(' '),
        [width('EXTRA-'), width('ORDINARY')] * 2, 20, ' ', cuts) is None


def test_safe_balanced_wrap_preserves_the_first_word():
    got = text_render_eng._balanced_wrap([80, 20, 20], 5, [50, 50], require_all_words=True)
    assert got is None  # The first word cannot fit either band; it cannot be omitted.


def test_blank_translation_is_skipped():
    region = make_region((80, 80, 180, 150), '  \n  ')

    _, changed = render([region])

    assert not changed.any()
    assert getattr(region, '_drawn_rect', None) is None


@pytest.mark.parametrize('box', [
    (330, 40, 350, 100), (-30, 40, -5, 100),
    (50, 250, 100, 275), (50, -30, 100, -5),
])
def test_region_entirely_outside_page_is_not_resurrected_by_margin(box):
    page = np.full((240, 320, 3), 255, dtype=np.uint8)
    region = make_region(box, 'Wait for me!', 24)

    output = text_render_eng.render_textblock_list_eng(
        page.copy(), [region], original_img=page, page_bubbles=[], safe_layout=True)

    np.testing.assert_array_equal(output, page)
    assert getattr(region, '_drawn_rect', None) is None


def test_neighboring_trusted_and_untrusted_regions_do_not_overlap():
    bubble = np.zeros((240, 320), dtype=np.uint8)
    bubble[40:205, 35:195] = 255
    regions = [
        make_region((75, 85, 145, 145), 'The first translation has a reliable bubble.', 32),
        make_region((160, 90, 280, 155), 'The neighboring translation has no reliable bubble.', 32),
    ]

    render(regions, page_bubbles=[bubble])

    assert regions[0]._bubble_source == 'segmented'
    assert regions[1]._bubble_source != 'segmented'
    assert_disjoint(regions)
    assert_within_layout(regions[1])


def test_omitting_safe_layout_preserves_explicit_legacy_behavior():
    page = np.full((240, 320, 3), 255, dtype=np.uint8)
    outputs, regions = [], []
    for options in ({}, {'safe_layout': False}):
        region = make_region((80, 65, 230, 165), 'Legacy layout remains unchanged.', 24)
        outputs.append(text_render_eng.render_textblock_list_eng(
            page.copy(), [region], original_img=page, page_bubbles=[], **options,
        ))
        regions.append(region)

    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert regions[0]._drawn_lines == regions[1]._drawn_lines
    assert regions[0]._drawn_rect == regions[1]._drawn_rect
    assert regions[0]._drawn_font_size == regions[1]._drawn_font_size


@pytest.mark.parametrize(('setting', 'expected'), [(None, True), ('0', False)])
def test_manga2eng_production_route_and_rollback(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv('MT_MANGA2ENG_SAFE_LAYOUT', raising=False)
    else:
        monkeypatch.setenv('MT_MANGA2ENG_SAFE_LAYOUT', setting)
    calls = []

    async def render_spy(*args, **kwargs):
        calls.append(kwargs)
        return args[0]

    monkeypatch.setattr(pipeline, 'dispatch_eng_render', render_spy)
    translator = pipeline.MangaTranslator.__new__(pipeline.MangaTranslator)
    translator._model_usage_timestamps = {}
    translator.verbose = False
    translator.font_path = ''
    translator._accum_time = lambda *args: None
    page = np.full((240, 320, 3), 255, dtype=np.uint8)
    ctx = Context(
        img_inpainted=page, img_rgb=page,
        text_regions=[make_region((80, 65, 230, 165), 'A translated line.')],
    )
    config = Config(render={'renderer': Renderer.manga2Eng})

    output = asyncio.run(translator._run_text_rendering(config, ctx))

    assert output is page
    assert len(calls) == 1
    assert calls[0]['safe_layout'] is expected


def test_hybrid_renderer_uses_shiori_for_all_regions(monkeypatch):
    from manga_translator.rendering import shiori_render

    monkeypatch.setenv('MT_MANGA2ENG_SAFE_LAYOUT', '1')
    page = np.full((240, 320, 3), 255, dtype=np.uint8)
    bubble = np.full((240, 320), 255, dtype=np.uint8)
    calls = []

    async def render_spy(*args, **kwargs):
        calls.append(kwargs)
        return args[0]

    monkeypatch.setattr(shiori_render, 'dispatch_shiori_render', render_spy)
    config = Config(render={'renderer': 'shiori_v2', 'font_size': 32})

    output = asyncio.run(shiori_render.dispatch_shiori_render_v2(
        page, page, [make_region((80, 65, 230, 165), 'A translated line.')],
        bubbles=[bubble], render_config=config.render, text_mask=bubble,
    ))

    assert output is page
    assert len(calls) == 1
    assert calls[0].get('safe_layout', False) is False
    assert calls[0]['manga2eng_paint'] is True
    assert calls[0]['bubbles'][0] is bubble
    assert calls[0]['text_mask'] is bubble
    assert calls[0]['render_config'] is config.render
