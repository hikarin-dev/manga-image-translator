import cv2
import numpy as np

from manga_translator.ocr.colors import (
    _ideal_outline_color,
    _normalize_outline_colors,
    apply_estimated_colors,
    estimate_colors,
    snap_cluster_colors,
)
from manga_translator.utils import Quadrilateral


PINK = np.array([224, 66, 122])
WHITE = np.array([248, 248, 248])


def _assert_color_close(actual, expected, tolerance=24):
    assert np.max(np.abs(np.asarray(actual) - np.asarray(expected))) <= tolerance


def test_white_outline_is_taken_from_glyph_edge_not_nearby_art():
    crop = np.full((48, 220, 3), (190, 132, 76), dtype=np.uint8)
    crop[:, :70] = (42, 35, 31)
    crop[:, 175:] = (102, 72, 48)

    cv2.putText(crop, 'TEST', (47, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, tuple(WHITE.tolist()), 9, cv2.LINE_AA)
    cv2.putText(crop, 'TEST', (47, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, tuple(PINK.tolist()), 4, cv2.LINE_AA)

    glyph_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.putText(glyph_mask, 'TEST', (47, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, 255, 9, cv2.LINE_AA)

    fill, outline = estimate_colors(crop, glyph_mask)

    _assert_color_close(fill, PINK)
    _assert_color_close(outline, WHITE)


def test_antialiased_light_pink_edge_resolves_to_solid_white_outline():
    crop = np.full((48, 220, 3), (78, 59, 47), dtype=np.uint8)
    glyph_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    outline_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.putText(outline_mask, 'TEST', (47, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, 255, 10, cv2.LINE_8)
    cv2.putText(glyph_mask, 'TEST', (47, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, 255, 5, cv2.LINE_8)

    crop[outline_mask > 0] = WHITE
    crop[glyph_mask > 0] = PINK
    inner_edge = ((glyph_mask > 0)
                  & (cv2.erode((glyph_mask > 0).astype(np.uint8),
                               np.ones((3, 3), np.uint8)) == 0))
    crop[inner_edge] = (238, 157, 185)

    fill, outline = estimate_colors(crop, glyph_mask)

    _assert_color_close(fill, PINK)
    _assert_color_close(outline, WHITE)


def test_outline_clustering_snaps_pale_pink_antialias_to_white():
    pale_pink = np.array([248, 208, 225])

    snapped = snap_cluster_colors([WHITE, pale_pink], preserve_chroma=False)

    assert np.array_equal(snapped[0], WHITE)
    assert np.array_equal(snapped[1], WHITE)


def test_exported_pink_antialias_samples_resolve_to_exact_white():
    samples = [
        ((194, 61, 107), (228, 156, 181)),
        ((182, 60, 107), (226, 154, 183)),
        ((196, 65, 117), (226, 156, 181)),
        ((199, 76, 127), (253, 230, 243)),
    ]

    for fill, detected_outline in samples:
        outline = _ideal_outline_color(np.array(fill), np.array(detected_outline))
        assert np.array_equal(outline, (255, 255, 255))


def test_unrelated_colored_outline_is_not_clamped():
    fill = np.array([210, 65, 115])
    colored_outline = np.array([75, 125, 205])

    outline = _ideal_outline_color(fill, colored_outline)

    assert np.array_equal(outline, colored_outline)


def test_same_page_white_consensus_repairs_gross_pink_outliers():
    estimates = [
        (np.array([200, 60, 112]), np.array([247, 248, 247])),
        (np.array([204, 66, 116]), np.array([246, 247, 249])),
        (np.array([194, 61, 106]), np.array([245, 246, 247])),
        (np.array([194, 61, 106]), np.array([230, 146, 128])),
        (np.array([215, 63, 115]), np.array([0, 0, 0])),
        (np.array([30, 30, 30]), np.array([0, 0, 0])),
    ]

    outlines = _normalize_outline_colors(estimates)

    for outline in outlines[:5]:
        assert np.array_equal(outline, (255, 255, 255))
    assert np.array_equal(outlines[5], (0, 0, 0))


def test_no_outline_uses_local_white_surrounding_instead_of_distant_dark_art():
    crop = np.full((48, 300, 3), (38, 34, 31), dtype=np.uint8)
    cv2.rectangle(crop, (25, 0), (165, 47), tuple(WHITE.tolist()), -1)
    cv2.putText(crop, 'TEST', (37, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, tuple(PINK.tolist()), 4, cv2.LINE_AA)

    glyph_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.putText(glyph_mask, 'TEST', (37, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, 255, 4, cv2.LINE_AA)

    fill, outline = estimate_colors(crop, glyph_mask)

    _assert_color_close(fill, PINK)
    _assert_color_close(outline, WHITE)
    assert np.mean(outline) > 220


def test_real_dark_outline_is_preserved():
    crop = np.full((48, 180, 3), tuple(WHITE.tolist()), dtype=np.uint8)
    cv2.putText(crop, 'TEST', (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, (18, 18, 18), 9, cv2.LINE_AA)
    cv2.putText(crop, 'TEST', (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, tuple(PINK.tolist()), 4, cv2.LINE_AA)

    glyph_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.putText(glyph_mask, 'TEST', (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, 255, 9, cv2.LINE_AA)

    fill, outline = estimate_colors(crop, glyph_mask)

    _assert_color_close(fill, PINK)
    _assert_color_close(outline, (18, 18, 18))


def test_filled_detector_box_is_rejected():
    crop = np.full((48, 180, 3), tuple(WHITE.tolist()), dtype=np.uint8)
    cv2.putText(crop, 'TEST', (20, 38), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, (20, 20, 20), 4, cv2.LINE_AA)
    filled_box = np.full(crop.shape[:2], 255, dtype=np.uint8)

    image_only = estimate_colors(crop)
    with_box = estimate_colors(crop, filled_box)

    assert np.array_equal(with_box[0], image_only[0])
    assert np.array_equal(with_box[1], image_only[1])


def test_apply_colors_handles_undersized_and_empty_detector_masks():
    image = np.full((100, 100, 3), 250, dtype=np.uint8)
    cv2.putText(image, 'TEST', (17, 62), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, tuple(PINK.tolist()), 4, cv2.LINE_AA)
    points = np.array([[10, 10], [90, 10], [90, 75], [10, 75]])

    for mask in (np.zeros((12, 12), dtype=np.uint8),
                 np.empty((0, 0), dtype=np.uint8)):
        region = Quadrilateral(points.copy(), 'TEST', 1.0, bg_r=1, bg_g=2, bg_b=3)
        apply_estimated_colors(image, [region], False, True, text_mask=mask)
        assert min(region.bg_r, region.bg_g, region.bg_b) >= 225


def test_apply_colors_preserves_ocr_colors_for_out_of_page_region():
    image = np.full((100, 100, 3), 250, dtype=np.uint8)
    points = np.array([[120, 120], [150, 120], [150, 145], [120, 145]])
    region = Quadrilateral(points, 'TEST', 1.0,
                           fg_r=10, fg_g=20, fg_b=30,
                           bg_r=220, bg_g=221, bg_b=222)

    apply_estimated_colors(image, [region], True, True,
                           text_mask=np.zeros((20, 20), dtype=np.uint8))

    assert (region.fg_r, region.fg_g, region.fg_b) == (10, 20, 30)
    assert (region.bg_r, region.bg_g, region.bg_b) == (220, 221, 222)
