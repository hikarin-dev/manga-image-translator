from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from manga_translator.rendering.text_render_eng import _bounded_layout_boxes


def source(box, translation='A translated sentence.'):
    box = np.array(box, dtype=np.int32)
    return SimpleNamespace(xyxy=box, center=(box[:2] + box[2:]) / 2,
                           translation=translation)


def allocation(box):
    x1, y1, x2, y2 = box
    return np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8), box


def assert_preserved_and_separate(regions, boxes, width, height):
    for i, (region, box) in enumerate(zip(regions, boxes)):
        assert np.all(box[:2] <= region.xyxy[:2])
        assert np.all(box[2:] >= region.xyxy[2:])
        assert 0 <= box[0] < box[2] <= width
        assert 0 <= box[1] < box[3] <= height
        for other in boxes[:i]:
            assert np.any(np.minimum(box[2:], other[2:]) <= np.maximum(box[:2], other[:2]))


@pytest.mark.parametrize('reverse', [False, True])
def test_joined_diagonal_lobes_keep_their_source_columns(reverse):
    # The two source clusters overlap vertically but occupy different lobes.
    regions = [source((1142, 74, 1206, 240)), source((1054, 115, 1120, 359))]
    if reverse:
        regions.reverse()
    shared = allocation((1007, 14, 1264, 384))
    boxes = _bounded_layout_boxes(regions, [shared] * 2, 1280, 1800, [shared] * 2)

    assert_preserved_and_separate(regions, boxes, 1280, 1800)
    by_x = sorted(boxes, key=lambda box: box[0])
    assert by_x[0][2] <= by_x[1][0]
    assert all(box[1] == 14 and box[3] == 384 for box in boxes)


@pytest.mark.parametrize('order', list(permutations(range(3))))
def test_three_joined_lobes_retain_the_center_cluster(order):
    originals = [source((50, 70, 90, 210)), source((145, 35, 180, 170)),
                 source((235, 80, 270, 225))]
    regions = [originals[i] for i in order]
    shared = allocation((20, 10, 300, 250))
    boxes = _bounded_layout_boxes(regions, [shared] * 3, 320, 260, [shared] * 3)

    assert_preserved_and_separate(regions, boxes, 320, 260)
    center_box = boxes[order.index(1)]
    assert center_box[0] >= originals[0].xyxy[2]
    assert center_box[2] <= originals[2].xyxy[0]


@pytest.mark.parametrize('reverse', [False, True])
def test_stacked_joined_lobes_keep_vertical_source_positions(reverse):
    regions = [source((80, 35, 140, 95)), source((90, 160, 155, 240))]
    if reverse:
        regions.reverse()
    shared = allocation((20, 10, 210, 270))
    boxes = _bounded_layout_boxes(regions, [shared] * 2, 240, 280, [shared] * 2)

    assert_preserved_and_separate(regions, boxes, 240, 280)
    assert all(box[0] == 20 and box[2] == 210 for box in boxes)


@pytest.mark.parametrize('reverse', [False, True])
def test_uncertain_overlapping_allocations_follow_source_gap(reverse):
    # Area retention alone favors a horizontal cut, crossing both source boxes.
    regions = [source((75, 50, 105, 160)), source((130, 70, 160, 175))]
    areas = [allocation((20, 20, 205, 180)), allocation((50, 60, 220, 240))]
    if reverse:
        regions.reverse()
        areas.reverse()
    boxes = _bounded_layout_boxes(regions, areas, 240, 260)

    assert_preserved_and_separate(regions, boxes, 240, 260)
    by_x = sorted(boxes, key=lambda box: box[0])
    assert by_x[0][2] <= by_x[1][0]


def test_shared_soft_margins_stay_within_each_available_mask():
    regions = [source((45, 25, 80, 130)), source((140, 40, 175, 150))]
    areas = [allocation((0, 0, 210, 185)), allocation((5, 5, 215, 190))]
    shared = allocation((10, 10, 205, 180))
    boxes = _bounded_layout_boxes(regions, areas, 220, 200, [shared] * 2)

    assert_preserved_and_separate(regions, boxes, 220, 200)
    for box, (_, available) in zip(boxes, areas):
        assert np.all(box[:2] >= available[:2])
        assert np.all(box[2:] <= available[2:])


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_touching_source_clusters_keep_the_exact_seam(shared, reverse):
    regions = [source((40, 40, 100, 140)), source((100, 40, 160, 140))]
    areas = [allocation((0, 0, 180, 180)), allocation((20, 0, 220, 180))]
    if reverse:
        regions.reverse()
        areas.reverse()
    bubble = allocation((20, 20, 170, 160))
    boxes = _bounded_layout_boxes(regions, areas, 240, 200, [bubble] * 2 if shared else None)

    assert_preserved_and_separate(regions, boxes, 240, 200)
    left, right = sorted(boxes, key=lambda box: box[0])
    assert left[2] == right[0] == 100
