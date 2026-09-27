"""Shiori: the commit-pinned Koharu scene renderer, fitting, shaping and Vello output.

The adapter supplies this pipeline's OCR geometry, wording, colors and segmented
balloons. Upstream owns contour fitting, conjoined-lobe partitioning, line breaking,
font fallback, writing modes, glyph strokes and compositing. See
shiori-renderer/UPSTREAM.md for the precise parity boundary.
"""
import json
import math
import os
import threading
from typing import List

import cv2
import numpy as np

from ..utils import BASE_PATH, TextBlock, get_logger
from ..utils.executors import run_cpu
from ..translators.common import ISO_639_1_TO_VALID_LANGUAGES
from .bubble_seg import detect_bubbles
from .text_render_eng import manga2eng_stroke_width

logger = get_logger('shiori_render')
UPSTREAM_REVISION = '4a133539f204ab1182ff64901ba5226b4e868fb0'
LANGUAGE_TAGS = {value: key for key, value in ISO_639_1_TO_VALID_LANGUAGES.items()}
LANGUAGE_TAGS.update(CHS='zh-Hans', CHT='zh-Hant', PTB='pt-BR')
# Readability policy outside the upstream fitter: an outward halo in page pixels.
BLACK_OUTLINE_WIDTH = 1.5
BLACK_COLOR_MAX = 32

_renderer = None
_document_font = None
_document_font_path = None
_renderer_lock = threading.RLock()


def _get_renderer(font_path: str):
    global _renderer, _document_font, _document_font_path
    font_path = os.path.realpath(font_path)
    with _renderer_lock:
        if _renderer is None or _document_font_path != font_path:
            import shiori_renderer
            if getattr(shiori_renderer, 'UPSTREAM_REVISION', None) != UPSTREAM_REVISION:
                raise RuntimeError('Shiori native renderer is out of date; rebuild/install shiori-renderer and restart the worker')
            renderer = shiori_renderer.PageRenderer([])
            family, ps_name = renderer.register_font(font_path)
            logger.info(f'Shiori renderer ready (Koharu {UPSTREAM_REVISION[:7]}, font: {family} / {ps_name})')
            _renderer, _document_font, _document_font_path = renderer, family, font_path
        return _renderer, _document_font


def _page_bubbles(renderer, original_img, masks):
    """Keep each instance's contour, including overlapping masks and IDs beyond 255."""
    if masks is None:
        masks = detect_bubbles(original_img)
    h, w = original_img.shape[:2]
    result = []
    for i, mask in enumerate(masks or []):
        binary = np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8)
        if binary.shape != (h, w):
            raise ValueError('Shiori bubble mask dimensions differ from the page')
        points = json.loads(renderer.bubble_geometry(binary.tobytes(), w, h))
        if points:
            result.append(({'id': i, 'points': points}, binary, int(binary.sum())))
    return result


def _region_mask(region, page_shape, text_mask=None):
    h, w = page_shape
    x1, y1, x2, y2 = (int(v) for v in region.xyxy)
    x1, y1, x2, y2 = min(w, max(0, x1)), min(h, max(0, y1)), min(w, max(0, x2 + 1)), min(h, max(0, y2 + 1))
    local = np.zeros((max(0, y2-y1), max(0, x2-x1)), np.uint8)
    if local.size:
        for line in np.asarray(region.lines):
            cv2.fillPoly(local, [np.rint(line - [x1, y1]).astype(np.int32)], 1)
        if text_mask is not None:
            local &= text_mask[y1:y2, x1:x2]
    return x1, y1, local


def _bubble_for_region(region, bubbles, page_shape, region_mask=None):
    """Upstream chooses the smallest balloon containing at least 90% of text pixels.

    This pipeline supplies a detector text mask restricted to OCR quadrilaterals
    rather than RF-DETR instance masks. If unavailable, use the quadrilaterals.
    Siblings keep the same balloon ID so upstream can divide its physical lobes.
    """
    if not bubbles:
        return None
    x1, y1, text_mask = region_mask if region_mask is not None else _region_mask(region, page_shape)
    y2, x2 = y1 + text_mask.shape[0], x1 + text_mask.shape[1]
    count = int(text_mask.sum())
    if not count:
        return None
    matches = [(area, bubble['id']) for bubble, mask, area in bubbles
               if np.count_nonzero(mask[y1:y2, x1:x2] & text_mask) / count >= 0.9]
    return min(matches)[1] if matches else None


def _source_direction(region):
    # TextBlock.vertical is target-language dependent; use the original line geometry.
    lines = np.asarray(region.lines, dtype=float)
    if not len(lines):
        return 'auto'
    widths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    heights = np.linalg.norm(lines[:, 3] - lines[:, 0], axis=1)
    i = int(np.argmax(widths * heights))
    return 'vertical' if heights[i] > widths[i] else 'horizontal'


def _block_input(region, index, bubble_id, render_config=None, inferred=None, manga2eng_paint=False):
    x1, y1, x2, y2 = (float(v) for v in region.xyxy)
    fg, bg = region.get_font_colors()
    block = {
        'nodeId': index,
        'transform': {'x': x1, 'y': y1, 'width': max(1.0, x2-x1), 'height': max(1.0, y2-y1)},
        'points': [{'x': float(x), 'y': float(y)} for x, y in region.min_rect.reshape(-1, 2)],
        'translation': region.translation or '',
        'sourceText': region.text or '',
        'bubbleId': bubble_id, 'sourceDirection': _source_direction(region),
        'color': [int(c) for c in np.clip(fg, 0, 255)] + [255],
        'strokeColor': [int(c) for c in np.clip(bg, 0, 255)] + [255],
    }
    if inferred:
        block.pop('points')
        block['transform']['rotationDeg'] = inferred['angleDegrees']
        block['sourceDirection'] = inferred['writingMode'].lower()
        if not manga2eng_paint:
            block['color'] = inferred['color'] + [255]
            block['strokeColor'] = (inferred['strokeColor'] or inferred['color']) + [255]
            block['strokeWidth'] = inferred['strokeWidth']
    if manga2eng_paint:
        # Resolve the manga2eng radius after upstream has chosen the font size.
        block['strokeWidth'] = 0.0
    if region.bold:
        block['fontWeight'] = 700
    if region.italic:
        block['fontStyle'] = 'italic'
    if render_config is not None:
        if render_config.font_color_fg is not None:
            block['color'] = list(render_config.font_color_fg) + [255]
    if not manga2eng_paint and max(block['color'][:3]) <= BLACK_COLOR_MAX:
        block['strokeColor'] = [255, 255, 255, 255]
        block['strokeWidth'] = max(BLACK_OUTLINE_WIDTH, block.get('strokeWidth') or 0.0)
    if render_config is not None:
        if render_config.font_color_bg is not None:
            block['strokeColor'] = list(render_config.font_color_bg) + [255]
        if render_config.disable_font_border:
            block['strokeWidth'] = 0.0
        size = getattr(render_config, 'font_size', None)
        if size and size > 0:
            block['fontSize'] = float(size)
        direction = getattr(render_config.direction, 'value', render_config.direction)
        if direction in ('horizontal', 'vertical'):
            block['writingMode'] = direction
        alignment = getattr(render_config.alignment, 'value', render_config.alignment)
        if alignment in ('left', 'center', 'right'):
            block['alignment'] = {'left': 'Start', 'center': 'Center', 'right': 'End'}[alignment]
    return block


async def dispatch_shiori_render(img_canvas: np.ndarray, original_img: np.ndarray,
                                 text_regions: List[TextBlock], font_path: str = '',
                                 device: str = 'cpu', verbose: bool = False,
                                 bubbles: List[np.ndarray] = None,
                                 render_config=None, text_mask=None, manga2eng_paint=False) -> np.ndarray:
    if not text_regions:
        return img_canvas
    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/ccvictoryspeech.ttf')

    def _sync():
        # One native renderer, with its registered fonts, serves every page. Only creating it
        # needs the lock: the native page render serializes itself, and the balloon geometry and
        # typography analysis before it are per-page, so pages overlap them instead of queueing.
        renderer, family = _get_renderer(font_path)
        if manga2eng_paint and not hasattr(renderer, 'layout_page'):
            raise RuntimeError('Shiori hybrid requires the updated native renderer; rebuild/install shiori-renderer and restart the worker')
        page_bubbles = _page_bubbles(renderer, original_img, bubbles)
        h, w = img_canvas.shape[:2]
        if text_mask is not None and text_mask.shape != (h, w):
            text_mask_input = cv2.resize(text_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            text_mask_input = text_mask
        if text_mask_input is not None:
            threshold = 0.5 if text_mask_input.max(initial=0) <= 1 else 127
            text_mask_input = np.ascontiguousarray(text_mask_input > threshold, dtype=np.uint8)
        region_masks = [_region_mask(region, (h, w), text_mask_input) for region in text_regions]
        predictions = json.loads(renderer.analyze_typography(
            np.ascontiguousarray(original_img).tobytes(), w, h,
            [region.xyxy.astype(float).tolist() for region in text_regions],
            [(x, y, mask.shape[1], mask.shape[0], mask.tobytes()) for x, y, mask in region_masks])) if text_mask_input is not None else [None] * len(text_regions)
        blocks = [_block_input(region, i, _bubble_for_region(region, page_bubbles, (h, w), region_masks[i]), render_config, predictions[i], manga2eng_paint)
                  for i, region in enumerate(text_regions)]
        target = text_regions[0].target_lang or 'ENG'
        options = {'documentFont': family, 'targetLanguage': LANGUAGE_TAGS.get(target, target.replace('_', '-').lower()),
                   'bubbles': [bubble for bubble, _, _ in page_bubbles], 'supersampling': 1}
        rgba = np.dstack([img_canvas, np.full((h, w), 255, dtype=np.uint8)])
        rgba_bytes, options_json = rgba.tobytes(), json.dumps(options)
        if manga2eng_paint and not (render_config and render_config.disable_font_border):
            layout = json.loads(renderer.layout_page(rgba_bytes, w, h, json.dumps(blocks), options_json))
            for info in layout:
                blocks[info['nodeId']]['strokeWidth'] = manga2eng_stroke_width(info['fontSize'])
        out_bytes, info_json = renderer.render_page(rgba_bytes, w, h, json.dumps(blocks), options_json)
        info_list = json.loads(info_json)
        for info in info_list:
            region = text_regions[info['nodeId']]
            # The area upstream laid this text into: its balloon, its own lobe of a balloon
            # it shares (divided at the necks between lobes), or its region when it has none.
            region._drawn_shape = [(point['x'], point['y']) for point in info['geometry']]
            region._drawn_lines = [line['text'] for line in info['lines']]
            region._drawn_inserted_hyphens = [i for i, line in enumerate(info['lines']) if line['inserted_hyphen']]
            region._drawn_font_size = info['fontSize']
            region._drawn_line_height = 1.2
            region._drawn_rect = (math.floor(info['x']), math.floor(info['y']),
                                  math.ceil(info['x'] + info['width']), math.ceil(info['y'] + info['height']))
            region._drawn_fg, region._drawn_bg = info['textColor'], info['strokeColor']
            region._drawn_stroke_width = info['strokeWidth']
            region._drawn_paint_policy = 'manga2eng' if manga2eng_paint else 'shiori'
        return np.frombuffer(out_bytes, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    return await run_cpu(_sync)


async def dispatch_shiori_render_v2(img_canvas: np.ndarray, original_img: np.ndarray,
                                    text_regions: List[TextBlock], font_path: str = '',
                                    line_spacing: int = 0, device: str = 'cpu',
                                    verbose: bool = False, render_config=None,
                                    text_mask=None, bubbles=None) -> np.ndarray:
    """Shiori layout for every region, with manga2eng's colors and stroke radius.

    Retain the existing config ID and call signature; line spacing comes from
    Shiori along with fitting, line breaking, shaping, and positioning.
    """
    return await dispatch_shiori_render(
        img_canvas, original_img, text_regions, font_path, device=device, verbose=verbose,
        bubbles=bubbles, render_config=render_config, text_mask=text_mask,
        manga2eng_paint=True)
