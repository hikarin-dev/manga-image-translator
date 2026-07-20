"""Shiori renderer — koharu's text renderer driven from this pipeline.

Completely separate from `text_render_eng.py`: layout, fitting, wrapping,
rasterization, and compositing all happen inside the `shiori_renderer` Rust
extension (vendored, unmodified koharu-renderer + a faithful port of
koharu-app's render driver; see `shiori-renderer/` at the repo root).

This module only adapts pipeline data to that driver's inputs:
- the inpainted page (RGBA) as the base plane,
- a bubble-ID mask built from our balloon segmentation (distinct grayscale ID
  per balloon, background 0 — koharu's Bubble mask contract),
- per-region blocks: detection bbox as the seed transform, the translation,
  the source reading direction, and a FontPrediction from `shiori_style`
  (YuzuMarker model — text/stroke colors and stroke width, independent of OCR).

Also hosts `dispatch_shiori_render_v2`, a hybrid that routes regions whose
balloon is well defined through manga2eng's balloon-contain fit and everything
else through this driver.
"""
import json
import os
import threading
from typing import List

import numpy as np

from ..utils import BASE_PATH, LANGUAGE_ORIENTATION_PRESETS, TextBlock, get_logger
from ..utils.executors import run_cpu
from .bubble_seg import assign_regions, detect_bubbles
from . import dispatch_eng_render, dispatch_eng_render_pillow, shiori_style

logger = get_logger('shiori_render')

_renderer = None
_document_font = None
_renderer_lock = threading.Lock()


def _get_renderer(font_path: str):
    global _renderer, _document_font
    if _renderer is None:
        with _renderer_lock:
            if _renderer is None:
                import shiori_renderer
                r = shiori_renderer.PageRenderer()
                family, ps_name = r.register_font(font_path)
                logger.info(f'shiori renderer ready (document font: {family} / {ps_name})')
                _renderer, _document_font = r, family
    return _renderer, _document_font


def _bubble_id_mask(original_img: np.ndarray, masks=None):
    """Distinct non-zero ID per balloon, 0 background. None if segmentation is
    unavailable — the driver then keeps every block at its detector box.
    `masks` lets a caller reuse an already-computed `detect_bubbles` result."""
    if masks is None:
        masks = detect_bubbles(original_img)
    if not masks:
        return None
    id_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    for i, m in enumerate(masks[:255]):
        id_mask[(m > 0) & (id_mask == 0)] = i + 1
    return id_mask


async def dispatch_shiori_render(img_canvas: np.ndarray, original_img: np.ndarray,
                                 text_regions: List[TextBlock], font_path: str = '',
                                 device: str = 'cpu', verbose: bool = False,
                                 bubbles: List[np.ndarray] = None) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/ccvictoryspeech.ttf')

    def _sync():
        renderer, family = _get_renderer(font_path)
        preds = shiori_style.predict(original_img, [r.xyxy for r in text_regions], device=device)
        id_mask = _bubble_id_mask(original_img, bubbles)

        blocks = []
        for i, (region, pred) in enumerate(zip(text_regions, preds)):
            x1, y1, x2, y2 = (float(v) for v in region.xyxy)
            # --- original: keep the YuzuMarker-predicted text/stroke colors ---
            #     (pred['textColor'] / pred['strokeColor'] pass through unchanged;
            #     revert by deleting the three lines below)
            # --- override: use the OCR-sampled region colors instead ---
            ocr_fg, ocr_bg = region.get_font_colors()
            pred['textColor'] = [int(c) for c in np.clip(np.asarray(ocr_fg), 0, 255)]
            pred['strokeColor'] = [int(c) for c in np.clip(np.asarray(ocr_bg), 0, 255)]
            blocks.append({
                'nodeId': i,
                'transform': {'x': x1, 'y': y1,
                              'width': max(1.0, x2 - x1), 'height': max(1.0, y2 - y1),
                              'rotationDeg': 0.0},
                'translation': region.translation or '',
                'fontPrediction': pred,
                'sourceDirection': 'vertical' if region.vertical else 'horizontal',
            })
        options = {'documentFont': family, 'targetLanguage': 'en'}

        h, w = img_canvas.shape[:2]
        rgba = np.dstack([img_canvas, np.full((h, w), 255, dtype=np.uint8)])
        out_bytes, info_json = renderer.render_page(
            rgba.tobytes(), w, h, json.dumps(blocks), json.dumps(options),
            id_mask.tobytes() if id_mask is not None else None)
        out = np.frombuffer(out_bytes, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()

        # Style hints for study-mode DOM text, mirroring what text_render_eng records.
        # The drawn colors become the region's ONE fg/bg pair — the study payload and
        # both DOM texts (original and translation) read the same variables, so what
        # the render pass drew is what every text display uses.
        for info in json.loads(info_json):
            region = text_regions[info['nodeId']]
            region._drawn_font_size = info['fontSize']
            region._drawn_rect = (int(info['x']), int(info['y']),
                                  int(info['x'] + info['width']), int(info['y'] + info['height']))
            region._drawn_fg = [int(c) for c in info['textColor']]
            region._drawn_bg = [int(c) for c in info['strokeColor']]
        return out

    return await run_cpu(_sync)


async def dispatch_shiori_render_v2(img_canvas: np.ndarray, original_img: np.ndarray,
                                    text_regions: List[TextBlock], font_path: str = '',
                                    line_spacing: int = 0, device: str = 'cpu',
                                    verbose: bool = False) -> np.ndarray:
    """Hybrid dispatch: manga2eng's balloon-contain fit where a region's balloon is
    well defined, the shiori engine for everything else.

    "Well defined" is the eng renderer's own trust test — the balloon segmenter
    isolates one balloon that covers the region and holds no other region
    (`assign_regions`). Those regions go through `dispatch_eng_render` unchanged;
    the rest through `dispatch_shiori_render`. The eng fit only typesets
    horizontal targets, so other target languages render fully shiori.
    """
    if len(text_regions) == 0:
        return img_canvas

    bubbles = None
    eng_regions, shiori_regions = [], []
    if LANGUAGE_ORIENTATION_PRESETS.get(text_regions[0].target_lang) == 'h':
        bubbles = await run_cpu(lambda: detect_bubbles(original_img))
        assigned = (assign_regions(bubbles, [r.xyxy for r in text_regions])
                    if bubbles else [None] * len(text_regions))
        for region, bubble in zip(text_regions, assigned):
            (eng_regions if bubble is not None else shiori_regions).append(region)
    else:
        shiori_regions = list(text_regions)

    output = img_canvas
    if shiori_regions:
        output = await dispatch_shiori_render(output, original_img, shiori_regions, font_path,
                                              device=device, verbose=verbose, bubbles=bubbles)
    if eng_regions:
        for region in eng_regions:
            region._typeset_eng = True  # study-hint flag: this region used manga2eng typesetting
        try:
            output = await dispatch_eng_render(output, original_img, eng_regions, font_path,
                                               line_spacing, verbose=verbose, page_bubbles=bubbles)
        except Exception as e:
            # Freetype path failed (e.g. a face/glyph fault) — retry these regions with the
            # Pillow renderer before giving up, mirroring the manga2eng dispatch.
            logger.warning(f'manga2eng freetype rendering failed ({e}); retrying regions with the Pillow renderer')
            output = await dispatch_eng_render_pillow(output, original_img, eng_regions, font_path, line_spacing)
    return output
