"""Speech-balloon instance segmentation (YOLO11n-seg, ONNX).

Supplies `text_render_eng` with a balloon mask that is far more reliable than the
Canny/contour heuristic in `ballon_extractor.py`. A trustworthy mask is what makes the
overflow correction in the renderer safe to apply -- see plans/bubble-aware-rendering.md.

Runs on the CPU execution provider (onnxruntime here is the CPU build) inside the render
lane, which is idle-heavy relative to the GPU/LLM lanes.
"""
import os
import threading
from typing import List, Optional

import cv2
import numpy as np

from ..utils import BASE_PATH, get_logger

logger = get_logger('bubble_seg')

MODEL_PATH = os.path.join(BASE_PATH, 'models', 'bubble_seg', 'manga109_yolo11n_seg.onnx')

# Trained at 1600 with stride 32 (huyvux3005/manga109-segmentation-bubble).
INPUT_SIZE = 1600
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

_session = None
_session_lock = threading.Lock()
_unavailable = False


def _get_session():
    """Lazily build the ORT session. Returns None if the model/runtime is unavailable,
    in which case callers fall back to the classical extractor."""
    global _session, _unavailable
    if _unavailable:
        return None
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        if os.environ.get('MT_BUBBLE_SEG', '1') == '0':
            logger.info('bubble segmentation disabled via MT_BUBBLE_SEG=0; using contour fallback')
            _unavailable = True
            return None
        if not os.path.isfile(MODEL_PATH):
            logger.warning(f'bubble segmentation model not found at {MODEL_PATH}; using contour fallback')
            _unavailable = True
            return None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            _session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
            logger.info('loaded speech-balloon segmentation model')
        except Exception as e:
            logger.warning(f'failed to load bubble segmentation model ({e}); using contour fallback')
            _unavailable = True
            return None
    return _session


def _letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, scale, dw, dh


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def detect_bubbles(img: np.ndarray) -> Optional[List[np.ndarray]]:
    """Segment speech balloons in an RGB page.

    Returns a list of full-page uint8 masks (255 inside a balloon), highest score first,
    or None when the model is unavailable so the caller can fall back.
    """
    session = _get_session()
    if session is None:
        return None
    try:
        return _detect(session, img)
    except Exception as e:
        logger.warning(f'bubble segmentation failed ({e}); using contour fallback for this page')
        return None


def assign_regions(masks: List[np.ndarray], region_xyxys: List[np.ndarray], min_cover: float = 0.5):
    """Map each text region to the balloon that contains it.

    Returns a list, one entry per region, of either (mask_cropped_to_balloon, [x1, y1, x2, y2])
    or None when the region should fall back to the contour extractor.

    A balloon holding more than one region is dropped: the renderer centres text in the whole
    balloon, so two regions sharing one would be drawn on top of each other. The contour
    extractor's per-region windows already handle that case.
    """
    assigned: List[Optional[int]] = []
    for xyxy in region_xyxys:
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        area = max(1, (x2 - x1) * (y2 - y1))
        best, best_cover = None, 0.0
        for i, m in enumerate(masks):
            sub = m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if sub.size == 0:
                continue
            cover = float((sub > 0).sum()) / area
            if cover > best_cover:
                best_cover, best = cover, i
        assigned.append(best if best_cover >= min_cover else None)

    counts = {}
    for i in assigned:
        if i is not None:
            counts[i] = counts.get(i, 0) + 1

    results = []
    for i in assigned:
        if i is None or counts[i] > 1:
            results.append(None)
            continue
        x, y, w, h = cv2.boundingRect(masks[i])
        results.append((masks[i][y:y + h, x:x + w].copy(), [x, y, x + w, y + h]))
    return results


def _detect(session, img: np.ndarray) -> List[np.ndarray]:
    ih, iw = img.shape[:2]
    canvas, scale, dw, dh = _letterbox(img, INPUT_SIZE)
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob)

    out0, protos = session.run(None, {session.get_inputs()[0].name: blob})

    # (1, 37, anchors) -> (anchors, 37): cx, cy, w, h, conf, 32 mask coefficients
    preds = out0[0].T
    scores = preds[:, 4]
    keep = scores > CONF_THRESHOLD
    preds = preds[keep]
    if preds.shape[0] == 0:
        return []
    scores = preds[:, 4]

    cxcy = preds[:, 0:2]
    wh = preds[:, 2:4]
    boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1)  # letterbox xyxy

    idx = _nms(boxes, scores, IOU_THRESHOLD)
    boxes, scores, coeffs = boxes[idx], scores[idx], preds[idx, 5:]

    # Mask prototypes are at input_size / 4 (400 for 1600).
    protos = protos[0]
    ch, mh, mw = protos.shape
    mask_maps = coeffs @ protos.reshape(ch, -1)
    mask_maps = 1.0 / (1.0 + np.exp(-mask_maps))
    mask_maps = mask_maps.reshape(-1, mh, mw)

    mx = INPUT_SIZE / mw
    my = INPUT_SIZE / mh

    masks = []
    for box, mask_map in zip(boxes, mask_maps):
        # Crop in proto space first so a coefficient blob can't bleed outside its box.
        bx1, by1, bx2, by2 = box
        cx1 = int(np.clip(np.floor(bx1 / mx), 0, mw))
        cy1 = int(np.clip(np.floor(by1 / my), 0, mh))
        cx2 = int(np.clip(np.ceil(bx2 / mx), 0, mw))
        cy2 = int(np.clip(np.ceil(by2 / my), 0, mh))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        cropped = np.zeros((mh, mw), dtype=np.float32)
        cropped[cy1:cy2, cx1:cx2] = mask_map[cy1:cy2, cx1:cx2]

        full = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        # Undo the letterbox: strip padding, then back to source resolution.
        uw, uh = int(round(iw * scale)), int(round(ih * scale))
        full = full[dh:dh + uh, dw:dw + uw]
        if full.size == 0:
            continue
        full = cv2.resize(full, (iw, ih), interpolation=cv2.INTER_LINEAR)
        masks.append(((full > MASK_THRESHOLD) * 255).astype(np.uint8))

    return masks
