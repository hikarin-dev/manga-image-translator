"""YuzuMarker font detection — text/stroke color, stroke width, direction per region.

Faithful PyTorch port of koharu's font detector stage (koharu v0.61.2, GPL-3.0:
`koharu-ml/src/font_detector/{mod,models}.rs` + the color normalization from
`koharu-app/src/pipeline/engines/yuzumarker_font.rs`), using the same weights
(`fffonion/yuzumarker-font-detection` on Hugging Face). Independent of OCR.

Model: torchvision-layout ResNet-50 with a 6162-way head — 6150 font classes,
2 direction logits, 10 regression outputs (text RGB, font size, stroke width,
stroke RGB, line spacing, angle; all sigmoid, sizes relative to crop width).

Two deliberate deviations from koharu's candle implementation, both toward the
ORIGINAL PyTorch training code the weights come from:
- max-pool after conv1 uses padding=1 (torchvision standard; candle's
  `max_pool2d_with_stride` cannot pad, so koharu silently drops it),
- inference runs in fp32 (koharu uses f16 on CUDA).
Preprocessing matches koharu: exact resize to 512x512 with a Catmull-Rom
bicubic (PIL BICUBIC), RGB / 255, no ImageNet normalization.
"""
import os
import threading
from typing import List

import numpy as np
from PIL import Image

from ..utils import BASE_PATH, get_logger

logger = get_logger('shiori_style')

WEIGHTS_URL = ('https://huggingface.co/fffonion/yuzumarker-font-detection/'
               'resolve/main/yuzumarker-font-detection.safetensors')
WEIGHTS_PATH = os.path.join(BASE_PATH, 'models', 'shiori', 'yuzumarker-font-detection.safetensors')
KEY_PREFIX = 'model._orig_mod.model.'

FONT_COUNT = 6150
REGRESSION_DIM = 10
INPUT_SIZE = 512
BATCH = 8

_model = None
_model_device = None
_lock = threading.Lock()


def _download_weights():
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    logger.info('downloading yuzumarker-font-detection weights (~150 MB, one-time)')
    import requests
    tmp = WEIGHTS_PATH + '.part'
    with requests.get(WEIGHTS_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    os.replace(tmp, WEIGHTS_PATH)


def _load_model(device: str):
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model
    with _lock:
        if _model is not None and _model_device == device:
            return _model
        import torch
        from torchvision.models import resnet50
        from safetensors.torch import load_file

        if not os.path.isfile(WEIGHTS_PATH):
            _download_weights()

        sd = load_file(WEIGHTS_PATH)
        stripped = {k[len(KEY_PREFIX):]: v for k, v in sd.items() if k.startswith(KEY_PREFIX)}
        if not stripped:
            raise RuntimeError(f'no "{KEY_PREFIX}*" tensors in {WEIGHTS_PATH}')

        model = resnet50(num_classes=FONT_COUNT + 2 + REGRESSION_DIM)
        result = model.load_state_dict(stripped, strict=False)
        # Faithfulness gate: the checkpoint must account for every model tensor.
        # BatchNorm num_batches_tracked is bookkeeping torch tolerates missing.
        real_missing = [k for k in result.missing_keys if not k.endswith('num_batches_tracked')]
        if real_missing or result.unexpected_keys:
            raise RuntimeError(f'font detector state_dict mismatch: '
                               f'missing={real_missing} unexpected={result.unexpected_keys}')
        model.eval().to(device)
        _model, _model_device = model, device
        logger.info(f'loaded yuzumarker font detector on {device}')
    return _model


def _preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    resized = Image.fromarray(crop_rgb).resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # (3, H, W)


# --- color normalization, ported from koharu-app engines/yuzumarker_font.rs ---

def _gray(c) -> bool:
    return max(c) - min(c) <= 10


def _clamp_black(c):
    t = 60 if _gray(c) else 12
    return (0, 0, 0) if all(v <= t for v in c) else c


def _clamp_white(c):
    t = 255 - (60 if _gray(c) else 12)
    return (255, 255, 255) if all(v >= t for v in c) else c


def _colors_similar(a, b) -> bool:
    return all(abs(a[i] - b[i]) <= 16 for i in range(3))


def _normalize(pred: dict) -> dict:
    pred['textColor'] = list(_clamp_white(_clamp_black(tuple(pred['textColor']))))
    pred['strokeColor'] = list(_clamp_white(_clamp_black(tuple(pred['strokeColor']))))
    if pred['strokeWidthPx'] > 0.0 and _colors_similar(pred['textColor'], pred['strokeColor']):
        pred['strokeWidthPx'] = 0.0
        pred['strokeColor'] = list(pred['textColor'])
    return pred


def predict(img_rgb: np.ndarray, xyxys: List, device: str = 'cpu') -> List[dict]:
    """Run the detector on each region's bbox crop of the SOURCE page.

    Returns one camelCase FontPrediction dict per region, ready to embed in the
    shiori renderer's block JSON. Crops follow koharu: plain unpadded bbox.
    """
    if not len(xyxys):
        return []
    import torch

    model = _load_model(device)
    ih, iw = img_rgb.shape[:2]
    crops, widths = [], []
    for xyxy in xyxys:
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, max(x1 + 1, x2)), min(ih, max(y1 + 1, y2))
        crop = img_rgb[y1:y2, x1:x2]
        widths.append(crop.shape[1])
        crops.append(_preprocess(np.ascontiguousarray(crop)))

    rows = []
    with torch.no_grad():
        for i in range(0, len(crops), BATCH):
            batch = torch.from_numpy(np.stack(crops[i:i + BATCH])).to(device)
            rows.append(model(batch).float().cpu().numpy())
    rows = np.concatenate(rows, axis=0)

    preds = []
    for row, width in zip(rows, widths):
        direction = 'vertical' if row[FONT_COUNT + 1] > row[FONT_COUNT] else 'horizontal'
        reg = 1.0 / (1.0 + np.exp(-row[FONT_COUNT + 2:FONT_COUNT + 2 + REGRESSION_DIM]))
        reg = np.clip(reg, 0.0, 1.0)
        text_color = [int(round(float(v) * 255.0)) for v in reg[0:3]]
        font_size_px = float(reg[3]) * width
        stroke_width_px = float(reg[4]) * width
        stroke_color = [int(round(float(v) * 255.0)) for v in reg[5:8]]
        line_spacing_px = float(reg[8]) * width
        line_height = 1.0 + line_spacing_px / font_size_px if font_size_px > 0.0 else 1.2
        angle_deg = (float(reg[9]) - 0.5) * 180.0

        preds.append(_normalize({
            'direction': direction,
            'textColor': text_color,
            'strokeColor': stroke_color,
            'fontSizePx': font_size_px,
            'strokeWidthPx': stroke_width_px,
            'lineHeight': line_height,
            'angleDeg': angle_deg,
        }))
    return preds
