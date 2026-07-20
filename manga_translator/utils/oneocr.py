"""
ctypes wrapper around the Windows 11 OneOCR engine (oneocr.dll).

The runtime consists of three files shipped with the Snipping Tool app package:
oneocr.dll, oneocr.onemodel and onnxruntime.dll. They are looked up in
ONEOCR_DIR (env), then <BASE_PATH>/models/oneocr/, and as a last resort copied
there automatically from the installed Snipping Tool package.

API surface reverse engineered by https://github.com/b1tg/win11-oneocr and
refined by https://github.com/AuroraWright/oneocr and
https://github.com/wangfu91/oneocr-rs (GetOcrLineStyle signature).
"""

import ctypes
import os
import shutil
import subprocess
import threading
from ctypes import POINTER, Structure, byref, c_char, c_char_p, c_float, c_int32, c_int64, c_ubyte
from typing import Optional

import cv2
import numpy as np

from .generic import BASE_PATH
from .log import get_logger

logger = get_logger('oneocr')

MODEL_KEY = b'kj)TGtrK>f]b[Piow.gU+nC@s""""""4'
REQUIRED_FILES = ('oneocr.dll', 'oneocr.onemodel', 'onnxruntime.dll')
# The engine rejects images below 50px or above 10000px on either side
MIN_SIDE = 50
MAX_SIDE = 10000

c_int64_p = POINTER(c_int64)
c_float_p = POINTER(c_float)
c_int32_p = POINTER(c_int32)
c_ubyte_p = POINTER(c_ubyte)


class _Image(Structure):
    _fields_ = [
        ('type', c_int32),
        ('width', c_int32),
        ('height', c_int32),
        ('_reserved', c_int32),
        ('step', c_int64),
        ('data_ptr', c_ubyte_p),
    ]


class _BBox(Structure):
    # Four corners, clockwise from top-left
    _fields_ = [(name, c_float) for name in ('x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4')]


_BBox_p = POINTER(_BBox)

_FUNCTIONS = [
    ('CreateOcrInitOptions', [c_int64_p], c_int64),
    ('OcrInitOptionsSetUseModelDelayLoad', [c_int64, c_char], c_int64),
    ('CreateOcrPipeline', [c_char_p, c_char_p, c_int64, c_int64_p], c_int64),
    ('CreateOcrProcessOptions', [c_int64_p], c_int64),
    ('OcrProcessOptionsSetMaxRecognitionLineCount', [c_int64, c_int64], c_int64),
    ('RunOcrPipeline', [c_int64, POINTER(_Image), c_int64, c_int64_p], c_int64),
    ('GetImageAngle', [c_int64, c_float_p], c_int64),
    ('GetOcrLineCount', [c_int64, c_int64_p], c_int64),
    ('GetOcrLine', [c_int64, c_int64, c_int64_p], c_int64),
    ('GetOcrLineContent', [c_int64, POINTER(c_char_p)], c_int64),
    ('GetOcrLineBoundingBox', [c_int64, POINTER(_BBox_p)], c_int64),
    ('GetOcrLineStyle', [c_int64, c_int32_p, c_float_p], c_int64),
    ('GetOcrLineWordCount', [c_int64, c_int64_p], c_int64),
    ('GetOcrWord', [c_int64, c_int64, c_int64_p], c_int64),
    ('GetOcrWordContent', [c_int64, POINTER(c_char_p)], c_int64),
    ('GetOcrWordBoundingBox', [c_int64, POINTER(_BBox_p)], c_int64),
    ('GetOcrWordConfidence', [c_int64, c_float_p], c_int64),
    ('ReleaseOcrResult', [c_int64], None),
    ('ReleaseOcrInitOptions', [c_int64], None),
    ('ReleaseOcrPipeline', [c_int64], None),
    ('ReleaseOcrProcessOptions', [c_int64], None),
]


def _snipping_tool_dir() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command',
             '(Get-AppxPackage Microsoft.ScreenSketch | Select-Object -Last 1).InstallLocation'],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        return None
    loc = out.strip().splitlines()[-1].strip() if out.strip() else ''
    if not loc:
        return None
    src = os.path.join(loc, 'SnippingTool')
    return src if os.path.isfile(os.path.join(src, 'oneocr.dll')) else None


def _ensure_files() -> str:
    env_dir = os.environ.get('ONEOCR_DIR')
    target = env_dir or os.path.join(BASE_PATH, 'models', 'oneocr')
    if all(os.path.isfile(os.path.join(target, f)) for f in REQUIRED_FILES):
        return target
    src = _snipping_tool_dir()
    if src and all(os.path.isfile(os.path.join(src, f)) for f in REQUIRED_FILES):
        os.makedirs(target, exist_ok=True)
        for f in REQUIRED_FILES:
            logger.info(f'Copying {f} from the Snipping Tool package')
            shutil.copy2(os.path.join(src, f), os.path.join(target, f))
        return target
    raise RuntimeError(
        f'OneOCR runtime not found. Copy {", ".join(REQUIRED_FILES)} from the Windows 11 '
        f'Snipping Tool app package (WindowsApps/Microsoft.ScreenSketch_*/SnippingTool) into {target}')


class OneOcrEngine:
    '''Loads oneocr.dll once and exposes recognize() over RGB numpy images.'''

    def __init__(self):
        directory = _ensure_files()
        os.add_dll_directory(directory)
        self._dll = ctypes.WinDLL(os.path.join(directory, 'oneocr.dll'))
        for name, argtypes, restype in _FUNCTIONS:
            fn = getattr(self._dll, name)
            fn.argtypes = argtypes
            fn.restype = restype

        init_options = c_int64()
        self._check(self._dll.CreateOcrInitOptions(byref(init_options)), 'CreateOcrInitOptions')
        self._init_options = init_options
        self._check(self._dll.OcrInitOptionsSetUseModelDelayLoad(init_options, 0),
                    'OcrInitOptionsSetUseModelDelayLoad')
        pipeline = c_int64()
        model_path = os.path.join(directory, 'oneocr.onemodel').encode()
        self._check(self._dll.CreateOcrPipeline(model_path, MODEL_KEY, init_options, byref(pipeline)),
                    'CreateOcrPipeline')
        self._pipeline = pipeline
        options = c_int64()
        self._check(self._dll.CreateOcrProcessOptions(byref(options)), 'CreateOcrProcessOptions')
        self._options = options
        self._check(self._dll.OcrProcessOptionsSetMaxRecognitionLineCount(options, 1000),
                    'OcrProcessOptionsSetMaxRecognitionLineCount')
        self._lock = threading.Lock()
        logger.info('OneOCR pipeline loaded')

    @staticmethod
    def _check(code: int, what: str):
        if code != 0:
            raise RuntimeError(f'OneOCR {what} failed with code {code}')

    def recognize(self, image: np.ndarray) -> dict:
        '''
        Runs OCR over an RGB (HxWx3) or grayscale (HxW) uint8 image. Returns
        {'angle': float|None, 'lines': [{'text', 'pts' (4,2) float32, 'style',
        'style_conf', 'words': [{'text', 'pts', 'conf'}]}]}.
        '''
        if image.ndim == 2:
            bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        else:
            bgra = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)

        scale = 1.0
        h, w = bgra.shape[:2]
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            bgra = cv2.resize(bgra, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
            h, w = bgra.shape[:2]
        if min(h, w) < MIN_SIDE:
            bgra = cv2.copyMakeBorder(bgra, 0, max(0, MIN_SIDE - h), 0, max(0, MIN_SIDE - w),
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255, 255))
            h, w = bgra.shape[:2]

        bgra = np.ascontiguousarray(bgra)
        img = _Image(type=3, width=w, height=h, _reserved=0, step=w * 4,
                     data_ptr=bgra.ctypes.data_as(c_ubyte_p))
        with self._lock:
            result = c_int64()
            if self._dll.RunOcrPipeline(self._pipeline, byref(img), self._options, byref(result)) != 0:
                logger.warning('RunOcrPipeline failed, returning no lines')
                return {'angle': None, 'lines': []}
            try:
                return self._parse(result, 1.0 / scale)
            finally:
                self._dll.ReleaseOcrResult(result)

    def _parse(self, result, upscale: float) -> dict:
        angle = c_float()
        angle_val = angle.value if self._dll.GetImageAngle(result, byref(angle)) == 0 else None
        count = c_int64()
        if self._dll.GetOcrLineCount(result, byref(count)) != 0:
            return {'angle': angle_val, 'lines': []}
        lines = []
        for i in range(count.value):
            line = c_int64()
            if self._dll.GetOcrLine(result, i, byref(line)) != 0 or not line.value:
                continue
            content = c_char_p()
            text = ''
            if self._dll.GetOcrLineContent(line, byref(content)) == 0 and content.value:
                text = content.value.decode('utf-8', errors='ignore')
            style = style_conf = None
            style_val, conf = c_int32(), c_float()
            if self._dll.GetOcrLineStyle(line, byref(style_val), byref(conf)) == 0:
                style = 'handwritten' if style_val.value == 0 else 'printed'
                style_conf = conf.value
            words = []
            word_count = c_int64()
            if self._dll.GetOcrLineWordCount(line, byref(word_count)) == 0:
                for j in range(word_count.value):
                    word = c_int64()
                    if self._dll.GetOcrWord(line, j, byref(word)) != 0:
                        continue
                    wcontent = c_char_p()
                    wtext = ''
                    if self._dll.GetOcrWordContent(word, byref(wcontent)) == 0 and wcontent.value:
                        wtext = wcontent.value.decode('utf-8', errors='ignore')
                    wconf = c_float()
                    conf_val = wconf.value if self._dll.GetOcrWordConfidence(word, byref(wconf)) == 0 else None
                    words.append({'text': wtext,
                                  'pts': self._bbox(word, self._dll.GetOcrWordBoundingBox, upscale),
                                  'conf': conf_val})
            lines.append({'text': text,
                          'pts': self._bbox(line, self._dll.GetOcrLineBoundingBox, upscale),
                          'style': style, 'style_conf': style_conf,
                          'words': words})
        return {'angle': angle_val, 'lines': lines}

    @staticmethod
    def _bbox(handle, getter, upscale: float) -> Optional[np.ndarray]:
        ptr = _BBox_p()
        if getter(handle, byref(ptr)) != 0 or not ptr:
            return None
        b = ptr.contents
        pts = np.array([[b.x1, b.y1], [b.x2, b.y2], [b.x3, b.y3], [b.x4, b.y4]], dtype=np.float32)
        return pts * upscale


_engine: Optional[OneOcrEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> OneOcrEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = OneOcrEngine()
    return _engine
