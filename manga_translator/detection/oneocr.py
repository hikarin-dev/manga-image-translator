import cv2
import numpy as np

from .common import CommonDetector
from ..utils import Quadrilateral
from ..utils.executors import run_cpu
from ..utils.oneocr import get_engine


class OneOcrDetector(CommonDetector):
    '''
    Detects text with the Windows OneOCR engine in a single full-page pass.
    Returned textlines already carry the recognized text and per-line style
    (see the oneocr OCR model, which passes them through instead of re-reading).
    detect_size, box_threshold and unclip_ratio have no equivalent here and are
    ignored; text_threshold filters lines by mean word confidence.
    '''

    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                      unclip_ratio: float, verbose: bool = False):
        result = await run_cpu(lambda: get_engine().recognize(image))

        textlines = []
        raw_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for line in result['lines']:
            if not line['text'] or line['pts'] is None:
                continue
            confs = [w['conf'] for w in line['words'] if w['conf'] is not None]
            # Word confidences are normally present; keep the line rather than
            # silently dropping it when the engine omits them
            prob = float(np.mean(confs)) if confs else 0.9
            if prob < text_threshold:
                continue
            q = Quadrilateral(line['pts'].astype(np.int64), line['text'], prob)
            q.oneocr_style = line['style']
            q.oneocr_style_conf = line['style_conf']
            textlines.append(q)
            boxes = [w['pts'] for w in line['words'] if w['pts'] is not None] or [line['pts']]
            cv2.fillPoly(raw_mask, [b.astype(np.int32) for b in boxes], 255)
            if verbose:
                self.logger.info(f'[{line["style"]} {line["style_conf"]:.2f}] {prob:.2f} {line["text"]}')

        return textlines, raw_mask, None
