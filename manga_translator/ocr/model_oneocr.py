from typing import List

import numpy as np

from .colors import estimate_colors as _estimate_colors
from .common import CommonOCR
from ..config import OcrConfig
from ..utils import Quadrilateral
from ..utils.executors import run_cpu
from ..utils.oneocr import get_engine


_CJK_RANGES = ((0x3000, 0x30ff), (0x3400, 0x4dbf), (0x4e00, 0x9fff), (0xf900, 0xfaff), (0xff00, 0xffef))


def _is_cjk(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES)


def _join_line_texts(parts: List[str]) -> str:
    '''Joins the engine's line fragments, omitting spaces between CJK characters.'''
    out = ''
    for part in parts:
        if out and not (_is_cjk(out[-1]) and _is_cjk(part[0])):
            out += ' '
        out += part
    return out


class OneOcrOCR(CommonOCR):
    '''
    Reads detected text regions with the Windows OneOCR engine.
    When the textlines already carry text (produced by the oneocr detector's
    combined detect+read pass) they are passed through without re-reading;
    otherwise each region is unwarped to a horizontal strip and recognized
    individually. Either way a cheap fg/bg color estimate is attached, plus the
    line style (printed/handwritten) reported by the engine.
    '''

    _CROP_HEIGHT = 48

    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig,
                         verbose: bool = False, result_dir: str = None) -> List[Quadrilateral]:
        if not textlines:
            return textlines
        passthrough = all(getattr(q, 'text', '') for q in textlines)
        min_prob = config.prob if config.prob is not None else 0.2
        quadrilaterals = list(self._generate_text_direction(textlines))

        def _run():
            engine = None if passthrough else get_engine()
            output = []
            for q, direction in quadrilaterals:
                if not isinstance(q, Quadrilateral):
                    # TextBlock lines are not produced by the detectors this model targets
                    continue
                crop = q.get_transformed_region(image, direction, self._CROP_HEIGHT)
                if not passthrough:
                    result = engine.recognize(crop)
                    lines = [ln for ln in result['lines'] if ln['text']]
                    if not lines:
                        continue
                    confs = [w['conf'] for ln in lines for w in ln['words'] if w['conf'] is not None]
                    prob = float(np.mean(confs)) if confs else 0.9
                    if prob < min_prob:
                        continue
                    # The crop is a horizontal strip; the engine may split it
                    # into several lines, so order them by x before joining
                    lines.sort(key=lambda ln: float('inf') if ln['pts'] is None else ln['pts'][:, 0].mean())
                    q.text = _join_line_texts([ln['text'] for ln in lines])
                    q.prob = prob
                    q.oneocr_style = lines[0]['style']
                    q.oneocr_style_conf = lines[0]['style_conf']
                fg, bg = _estimate_colors(crop)
                q.color_estimate = (fg, bg)
                q.fg_r, q.fg_g, q.fg_b = fg
                q.bg_r, q.bg_g, q.bg_b = bg
                if verbose:
                    style = getattr(q, 'oneocr_style', None)
                    self.logger.info(f'prob: {q.prob} [{style}] {q.text} fg: {tuple(fg)} bg: {tuple(bg)}')
                output.append(q)
            return output

        return await run_cpu(_run)
