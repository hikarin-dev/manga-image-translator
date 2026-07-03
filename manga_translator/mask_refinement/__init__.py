import logging
import time
from typing import List
import cv2
import numpy as np

from .text_mask_utils import complete_mask_fill, complete_mask
from ..utils import TextBlock, Quadrilateral
from ..utils.bubble import is_ignore
from ..utils.executors import run_cpu, run_proc
from ..utils.profiling import add_substage

_proc_broken = False

async def dispatch(text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text', dilation_offset: int = 0, ignore_bubble: int = 0, verbose: bool = False,kernel_size:int=3) -> np.ndarray:
    # complete_mask is ~1s/page of Python-heavy connected-component/watershed work; on a
    # thread it serializes with everything else on the GIL, capping pipeline throughput.
    # Run the EXACT same function on the process pool (identical output, no GIL); fall back
    # to the thread pool if the pool ever breaks (e.g. a pickling regression).
    global _proc_broken
    t0 = time.perf_counter()
    if method == 'fit_text' and ignore_bubble < 1 and not _proc_broken:
        try:
            pre = await run_cpu(_prepare, text_regions, raw_image, raw_mask)
            img_resized, mask_resized, textlines, scale_factor = pre
            final_mask = await run_proc(complete_mask, img_resized, mask_resized, textlines,
                                        1e-2, dilation_offset, kernel_size)
            out = await run_cpu(_finalize, final_mask, raw_image)
            add_substage('mask_refine', time.perf_counter() - t0)
            return out
        except Exception as e:
            _proc_broken = True
            logging.getLogger('mask-refinement').warning(
                f'process-pool mask refinement failed ({e}); falling back to in-process')
    out = await run_cpu(_dispatch_sync, text_regions, raw_image, raw_mask, method,
                        dilation_offset, ignore_bubble, verbose, kernel_size)
    add_substage('mask_refine', time.perf_counter() - t0)
    return out


def _prepare(text_regions, raw_image, raw_mask):
    """Thread-side pre: downscale + build the per-line quads (small, picklable)."""
    scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)
    img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)
    mask_resized[mask_resized > 0] = 255
    textlines = []
    for region in text_regions:
        for l in region.lines:
            textlines.append(Quadrilateral(l * scale_factor, '', 0))
    return img_resized, mask_resized, textlines, scale_factor


def _finalize(final_mask, raw_image):
    """Thread-side post: upscale the refined mask back to page size."""
    if final_mask is None:
        return np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype = np.uint8)
    final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation = cv2.INTER_LINEAR)
    final_mask[final_mask > 0] = 255
    return final_mask


def _dispatch_sync(text_regions: List[TextBlock], raw_image: np.ndarray, raw_mask: np.ndarray, method: str = 'fit_text', dilation_offset: int = 0, ignore_bubble: int = 0, verbose: bool = False,kernel_size:int=3) -> np.ndarray:
    # Larger sized mask images will probably have crisper and thinner mask segments due to being able to fit the text pixels better
    # so we dont want to size them down as much to not lose information
    scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)

    img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)

    mask_resized[mask_resized > 0] = 255
    textlines = []
    for region in text_regions:
        for l in region.lines:
            q = Quadrilateral(l * scale_factor, '', 0)
            textlines.append(q)

    final_mask = complete_mask(img_resized, mask_resized, textlines, dilation_offset=dilation_offset,kernel_size=kernel_size) if method == 'fit_text' else complete_mask_fill([txtln.aabb.xywh for txtln in textlines])
    if final_mask is None:
        final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype = np.uint8)
    else:
        final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        final_mask[final_mask > 0] = 255

    if ignore_bubble < 1 or ignore_bubble > 50:
        return final_mask

    # bubble
    kernel_size = int(max(final_mask.shape) * 0.025)  # 选择一个合适的核大小
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)  # 根据需要调整迭代次数
    # border
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        temp_mask = np.zeros_like(final_mask)
        # rect min
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, -1)
        # get textblock
        textblock=cv2.bitwise_and(raw_image, raw_image, mask=temp_mask)
        if is_ignore(textblock, ignore_bubble):
            cv2.drawContours(final_mask, [cnt], -1, 0, -1)

    return final_mask
