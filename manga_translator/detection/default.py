import os
import shutil
import time
import numpy as np
import torch
import cv2
import einops
from typing import List, Tuple

from .default_utils.DBNet_resnet34 import TextDetection as TextDetectionDefault
from .default_utils import imgproc, dbnet_utils, craft_utils
from .common import OfflineDetector
from ..utils import TextBlock, Quadrilateral, det_rearrange_forward
from ..utils.executors import run_cpu, submit_gpu
from ..utils.profiling import add_substage

MODEL = None
def det_batch_forward_default(batch: np.ndarray, device: str):
    global MODEL
    if isinstance(batch, list):
        batch = np.array(batch)
    batch = einops.rearrange(batch.astype(np.float32) / 127.5 - 1.0, 'n h w c -> n c h w')
    batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        db, mask = MODEL(batch)
        db = db.sigmoid().cpu().numpy()
        mask = mask.cpu().numpy()
    return db, mask

class DefaultDetector(OfflineDetector):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt',
            'hash': '67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e',
            'file': '.',
        }
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('detect-20241225.ckpt'):
            shutil.move('detect-20241225.ckpt', self._get_file_path('detect-20241225.ckpt'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        self.model = TextDetectionDefault()
        sd = torch.load(self._get_file_path('detect-20241225.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        self.device = device
        if device == 'cuda' or device == 'mps':
            self.model = self.model.to(self.device)
        global MODEL
        MODEL = self.model

    async def _unload(self):
        del self.model

    # _infer orchestrates its own placement: heavy host work (the d=17 bilateral filter,
    # DBNet box extraction) runs on the CPU pool, only the forward runs on the GPU thread.
    # Same ops in the same order — results identical; the GPU lane just stops paying for host work.
    _SELF_ORCHESTRATED = True
    # Lane 1: detection kernels overlap the OCR beam loop's host gaps on lane 0.
    _GPU_LANE = 1

    async def _rearrange_forward_gpu(self, image: np.ndarray, detect_size: int, verbose: bool):
        # TODO: Move det_rearrange_forward to common.py and refactor
        return det_rearrange_forward(image, det_batch_forward_default, detect_size, 4, device=self.device, verbose=verbose)

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):

        # Tall-strip rearrange path (webtoons): host/GPU interleaved, keep whole on the GPU
        # thread as before. For normal pages it returns (None, None) immediately.
        db, mask = await submit_gpu(self._rearrange_forward_gpu(image, detect_size, verbose), self._GPU_LANE)

        if db is None:
            # rearrangement is not required, fallback to default forward
            def _pre():
                t0 = time.perf_counter()
                out = imgproc.resize_aspect_ratio(cv2.bilateralFilter(image, 17, 80, 80), detect_size, cv2.INTER_LINEAR, mag_ratio = 1)
                add_substage('det_pre', time.perf_counter() - t0)
                return out
            img_resized, target_ratio, _, pad_w, pad_h = await run_cpu(_pre)
            img_resized_h, img_resized_w = img_resized.shape[:2]
            ratio_h = ratio_w = 1 / target_ratio

            async def _fwd():
                t0 = time.perf_counter()
                out = det_batch_forward_default([img_resized], self.device)
                add_substage('det_gpu', time.perf_counter() - t0)
                return out
            db, mask = await submit_gpu(_fwd(), self._GPU_LANE)
        else:
            img_resized_h, img_resized_w = image.shape[:2]
            ratio_w = ratio_h = 1
            pad_h = pad_w = 0
        self.logger.debug(f'Detection resolution: {img_resized_w}x{img_resized_h}')

        def _post():
            t0 = time.perf_counter()
            mask_ = mask[0, 0, :, :]
            det = dbnet_utils.SegDetectorRepresenter(text_threshold, box_threshold, unclip_ratio=unclip_ratio)
            boxes, scores = det({'shape':[(img_resized_h, img_resized_w)]}, db)
            boxes, scores = boxes[0], scores[0]
            if boxes.size == 0:
                polys = []
            else:
                idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
                polys, _ = boxes[idx], scores[idx]
                polys = polys.astype(np.float64)
                polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=1)
                polys = polys.astype(np.int64)

            textlines = [Quadrilateral(pts.astype(int), '', score) for pts, score in zip(polys, scores)]
            textlines = list(filter(lambda q: q.area > 16, textlines))
            mask_resized = cv2.resize(mask_, (mask_.shape[1] * 2, mask_.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
            if pad_h > 0:
                mask_resized = mask_resized[:-pad_h, :]
            elif pad_w > 0:
                mask_resized = mask_resized[:, :-pad_w]
            raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)
            add_substage('det_post', time.perf_counter() - t0)
            return textlines, raw_mask

        textlines, raw_mask = await run_cpu(_post)
        return textlines, raw_mask, None
