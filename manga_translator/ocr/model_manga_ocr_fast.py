'''
Fast variant of the mocr OCR slot. Text comes from manga-ocr run in batches
(the ViT processor resizes every region to a fixed 224px input, so batching
needs no padding) instead of once per region in a Python loop. Per-region
probability and font colors come from the non-autoregressive 48px CTC model's
trained color head — a single batched forward — replacing the autoregressive
48px beam decode that dominates the stock model's runtime.
'''

import os
import shutil
from typing import List
import cv2
from PIL import Image
import numpy as np
import einops

import torch

from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process as mocr_post_process

from .common import OfflineOCR
from .model_48px_ctc import OCR as CTCOCR
from .model_manga_ocr import merge_bboxes
from ..config import OcrConfig
from ..utils import TextBlock, Quadrilateral, AvgMeter, chunks

class ModelMangaOCRFast(OfflineOCR):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip',
            'hash': 'fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101',
            'archive': {
                'ocr-ctc.ckpt': '.',
                'alphabet-all-v5.txt': '.',
            },
        },
    }

    _MOCR_BATCH_SIZE = 16

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('ocr-ctc.ckpt'):
            shutil.move('ocr-ctc.ckpt', self._get_file_path('ocr-ctc.ckpt'))
        if os.path.exists('alphabet-all-v5.txt'):
            shutil.move('alphabet-all-v5.txt', self._get_file_path('alphabet-all-v5.txt'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str):
        with open(self._get_file_path('alphabet-all-v5.txt'), 'r', encoding = 'utf-8') as fp:
            dictionary = [s[:-1] for s in fp.readlines()]

        self.model = CTCOCR(dictionary, 768)
        sd = torch.load(self._get_file_path('ocr-ctc.ckpt'), map_location = 'cpu')
        sd = sd['model'] if 'model' in sd else sd
        del sd['encoders.layers.0.pe.pe']
        del sd['encoders.layers.1.pe.pe']
        del sd['encoders.layers.2.pe.pe']
        self.model.load_state_dict(sd, strict = False)
        self.model.eval()
        self.mocr = MangaOcr()
        self.device = device
        if (device == 'cuda' or device == 'mps'):
            self.use_gpu = True
        else:
            self.use_gpu = False
        if self.use_gpu:
            self.model = self.model.to(device)

    async def _unload(self):
        del self.model
        del self.mocr

    def _mocr_batch(self, images: List[np.ndarray]) -> List[str]:
        '''Batched equivalent of calling self.mocr(img) once per image.'''
        mocr = self.mocr
        texts = []
        for batch in chunks(images, self._MOCR_BATCH_SIZE):
            pil_images = [Image.fromarray(img).convert('L').convert('RGB') for img in batch]
            pixel_values = mocr.processor(pil_images, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(device=mocr.model.device, dtype=mocr.model.dtype)
            with torch.no_grad():
                token_ids = mocr.model.generate(pixel_values, max_length=300)
            for row in token_ids:
                texts.append(mocr_post_process(mocr.tokenizer.decode(row, skip_special_tokens=True)))
        return texts

    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False, ignore_bubble: int = 0, result_dir: str = None) -> List[TextBlock]:
        text_height = 48
        max_chunk_size = 16
        threshold = config.prob if config.prob is not None else 0.2

        quadrilaterals = list(self._generate_text_direction(textlines))
        region_imgs = [q.get_transformed_region(image, d, text_height) for q, d in quadrilaterals]

        perm = range(len(region_imgs))
        is_quadrilaterals = False
        if len(quadrilaterals) > 0 and isinstance(quadrilaterals[0][0], Quadrilateral):
            perm = sorted(range(len(region_imgs)), key = lambda x: region_imgs[x].shape[1])
            is_quadrilaterals = True

        if config.use_mocr_merge:
            merged_textlines, merged_idx = await merge_bboxes(textlines, image.shape[1], image.shape[0])
            merged_quadrilaterals = list(self._generate_text_direction(merged_textlines))
        else:
            merged_idx = [[i] for i in range(len(region_imgs))]
            merged_quadrilaterals = quadrilaterals
        merged_region_imgs = []
        for q, d in merged_quadrilaterals:
            if d == 'h':
                merged_text_height = q.aabb.w
                merged_d = 'h'
            elif d == 'v':
                merged_text_height = q.aabb.h
                merged_d = 'h'
            merged_region_imgs.append(q.get_transformed_region(image, merged_d, merged_text_height))
        texts = {idx: text for idx, text in enumerate(self._mocr_batch(merged_region_imgs))}

        # CTC pass over the textline strips for probability + font colors
        ix = 0
        line_stats = {}
        for indices in chunks(perm, max_chunk_size):
            N = len(indices)
            widths = [region_imgs[i].shape[1] for i in indices]
            max_width = (4 * (max(widths) + 7) // 4) + 128
            region = np.zeros((N, text_height, max_width, 3), dtype = np.uint8)
            for i, idx in enumerate(indices):
                W = region_imgs[idx].shape[1]
                region[i, :, : W, :] = region_imgs[idx]
                if verbose:
                    ocr_result_dir = result_dir or os.environ.get('MANGA_OCR_RESULT_DIR', 'result/ocrs/')
                    os.makedirs(ocr_result_dir, exist_ok=True)
                    if quadrilaterals[idx][1] == 'v':
                        cv2.imwrite(os.path.join(ocr_result_dir, f'{ix}.png'), cv2.rotate(cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE))
                    else:
                        cv2.imwrite(os.path.join(ocr_result_dir, f'{ix}.png'), cv2.cvtColor(region[i, :, :, :], cv2.COLOR_RGB2BGR))
                ix += 1
            images = (torch.from_numpy(region).float() - 127.5) / 127.5
            images = einops.rearrange(images, 'N H W C -> N C H W')
            if self.use_gpu:
                images = images.to(self.device)
            with torch.inference_mode():
                decoded = self.model.decode(images, widths, 0, verbose = verbose)
            for i, single_line in enumerate(decoded):
                if not single_line:
                    continue
                total_fr = AvgMeter()
                total_fg = AvgMeter()
                total_fb = AvgMeter()
                total_br = AvgMeter()
                total_bg = AvgMeter()
                total_bb = AvgMeter()
                total_logprob = AvgMeter()
                for (chid, logprob, fr, fg, fb, br, bg, bb) in single_line:
                    ch = self.model.dictionary[chid]
                    total_logprob(logprob)
                    if ch != '<SP>':
                        total_fr(int(fr * 255))
                        total_fg(int(fg * 255))
                        total_fb(int(fb * 255))
                        total_br(int(br * 255))
                        total_bg(int(bg * 255))
                        total_bb(int(bb * 255))
                prob = np.exp(total_logprob())
                if prob < threshold:
                    continue
                clamp = lambda v: min(max(int(v), 0), 255)
                cur_region = quadrilaterals[indices[i]][0]
                area = cur_region.area if isinstance(cur_region, Quadrilateral) else 1
                line_stats[indices[i]] = (prob, area,
                                          (clamp(total_fr()), clamp(total_fg()), clamp(total_fb())),
                                          (clamp(total_br()), clamp(total_bg()), clamp(total_bb())))

        output_regions = []
        for i, nodes in enumerate(merged_idx):
            total_logprobs = 0
            total_area = 0
            fg_list = []
            bg_list = []
            for idx in nodes:
                if idx not in line_stats:
                    continue
                prob, area, fg, bg = line_stats[idx]
                total_logprobs += np.log(prob) * area
                total_area += area
                fg_list.append(fg)
                bg_list.append(bg)
            if total_area > 0:
                prob = float(np.exp(total_logprobs / total_area))
            else:
                prob = 0.0
            fr, fg, fb = ([int(round(v)) for v in np.mean(fg_list, axis=0)] if fg_list else (0, 0, 0))
            br, bg, bb = ([int(round(v)) for v in np.mean(bg_list, axis=0)] if bg_list else (0, 0, 0))

            txt = texts[i]
            self.logger.info(f'prob: {prob} {txt} fg: ({fr}, {fg}, {fb}) bg: ({br}, {bg}, {bb})')
            cur_region = merged_quadrilaterals[i][0]
            if isinstance(cur_region, Quadrilateral):
                cur_region.text = txt
                cur_region.prob = prob
                cur_region.fg_r = fr
                cur_region.fg_g = fg
                cur_region.fg_b = fb
                cur_region.bg_r = br
                cur_region.bg_g = bg
                cur_region.bg_b = bb
            else: # TextBlock
                cur_region.text.append(txt)
                cur_region.update_font_colors(np.array([fr, fg, fb]), np.array([br, bg, bb]))
            output_regions.append(cur_region)

        if is_quadrilaterals:
            return output_regions
        return textlines
