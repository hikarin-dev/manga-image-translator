'''
Hayai OCR v2.5 "Nova" variant of the mocr OCR slot. Text comes from
JustANormalTinkerer/hayai-ocr-v2.5-nova (a manga-ocr fork: SigLIP2 NaFlex
encoder that keeps the crop's aspect ratio + 12-layer GQA decoder), run through
the self-contained torch port in hayai_nova.py because the upstream remote code
needs transformers >= 4.49. Region handling and per-region probability/font
colors are inherited from ModelMangaOCRFast (48px CTC color head).
'''

from typing import List
import numpy as np

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from manga_ocr.ocr import post_process as mocr_post_process

from .hayai_nova import HayaiModel, preprocess
from .model_48px_ctc import OCR as CTCOCR
from .model_manga_ocr_fast import ModelMangaOCRFast
from ..utils import chunks
from ..utils.executors import run_cpu, run_gpu

_HF_BASE = 'https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova/resolve/e34d7755ed11e626c5ba39544af5d66f20ee57cc/'

class ModelHayaiOCR(ModelMangaOCRFast):
    _MODEL_MAPPING = {
        'model': ModelMangaOCRFast._MODEL_MAPPING['model'],
        'hayai-weights': {
            'url': _HF_BASE + 'model.safetensors',
            'hash': 'ac63cd177c5b68bd91e18dd4408e0400461c7cf19f484b0da1643e232d1ffc3a',
            'file': 'hayai-nova.safetensors',
        },
        'hayai-tokenizer': {
            'url': _HF_BASE + 'tokenizer.json',
            'hash': 'f8a0a909c628a684fe463094614e236a8b1d3609e7770f77e7beafaf1056bf13',
            'file': 'hayai-nova-tokenizer.json',
        },
    }

    # NaFlex patch budget: 512 is upstream's Nova default and its best-scoring
    # setting (JMangaBench CER 3.10% vs 3.65% at 384), ~1.34x the cost of 256.
    _MAX_NUM_PATCHES = 512
    _MAX_NEW_TOKENS = 128
    _HAYAI_BATCH_SIZE = 64
    # Hayai reads dense, vertical and multi-line text in one pass; its benchmark scores are on
    # region crops, not single lines (model card).
    _READS_REGIONS = True

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
        self.device = device
        if (device == 'cuda' or device == 'mps'):
            self.use_gpu = True
        else:
            self.use_gpu = False
        if self.use_gpu:
            self.model = self.model.to(device)

        self.hayai = HayaiModel()
        sd = load_file(self._get_file_path('hayai-nova.safetensors'))
        missing, unexpected = self.hayai.load_state_dict(sd, strict = False)
        # Only the unused SigLIP2 pooling head and the training-only IDS radical head may be left over.
        unexpected = [k for k in unexpected if not k.startswith(('vision_encoder.vision_model.head.', 'decoder.ids_'))]
        if missing or unexpected:
            raise RuntimeError(f'Hayai checkpoint mismatch: missing={missing} unexpected={unexpected}')
        self.hayai.eval().to(device)

        self.hayai_tokenizer = Tokenizer.from_file(self._get_file_path('hayai-nova-tokenizer.json'))
        self.hayai_bos = self.hayai_tokenizer.token_to_id('<bos>')
        self.hayai_eos = self.hayai_tokenizer.token_to_id('<eos>')
        self.hayai_pad = self.hayai_tokenizer.token_to_id('<pad>')

    async def _unload(self):
        del self.model
        del self.hayai
        del self.hayai_tokenizer

    def _hayai_prepare(self, images: List[np.ndarray]):
        '''CPU: NaFlex resize/normalize, one tensor set per batch.'''
        return [preprocess(batch, self._MAX_NUM_PATCHES) for batch in chunks(images, self._HAYAI_BATCH_SIZE)]

    def _hayai_generate(self, batches) -> List[List[int]]:
        '''GPU lane: greedy decode of every prepared batch.'''
        device = next(self.hayai.parameters()).device
        ids = []
        for pixel_values, pixel_mask, shapes in batches:
            with torch.inference_mode():
                ids.extend(self.hayai.generate(pixel_values.to(device), pixel_mask.to(device), shapes.to(device),
                                               self.hayai_bos, self.hayai_eos, self.hayai_pad, self._MAX_NEW_TOKENS))
        return ids

    def _hayai_texts(self, ids: List[List[int]]) -> List[str]:
        '''CPU: token ids to text.'''
        return [self.hayai_tokenizer.decode(seq, skip_special_tokens = True) for seq in ids]

    def _hayai_raw(self, images: List[np.ndarray]) -> List[str]:
        '''Hayai text for each RGB crop, before manga-ocr post-processing.'''
        return self._hayai_texts(self._hayai_generate(self._hayai_prepare(images)))

    def _mocr_batch(self, images: List[np.ndarray]) -> List[str]:
        '''Same contract as ModelMangaOCRFast._mocr_batch, but through Hayai OCR.'''
        return [mocr_post_process(t) for t in self._hayai_raw(images)]

    async def _texts(self, images: List[np.ndarray]) -> List[str]:
        '''Only the decode runs on the GPU lane; preprocessing and detokenizing stay on the CPU pool.'''
        batches = await run_cpu(self._hayai_prepare, images)
        ids = await run_gpu(self._hayai_generate, batches, lane = self._GPU_LANE)
        return await run_cpu(lambda: [mocr_post_process(t) for t in self._hayai_texts(ids)])
