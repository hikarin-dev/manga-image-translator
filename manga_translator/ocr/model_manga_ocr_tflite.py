'''
TFLite variant of the mocr OCR slot. Text comes from the KV-cached LiteRT
conversion of manga-ocr (jgalamba/manga-ocr-kvcache-tflite: int8 ViT encoder +
fp16 2-layer decoder with an explicit KV-cache init/step signature pair),
running on CPU via ai-edge-litert instead of the torch manga-ocr model.
Region handling and per-region probability/font colors are inherited from
ModelMangaOCRFast (48px CTC color head).
'''

import csv
from typing import List
from PIL import Image
import numpy as np

import torch

from manga_ocr.ocr import post_process as mocr_post_process

from .model_48px_ctc import OCR as CTCOCR
from .model_manga_ocr_fast import ModelMangaOCRFast

_HF_BASE = 'https://huggingface.co/jgalamba/manga-ocr-kvcache-tflite/resolve/main/'

class ModelMangaOCRTflite(ModelMangaOCRFast):
    _MODEL_MAPPING = {
        'model': ModelMangaOCRFast._MODEL_MAPPING['model'],
        'tflite-encoder': {
            'url': _HF_BASE + 'encoder_int8.tflite',
            'hash': 'bf858e9189b66d2da915df36c1a3fa056a0795b9a7948461085dc06216747b9a',
            'file': 'encoder_int8.tflite',
        },
        'tflite-decoder': {
            'url': _HF_BASE + 'decoder_cache_fp16.tflite',
            'hash': '4695855693df18652a3b896fe97c492b943cd1128ed461fddd17155320e30025',
            'file': 'decoder_cache_fp16.tflite',
        },
        'tflite-vocab': {
            'url': _HF_BASE + 'mocr2025_vocab.csv',
            'hash': 'b16c1cd27ecce8ede5fd358c706eec910cb632d5e032a8709e7703d0365b6d51',
            'file': 'mocr2025_vocab.csv',
        },
    }

    _BOS = 2
    _EOS = 3
    _MIN_EMIT_ID = 5
    _SELF_KV_MAX = 256

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

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            raise ImportError('The mocr_tflite OCR model requires the ai-edge-litert package (pip install ai-edge-litert)')

        import os
        threads = os.cpu_count() or 4
        self.tfl_encoder = Interpreter(model_path = self._get_file_path('encoder_int8.tflite'), num_threads = threads)
        self.tfl_decoder = Interpreter(model_path = self._get_file_path('decoder_cache_fp16.tflite'), num_threads = threads)
        self.enc_run = self.tfl_encoder.get_signature_runner('serving_default')
        self.dec_init = self.tfl_decoder.get_signature_runner('init')
        self.dec_step = self.tfl_decoder.get_signature_runner('step')
        self._token_dtype = self.dec_init.get_input_details()['args_1']['dtype']
        self._pos_dtype = self.dec_step.get_input_details()['args_2']['dtype']
        self._kv_dtype = self.dec_step.get_input_details()['args_3']['dtype']

        self.vocab = {}
        with open(self._get_file_path('mocr2025_vocab.csv'), 'r', encoding = 'utf-8', newline = '') as fp:
            reader = csv.reader(fp)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].isdigit():
                    self.vocab[int(row[0])] = row[1]

    async def _unload(self):
        del self.model
        del self.enc_run
        del self.dec_init
        del self.dec_step
        del self.tfl_encoder
        del self.tfl_decoder

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img).convert('L').convert('RGB').resize((224, 224), Image.BILINEAR)
        px = np.asarray(pil, dtype = np.float32)
        px = (px / 255.0 - 0.5) / 0.5
        return px.transpose(2, 0, 1)[None]

    def _decode_one(self, enc_out: np.ndarray, max_tokens: int = 300) -> str:
        kv_shape = (1, 12, self._SELF_KV_MAX, 64)
        self_k = [np.zeros(kv_shape, dtype = self._kv_dtype) for _ in range(2)]
        self_v = [np.zeros(kv_shape, dtype = self._kv_dtype) for _ in range(2)]

        token = np.array([[self._BOS]], dtype = self._token_dtype)
        out = self.dec_init(args_0 = enc_out, args_1 = token)
        for l in range(2):
            self_k[l][:, :, 0:1, :] = out[f'output_{1 + l}']
            self_v[l][:, :, 0:1, :] = out[f'output_{3 + l}']
        cross = {
            'args_7': out['output_5'], 'args_8': out['output_6'],
            'args_9': out['output_7'], 'args_10': out['output_8'],
        }

        ids = []
        next_id = int(np.argmax(out['output_0'][0]))
        pos = 1
        limit = min(max_tokens, self._SELF_KV_MAX - 1)
        while next_id != self._EOS and pos <= limit:
            ids.append(next_id)
            token = np.array([[next_id]], dtype = self._token_dtype)
            out = self.dec_step(args_0 = enc_out, args_1 = token,
                                args_2 = np.array([pos], dtype = self._pos_dtype),
                                args_3 = self_k[0], args_4 = self_k[1],
                                args_5 = self_v[0], args_6 = self_v[1],
                                **cross)
            for l in range(2):
                self_k[l][:, :, pos:pos + 1, :] = out[f'output_{1 + l}']
                self_v[l][:, :, pos:pos + 1, :] = out[f'output_{3 + l}']
            next_id = int(np.argmax(out['output_0'][0]))
            pos += 1

        text = ''.join(self.vocab.get(i, '') for i in ids if i >= self._MIN_EMIT_ID)
        return mocr_post_process(text)

    def _mocr_batch(self, images: List[np.ndarray]) -> List[str]:
        '''Same contract as ModelMangaOCRFast._mocr_batch, but through LiteRT.'''
        texts = []
        for img in images:
            enc_out = self.enc_run(args_0 = self._preprocess(img))['output_0']
            texts.append(self._decode_one(enc_out))
        return texts
