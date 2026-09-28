'''
Self-contained torch port of Hayai OCR v2.5 "Nova"
(https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova, Apache-2.0,
pinned revision in model_hayai.py).

The upstream remote code imports `Siglip2VisionModel` from transformers >= 4.49
and its processor comes from transformers 5; this venv pins transformers 4.46.3
(manga-ocr breaks on 5.x). So the SigLIP2 NaFlex vision tower and its image
processor are re-implemented here in plain torch with the upstream parameter
names (checkpoint loads unchanged), ported from transformers 5.17. The decoder
is upstream `modeling_hayai.py` with two deliberate deviations, marked [shiori]:
  - no transformers PreTrainedModel/config plumbing;
  - batched generation masks the zero-padded vision tokens of shorter images
    (upstream leaves them attendable), so a crop reads the same whether it is
    decoded alone or in a batch;
  - on CUDA the decode steps replay as captured CUDA graphs (_DecodeGraph).
'''

import math
import os
import warnings
from collections import OrderedDict
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tvF


# =======================================================
# SigLIP2 NaFlex image processor (transformers 5.17, torchvision backend)
# =======================================================
PATCH_SIZE = 16


@lru_cache(maxsize=256)
def get_image_size_for_max_num_patches(image_height: int, image_width: int, patch_size: int,
                                       max_num_patches: int, eps: float = 1e-5) -> Tuple[int, int]:
    def get_scaled_image_size(scale: float, size: int, patch_size: int) -> int:
        scaled_size = size * scale
        scaled_size = math.ceil(scaled_size / patch_size) * patch_size
        scaled_size = max(patch_size, scaled_size)
        return int(scaled_size)

    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)
        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale

    scale = scale_min
    return (get_scaled_image_size(scale, image_height, patch_size),
            get_scaled_image_size(scale, image_width, patch_size))


def preprocess(images: List[np.ndarray], max_num_patches: int):
    '''RGB uint8 HWC arrays -> (pixel_values, pixel_attention_mask, spatial_shapes).'''
    pixel_values, masks, shapes = [], [], []
    for img in images:
        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        height, width = get_image_size_for_max_num_patches(image.shape[-2], image.shape[-1], PATCH_SIZE, max_num_patches)
        image = tvF.resize(image, [height, width], interpolation = tvF.InterpolationMode.BILINEAR, antialias = True)
        # fused rescale (1/255) + normalize (mean 0.5, std 0.5), as the torchvision backend does it
        image = tvF.normalize(image.to(dtype = torch.float32), [127.5] * 3, [127.5] * 3)
        c, h, w = image.shape
        nh, nw = h // PATCH_SIZE, w // PATCH_SIZE
        patches = image.reshape(c, nh, PATCH_SIZE, nw, PATCH_SIZE).permute(1, 3, 2, 4, 0).reshape(nh * nw, -1)
        mask = torch.ones((max_num_patches,), dtype = torch.int32)
        pad = max_num_patches - patches.shape[0]
        if pad > 0:
            patches = F.pad(patches, [0, 0, 0, pad], mode = 'constant', value = 0)
            mask[-pad:] = 0
        pixel_values.append(patches)
        masks.append(mask)
        shapes.append((nh, nw))
    return torch.stack(pixel_values), torch.stack(masks), torch.tensor(shapes)


# =======================================================
# SigLIP2 vision tower (google/siglip2-base-patch16-naflex vision config)
# =======================================================
class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, embed_dim: int = 768, num_patches: int = 256, num_channels: int = 3):
        super().__init__()
        self.patch_embedding = nn.Linear(num_channels * PATCH_SIZE * PATCH_SIZE, embed_dim)
        self.position_embedding_size = int(num_patches ** 0.5)
        self.position_embedding = nn.Embedding(num_patches, embed_dim)

    @staticmethod
    def resize_positional_embeddings(positional_embeddings: torch.Tensor, spatial_shapes: torch.Tensor,
                                     max_length: int) -> torch.Tensor:
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype
        result = torch.empty((batch_size, max_length, embed_dim), device = positional_embeddings.device, dtype = source_dtype)
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        if positional_embeddings.device.type == 'cpu':
            positional_embeddings = positional_embeddings.to(torch.float32)
        for i in range(batch_size):
            height, width = spatial_shapes[i].tolist()
            resized = F.interpolate(positional_embeddings, size = (height, width), mode = 'bilinear',
                                    align_corners = False, antialias = True)
            resized = resized.reshape(embed_dim, height * width).transpose(0, 1).to(source_dtype)
            result[i, : height * width] = resized
            result[i, height * width:] = resized[0]
        return result

    def forward(self, pixel_values: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values.to(dtype = self.patch_embedding.weight.dtype))
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1)
        return patch_embeds + self.resize_positional_embeddings(positional_embeddings, spatial_shapes, pixel_values.shape[1])


class Siglip2Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask, scale = self.head_dim ** -0.5)
        return self.out_proj(out.transpose(1, 2).reshape(b, s, -1))


class Siglip2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate = 'tanh'))


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, intermediate_size: int, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps = eps)
        self.self_attn = Siglip2Attention(embed_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps = eps)
        self.mlp = Siglip2MLP(embed_dim, intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), attention_mask)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Siglip2Encoder(nn.Module):
    def __init__(self, num_layers: int, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([Siglip2EncoderLayer(**layer_kwargs) for _ in range(num_layers)])


class Siglip2VisionTransformer(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12,
                 intermediate_size: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.embeddings = Siglip2VisionEmbeddings(embed_dim)
        self.encoder = Siglip2Encoder(num_layers, embed_dim = embed_dim, num_heads = num_heads,
                                      intermediate_size = intermediate_size, eps = eps)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = eps)
        # The pooling head in the checkpoint (vision_model.head.*) is unused: Hayai reads last_hidden_state.

    def forward(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                spatial_shapes: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, spatial_shapes)
        # bidirectional key-padding mask
        attention_mask = pixel_attention_mask.bool()[:, None, None, :]
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.post_layernorm(hidden_states)


class Siglip2VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = Siglip2VisionTransformer()

    def forward(self, **kwargs) -> torch.Tensor:
        return self.vision_model(**kwargs)


# =======================================================
# Hayai decoder (upstream modeling_hayai.py)
# =======================================================
@lru_cache(maxsize=None)
def _get_2d_visual_freqs(h: int, w: int, d_axis: int, theta: float, device_str: str):
    device = torch.device(device_str)
    freqs = 1.0 / (theta ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
    grid_y = torch.arange(h, device=device, dtype=torch.float32)
    grid_x = torch.arange(w, device=device, dtype=torch.float32)
    freqs_y = torch.outer(grid_y, freqs)
    freqs_x = torch.outer(grid_x, freqs)
    grid_y_ext = freqs_y.unsqueeze(1).expand(h, w, -1)
    grid_x_ext = freqs_x.unsqueeze(0).expand(h, w, -1)
    vis_freqs = torch.cat([grid_y_ext, grid_x_ext], dim=-1).flatten(0, 1)
    return torch.cos(vis_freqs), torch.sin(vis_freqs)


@lru_cache(maxsize=None)
def _get_1d_text_freqs(n_text: int, d_axis: int, theta: float, device_str: str):
    device = torch.device(device_str)
    freqs = 1.0 / (theta ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
    t_text = torch.arange(n_text, device=device, dtype=torch.float32)
    text_freqs_1d = torch.outer(t_text, freqs)
    text_freqs = torch.cat([text_freqs_1d, text_freqs_1d], dim=-1)
    return torch.cos(text_freqs), torch.sin(text_freqs)


def compute_batch_2d_mrope_freqs(spatial_shapes: torch.Tensor, m_vision: int, n_text: int,
                                 d_head: int = 64, theta: float = 10000.0, device="cuda"):
    b = spatial_shapes.shape[0]
    d_axis = d_head // 2
    total_len = m_vision + n_text

    cos_batch = torch.ones((b, total_len, d_axis), device=device)
    sin_batch = torch.zeros((b, total_len, d_axis), device=device)

    cos_text, sin_text = _get_1d_text_freqs(n_text, d_axis, theta, str(device))
    cos_batch[:, m_vision:, :] = cos_text.unsqueeze(0).expand(b, -1, -1)
    sin_batch[:, m_vision:, :] = sin_text.unsqueeze(0).expand(b, -1, -1)

    for i in range(b):
        h = spatial_shapes[i, 0].item()
        w = spatial_shapes[i, 1].item()
        actual_vis = min(h * w, m_vision)
        if actual_vis <= 0:
            continue
        cos_vis, sin_vis = _get_2d_visual_freqs(h, w, d_axis, theta, str(device))
        cos_batch[i, :actual_vis] = cos_vis[:actual_vis]
        sin_batch[i, :actual_vis] = sin_vis[:actual_vis]

    return cos_batch, sin_batch


def apply_rotary_emb_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    freqs_complex = torch.view_as_complex(torch.stack([cos, sin], dim=-1).float().contiguous())
    out_complex = x_complex * freqs_complex.unsqueeze(2)
    out = torch.view_as_real(out_complex).flatten(-2)
    return out.to(orig_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps)
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        return (x_f32 * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class DSCProjector(nn.Module):
    def __init__(self, d_vision: int = 768, d_model: int = 512, downscale_factor: int = 2):
        super().__init__()
        self.downscale = downscale_factor
        unshuffle_dim = d_vision * (downscale_factor ** 2)
        self.norm = nn.LayerNorm(unshuffle_dim)
        self.mlp = nn.Sequential(
            nn.Linear(unshuffle_dim, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        self.out_norm = RMSNorm(d_model)

    def forward(self, visual_features: torch.Tensor, spatial_shapes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, m_vision, d_vision = visual_features.shape

        compressed_list = []
        new_shapes_list = []

        for i in range(b):
            hp = spatial_shapes[i, 0].item()
            wp = spatial_shapes[i, 1].item()
            valid_len = hp * wp

            feat = visual_features[i:i+1, :valid_len, :].view(1, hp, wp, d_vision).permute(0, 3, 1, 2)

            pad_h = (self.downscale - (hp % self.downscale)) % self.downscale
            pad_w = (self.downscale - (wp % self.downscale)) % self.downscale
            if pad_h > 0 or pad_w > 0:
                feat = F.pad(feat, (0, pad_w, 0, pad_h), mode="replicate")

            feat = F.pixel_unshuffle(feat, downscale_factor=self.downscale)
            _, c_out, h_out, w_out = feat.shape

            tokens = feat.permute(0, 2, 3, 1).reshape(h_out * w_out, c_out)
            compressed_list.append(tokens)
            new_shapes_list.append([h_out, w_out])

        max_len = max(t.size(0) for t in compressed_list)
        batch_tokens = torch.zeros((b, max_len, compressed_list[0].size(-1)), dtype=visual_features.dtype, device=visual_features.device)
        for i, t in enumerate(compressed_list):
            batch_tokens[i, :t.size(0), :] = t

        out = self.out_norm(self.mlp(self.norm(batch_tokens)))
        new_shapes = torch.tensor(new_shapes_list, dtype=torch.long, device=spatial_shapes.device)
        return out, new_shapes


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, h_q: int = 8, h_kv: int = 2, d_head: int = 64):
        super().__init__()
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_head = d_head
        self.num_queries_per_kv = h_q // h_kv
        self.w_q = nn.Linear(d_model, h_q * d_head, bias=False)
        self.w_k = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.w_v = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.w_o = nn.Linear(h_q * d_head, d_model, bias=False)
        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                cos_sin: tuple = None, kv_cache: dict = None,
                layer_idx: int = None, cache_seqlens: int = 0) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.w_q(x).view(b, s, self.h_q, self.d_head)
        k = self.w_k(x).view(b, s, self.h_kv, self.d_head)
        v = self.w_v(x).view(b, s, self.h_kv, self.d_head)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cos_sin is not None:
            cos, sin = cos_sin
            q = apply_rotary_emb_2d(q, cos, sin)
            k = apply_rotary_emb_2d(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache[layer_idx]
            k_cache[:, :, cache_seqlens:cache_seqlens + s] = k.transpose(1, 2)
            v_cache[:, :, cache_seqlens:cache_seqlens + s] = v.transpose(1, 2)
            total_len = cache_seqlens + s
            k = k_cache[:, :, :total_len].contiguous()
            v = v_cache[:, :, :total_len].contiguous()

        q_t = q.transpose(1, 2)

        if kv_cache is None:
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
        else:
            k_t = k
            v_t = v

        sdpa_kwargs = {"attn_mask": mask, "dropout_p": 0.0, "is_causal": False}

        if self.h_kv != self.h_q:
            try:
                context = F.scaled_dot_product_attention(q_t, k_t, v_t, enable_gqa=True, **sdpa_kwargs)
            except TypeError:
                k_t_rep = k_t.repeat_interleave(self.num_queries_per_kv, dim=1)
                v_t_rep = v_t.repeat_interleave(self.num_queries_per_kv, dim=1)
                context = F.scaled_dot_product_attention(q_t, k_t_rep, v_t_rep, **sdpa_kwargs)
        else:
            context = F.scaled_dot_product_attention(q_t, k_t, v_t, **sdpa_kwargs)

        return self.w_o(context.transpose(1, 2).contiguous().view(b, s, -1))

    def step(self, x, cos, sin, k_cache, v_cache, pos, mask):
        '''[shiori] One decode position for a captured graph: this token's key and value go to the
        cache at `pos` (a 1-element device tensor, so one graph serves every position) and the
        query attends over the whole fixed-length cache under `mask`.'''
        b = x.size(0)
        q = self.q_norm(self.w_q(x).view(b, 1, self.h_q, self.d_head))
        k = self.k_norm(self.w_k(x).view(b, 1, self.h_kv, self.d_head))
        v = self.w_v(x).view(b, 1, self.h_kv, self.d_head)
        q = apply_rotary_emb_2d(q, cos, sin)
        k = apply_rotary_emb_2d(k, cos, sin)
        k_cache.index_copy_(2, pos, k.transpose(1, 2).to(k_cache.dtype))
        v_cache.index_copy_(2, pos, v.transpose(1, 2).to(v_cache.dtype))
        context = F.scaled_dot_product_attention(q.transpose(1, 2), k_cache, v_cache, attn_mask=mask,
                                                 dropout_p=0.0, is_causal=False, enable_gqa=True)
        return self.w_o(context.transpose(1, 2).contiguous().view(b, 1, -1))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h_q: int, h_kv: int, d_ffn: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, h_q, h_kv)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)
        self.attn_res_scale = nn.Parameter(torch.ones(d_model))
        self.ffn_res_scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                cos_sin: tuple = None, kv_cache: dict = None,
                layer_idx: int = None, cache_seqlens: int = 0) -> torch.Tensor:
        x = x + self.attn_res_scale * self.attn(
            self.attn_norm(x), mask=mask, cos_sin=cos_sin,
            kv_cache=kv_cache, layer_idx=layer_idx, cache_seqlens=cache_seqlens
        )
        x = x + self.ffn_res_scale * self.ffn(self.ffn_norm(x))
        return x

    def step(self, x, cos, sin, k_cache, v_cache, pos, mask):
        '''[shiori] forward() for one decode position of a captured graph (see attention step).'''
        x = x + self.attn_res_scale * self.attn.step(self.attn_norm(x), cos, sin, k_cache, v_cache, pos, mask)
        x = x + self.ffn_res_scale * self.ffn(self.ffn_norm(x))
        return x


class VisualCausalOCRDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, d_vision: int = 768,
                 d_ffn: int = 2048, n_layers: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.projector = DSCProjector(d_vision, d_model, downscale_factor=2)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h_q=8, h_kv=2, d_ffn=d_ffn)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self._mask_cache = {}

    def generate_block_causal_mask(self, m_vision: int, n_text: int, device) -> torch.Tensor:
        key = (m_vision, n_text, str(device))
        if key not in self._mask_cache:
            total_len = m_vision + n_text
            mask = torch.zeros((total_len, total_len), dtype=torch.float32, device=device)
            mask[:m_vision, m_vision:] = -1e9
            text_causal = torch.triu(torch.ones((n_text, n_text), dtype=torch.float32, device=device), diagonal=1) * -1e9
            mask[m_vision:, m_vision:] = text_causal
            self._mask_cache[key] = mask.unsqueeze(0).unsqueeze(0)
        return self._mask_cache[key]


class _DecodeGraph:
    '''[shiori] Greedy decode steps replayed as one captured CUDA graph each.

    Eagerly a step is ~500 small kernels launched from Python, and on Windows the launches cost
    more than the work, so decoding is launch-bound. Here one step (embed, 12 layers, head,
    argmax, bookkeeping) is captured once per padded batch size and cache length, then
    replayed: one launch per step. Shapes must stay fixed for that, so the cache has a fixed
    length, the position lives on the device, and a mask hides the slots not written yet. The
    step computes what the eager loop computes; attending over the fixed length (masked) instead
    of the written prefix can round differently in the last bits.'''

    GRAPHS_KEPT = 16

    def __init__(self, model, batch, length, dtype, device, max_new_tokens, eos_id, pad_id):
        attn = model.decoder.layers[0].attn
        shape = (batch, attn.h_kv, length, attn.d_head)
        self.model, self.batch, self.length = model, batch, length
        self.eos_id, self.pad_id = eos_id, pad_id
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in model.decoder.layers]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in model.decoder.layers]
        self.tok = torch.zeros(batch, dtype=torch.long, device=device)
        self.out = torch.zeros(batch, dtype=torch.long, device=device)
        self.unfinished = torch.zeros(batch, dtype=torch.bool, device=device)
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.idx = torch.zeros(1, dtype=torch.long, device=device)
        self.mask = torch.zeros((batch, 1, 1, length), dtype=torch.float32, device=device)
        d_axis = 32
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
        t_text_all = torch.arange(max_new_tokens + 1, device=device, dtype=torch.float32)
        text_freqs_all = torch.cat([torch.outer(t_text_all, freqs), torch.outer(t_text_all, freqs)], dim=-1)
        self.cos, self.sin = torch.cos(text_freqs_all), torch.sin(text_freqs_all)
        self.graph = None

    def _step(self):
        dec = self.model.decoder
        # Weight-cast caching is incompatible with capture; the casts become part of the graph.
        with torch.autocast(device_type='cuda', dtype=torch.float16, cache_enabled=False):
            x = dec.token_embeddings(self.tok.unsqueeze(1))
            cos = self.cos.index_select(0, self.idx).expand(self.batch, 1, -1)
            sin = self.sin.index_select(0, self.idx).expand(self.batch, 1, -1)
            self.mask.index_fill_(3, self.pos, 0.0)
            for i, layer in enumerate(dec.layers):
                x = layer.step(x, cos, sin, self.k[i], self.v[i], self.pos, self.mask)
            logits = dec.output_head(dec.final_norm(x))[:, -1, :]
            nxt = torch.argmax(logits, dim=-1) * self.unfinished + self.pad_id * (~self.unfinished)
        self.out.copy_(nxt)
        self.unfinished.copy_(self.unfinished & (nxt != self.eos_id) & (nxt != self.pad_id))
        self.tok.copy_(nxt)
        self.pos.add_(1)
        self.idx.add_(1)

    def start(self, next_tokens, unfinished, valid_vis, m_vision):
        '''Load the state the prefill left (it wrote this graph's cache directly) and capture on
        first use. Rows past the real batch attend one slot so their math stays finite.'''
        b = next_tokens.size(0)
        self.tok.fill_(self.pad_id)
        self.tok[:b] = next_tokens
        self.unfinished.zero_()
        self.unfinished[:b] = unfinished
        self.pos.fill_(m_vision + 1)
        self.idx.fill_(1)
        self.mask.fill_(-1e9)
        self.mask[:, 0, 0, 0] = 0.0
        slots = torch.arange(self.length, device=self.mask.device)
        self.mask[:b, 0, 0, :].masked_fill_(slots[None, :] < valid_vis[:, None], 0.0)
        self.mask[:b, 0, 0, m_vision] = 0.0     # the bos token
        if self.graph is None:
            saved = [t.clone() for t in (self.tok, self.unfinished, self.pos, self.idx, self.mask)]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):          # warm-up before capture, as CUDA graphs require
                for _ in range(2):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            for t, value in zip((self.tok, self.unfinished, self.pos, self.idx, self.mask), saved):
                t.copy_(value)                     # slots the warm-up wrote are rewritten before read
            graph = torch.cuda.CUDAGraph()
            # thread_local: the other GPU lanes keep launching work while this lane captures.
            with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                self._step()
            self.graph = graph

    def decode(self, generated_tokens, max_new_tokens):
        b = generated_tokens.size(0)
        for step in range(1, max_new_tokens):
            if not self.unfinished[:b].any():
                break
            self.graph.replay()
            generated_tokens[:, step + 1] = self.out[:b]


class HayaiModel(nn.Module):
    def __init__(self, vocab_size: int = 16004, d_model: int = 512, d_vision: int = 768,
                 d_ffn: int = 2048, n_layers: int = 12):
        super().__init__()
        self.vision_encoder = Siglip2VisionModel()
        self.decoder = VisualCausalOCRDecoder(vocab_size=vocab_size, d_model=d_model, d_vision=d_vision,
                                              d_ffn=d_ffn, n_layers=n_layers)
        # [shiori] CUDA-graph decoding (_DecodeGraph); MT_HAYAI_GRAPHS=0 turns it off.
        self.graphed_decode = os.environ.get('MT_HAYAI_GRAPHS', '1') != '0'
        self._graphs = OrderedDict()

    def _decode_graph(self, b, m_vision, max_new_tokens, dtype, device, eos_id, pad_id):
        '''The graph for this batch, padded to a power of two, and cache length, bucketed by 64.'''
        batch = 1 << (b - 1).bit_length()
        length = -(-(m_vision + 1) // 64) * 64 + max_new_tokens
        key = (batch, length, dtype, max_new_tokens, eos_id, pad_id)
        graph = self._graphs.get(key)
        if graph is None:
            graph = self._graphs[key] = _DecodeGraph(self, batch, length, dtype, device, max_new_tokens,
                                                     eos_id, pad_id)
            while len(self._graphs) > _DecodeGraph.GRAPHS_KEPT:
                self._graphs.popitem(last=False)
        self._graphs.move_to_end(key)
        return graph

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                 spatial_shapes: torch.Tensor, bos_id: int, eos_id: int, pad_id: int,
                 max_new_tokens: int = 128) -> List[List[int]]:
        '''Greedy decode; returns token ids per image with bos/eos/pad removed.'''
        device = pixel_values.device
        b = pixel_values.size(0)

        amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            vision_hidden = self.vision_encoder(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes
            )
            vision_embeddings, new_shapes = self.decoder.projector(vision_hidden, spatial_shapes)
            m_vision = vision_embeddings.size(1)

            bos_tokens = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
            bos_embeddings = self.decoder.token_embeddings(bos_tokens)
            x = torch.cat([vision_embeddings, bos_embeddings], dim=1)

            mask = self.decoder.generate_block_causal_mask(m_vision, 1, device)
            cos_batch, sin_batch = compute_batch_2d_mrope_freqs(new_shapes, m_vision, 1, d_head=64, device=device)

            # [shiori] mask the zero-padded vision tokens of shorter images in the batch
            valid_vis = (new_shapes[:, 0] * new_shapes[:, 1]).to(device)
            max_seq_len = m_vision + max_new_tokens + 1
            pad_positions = torch.arange(m_vision, device=device)[None, :] >= valid_vis[:, None]
            if bool(pad_positions.any()):
                key_pad = torch.zeros((b, 1, 1, max_seq_len), dtype=torch.float32, device=device)
                key_pad[:, 0, 0, :m_vision] = pad_positions.float() * -1e9
                prefill_mask = mask + key_pad[:, :, :, :m_vision + 1]
                step_pad = key_pad
            else:
                prefill_mask = mask
                step_pad = None

            d_axis = 32
            freqs = 1.0 / (10000.0 ** (torch.arange(0, d_axis, 2, device=device).float() / d_axis))
            t_text_all = torch.arange(max_new_tokens + 1, device=device, dtype=torch.float32)
            text_freqs_all = torch.cat([torch.outer(t_text_all, freqs), torch.outer(t_text_all, freqs)], dim=-1)
            cos_text_all = torch.cos(text_freqs_all)
            sin_text_all = torch.sin(text_freqs_all)

            graph = None
            if self.graphed_decode and device.type == 'cuda':
                graph = self._decode_graph(b, m_vision, max_new_tokens, x.dtype, device, eos_id, pad_id)
            kv_cache = {}
            for i, layer in enumerate(self.decoder.layers):
                if graph is not None:   # [shiori] the prefill fills the graph's own cache
                    kv_cache[i] = (graph.k[i][:b, :, :max_seq_len], graph.v[i][:b, :, :max_seq_len])
                    continue
                k_cache = torch.zeros((b, layer.attn.h_kv, max_seq_len, layer.attn.d_head), dtype=x.dtype, device=device)
                v_cache = torch.zeros((b, layer.attn.h_kv, max_seq_len, layer.attn.d_head), dtype=x.dtype, device=device)
                kv_cache[i] = (k_cache, v_cache)

            cache_seqlens = 0
            for i, layer in enumerate(self.decoder.layers):
                x = layer(x, mask=prefill_mask, cos_sin=(cos_batch, sin_batch), kv_cache=kv_cache, layer_idx=i, cache_seqlens=cache_seqlens)
            cache_seqlens += x.size(1)

            x_bos = self.decoder.final_norm(x[:, -1:])
            logits = self.decoder.output_head(x_bos)[:, -1, :]

            next_tokens = torch.argmax(logits, dim=-1)
            generated_tokens = torch.full((b, max_new_tokens + 1), pad_id, dtype=torch.long, device=device)
            generated_tokens[:, 0] = bos_id
            generated_tokens[:, 1] = next_tokens
            unfinished = (next_tokens != eos_id) & (next_tokens != pad_id)

            if graph is not None:
                try:
                    graph.start(next_tokens, unfinished, valid_vis, m_vision)
                except Exception as e:  # capture unsupported here: decode eagerly from now on
                    warnings.warn(f'Hayai CUDA-graph decoding disabled ({type(e).__name__}: {e})')
                    self.graphed_decode = False
                    self._graphs.clear()
                    graph = None
            if graph is not None:
                graph.decode(generated_tokens, max_new_tokens)
            else:
                self._decode_eager(next_tokens, unfinished, generated_tokens, kv_cache, cache_seqlens,
                                   step_pad, cos_text_all, sin_text_all, eos_id, pad_id, max_new_tokens)

        return [[t for t in seq.tolist()[1:] if t not in (eos_id, pad_id)] for seq in generated_tokens]

    def _decode_eager(self, next_tokens, unfinished, generated_tokens, kv_cache, cache_seqlens, step_pad,
                      cos_text_all, sin_text_all, eos_id, pad_id, max_new_tokens):
        '''The upstream decode loop, unchanged, for when no CUDA graph is used.'''
        b = next_tokens.size(0)
        device = next_tokens.device
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32,
                            enabled=(device.type == "cuda")):
            for step in range(1, max_new_tokens):
                if not unfinished.any(): break
                x_step = self.decoder.token_embeddings(next_tokens.unsqueeze(1))
                cos_step = cos_text_all[step].unsqueeze(0).expand(b, 1, -1)
                sin_step = sin_text_all[step].unsqueeze(0).expand(b, 1, -1)

                step_mask = step_pad[:, :, :, :cache_seqlens + 1] if step_pad is not None else None
                for i, layer in enumerate(self.decoder.layers):
                    x_step = layer(x_step, mask=step_mask, cos_sin=(cos_step, sin_step), kv_cache=kv_cache, layer_idx=i, cache_seqlens=cache_seqlens)
                cache_seqlens += 1

                x_step = self.decoder.final_norm(x_step)
                logits_step = self.decoder.output_head(x_step)[:, -1, :]
                next_tokens = torch.argmax(logits_step, dim=-1) * unfinished + pad_id * (~unfinished)
                generated_tokens[:, step+1] = next_tokens
                unfinished = unfinished & (next_tokens != eos_id) & (next_tokens != pad_id)
