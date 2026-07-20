import asyncio
import cv2
import json
import langcodes
import os
import regex as re
import time
import torch
import logging
import sys
import traceback
import numpy as np
from PIL import Image
from typing import Optional, Any, List
import py3langid as langid

from .config import Config, Colorizer, Detector, Translator, Renderer, Inpainter
from .utils import (
    BASE_PATH,
    LANGUAGE_ORIENTATION_PRESETS,
    ModelWrapper,
    Context,
    load_image,
    dump_image,
    visualize_textblocks,
    is_valuable_text,
    sort_regions,
)

from .detection import dispatch as dispatch_detection, prepare as prepare_detection, unload as unload_detection
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling, unload as unload_upscaling
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr, unload as unload_ocr
from .ocr.colors import apply_estimated_colors
from .textline_merge import dispatch as dispatch_textline_merge
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting, unload as unload_inpainting
from .translators import (
    dispatch as dispatch_translation,
    prepare as prepare_translation,
    unload as unload_translation,
)
from .translators.common import ISO_639_1_TO_VALID_LANGUAGES
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization, unload as unload_colorization
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render, dispatch_eng_render_pillow, render_bubble_debug
from .rendering.shiori_render import dispatch_shiori_render, dispatch_shiori_render_v2
from .utils.executors import run_cpu, run_proc, submit_gpu, prewarm_proc_pool
from .utils.profiling import Profiler

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')

# 全局console实例，用于日志重定向
_global_console = None
_log_console = None

def set_main_logger(l):
    global logger
    logger = l

class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass

def load_dictionary(file_path):
    dictionary = []
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                # Ignore empty lines and lines starting with '#' or '//'
                if not line.strip() or line.strip().startswith('#') or line.strip().startswith('//'):
                    continue
                # Remove comment parts
                line = line.split('#')[0].strip()
                line = line.split('//')[0].strip()
                parts = line.split()
                if len(parts) == 1:
                    # If there is only the left part, the right part defaults to an empty string, meaning delete the left part
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, '', line_number))
                elif len(parts) == 2:
                    # If both left and right parts are present, perform the replacement
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, parts[1], line_number))
                else:
                    logger.error(f'Invalid dictionary entry at line {line_number}: {line.strip()}')
    return dictionary

def apply_dictionary(text, dictionary):
    for pattern, value, line_number in dictionary:
        original_text = text
        text = pattern.sub(value, text)
        if text != original_text:
            logger.info(f'Line {line_number}: Replaced "{original_text}" with "{text}" using pattern "{pattern.pattern}" and value "{value}"')
    return text


# ── study-layer builders (module-level so they run in the GIL-free process pool) ──────────
# text_and_image study mode partitions the final render's changed pixels among a page's bubbles
# and encodes a full-page transparent layer per bubble + the shared inpaint bg: per-bubble numpy
# fills + PNG/WebP encodes + base64. On the thread pool that Python-heavy work holds the GIL and
# serializes the whole pipeline (study_overlay measured ~40% of a dense gallery's wall, with CPU
# cores idle — the GIL-serialization tell). Running it out-of-process (like mask refinement) frees
# the GIL. These are module-level, not closures/methods, so ProcessPoolExecutor can pickle them.

def _study_norm(x1, y1, x2, y2, W, H):
    return {'x': x1 / W, 'y': y1 / H, 'w': (x2 - x1) / W, 'h': (y2 - y1) / H}


def _study_meta_bubble(info, region_xyxy, W, H):
    """Per-bubble geometry + text + style, normalized to the page. Reads no pixels."""
    dx1, dy1, dx2, dy2 = info['det']
    bx1, by1, bx2, by2 = info['rbox']
    bubble = {
        'box': _study_norm(dx1, dy1, dx2, dy2, W, H),
        'rbox': _study_norm(bx1, by1, bx2, by2, W, H),
        'region': _study_norm(*region_xyxy, W, H),
        'tr': info['tr'], 'src': info['src'], 'style': info['style'],
    }
    # Optional DOM-text metadata: per-source-line ruby segments and the rect where the renderer
    # pasted its glyph canvas.
    for k in ('furi',):
        if info.get(k):
            bubble[k] = info[k]
    if info.get('tbox'):
        bubble['tbox'] = _study_norm(*info['tbox'], W, H)
    return bubble


# ── furigana (ruby) segmentation for study-mode source text ───────────────────────────────
# pykakasi splits Japanese text into words with kana readings; each source line becomes a list of
# [text, reading|None] segments where a reading covers only the kanji run (shared kana at the
# word's edges — okurigana/prefixes — are trimmed out of the ruby). The reader decides whether
# to SHOW furigana (it gates on the gallery's language), so this only skips work when the text
# plainly has no kanji or pykakasi isn't installed.

_KAKASI = None


def _has_kanji(s):
    # CJK unified ideographs (incl. ext. A) + compatibility ideographs + iteration marks.
    return any('㐀' <= c <= '鿿' or '豈' <= c <= '﫿' or c in '々〆' for c in s)


def _furi_seg_append(segs, text, ruby):
    """Append a segment, merging consecutive no-ruby runs to keep the payload compact."""
    if ruby is None and segs and segs[-1][1] is None:
        segs[-1][0] += text
    else:
        segs.append([text, ruby])


def _furi_lines(lines):
    """Per-line ruby segments for `lines`, or None when nothing gets a reading."""
    global _KAKASI
    if not any(_has_kanji(line) for line in lines):
        return None
    if _KAKASI is None:
        try:
            import pykakasi
            _KAKASI = pykakasi.kakasi()
        except Exception:
            _KAKASI = False
    if _KAKASI is False:
        return None
    out = []
    any_ruby = False
    for line in lines:
        segs = []
        try:
            items = _KAKASI.convert(line)
        except Exception:
            items = []
        if not items:
            segs.append([line, None])
            out.append(segs)
            continue
        for item in items:
            orig = item.get('orig') or ''
            hira = item.get('hira') or ''
            if not orig:
                continue
            if not hira or not _has_kanji(orig):
                _furi_seg_append(segs, orig, None)
                continue
            # Trim kana the word shares with its reading at both ends so the ruby sits over
            # the kanji only (お預け/おあずけ → お + 預け⟨あず⟩… etc.).
            p = 0
            while p < len(orig) and p < len(hira) and orig[p] == hira[p]:
                p += 1
            s = 0
            while s < len(orig) - p and s < len(hira) - p and orig[len(orig) - 1 - s] == hira[len(hira) - 1 - s]:
                s += 1
            core_o, core_h = orig[p:len(orig) - s], hira[p:len(hira) - s]
            if not core_o or not core_h:
                _furi_seg_append(segs, orig, None)
                continue
            if p:
                _furi_seg_append(segs, orig[:p], None)
            _furi_seg_append(segs, core_o, core_h)
            any_ruby = True
            if s:
                _furi_seg_append(segs, orig[len(orig) - s:], None)
        out.append(segs)
    return out if any_ruby else None


def _study_img_data_url(arr, mode_, fmt='PNG', **save_kw):
    import io as _io
    import base64
    buf = _io.BytesIO()
    Image.fromarray(arr, mode_).save(buf, format=fmt, **save_kw)
    mime = 'image/webp' if fmt == 'WEBP' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _build_page_layers_job(rendered, inpainted, infos, W, H):
    """CPU-bound study layers for one page — the process-pool job. EXACTNESS INVARIANT
    (unchanged): every pixel where the final render differs from the inpaint is assigned to
    exactly ONE bubble, whose layer stores the final render's RGB at full alpha, so reassembling
    all bubble layers over the inpaint reproduces the final page pixel-for-pixel. Ownership is
    per pixel: the render box that contains it (nearest box center on overlap), else the nearest
    render box — so adjacent bubbles split cleanly at their boundary."""
    diff = np.abs(rendered.astype(np.int16) - inpainted.astype(np.int16)).max(axis=2)
    changed = diff > 0
    # Defensive: a renderer that perturbs untouched pixels (full-page roundtrip) would mark
    # everything changed; fall back to the visible-change threshold in that case.
    if changed.mean() > 0.35:
        changed = diff > 8
    ys, xs = np.nonzero(changed)
    if xs.size == 0:
        return None

    xf = xs.astype(np.float32)
    yf = ys.astype(np.float32)
    best_d = np.full(xs.shape, np.inf, dtype=np.float32)
    owner = np.zeros(xs.shape, dtype=np.int32)
    for ri, info in enumerate(infos):
        bx1, by1, bx2, by2 = info['rbox']
        # squared rect-distance to the render box (0 inside) + a tiny center-distance term that
        # deterministically breaks ties between overlapping boxes.
        dx = np.maximum(np.maximum(bx1 - xf, 0.0), xf - (bx2 - 1))
        dy = np.maximum(np.maximum(by1 - yf, 0.0), yf - (by2 - 1))
        d = dx * dx + dy * dy
        cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        d += ((xf - cx) ** 2 + (yf - cy) ** 2) * np.float32(1e-7)
        m = d < best_d
        best_d[m] = d[m]
        owner[m] = ri

    bubbles = []
    for ri, info in enumerate(infos):
        sel = owner == ri
        if not sel.any():
            continue
        rx, ry = xs[sel], ys[sel]
        gx1, gy1 = int(rx.min()), int(ry.min())
        gx2, gy2 = int(rx.max()) + 1, int(ry.max()) + 1
        # Full-page transparent layer holding exactly this bubble's pixels: RGB from the final
        # render (antialiasing against the inpaint already baked in), alpha 255 so compositing
        # over the inpaint bg reproduces the render exactly.
        text_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        text_rgba[ry, rx, :3] = rendered[ry, rx]
        text_rgba[ry, rx, 3] = 255
        # bg clip region = detection box unioned with the full glyph extent.
        dx1, dy1, dx2, dy2 = info['det']
        bubble = _study_meta_bubble(info, (min(dx1, gx1), min(dy1, gy1), max(dx2, gx2), max(dy2, gy2)), W, H)
        bubble['text'] = _study_img_data_url(text_rgba, 'RGBA', 'PNG')
        bubbles.append(bubble)
    if not bubbles:
        return None
    # Shared inpaint bg is opaque art (no text) → WebP is far smaller than PNG at quality the eye
    # can't tell apart, and it only ever shows behind a revealed bubble.
    bg = _study_img_data_url(inpainted, 'RGB', 'WEBP', quality=90, method=6)
    return {'page': {'w': W, 'h': H}, 'bg': bg, 'bubbles': bubbles}

class MangaTranslator:
    verbose: bool
    ignore_errors: bool
    _gpu_limited_memory: bool
    device: Optional[str]
    kernel_size: Optional[int]
    models_ttl: int
    _progress_hooks: list[Any]
    result_sub_folder: str
    batch_size: int

    def __init__(self, params: dict = None):
        self.pre_dict = params.get('pre_dict', None)
        self.post_dict = params.get('post_dict', None)
        self.font_path = None
        self.use_mtpe = False
        self.kernel_size = None
        self.device = None
        self._gpu_limited_memory = False
        self.ignore_errors = False
        self.verbose = False
        self.models_ttl = 0
        self.batch_size = 1  # 默认不批量处理

        self._progress_hooks = []
        self._page_result_hooks = []
        self._page_bubbles_hooks = []
        self._add_logger_hook()

        params = params or {}
        
        self._batch_contexts = []  # 存储批量处理的上下文
        self._batch_configs = []   # 存储批量处理的配置
        self.disable_memory_optimization = params.get('disable_memory_optimization', False)
        # batch_concurrent 会在 parse_init_params 中验证并设置
        self.batch_concurrent = params.get('batch_concurrent', False)
        
        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        self._model_usage_timestamps = {}
        self._stage_times = {}  # per-stage wall-clock accumulator (s); reset per gallery run
        self._detector_cleanup_task = None
        self.prep_manual = params.get('prep_manual', None)
        self.context_size = params.get('context_size', 0)
        self.all_page_translations = []
        self._original_page_texts = []  # 存储原文页面数据，用于并发模式下的上下文

        # 调试图片管理相关属性
        self._current_image_context = None  # 存储当前处理图片的上下文信息
        self._saved_image_contexts = {}     # 存储批量处理中每个图片的上下文信息
        
        # 设置日志文件
        self._setup_log_file()

    def _setup_log_file(self):
        """设置日志文件，在result文件夹下创建带时间戳的log文件"""
        try:
            # 创建result目录
            result_dir = os.path.join(BASE_PATH, 'result')
            os.makedirs(result_dir, exist_ok=True)
            
            # 生成带时间戳的日志文件名
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            log_filename = f"log_{timestamp}.txt"
            log_path = os.path.join(result_dir, log_filename)
            
            # 配置文件日志处理器
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            # 使用自定义格式器，保持与控制台输出一致
            from .utils.log import Formatter
            formatter = Formatter()
            file_handler.setFormatter(formatter)
            
            # 添加到manga-translator根logger以捕获所有输出
            mt_logger = logging.getLogger('manga-translator')
            mt_logger.addHandler(file_handler)
            if not mt_logger.level or mt_logger.level > logging.DEBUG:
                mt_logger.setLevel(logging.DEBUG)
            
            # 保存日志文件路径供后续使用
            self._log_file_path = log_path
            
            # 简单的print重定向
            import builtins
            original_print = builtins.print
            
            def log_print(*args, **kwargs):
                # 正常打印到控制台
                original_print(*args, **kwargs)
                # 同时写入日志文件
                try:
                    import io
                    buffer = io.StringIO()
                    original_print(*args, file=buffer, **kwargs)
                    output = buffer.getvalue()
                    if output.strip():
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(output)
                except Exception:
                    pass
            
            builtins.print = log_print
            
            # Rich Console输出重定向
            try:
                from rich.console import Console
                import sys
                
                # 创建一个自定义的文件对象，同时写入控制台和日志文件
                class TeeFile:
                    def __init__(self, log_file_path, original_file):
                        self.log_file_path = log_file_path
                        self.original_file = original_file
                    
                    def write(self, text):
                        # 写入原始输出
                        self.original_file.write(text)
                        # 写入日志文件
                        try:
                            if text.strip():
                                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                                    f.write(text)
                        except Exception:
                            pass
                        return len(text)
                    
                    def flush(self):
                        self.original_file.flush()
                    
                    def __getattr__(self, name):
                        return getattr(self.original_file, name)
                
                # 创建一个仅用于日志记录的Console（无颜色、无样式）
                class LogOnlyFile:
                    def __init__(self, log_file_path):
                        self.log_file_path = log_file_path
                    
                    def write(self, text):
                        try:
                            if text.strip():
                                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                                    f.write(text)
                        except Exception:
                            pass
                        return len(text)
                    
                    def flush(self):
                        pass
                    
                    def isatty(self):
                        return False
                
                # 为日志创建纯文本console
                log_file_only = LogOnlyFile(log_path)
                log_console = Console(file=log_file_only, force_terminal=False, no_color=True, width=80)
                
                # 创建带颜色的控制台console
                display_console = Console(force_terminal=True)
                
                # 全局设置console实例，供translator使用
                global _global_console, _log_console
                _global_console = display_console  # 控制台显示用
                _log_console = log_console         # 日志记录用
                
            except Exception as e:
                logger.debug(f"Failed to setup rich console logging: {e}")
            
            logger.info(f"Log file created: {log_path}")
        except Exception as e:
            print(f"Failed to setup log file: {e}")

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.use_mtpe = params.get('use_mtpe', False)
        self.font_path = params.get('font_path', None)
        self.models_ttl = params.get('models_ttl', 0)
        self.batch_size = params.get('batch_size', 1)  # 添加批量大小参数
        
        # 验证batch_concurrent参数
        if self.batch_concurrent and self.batch_size < 2:
            logger.warning('--batch-concurrent requires --batch-size to be at least 2. When batch_size is 1, concurrent mode has no effect.')
            logger.info('Suggestion: Use --batch-size 2 (or higher) with --batch-concurrent, or remove --batch-concurrent flag.')
            # 自动禁用并发模式
            self.batch_concurrent = False
            
        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device if params.get('use_gpu', False) else 'cpu'
        self._gpu_limited_memory = params.get('use_gpu_limited', False)
        if self._gpu_limited_memory and not self.using_gpu:
            self.device = device
        if self.using_gpu and ( not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            raise Exception(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu args was set.\n'
                'Is the correct pytorch version installed? (See https://pytorch.org/)')
        if params.get('model_dir'):
            ModelWrapper._MODEL_DIR = params.get('model_dir')
        #todo: fix why is kernel size loaded in the constructor
        self.kernel_size=int(params.get('kernel_size'))
        # Set input files
        self.input_files = params.get('input', [])
        # Set save_text
        self.save_text = params.get('save_text', False)
        # Set load_text
        self.load_text = params.get('load_text', False)
        
        # batch_concurrent 已在初始化时设置并验证
        

        
    def _set_image_context(self, config: Config, image=None):
        """设置当前处理图片的上下文信息，用于生成调试图片子文件夹"""
        from .utils.generic import get_image_md5

        # 使用毫秒级时间戳确保唯一性
        timestamp = str(int(time.time() * 1000))
        detection_size = str(getattr(config.detector, 'detection_size', 1024))
        target_lang = getattr(config.translator, 'target_lang', 'unknown')
        translator = getattr(config.translator, 'translator', 'unknown')

        # 计算图片MD5哈希值
        if image is not None:
            file_md5 = get_image_md5(image)
        else:
            file_md5 = "unknown"

        # 生成子文件夹名：{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}
        subfolder_name = f"{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}"

        self._current_image_context = {
            'subfolder': subfolder_name,
            'file_md5': file_md5,
            'config': config
        }
        
    def _get_image_subfolder(self) -> str:
        """获取当前图片的调试子文件夹名"""
        if self._current_image_context:
            return self._current_image_context['subfolder']
        return ''
    
    def _save_current_image_context(self, image_md5: str):
        """保存当前图片上下文，用于批量处理中保持一致性"""
        if self._current_image_context:
            self._saved_image_contexts[image_md5] = self._current_image_context.copy()

    def _restore_image_context(self, image_md5: str):
        """恢复保存的图片上下文"""
        if image_md5 in self._saved_image_contexts:
            self._current_image_context = self._saved_image_contexts[image_md5].copy()
            return True
        return False

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def translate(self, image: Image.Image, config: Config, image_name: str = None, skip_context_save: bool = False) -> Context:
        """
        Translates a single image.

        :param image: Input image.
        :param config: Translation config.
        :param image_name: Deprecated parameter, kept for compatibility.
        :return: Translation context.
        """
        await self._report_progress('running_pre_translation_hooks')
        for hook in self._progress_hooks:
            try:
                hook('running_pre_translation_hooks', False)
            except Exception as e:
                logger.error(f"Error in progress hook: {e}")

        ctx = Context()
        ctx.input = image
        ctx.result = None
        ctx.verbose = self.verbose

        # 设置图片上下文以生成调试图片子文件夹
        self._set_image_context(config, image)
        # Pin this page's context onto the ctx so every debug dump resolves its folder from the
        # ctx even if another page mutates self._current_image_context mid-flight.
        ctx.image_context = dict(self._current_image_context)

        # 保存debug文件夹信息到Context中（用于Web模式的缓存访问）
        # 在web模式下总是保存，不仅仅是verbose模式
        ctx.debug_folder = self._get_image_subfolder()

        # 保存原始输入图片用于调试
        if self.verbose:
            try:
                input_img = np.array(image)
                if len(input_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                result_path = self._result_path('input.png', ctx)
                success = cv2.imwrite(result_path, input_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {result_path}")
            except Exception as e:
                logger.error(f"Error saving input.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if ( self.models_ttl == 0 ):
            logger.info('Loading models')
            if config.upscale.upscale_ratio:
                await prepare_upscaling(config.upscale.upscaler)
            await prepare_detection(config.detector.detector)
            await prepare_ocr(config.ocr.ocr, self.device)
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            await prepare_translation(config.translator.translator_gen)
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)

        # translate
        ctx = await self._translate(config, ctx)

        # 在翻译流程的最后保存翻译结果，确保保存的是最终结果（包括重试后的结果）
        # Save translation results at the end of translation process to ensure final results are saved
        if not skip_context_save and ctx.text_regions:
            # 汇总本页翻译，供下一页做上文
            page_translations = {r.text_raw if hasattr(r, "text_raw") else r.text: r.translation
                                 for r in ctx.text_regions}
            self.all_page_translations.append(page_translations)

            # 同时保存原文用于并发模式的上下文
            page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                  for i, r in enumerate(ctx.text_regions)}
            self._original_page_texts.append(page_original_texts)

        return ctx

    async def _translate(self, config: Config, ctx: Context) -> Context:
        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())
        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input  # Fallback to input image if colorization fails

        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized # Fallback to colorized (or input) image if upscaling fails
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:  
            logger.error(f"Error during detection:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] 
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
            cv2.imwrite(self._result_path('mask_raw.png', ctx), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png', ctx), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        # -- OCR
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as e:  
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] # Fallback to empty textlines if OCR fails

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:  
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = [] # Fallback to empty text_regions if textline merge fails

        if self.verbose and ctx.text_regions:
            show_panels = not config.force_simple_sort  # 当不使用简单排序时显示panel
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions, 
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl)
            cv2.imwrite(self._result_path('bboxes.png', ctx), bboxes)

        # Apply pre-dictionary after textline merge
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text  
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")
            
        # -- Translation
        await self._report_progress('translating')
        try:
            ctx.text_regions = await self._run_text_translation(config, ctx)
        except Exception as e:  
            logger.error(f"Error during translating:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = [] # Fallback to empty text_regions if translation fails

        await self._report_progress('after-translating')

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Mask refinement
        # (Delayed to take advantage of the region filtering done after ocr and translation)
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:  
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0] # Fallback to raw mask or empty mask

        if self.verbose and ctx.mask is not None:
            inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                          self.device, self.verbose)
            cv2.imwrite(self._result_path('inpaint_input.png', ctx), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png', ctx), ctx.mask)

        # -- Inpainting
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)
        except Exception as e:  
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb
        if getattr(ctx, '_gallery', False):
            # Only the GIMP save format reads gimp_mask; the gallery path never does, and the
            # full-page dstack+cvtColor per page is pure loop-blocking waste there.
            ctx.gimp_mask = None
        else:
            ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png', ctx)
                success = cv2.imwrite(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))
                if not success:
                    logger.warning(f"Failed to save debug image: {inpainted_path}")
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
        # -- Rendering
        await self._report_progress('rendering')

        # 在rendering状态后立即发送文件夹信息，用于前端精确检查final.png
        ic = getattr(ctx, 'image_context', None) or self._current_image_context
        if hasattr(self, '_progress_hooks') and ic:
            # 发送特殊格式的消息，前端可以解析
            await self._report_progress(f"rendering_folder:{ic['subfolder']}")

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted # Fallback to inpainted (or original RGB) image if rendering fails

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)

        return await self._revert_upscale(config, ctx)
    
    # If `revert_upscaling` is True, revert to input size
    # Else leave `ctx` as-is
    async def _revert_upscale(self, config: Config, ctx: Context):
        if config.upscale.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)

        # 在verbose模式下保存final.png到调试文件夹
        if ctx.result and self.verbose:
            try:
                final_img = np.array(ctx.result)
                if len(final_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
                final_path = self._result_path('final.png', ctx)
                success = cv2.imwrite(final_path, final_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {final_path}")
            except Exception as e:
                logger.error(f"Error saving final.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # Web流式模式优化：保存final.png并使用占位符
        if ctx.result and not self.result_sub_folder and hasattr(self, '_is_streaming_mode') and self._is_streaming_mode:
            # 保存final.png文件
            final_img = np.array(ctx.result)
            if len(final_img.shape) == 3:  # 彩色图片，转换BGR顺序
                final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self._result_path('final.png', ctx), final_img)

            # 通知前端文件已就绪
            ic = getattr(ctx, 'image_context', None) or self._current_image_context
            if hasattr(self, '_progress_hooks') and ic:
                await self._report_progress(f"final_ready:{ic['subfolder']}")

            # 创建占位符结果并立即返回
            from PIL import Image
            placeholder = Image.new('RGB', (1, 1), color='white')
            ctx.result = placeholder
            ctx.use_placeholder = True
            return ctx

        return ctx

    async def _run_colorizer(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("colorizer", config.colorizer.colorizer)] = current_time
        #todo: im pretty sure the ctx is never used. does it need to be passed in?
        return await dispatch_colorization(
            config.colorizer.colorizer,
            colorization_size=config.colorizer.colorization_size,
            denoise_sigma=config.colorizer.denoise_sigma,
            device=self.device,
            image=ctx.input,
            **ctx
        )

    async def _run_upscaling(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("upscaling", config.upscale.upscaler)] = current_time
        return (await dispatch_upscaling(config.upscale.upscaler, [ctx.img_colorized], config.upscale.upscale_ratio, self.device))[0]

    def _accum_time(self, stage: str, dt: float):
        """Accumulate per-stage wall-clock seconds for the current gallery run (Step-0 profiling)."""
        self._stage_times[stage] = self._stage_times.get(stage, 0.0) + dt

    async def _run_detection(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("detection", config.detector.detector)] = current_time
        t0 = time.perf_counter()
        result = await dispatch_detection(config.detector.detector, ctx.img_rgb, config.detector.detection_size, config.detector.text_threshold,
                                        config.detector.box_threshold,
                                        config.detector.unclip_ratio, config.detector.det_invert, config.detector.det_gamma_correct, config.detector.det_rotate,
                                        config.detector.det_auto_rotate,
                                        self.device, self.verbose)
        self._accum_time('detection', time.perf_counter() - t0)
        return result

    async def _unload_model(self, tool: str, model: str):
        logger.info(f"Unloading {tool} model: {model}")
        match tool:
            case 'colorization':
                await unload_colorization(model)
            case 'detection':
                await unload_detection(model)
            case 'inpainting':
                await unload_inpainting(model)
            case 'ocr':
                await unload_ocr(model)
            case 'upscaling':
                await unload_upscaling(model)
            case 'translation':
                await unload_translation(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # empty CUDA cache

    # Background models cleanup job.
    async def _detector_cleanup_job(self):
        while True:
            if self.models_ttl == 0:
                await asyncio.sleep(1)
                continue
            now = time.time()
            for (tool, model), last_used in list(self._model_usage_timestamps.items()):
                if now - last_used > self.models_ttl:
                    await self._unload_model(tool, model)
                    del self._model_usage_timestamps[(tool, model)]
            await asyncio.sleep(1)

    async def _run_ocr(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("ocr", config.ocr.ocr)] = current_time
        
        # 为OCR创建子文件夹（只在verbose模式下）
        # The line-crop folder is resolved from this page's own ctx and handed down the dispatch
        # chain explicitly — the old MANGA_OCR_RESULT_DIR env-var handoff was process-global and
        # raced when several pages OCR'd concurrently in the gallery pipeline (crops from all
        # pages clobbered each other in one flat folder).
        if self.verbose:
            ocr_result_dir = self._result_path('ocrs', ctx)
            os.makedirs(ocr_result_dir, exist_ok=True)
        else:
            ocr_result_dir = None

        t0 = time.perf_counter()
        textlines = await dispatch_ocr(config.ocr.ocr, ctx.img_rgb, ctx.textlines, config.ocr, self.device, self.verbose,
                                       result_dir=ocr_result_dir)
        self._accum_time('ocr', time.perf_counter() - t0)

        if config.render.estimate_font_color or config.render.estimate_outline_color:
            try:
                apply_estimated_colors(ctx.img_rgb, [t for t in textlines if t.text.strip()],
                                       config.render.estimate_font_color, config.render.estimate_outline_color,
                                       text_mask=ctx.mask_raw)
            except Exception:
                logger.warning('Post-OCR color estimation failed', exc_info=True)

        new_textlines = []
        for textline in textlines:
            if textline.text.strip():
                if config.render.font_color_fg:
                    textline.fg_r, textline.fg_g, textline.fg_b = config.render.font_color_fg
                if config.render.font_color_bg:
                    textline.bg_r, textline.bg_g, textline.bg_b = config.render.font_color_bg
                new_textlines.append(textline)
        return new_textlines

    async def _run_textline_merge(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("textline_merge", "textline_merge")] = current_time
        t0 = time.perf_counter()
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],
                                                     verbose=self.verbose)
        self._accum_time('textline_merge', time.perf_counter() - t0)
        for region in text_regions:
            if not hasattr(region, "text_raw"):
                region.text_raw = region.text      # <- Save the initial OCR results to expand the render detection box. Also, prevent affecting the forbidden translation function.       
        # Filter out languages to skip  
        if config.translator.skip_lang is not None:  
            skip_langs = [lang.strip().upper() for lang in config.translator.skip_lang.split(',')]  
            filtered_textlines = []  
            for txtln in ctx.textlines:  
                try:  
                    detected_lang, confidence = langid.classify(txtln.text)
                    source_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
                    if source_language != 'UNKNOWN':
                        source_language = source_language.upper()
                except Exception:  
                    source_language = 'UNKNOWN'  
    
                # Print detected source_language and whether it's in skip_langs  
                # logger.info(f'Detected source language: {source_language}, in skip_langs: {source_language in skip_langs}, text: "{txtln.text}"')  
    
                if source_language in skip_langs:  
                    logger.info(f'Filtered out: {txtln.text}')  
                    logger.info(f'Reason: Detected language {source_language} is in skip_langs')  
                    continue  # Skip this region  
                filtered_textlines.append(txtln)  
            ctx.textlines = filtered_textlines  
    
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],  
                                                     verbose=self.verbose)  

        new_text_regions = []
        for region in text_regions:
            # Remove leading spaces after pre-translation dictionary replacement                
            original_text = region.text  
            stripped_text = original_text.strip()  
            
            # Record removed leading characters  
            removed_start_chars = original_text[:len(original_text) - len(stripped_text)]  
            if removed_start_chars:  
                logger.info(f'Removed leading characters: "{removed_start_chars}" from "{original_text}"')  
            
            # Modified filtering condition: handle incomplete parentheses  
            bracket_pairs = {  
                '(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '〔': '〕', '〈': '〉', '「': '」',  
                '"': '"', '＂': '＂', "'": "'", "“": "”", '《': '》', '『': '』', '"': '"', '〝': '〞', '﹁': '﹂', '﹃': '﹄',  
                '⸂': '⸃', '⸄': '⸅', '⸉': '⸊', '⸌': '⸍', '⸜': '⸝', '⸠': '⸡', '‹': '›', '«': '»', '＜': '＞', '<': '>'  
            }   
            left_symbols = set(bracket_pairs.keys())  
            right_symbols = set(bracket_pairs.values())  
            
            has_brackets = any(s in stripped_text for s in left_symbols) or any(s in stripped_text for s in right_symbols)  
            
            if has_brackets:  
                result_chars = []  
                stack = []  
                to_skip = []    
                
                # 第一次遍历：标记匹配的括号  
                # First traversal: mark matching brackets
                for i, char in enumerate(stripped_text):  
                    if char in left_symbols:  
                        stack.append((i, char))  
                    elif char in right_symbols:  
                        if stack:  
                            # 有对应的左括号，出栈  
                            # There is a corresponding left bracket, pop the stack
                            stack.pop()  
                        else:  
                            # 没有对应的左括号，标记为删除  
                            # No corresponding left parenthesis, marked for deletion
                            to_skip.append(i)  
                
                # 标记未匹配的左括号为删除
                # Mark unmatched left brackets as delete  
                for pos, _ in stack:  
                    to_skip.append(pos)  
                
                has_removed_symbols = len(to_skip) > 0  
                
                # 第二次遍历：处理匹配但不对应的括号
                # Second pass: Process matching but mismatched brackets
                stack = []  
                for i, char in enumerate(stripped_text):  
                    if i in to_skip:  
                        # 跳过孤立的括号
                        # Skip isolated parentheses
                        continue  
                        
                    if char in left_symbols:  
                        stack.append(char)  
                        result_chars.append(char)  
                    elif char in right_symbols:  
                        if stack:  
                            left_bracket = stack.pop()  
                            expected_right = bracket_pairs.get(left_bracket)  
                            
                            if char != expected_right:  
                                # 替换不匹配的右括号为对应左括号的正确右括号
                                # Replace mismatched right brackets with the correct right brackets corresponding to the left brackets
                                result_chars.append(expected_right)  
                                logger.info(f'Fixed mismatched bracket: replaced "{char}" with "{expected_right}"')  
                            else:  
                                result_chars.append(char)  
                    else:  
                        result_chars.append(char)  
                
                new_stripped_text = ''.join(result_chars)  
                
                if has_removed_symbols:  
                    logger.info(f'Removed unpaired bracket from "{stripped_text}"')  
                
                if new_stripped_text != stripped_text and not has_removed_symbols:  
                    logger.info(f'Fixed brackets: "{stripped_text}" → "{new_stripped_text}"')  
                
                stripped_text = new_stripped_text  
              
            region.text = stripped_text.strip()     
            
            if len(region.text) < config.ocr.min_text_length \
                    or not is_valuable_text(region.text) \
                    or (not config.translator.no_text_lang_skip and langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0):
                if region.text.strip():
                    logger.info(f'Filtered out: {region.text}')
                    if len(region.text) < config.ocr.min_text_length:
                        logger.info('Reason: Text length is less than the minimum required length.')
                    elif not is_valuable_text(region.text):
                        logger.info('Reason: Text is not considered valuable.')
                    elif langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0:
                        logger.info('Reason: Text language matches the target language and no_text_lang_skip is False.')
            else:
                if config.render.font_color_fg or config.render.font_color_bg:
                    if config.render.font_color_bg:
                        region.adjust_bg_color = False
                new_text_regions.append(region)
        text_regions = new_text_regions

        text_regions = sort_regions(
            text_regions,
            right_to_left=config.render.rtl,
            img=ctx.img_rgb,
            force_simple_sort=config.force_simple_sort
        )   
        
        return text_regions

    def reset_page_context(self):
        """Clear accumulated cross-page translation context. Call between galleries so one
        title's dialogue doesn't leak into the next as 'previous pages' reference."""
        self.all_page_translations = []
        self._original_page_texts = []
        return {"ok": True}

    def _build_prev_context(self, use_original_text=False, current_page_index=None, batch_index=None, batch_original_texts=None):
        """
        跳过句子数为0的页面，取最近 context_size 个非空页面，拼成：
        <|1|>句子
        <|2|>句子
        ...
        的格式；如果没有任何非空页面，返回空串。

        Args:
            use_original_text: 是否使用原文而不是译文作为上下文
            current_page_index: 当前页面索引，用于确定上下文范围
            batch_index: 当前页面在批次中的索引
            batch_original_texts: 当前批次的原文数据
        """
        if self.context_size <= 0:
            return ""

        # 在并发模式下，需要特殊处理上下文范围
        if batch_index is not None and batch_original_texts is not None:
            # 并发模式：使用已完成的页面 + 当前批次中已处理的页面
            available_pages = self.all_page_translations.copy()

            # 添加当前批次中在当前页面之前的页面
            for i in range(batch_index):
                if i < len(batch_original_texts) and batch_original_texts[i]:
                    # 在并发模式下，我们使用原文作为"已完成"的页面
                    if use_original_text:
                        available_pages.append(batch_original_texts[i])
                    else:
                        # 如果不使用原文，则跳过当前批次的页面（因为它们还没有翻译完成）
                        pass
        elif current_page_index is not None:
            # 使用指定页面索引之前的页面作为上下文
            available_pages = self.all_page_translations[:current_page_index] if self.all_page_translations else []
        else:
            # 使用所有已完成的页面
            available_pages = self.all_page_translations or []

        if not available_pages:
            return ""

        # 筛选出有句子的页面
        non_empty_pages = [
            page for page in available_pages
            if any(sent.strip() for sent in page.values())
        ]
        # 实际要用的页数
        pages_used = min(self.context_size, len(non_empty_pages))
        if pages_used == 0:
            return ""
        tail = non_empty_pages[-pages_used:]

        # 拼接 - 根据参数决定使用原文还是译文
        lines = []
        for page in tail:
            for sent in page.values():
                if sent.strip():
                    lines.append(sent.strip())

        # 如果使用原文，需要从原始数据中获取
        if use_original_text and hasattr(self, '_original_page_texts'):
            # 尝试获取对应的原文
            original_lines = []
            for i, page in enumerate(tail):
                page_idx = available_pages.index(page)
                if page_idx < len(self._original_page_texts):
                    original_page = self._original_page_texts[page_idx]
                    for sent in original_page.values():
                        if sent.strip():
                            original_lines.append(sent.strip())
            if original_lines:
                lines = original_lines

        numbered = [f"<|{i+1}|>{s}" for i, s in enumerate(lines)]
        context_type = "original text" if use_original_text else "translation results"
        return f"Here are the previous {context_type} for reference:\n" + "\n".join(numbered)

    async def _dispatch_with_context(self, config: Config, texts: list[str], ctx: Context):
        # 计算实际要使用的上下文页数和跳过的空页数
        # Calculate the actual number of context pages to use and empty pages to skip
        done_pages = self.all_page_translations
        if self.context_size > 0 and done_pages:
            pages_expected = min(self.context_size, len(done_pages))
            non_empty_pages = [
                page for page in done_pages
                if any(sent.strip() for sent in page.values())
            ]
            pages_used = min(self.context_size, len(non_empty_pages))
            skipped = pages_expected - pages_used
        else:
            pages_used = skipped = 0

        if self.context_size > 0:
            logger.info(f"Context-aware translation enabled with {self.context_size} pages of history")

        # 构建上下文字符串
        # Build the context string
        prev_ctx = self._build_prev_context()

        # 如果是 ChatGPT 或 ChatGPT2Stage 翻译器，则专门处理上下文注入
        # Special handling for ChatGPT and ChatGPT2Stage translators: inject context
        if config.translator.translator in [Translator.chatgpt, Translator.chatgpt_2stage]:
            if config.translator.translator == Translator.chatgpt:
                from .translators.chatgpt import OpenAITranslator
                translator = OpenAITranslator()
            else:  # chatgpt_2stage
                from .translators.chatgpt_2stage import ChatGPT2StageTranslator
                translator = ChatGPT2StageTranslator()
                
            translator.parse_args(config.translator)
            translator.set_prev_context(prev_ctx)

            if pages_used > 0:
                context_count = prev_ctx.count("<|")
                logger.info(f"Carrying {pages_used} pages of context, {context_count} sentences as translation reference")
            if skipped > 0:
                logger.warning(f"Skipped {skipped} pages with no sentences")
                

            
            # ChatGPT2Stage 需要传递 ctx 参数，普通 ChatGPT 不需要
            if config.translator.translator == Translator.chatgpt_2stage:
                # 添加result_path_callback到Context，让translator可以保存bboxes_fixed.png
                ctx.result_path_callback = self._result_path
                return await translator._translate(ctx.from_lang, config.translator.target_lang, texts, ctx)
            else:
                return await translator._translate(ctx.from_lang, config.translator.target_lang, texts)


        return await dispatch_translation(
            config.translator.translator_gen,
            texts,
            config.translator,
            self.use_mtpe,
            ctx,
            'cpu' if self._gpu_limited_memory else self.device
        )

    async def _run_screened_translation(self, config: Config, texts: list, ctx: Context) -> list:
        import copy
        from datetime import datetime
        from .translators import get_translator
        screen_key   = Translator(config.translator.content_screen_translator)
        fallback_key = Translator(config.translator.content_screen_fallback_translator)
        device = 'cpu' if self._gpu_limited_memory else self.device

        # Step 1: translate everything with the local fallback model first
        fallback_cfg = copy.deepcopy(config)
        fallback_cfg.translator.translator = fallback_key
        fallback_cfg.translator._translator_gen = None
        fallback_results = await self._dispatch_with_context(fallback_cfg, texts, ctx)

        # Step 2: screen the translated output (English is far easier to classify than raw OCR)
        screen_translator = get_translator(screen_key)
        await screen_translator.load(ctx.from_lang, config.translator.target_lang, device)
        self._model_usage_timestamps[("translation", screen_key)] = time.time()
        flags = await screen_translator.classify(fallback_results, config.translator.content_screen_prompt)
        self._model_usage_timestamps[("translation", screen_key)] = time.time()

        n_flagged = sum(flags)
        logger.info(f'Content screen: {n_flagged}/{len(flags)} bubble(s) flagged as explicit — '
                    f'sending {len(flags) - n_flagged} clean bubble(s) to primary translator')

        # Step 3: re-translate clean bubbles with the primary (cloud) translator
        clean_idx = [i for i, f in enumerate(flags) if not f]
        results = list(fallback_results)  # start from fallback, override clean ones

        if clean_idx:
            cloud_trans = await self._dispatch_with_context(
                config, [texts[i] for i in clean_idx], ctx)
            for i, t in zip(clean_idx, cloud_trans):
                results[i] = t

        try:
            lines = [
                f'Content Screen Log — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                f'Primary: {config.translator.translator}  |  '
                f'Screen model: {config.translator.content_screen_translator}  |  '
                f'Fallback: {config.translator.content_screen_fallback_translator}',
                f'Flagged: {n_flagged}/{len(flags)} bubble(s) kept as fallback',
                '',
                f'{"#":<4}  {"Label":<8}  {"OCR":<35}  {"Fallback":<35}  Final',
                '-' * 120,
            ]
            for i, (text, flag, fallback, final) in enumerate(
                    zip(texts, flags, fallback_results, results), 1):
                label = 'EXPLICIT' if flag else 'clean   '
                lines.append(
                    f'{i:<4}  {label}  {text[:35]:<35}  {fallback[:35]:<35}  {final}')
            with open(self._result_path('content_screen.txt', ctx), 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            logger.warning(f'Content screen: failed to write log: {e}')

        return results

    async def _run_text_translation(self, config: Config, ctx: Context):
        # 检查text_regions是否为None或空
        if not ctx.text_regions:
            return []
            
        # 如果设置了prep_manual则将translator设置为none，防止token浪费
        # Set translator to none to provent token waste if prep_manual is True  
        if self.prep_manual:  
            config.translator.translator = Translator.none
    
        current_time = time.time()
        self._model_usage_timestamps[("translation", config.translator.translator)] = current_time

        # 为none翻译器添加特殊处理  
        # Add special handling for none translator  
        if config.translator.translator == Translator.none:  
            # 使用none翻译器时，为所有文本区域设置必要的属性  
            # When using none translator, set necessary properties for all text regions  
            for region in ctx.text_regions:  
                region.translation = ""  # 空翻译将创建空白区域 / Empty translation will create blank areas  
                region.target_lang = config.translator.target_lang  
                region._alignment = config.render.alignment  
                region._direction = config.render.direction    
            return ctx.text_regions  

        # 以下翻译处理仅在非none翻译器或有none翻译器但没有prep_manual时执行  
        # Translation processing below only happens for non-none translator or none translator without prep_manual  
        if self.load_text:  
            input_filename = os.path.splitext(os.path.basename(self.input_files[0]))[0]  
            with open(self._result_path(f"{input_filename}_translations.txt"), "r") as f:  
                    translated_sentences = json.load(f)  
        else:  
            # 如果是none翻译器，不需要调用翻译服务，文本已经设置为空  
            # If using none translator, no need to call translation service, text is already set to empty  
            if config.translator.translator != Translator.none:
                # 自动给 ChatGPT 加上下文，其他翻译器不改变
                # Automatically add context to ChatGPT, no change for other translators
                texts = [region.text for region in ctx.text_regions]
                if config.translator.content_screen_enabled:
                    translated_sentences = \
                        await self._run_screened_translation(config, texts, ctx)
                else:
                    translated_sentences = \
                        await self._dispatch_with_context(config, texts, ctx)
            else:  
                # 对于none翻译器，创建一个空翻译列表  
                # For none translator, create an empty translation list  
                translated_sentences = ["" for _ in ctx.text_regions]  

            # Save translation if args.save_text is set and quit  
            if self.save_text:  
                input_filename = os.path.splitext(os.path.basename(self.input_files[0]))[0]  
                with open(self._result_path(f"{input_filename}_translations.txt"), "w") as f:  
                    json.dump(translated_sentences, f, indent=4, ensure_ascii=False)  
                print("Don't continue if --save-text is used")  
                exit(-1)  

        # 如果不是none翻译器或者是none翻译器但没有prep_manual  
        # If not none translator or none translator without prep_manual  
        if config.translator.translator != Translator.none or not self.prep_manual:  
            for region, translation in zip(ctx.text_regions, translated_sentences):  
                if config.render.uppercase:  
                    translation = translation.upper()  
                elif config.render.lowercase:  
                    translation = translation.lower()  # 修正：应该是lower而不是upper  
                region.translation = translation  
                region.target_lang = config.translator.target_lang  
                region._alignment = config.render.alignment  
                region._direction = config.render.direction  

        # Punctuation correction logic. for translators often incorrectly change quotation marks from the source language to those commonly used in the target language.
        check_items = [
            # 圆括号处理
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],
            
            # 方括号处理
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],
            
            # 引号处理
            ["「", "“", "‘", "『", "【"],
            ["」", "”", "’", "』", "】"],
            ["『", "“", "‘", "「", "【"],
            ["』", "”", "’", "」", "】"],
            
            # 新增【】处理
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        replace_items = [
            ["「", "“"],
            ["「", "‘"],
            ["」", "”"],
            ["」", "’"],
            ["【", "["],  
            ["】", "]"],  
        ]

        for region in ctx.text_regions:
            if region.text and region.translation:
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text: 
                    quote_type = '【】'
                else:
                    quote_type = None
                
                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')
                    
                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):
                        
                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':  
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # === 优化后的数量判断逻辑 ===
                # === Optimized quantity judgment logic ===
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])
                    
                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # 强制替换规则
                # Forced replacement rules
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

        # 注意：翻译结果的保存移动到了翻译流程的最后，确保保存的是最终结果而不是重试前的结果

        # Apply post dictionary after translating
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:  
            original = region.translation  
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:  
            logger.info("Post-translation replacements:")  
            for replacement in post_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("No post-translation replacements made.")

        # 译后检查和重试逻辑 - 第一阶段：单个region幻觉检测
        failed_regions = []
        if config.translator.enable_post_translation_check:
            logger.info("Starting post-translation check...")
            
            # 单个region级别的幻觉检测（在过滤前进行）
            for region in ctx.text_regions:
                if region.translation and region.translation.strip():
                    # 只检查重复内容幻觉，不进行页面级目标语言检查
                    if await self._check_repetition_hallucination(
                        region.translation, 
                        config.translator.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)
            
            # 对失败的区域进行重试
            if failed_regions:
                logger.warning(f"Found {len(failed_regions)} regions that failed repetition check, starting retry...")
                for region in failed_regions:
                    await self._retry_translation_with_validation(region, config, ctx)
                logger.info("Repetition check retry finished.")

        # 译后检查和重试逻辑 - 第二阶段：页面级目标语言检查（使用过滤后的区域）
        if config.translator.enable_post_translation_check:
            
            # 页面级目标语言检查（使用过滤后的区域数量）
            page_lang_check_result = True
            if ctx.text_regions and len(ctx.text_regions) > 5:
                logger.info(f"Starting page-level target language check with {len(ctx.text_regions)} regions...")
                page_lang_check_result = await self._check_target_language_ratio(
                    ctx.text_regions,
                    config.translator.target_lang,
                    min_ratio=0.5
                )
                
                if not page_lang_check_result:
                    logger.warning("Page-level target language ratio check failed")
                    
                    # 第二阶段：整个批次重新翻译逻辑
                    max_batch_retry = config.translator.post_check_max_retry_attempts
                    batch_retry_count = 0
                    
                    while batch_retry_count < max_batch_retry and not page_lang_check_result:
                        batch_retry_count += 1
                        logger.warning(f"Starting batch retry {batch_retry_count}/{max_batch_retry} for page-level target language check...")
                        
                        # 重新翻译所有区域
                        original_texts = []
                        for region in ctx.text_regions:
                            if hasattr(region, 'text') and region.text:
                                original_texts.append(region.text)
                            else:
                                original_texts.append("")
                        
                        if original_texts:
                            try:
                                # 重新批量翻译
                                logger.info(f"Retrying translation for {len(original_texts)} regions...")
                                new_translations = await self._batch_translate_texts(original_texts, config, ctx)
                                
                                # 更新翻译结果到regions
                                for i, region in enumerate(ctx.text_regions):
                                    if i < len(new_translations) and new_translations[i]:
                                        old_translation = region.translation
                                        region.translation = new_translations[i]
                                        logger.debug(f"Region {i+1} translation updated: '{old_translation}' -> '{new_translations[i]}'")
                                    
                                # 重新检查目标语言比例
                                logger.info(f"Re-checking page-level target language ratio after batch retry {batch_retry_count}...")
                                page_lang_check_result = await self._check_target_language_ratio(
                                    ctx.text_regions,
                                    config.translator.target_lang,
                                    min_ratio=0.5
                                )
                                
                                if page_lang_check_result:
                                    logger.info(f"Page-level target language check passed")
                                    break
                                else:
                                    logger.warning(f"Page-level target language check still failed")
                                    
                            except Exception as e:
                                logger.error(f"Error during batch retry {batch_retry_count}: {e}")
                                break
                        else:
                            logger.warning("No text found for batch retry")
                            break
                    
                    if not page_lang_check_result:
                        logger.error(f"Page-level target language check failed after all {max_batch_retry} batch retries")
                else:
                    logger.info("Page-level target language ratio check passed")
            else:
                logger.info(f"Skipping page-level target language check: only {len(ctx.text_regions)} regions (threshold: 5)")
            
            # 统一的成功信息
            if page_lang_check_result:
                logger.info("All translation regions passed post-translation check.")
            else:
                logger.warning("Some translation regions failed post-translation check.")

        # 过滤逻辑（简化版本，保留主要过滤条件）
        new_text_regions = []
        for region in ctx.text_regions:
            should_filter = False
            filter_reason = ""

            if not region.translation.strip():
                should_filter = True
                filter_reason = "Translation contain blank areas"
            elif config.translator.translator != Translator.none:
                if region.translation.isnumeric():
                    should_filter = True
                    filter_reason = "Numeric translation"
                elif config.filter_text and re.search(config.re_filter_text, region.translation):
                    should_filter = True
                    filter_reason = f"Matched filter text: {config.filter_text}"
                elif not config.translator.translator == Translator.original:
                    text_equal = region.text.lower().strip() == region.translation.lower().strip()
                    if text_equal:
                        should_filter = True
                        filter_reason = "Translation identical to original"

            if should_filter:
                if region.translation.strip():
                    logger.info(f'Filtered out: {region.translation}')
                    logger.info(f'Reason: {filter_reason}')
            else:
                new_text_regions.append(region)

        return new_text_regions

    async def _run_mask_refinement(self, config: Config, ctx: Context):
        return await dispatch_mask_refinement(ctx.text_regions, ctx.img_rgb, ctx.mask_raw, 'fit_text',
                                              config.mask_dilation_offset, config.ocr.ignore_bubble, self.verbose,self.kernel_size)

    async def _run_inpainting(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("inpainting", config.inpainter.inpainter)] = current_time
        t0 = time.perf_counter()
        result = await dispatch_inpainting(config.inpainter.inpainter, ctx.img_rgb, ctx.mask, config.inpainter, config.inpainter.inpainting_size, self.device,
                                         self.verbose)
        self._accum_time('inpainting', time.perf_counter() - t0)
        return result

    async def _run_text_rendering(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("rendering", config.render.renderer)] = current_time
        # Sample each region's clean (inpainted) bubble background so get_font_colors can
        # snap a spurious low-contrast outline to it. Read by every get_font_colors-based
        # renderer — including shiori while it takes OCR colors; none ignores it.
        # (original, for when shiori goes back to model colors:)
        # if ctx.text_regions and config.render.renderer not in (Renderer.none, Renderer.shiori):
        if ctx.text_regions and config.render.renderer is not Renderer.none:
            inpainted = ctx.img_inpainted
            ih, iw = inpainted.shape[:2]
            for region in ctx.text_regions:
                x1, y1, x2, y2 = (int(v) for v in region.xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                if x2 > x1 and y2 > y1:
                    crop = inpainted[y1:y2, x1:x2].reshape(-1, inpainted.shape[-1])
                    region._bubble_bg = np.median(crop, axis=0)
        t0 = time.perf_counter()
        if config.render.renderer == Renderer.none:
            output = ctx.img_inpainted
        elif config.render.renderer == Renderer.shiori and ctx.text_regions:
            output = await dispatch_shiori_render(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, device=self.device, verbose=self.verbose)
        elif config.render.renderer == Renderer.shioriV2 and ctx.text_regions:
            output = await dispatch_shiori_render_v2(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing, device=self.device, verbose=self.verbose)
        # manga2eng currently only supports horizontal left to right rendering
        elif (config.render.renderer == Renderer.manga2Eng or config.render.renderer == Renderer.manga2EngPillow) and ctx.text_regions and LANGUAGE_ORIENTATION_PRESETS.get(ctx.text_regions[0].target_lang) == 'h':
            if config.render.renderer == Renderer.manga2EngPillow:
                output = await dispatch_eng_render_pillow(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing)
            else:
                try:
                    output = await dispatch_eng_render(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing, verbose=self.verbose)
                except Exception as e:
                    # Freetype path failed (e.g. a face/glyph fault on this page) — retry the page
                    # with the Pillow renderer before giving up; a second failure raises as before.
                    logger.warning(f'manga2eng freetype rendering failed ({e}); retrying page with the Pillow renderer')
                    output = await dispatch_eng_render_pillow(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing)
        else:
            output = await dispatch_rendering(ctx.img_inpainted, ctx.text_regions, self.font_path, config.render.font_size,
                                              config.render.font_size_offset,
                                              config.render.font_size_minimum, not config.render.no_hyphenation, ctx.render_mask, config.render.line_spacing)
        # Verbose dump of the balloon each eng-rendered region used and how it was fitted, drawn
        # over the rendered output so the overlay shows the translated text inside its balloon.
        # The path MUST come from this page's own context: in the gallery pipeline several pages
        # are in flight at once and the mutable self._current_image_context belongs to whichever
        # page preprocessed last, so routing through self._result_path() directly sent most
        # pages' bubbles.png into one folder, overwriting each other.
        if self.verbose and ctx.text_regions and any(getattr(r, '_bubble_source', None) is not None for r in ctx.text_regions):
            try:
                rp = getattr(ctx, 'result_path_callback', None)
                if rp is not None:
                    bubbles_path = rp('bubbles.png')
                else:
                    bubbles_path = self._result_path('bubbles.png', ctx)
                cv2.imwrite(bubbles_path, render_bubble_debug(output, ctx.text_regions))
            except Exception as e:
                logger.warning(f'failed to write bubble debug overlay: {e}')
        self._accum_time('rendering', time.perf_counter() - t0)
        return output

    def _result_path(self, path: str, ctx: Context = None) -> str:
        """
        Returns path to result folder where intermediate images are saved when using verbose flag
        or web mode input/result images are cached.

        When `ctx` carries a per-page `image_context` it takes priority: in the gallery pipeline
        several pages are in flight at once and the mutable self._current_image_context belongs to
        whichever page preprocessed last, so any dump routed through it can land in (and overwrite)
        another page's folder.
        """
        ic = getattr(ctx, 'image_context', None) if ctx is not None else None
        if not ic:
            ic = self._current_image_context
        # 只有在verbose模式下才使用图片级子文件夹
        if self.verbose:
            image_subfolder = ic['subfolder'] if ic else ''
            if image_subfolder:
                if self.result_sub_folder:
                    result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, path)
                else:
                    result_path = os.path.join(BASE_PATH, 'result', image_subfolder, path)
                # 确保目录存在
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                return result_path

        # 在server/web模式下（result_sub_folder为空）且为非verbose模式时
        # 需要创建一个子文件夹来保存final.png
        if not self.result_sub_folder:
            if ic:
                # 直接使用已生成的子文件夹名
                sub_folder = ic['subfolder']
            else:
                # 没有上下文信息时使用默认值
                timestamp = str(int(time.time() * 1000))
                sub_folder = f"{timestamp}-unknown-1024-unknown-unknown"

            result_path = os.path.join(BASE_PATH, 'result', sub_folder, path)
        else:
            result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        return result_path

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def add_page_result_hook(self, ph):
        self._page_result_hooks.append(ph)

    async def _emit_page_result(self, index: int, image):
        """Emit a single finished page during a streaming batch translation.
        No-op unless a page-result hook is registered (e.g. by the share server),
        so the CLI batch path is unaffected."""
        for ph in self._page_result_hooks:
            await ph(index, image)

    def add_page_bubbles_hook(self, ph):
        self._page_bubbles_hooks.append(ph)

    async def _emit_page_bubbles(self, index: int, bubbles: list):
        """Emit one page's per-bubble translation overlays during a streaming gallery
        translation. No-op unless a bubbles hook is registered, so every other path
        (CLI, single-image, batch) is unaffected."""
        for ph in self._page_bubbles_hooks:
            await ph(index, bubbles)

    async def _build_bubble_overlays(self, ctx, config, mode: str = 'text_and_image') -> dict:
        """Build a page's study payload so a reader can reveal one translated bubble at a time
        over the clean original, satisfying: precise OCR border, uncropped & isolated text,
        page-sized (zoom-stable) layers, and overlap-safe stacking.

        `mode` (config.study_mode_generation, never 'disabled' here):
          • text_only      — metadata only: per-bubble geometry + original/translated text +
                             renderer style hints. No image layers, no diff, near-zero cost.
          • text_and_image — metadata plus the image layers:
              · bg   — the full inpainted page (text removed), shared by every bubble.
              · text — per bubble, a FULL-PAGE transparent PNG holding ONLY its glyphs.

        Every bubble carries:
          · box    — the OCR detection region (the hover/click border), normalized.
          · rbox   — the box the renderer actually laid this region's text in
                     (manga2eng's enlarged_xyxy; falls back to the detection box).
          · region — union(box, rbox or glyph extent); clips bg so the background covers
                     the Japanese AND the full English.
          · tr/src — translated / original text.
          · style  — renderer-derived hints (fontSize px, fg/bg rgb, align, dir, lineSpacing)
                     so a reader can render the bubble as DOM text.

        For text_and_image the glyphs are lifted out of the SINGLE final render
        (ctx.img_rendered = inpaint + every region's text drawn together) by PARTITIONING
        every changed pixel among the bubbles (see _build_page_layers_job): reassembling all
        bubble layers over the inpaint reproduces the final page pixel-for-pixel — no
        cropping, no bleed, no double-draw. The whole page (diff, ownership, per-bubble
        layers, encodes) runs as ONE CPU-pool job instead of one per bubble.
        Returns None when there's nothing to show. The pixel partition + encodes run in the
        GIL-free process pool (see _build_page_layers_job)."""
        inpainted = getattr(ctx, 'img_inpainted', None)
        rendered = getattr(ctx, 'img_rendered', None)
        regions = getattr(ctx, 'text_regions', None)
        if rendered is None or not regions:
            return None
        H, W = rendered.shape[:2]

        _is_m2e = config.render.renderer in (Renderer.manga2Eng, Renderer.manga2EngPillow)

        def _style_hints(r):
            """Best-effort renderer/OCR style metadata for DOM-text study rendering."""
            hints = {}
            try:
                # Prefer the size the renderer actually DREW (post fit-downscale) over the
                # pre-layout detection size — this is what makes DOM text match the image.
                fs = int(getattr(r, '_drawn_font_size', 0) or getattr(r, 'font_size', 0) or 0)
                if fs > 0:
                    hints['fontSize'] = fs
            except Exception:
                pass
            try:
                # The ORIGINAL text's detected size (pre-render), for typesetting the source
                # as DOM text at its own scale rather than the translation's.
                sfs = int(getattr(r, 'font_size', 0) or 0)
                if sfs > 0:
                    hints['srcFontSize'] = sfs
            except Exception:
                pass
            if _is_m2e or getattr(r, '_typeset_eng', False):
                # manga2eng letters in comic caps with a tight line advance (~0.8×size) and a
                # white border — hint it so DOM text can mimic the typesetting. shiori v2 marks
                # the regions it routed through the eng fit with _typeset_eng.
                hints['caps'] = True
            try:
                # One fg/bg pair, always the colors the render pass actually drew with:
                # renderers that record them (_drawn_fg/_drawn_bg) win over the OCR-derived
                # colors; both the original and the translation DOM text share this pair.
                fg = getattr(r, '_drawn_fg', None)
                bg = getattr(r, '_drawn_bg', None)
                if fg is None or bg is None:
                    fg, bg = r.get_font_colors()
                hints['fg'] = [int(c) for c in np.clip(np.asarray(fg), 0, 255)]
                hints['bg'] = [int(c) for c in np.clip(np.asarray(bg), 0, 255)]
            except Exception:
                pass
            try:
                align = getattr(r, 'alignment', None)
                if isinstance(align, str):
                    hints['align'] = align
            except Exception:
                pass
            try:
                direction = getattr(r, 'source_direction', None)
                if direction not in ('h', 'v'):
                    direction = getattr(r, 'direction', None)
                if isinstance(direction, str):
                    hints['dir'] = direction
            except Exception:
                pass
            try:
                ls = getattr(r, 'line_spacing', None)
                if ls:
                    hints['lineSpacing'] = float(ls)
            except Exception:
                pass
            try:
                # The pitch the renderer actually drew lines at, as a line-height ratio.
                dlh = getattr(r, '_drawn_line_height', None)
                if dlh:
                    hints['lineH'] = float(dlh)
            except Exception:
                pass
            return hints

        # Per-region geometry + text + style, gathered once for both modes. det = clamped OCR
        # box; rbox = clamped render box (enlarged_xyxy where the renderer actually drew, padded
        # a little so component-overlap assignment catches border strokes).
        infos = []
        for r in regions:
            tr = (getattr(r, 'translation', '') or '').strip()
            if not tr:
                continue
            try:
                dx1, dy1, dx2, dy2 = (int(v) for v in r.xyxy)
            except Exception:
                continue
            dx1, dx2 = max(0, min(W, dx1)), max(0, min(W, dx2))
            dy1, dy2 = max(0, min(H, dy1)), max(0, min(H, dy2))
            if dx2 <= dx1 or dy2 <= dy1:
                continue
            rb = getattr(r, 'enlarged_xyxy', None)
            try:
                bx1, by1, bx2, by2 = (int(v) for v in rb) if rb is not None else (dx1, dy1, dx2, dy2)
            except Exception:
                bx1, by1, bx2, by2 = dx1, dy1, dx2, dy2
            bx1 = max(0, bx1 - 6); by1 = max(0, by1 - 6)
            bx2 = min(W, bx2 + 6); by2 = min(H, by2 + 6)
            if bx2 <= bx1 or by2 <= by1:
                bx1, by1, bx2, by2 = dx1, dy1, dx2, dy2
            src_lines = []
            try:
                for line in (getattr(r, 'texts', None) or []):
                    line = str(line)
                    if line.strip():
                        src_lines.append(line)
            except Exception:
                src_lines = []
            src = '\n'.join(src_lines) if src_lines else (getattr(r, 'text', '') or '')
            info = {
                'det': (dx1, dy1, dx2, dy2), 'rbox': (bx1, by1, bx2, by2),
                'tr': tr, 'src': src, 'style': _style_hints(r),
            }
            # Where the renderer's glyph canvas actually landed — the DOM translation matches
            # the image only when positioned at this rect, not at the layout-allowance box.
            dr = getattr(r, '_drawn_rect', None)
            try:
                tx1, ty1, tx2, ty2 = (int(v) for v in dr)
                if tx2 > tx1 and ty2 > ty1:
                    info['tbox'] = (tx1, ty1, tx2, ty2)
            except Exception:
                pass
            drawn = getattr(r, '_drawn_lines', None)
            if isinstance(drawn, list) and any(str(s).strip() for s in drawn):
                info['tr'] = '\n'.join(str(s) for s in drawn)
            furi_lines = src_lines or (src.splitlines() if src.strip() else [])
            furi = _furi_lines(furi_lines)
            if furi:
                info['furi'] = furi
            infos.append(info)
        if not infos:
            return None

        if mode == 'text_only':
            # Geometry/text/style only — region = union(detection box, render box), no pixels read.
            bubbles = []
            for info in infos:
                dx1, dy1, dx2, dy2 = info['det']
                bx1, by1, bx2, by2 = info['rbox']
                bubbles.append(_study_meta_bubble(
                    info, (min(dx1, bx1), min(dy1, by1), max(dx2, bx2), max(dy2, by2)), W, H))
            return {'page': {'w': W, 'h': H}, 'bubbles': bubbles}

        if inpainted is None or inpainted.shape[:2] != (H, W):
            return None

        # The pixel partition + per-bubble PNG/WebP encodes are CPU-heavy and GIL-holding, so run
        # them out-of-process. The info-gathering above had to stay here (it reads region objects
        # that don't pickle); only the two page arrays + the small `infos` list cross to the worker.
        return await run_proc(_build_page_layers_job, rendered, inpainted, infos, W, H)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'translating': 'Running text translation',
            'rendering': 'Running rendering',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'error-translating': 'Text translator returned empty queries',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    async def translate_batch(self, images_with_configs: List[tuple], batch_size: int = None, image_names: List[str] = None) -> List[Context]:
        """
        批量翻译多张图片，在翻译阶段进行批量处理以提高效率
        Args:
            images_with_configs: List of (image, config) tuples
            batch_size: 批量大小，如果为None则使用实例的batch_size
            image_names: 已弃用的参数，保留用于兼容性
        Returns:
            List of Context objects with translation results
        """
        batch_size = batch_size or self.batch_size
        if batch_size <= 1:
            # 不使用批量处理时，回到原来的逐个处理方式
            logger.debug('Batch size <= 1, switching to individual processing mode')
            results = []
            for i, (image, config) in enumerate(images_with_configs):
                ctx = await self.translate(image, config)  # 单页翻译时正常保存上下文
                results.append(ctx)
            return results
        
        logger.debug(f'Starting batch translation: {len(images_with_configs)} images, batch size: {batch_size}')
        
        # 简化的内存检查
        memory_optimization_enabled = not self.disable_memory_optimization
        if not memory_optimization_enabled:
            logger.debug('Memory optimization disabled for batch translation')
        
        results = []
        
        # 处理所有图片到翻译之前的步骤
        logger.debug('Starting pre-processing phase...')
        pre_translation_contexts = []
        
        for i, (image, config) in enumerate(images_with_configs):
            logger.debug(f'Pre-processing image {i+1}/{len(images_with_configs)}')
            
            # 简化的内存检查
            if memory_optimization_enabled:
                try:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 85:
                        logger.warning(f'High memory usage during pre-processing: {memory_percent:.1f}%')
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except ImportError:
                    pass  # psutil 不可用时忽略
                except Exception as e:
                    logger.debug(f'Memory check failed: {e}')
                
            try:
                # 为批量处理中的每张图片设置上下文
                self._set_image_context(config, image)
                # 保存图片上下文，确保后处理阶段使用相同的文件夹
                if self._current_image_context:
                    image_md5 = self._current_image_context['file_md5']
                    self._save_current_image_context(image_md5)
                ctx = await self._translate_until_translation(image, config)
                # 保存图片上下文到Context对象中，用于后续批量处理
                if self._current_image_context:
                    ctx.image_context = self._current_image_context.copy()
                # 保存verbose标志到Context对象中
                ctx.verbose = self.verbose
                pre_translation_contexts.append((ctx, config))
                logger.debug(f'Image {i+1} pre-processing successful')
            except MemoryError as e:
                logger.error(f'Memory error in pre-processing image {i+1}: {e}')
                if not memory_optimization_enabled:
                    logger.error('Consider enabling memory optimization')
                    raise
                    
                # 尝试降级处理
                try:
                    logger.warning(f'Image {i+1} attempting fallback processing...')
                    import copy
                    recovery_config = copy.deepcopy(config)
                    
                    # 强制清理
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 重新设置图片上下文
                    self._set_image_context(recovery_config, image)
                    # 保存fallback图片上下文
                    if self._current_image_context:
                        image_md5 = self._current_image_context['file_md5']
                        self._save_current_image_context(image_md5)
                    ctx = await self._translate_until_translation(image, recovery_config)
                    # 保存图片上下文到Context对象中
                    if self._current_image_context:
                        ctx.image_context = self._current_image_context.copy()
                    # 保存verbose标志到Context对象中
                    ctx.verbose = self.verbose
                    pre_translation_contexts.append((ctx, recovery_config))
                    logger.info(f'Image {i+1} fallback processing successful')
                except Exception as retry_error:
                    logger.error(f'Image {i+1} fallback processing also failed: {retry_error}')
                    # 创建空context作为占位符
                    ctx = Context()
                    ctx.input = image
                    ctx.text_regions = []  # 确保text_regions被初始化为空列表
                    pre_translation_contexts.append((ctx, config))
            except Exception as e:
                logger.error(f'Image {i+1} pre-processing error: {e}')
                # 创建空context作为占位符
                ctx = Context()
                ctx.input = image
                ctx.text_regions = []  # 确保text_regions被初始化为空列表
                pre_translation_contexts.append((ctx, config))
        
        if not pre_translation_contexts:
            logger.warning('No images pre-processed successfully')
            return results
            
        logger.debug(f'Pre-processing completed: {len(pre_translation_contexts)} images')
            
        # 批量翻译处理
        logger.debug('Starting batch translation phase...')
        try:
            if self.batch_concurrent:
                logger.info(f'Using concurrent mode for batch translation')
                translated_contexts = await self._concurrent_translate_contexts(pre_translation_contexts)
            else:
                logger.debug(f'Using standard batch mode for translation')
                translated_contexts = await self._batch_translate_contexts(pre_translation_contexts, batch_size)
        except MemoryError as e:
            logger.error(f'Memory error in batch translation: {e}')
            if not memory_optimization_enabled:
                logger.error('Consider enabling memory optimization')
                raise
                
            logger.warning('Batch translation failed, switching to individual page translation mode...')
            # 降级到每页逐个翻译
            translated_contexts = []
            for ctx, config in pre_translation_contexts:
                try:
                    if ctx.text_regions:  # 检查text_regions是否不为None且不为空
                        # 对整页进行翻译处理
                        translated_texts = await self._batch_translate_texts([region.text for region in ctx.text_regions], config, ctx)
                        
                        # 将翻译结果应用到各个region
                        for region, translation in zip(ctx.text_regions, translated_texts):
                            region.translation = translation
                            region.target_lang = config.translator.target_lang
                            region._alignment = config.render.alignment
                            region._direction = config.render.direction
                    translated_contexts.append((ctx, config))
                    
                    # 每页翻译后都清理内存
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as individual_error:
                    logger.error(f'Individual page translation failed: {individual_error}')
                    translated_contexts.append((ctx, config))
        
        # 完成翻译后的处理
        logger.debug('Starting post-processing phase...')
        for i, (ctx, config) in enumerate(translated_contexts):
            try:
                if ctx.text_regions:
                    # 恢复预处理阶段保存的图片上下文，确保使用相同的文件夹
                    # 通过图片计算MD5来恢复上下文
                    from .utils.generic import get_image_md5
                    image = ctx.input  # 从context中获取原始图片
                    image_md5 = get_image_md5(image)
                    if not self._restore_image_context(image_md5):
                        # 如果恢复失败，作为fallback重新设置（理论上不应该发生）
                        logger.warning(f"Failed to restore image context for MD5 {image_md5}, creating new context")
                        self._set_image_context(config, image)
                    ctx = await self._complete_translation_pipeline(ctx, config)
                results.append(ctx)
                logger.debug(f'Image {i+1} post-processing completed')
                # Stream this finished page out immediately (no-op without a hook).
                try:
                    await self._emit_page_result(i, getattr(ctx, 'result', None))
                except Exception as emit_error:
                    logger.error(f'Image {i+1} page-result emit error: {emit_error}')
            except Exception as e:
                logger.error(f'Image {i+1} post-processing error: {e}')
                results.append(ctx)
                try:
                    await self._emit_page_result(i, getattr(ctx, 'result', None))
                except Exception as emit_error:
                    logger.error(f'Image {i+1} page-result emit error: {emit_error}')
        
        logger.info(f'Batch translation completed: processed {len(results)} images')

        # 批处理完成后，保存所有页面的最终翻译结果
        for ctx in results:
            if ctx.text_regions:
                # 汇总本页翻译，供下一页做上文
                page_translations = {r.text_raw if hasattr(r, "text_raw") else r.text: r.translation
                                     for r in ctx.text_regions}
                self.all_page_translations.append(page_translations)

                # 同时保存原文用于并发模式的上下文
                page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                      for i, r in enumerate(ctx.text_regions)}
                self._original_page_texts.append(page_original_texts)

        # 清理批量处理的图片上下文缓存
        self._saved_image_contexts.clear()

        return results

    async def translate_gallery_stream(self, images: List, config: Config, batch_size: int = 0, job_token: str = "") -> dict:
        """Translate a whole gallery as ONE pipelined streaming request.

        Used by the share server's /execute/translate_gallery_stream route. Returns a
        small summary only — the rendered images leave incrementally through the
        page-result hook (status-5 frames), never as a (huge) list of Contexts.

        Unlike translate_batch (strict preprocess-all → translate-all → render-all),
        the three stages overlap: pages are detected/OCR'd one by one on the GPU;
        preprocessed pages become shared translation calls that run on the network
        while later pages keep preprocessing; translated batches are inpainted/
        rendered and emitted as soon as they come back. For most translators a call
        is formed every `batch_size` pages (<= 0 means all pages share one call).
        For DeepSeek, `batch_size` is only the MAX pages per call: request boundaries
        adapt to the OCR output volume (token soft target, a small first request so
        the LLM lane starts early, an idle flush that keeps a request in flight, and
        end-of-gallery shrinking that cuts the pure network tail). Page entries may
        be compressed image bytes (decoded here, one at a time) or PIL Images.

        Cross-page context (chatgpt + --context-size) is request-local: batches are
        chained in input order and the instance-global context lists are restored at
        the end, so queued clients can't bleed context into each other and the client
        no longer needs to call /reset-context.
        """
        import hashlib
        import io as _io
        from .translators import OFFLINE_TRANSLATORS
        from .utils.profiling import snapshot_substages, reset_llm_usage, snapshot_llm_usage

        n = len(images)
        if n == 0:
            return {"count": 0, "failed": []}
        cap = int(batch_size) if batch_size else 0
        cap = n if cap <= 0 else min(cap, n)

        # Adaptive LLM batching (DeepSeek): request boundaries are driven purely by OCR
        # token volume against the model's own input-token ceiling — the frontend page
        # count does NOT size requests. Sizing balances the two costs the pipeline trades
        # off: round-trips (the wall bottleneck → pack toward a soft token target to send
        # as few requests as possible) against overlap (→ a small first request and a
        # shrinking last request so OCR/inpaint/render never stall on translation).
        adaptive = (config.translator.translator == Translator.deepseek) and n > 1
        soft_tokens = 0
        if adaptive:
            try:
                from .translators.deepseek import DeepseekTranslator
                # Hard ceiling = the model's real max input tokens; the prompt assembler
                # enforces it and splits anything over, so it's the backstop, not our target.
                # Soft target sits below it with headroom (a full request never trips the
                # ceiling) and is small enough that several requests stay in flight across
                # the LLM lanes instead of collapsing into one giant call that kills overlap.
                hard_in = DeepseekTranslator._MAX_TOKENS_IN
                soft_tokens = max(2000, min(12000, int(hard_in * 0.75)))
            except Exception:
                adaptive = False
        ADAPTIVE_FIRST_PAGES = 3   # start-fast trigger only (NOT a size cap): the first request
                                   # fires after this many pages even if still under the token
                                   # target, so a sparse gallery still starts the pipeline promptly
        ADAPTIVE_TAIL_PAGES = 4    # within this many pages of the end, lower the token target so
                                   # the final request is small (cuts the pure-network tail)
        ADAPTIVE_MAX_PAGES = cap   # HARD page ceiling per request — required for correctness, not
                                   # just pacing. The sequencer takes one `inflight` permit per page
                                   # BEFORE a batch flushes; a token-only batch on a sparse gallery
                                   # (little text, target never reached) would keep accumulating and
                                   # hold every permit, so the sequencer blocks on inflight.acquire()
                                   # with no batch in flight to drain it → deadlock (galleries hung
                                   # mid-run). Since inflight = max(cap*3, 8) > cap, flushing by
                                   # cap guarantees the batch releases before it can starve inflight.
                                   # It also keeps requests small → steady render progress, no long
                                   # silent wait while one giant request runs. Dense pages still
                                   # split smaller via the token target below.

        def _est_page_tokens(ctx) -> int:
            """Rough DeepSeek input-token estimate for one page's OCR text (per the DeepSeek
            docs: ~0.6 tokens/CJK char, ~0.3/Latin char — padded, plus marker overhead)."""
            total = 0
            for r in (getattr(ctx, 'text_regions', None) or []):
                t = getattr(r, 'text', '') or ''
                cjk = sum(1 for ch in t if ord(ch) > 0x2E7F)
                total += int(cjk * 0.7 + (len(t) - cjk) * 0.35) + 6
            return total

        study_mode = str(getattr(config, 'study_mode_generation', 'disabled') or 'disabled')
        # Run-attribution header: without it a profiler summary can't be tied to a configuration
        # after the fact (translator, cap and study mode are what move the numbers).
        logger.info(
            f'Gallery run: pages={n} batch_size={cap}{" (adaptive)" if adaptive else ""} '
            f'translator={config.translator.translator} '
            f'target={config.translator.target_lang} renderer={config.render.renderer.value} '
            f'study_mode_generation={study_mode} job_token={(job_token[:8] + "…") if job_token else "-"}')

        is_ctx_mode = self.context_size > 0 and config.translator.translator in (
            Translator.chatgpt, Translator.chatgpt_2stage)
        # Offline translators run on the GPU themselves — their translation step must
        # hold the GPU lock instead of overlapping with detection/inpainting.
        try:
            is_offline_tl = any(key in OFFLINE_TRANSLATORS for key, _ in config.translator.translator_gen.chain)
        except Exception:
            is_offline_tl = False

        saved_pages, saved_originals = self.all_page_translations, self._original_page_texts
        self.all_page_translations, self._original_page_texts = [], []
        prev_concurrent, self.batch_concurrent = self.batch_concurrent, False
        self._gallery_cancel = False
        self._gallery_job_token = job_token   # stamped onto every status-5 page frame + checked by /cancel_gallery

        self._stage_times = {}                # per-stage compute accumulator (filled by stage methods)
        prewarm_proc_pool()                   # mask-refinement workers spawn while models load
        prof = Profiler(interval=1.0)
        prof.start()
        _sub0 = snapshot_substages()          # host/kernel sub-splits reported by the models
        reset_llm_usage()                     # per-chunk LLM request/token/cost accounting (one chunk at a time)

        # All CUDA runs on the single GPU worker thread (submit_gpu) → kernels serialize there, so no
        # asyncio lock is needed for GPU ordering. Per-page context is carried on ctx (ctx.image_context);
        # the parallel stages never read the mutable self._current_image_context, so the render/study
        # workers run concurrently without racing. The pipeline is staged: producer (decode+detect+OCR)
        # → run_batch (translate) → inpaint_stage (GPU inpaint) → render_worker×N (render+study+emit).
        llm = asyncio.Semaphore(1 if (is_ctx_mode or is_offline_tl) else 6)
        # Bound pages alive between preprocess and emit — each holds full-res buffers.
        inflight = asyncio.Semaphore(max(cap * 3, 8))
        post_q: asyncio.Queue = asyncio.Queue()       # S2→S3: translated batches → inpaint
        render_q: asyncio.Queue = asyncio.Queue()     # S3→S4/S5: inpainted pages → render/study workers
        NUM_RENDER = max(2, min(4, (os.cpu_count() or 4) // 2))
        failed = set()
        emitted = 0
        study_meta_pages = 0    # pages that emitted a metadata-only status-6 frame (text_only)
        study_image_pages = 0   # pages that emitted a full image-layer status-6 frame
        bubbles_total = 0       # bubbles across all emitted study frames
        was_cancelled = False

        def _record_page_context(ctx):
            if not ctx.text_regions:
                return
            self.all_page_translations.append({
                (r.text_raw if hasattr(r, 'text_raw') else r.text): r.translation
                for r in ctx.text_regions})
            self._original_page_texts.append({
                k: (r.text_raw if hasattr(r, 'text_raw') else r.text)
                for k, r in enumerate(ctx.text_regions)})

        tl_pages_done = 0   # cumulative pages whose translation finished (for page-based progress)

        async def run_batch(pages_before: int, batch: list, prev_task):
            # gallery-tl / gallery-tl-done progress is in PAGE units (a/n), not batch
            # indices: adaptive batching makes the batch count unknowable up front, and
            # pages are the unit every consumer (scheduler adapter, client label/percent)
            # actually wants. `pages_before` is the page count of all earlier batches.
            nonlocal tl_pages_done
            try:
                if prev_task is not None:  # context mode: keep page order across batches
                    await prev_task
                async with llm:
                    if self._gallery_cancel:
                        raise asyncio.CancelledError()
                    await self._report_progress(f'gallery-tl:{pages_before + len(batch)}/{n}')
                    pairs = [(ctx, config) for _, ctx in batch]
                    _tl_t0 = time.perf_counter()
                    if is_offline_tl:
                        # Offline translators run on the GPU — keep them on the GPU worker thread so
                        # they serialize with detection/inpainting instead of racing for the device.
                        await submit_gpu(self._batch_translate_contexts(pairs, len(pairs)))
                    else:
                        await self._batch_translate_contexts(pairs, len(pairs))
                    self._accum_time('translation', time.perf_counter() - _tl_t0)
                for _, ctx in batch:
                    _record_page_context(ctx)
                tl_pages_done += len(batch)
                await self._report_progress(f'gallery-tl-done:{tl_pages_done}/{n}')
                await post_q.put(batch)
            except asyncio.CancelledError:
                await post_q.put([(idx, None) for idx, _ in batch])
            except Exception as e:
                logger.error(f'Gallery translation batch (pages {batch[0][0] + 1}-{batch[-1][0] + 1}) failed: {e}')
                await post_q.put([(idx, None) for idx, _ in batch])

        async def inpaint_stage():
            """S3: pull each translated batch, inpaint every page on the GPU thread, and hand the
            pages to the render/study workers. A single instance — the GPU serializes anyway — but
            it never blocks on rendering, so the GPU keeps moving to the next page's inpaint."""
            while True:
                _t = time.perf_counter()
                batch = await post_q.get()
                prof.add_wait('post_q', time.perf_counter() - _t)
                if batch is None:
                    for _ in range(NUM_RENDER):
                        await render_q.put(None)   # fan-out shutdown to the render workers
                    return
                # Prefetch mask refinement (heavy CPU: watershed/CC per page) for the whole
                # batch concurrently on the CPU pool, so the GPU inpaints page k while page
                # k+1's mask is still being refined instead of waiting ~1s per page for it.
                async def _prefetch_mask(ctx):
                    try:
                        if ctx is not None and not self._gallery_cancel and ctx.text_regions and ctx.mask is None:
                            ctx.mask = await self._run_mask_refinement(config, ctx)
                    except Exception:
                        pass   # left None — _inpaint_stage retries and owns the error path
                prefetch = [asyncio.create_task(_prefetch_mask(ctx)) for _, ctx in batch]
                for (idx, ctx), pf in zip(batch, prefetch):
                    try:
                        await pf
                        if ctx is not None and not self._gallery_cancel and ctx.text_regions:
                            ctx = await self._inpaint_stage(ctx, config)
                        elif ctx is not None and not self._gallery_cancel:
                            # Text-less page: no OCR regions, or every region was filtered as
                            # low-value (sfx/decorative). There's nothing to translate or inpaint,
                            # but it must still emit its ORIGINAL image — otherwise the render
                            # worker sees result=None and marks a perfectly good art page "failed",
                            # which also leaves it unstored so every later Translate re-runs
                            # detection/OCR on it only to fail again. (The single-image path does
                            # this inside _inpaint_stage; the gallery path skips that call here.)
                            ctx.result = ctx.upscaled
                            ctx._skip_render = True
                            ctx = await self._revert_upscale(config, ctx)
                    except Exception as e:
                        logger.error(f'Gallery page {idx + 1} inpainting failed: {e}')
                    await render_q.put((idx, ctx))

        async def render_worker():
            """S4+S5: render + final compose + study overlay + emit. NUM_RENDER of these run in
            parallel on the CPU pool, so several pages typeset/encode at once while the GPU works
            ahead on later pages. Pages may emit out of order — each frame carries its page index."""
            nonlocal emitted, study_meta_pages, study_image_pages, bubbles_total
            while True:
                _t = time.perf_counter()
                item = await render_q.get()
                prof.add_wait('render_q', time.perf_counter() - _t)
                if item is None:
                    return
                idx, ctx = item
                try:
                    if ctx is None or self._gallery_cancel:
                        failed.add(idx)
                        continue
                    if ctx.text_regions and not getattr(ctx, '_skip_render', False):
                        ctx = await self._render_stage(ctx, config)
                    result = getattr(ctx, 'result', None)
                    if result is not None:
                        await self._emit_page_result(idx, result)
                        # Per-page study payload (metadata and/or layers, per study_mode_generation)
                        # — best-effort: a failure never fails the page.
                        if study_mode != 'disabled':
                            try:
                                _study_t0 = time.perf_counter()
                                study = await self._build_bubble_overlays(ctx, config, study_mode)
                                self._accum_time('study_overlay', time.perf_counter() - _study_t0)
                                if study:
                                    await self._emit_page_bubbles(idx, study)
                                    if study.get('bg') is not None:
                                        study_image_pages += 1
                                    else:
                                        study_meta_pages += 1
                                    bubbles_total += len(study.get('bubbles') or [])
                            except Exception as e:
                                logger.error(f'Gallery page {idx + 1} study overlay build failed: {e}')
                        emitted += 1
                    else:
                        failed.add(idx)
                except Exception as e:
                    logger.error(f'Gallery page {idx + 1} post-processing failed: {e}')
                    failed.add(idx)
                finally:
                    images[idx] = None  # free page memory as we go
                    if ctx is not None:
                        for attr in ('input', 'img_colorized', 'upscaled', 'img_rgb', 'img_alpha',
                                     'mask_raw', 'mask', 'img_inpainted', 'gimp_mask', 'img_rendered',
                                     'result', 'textlines', 'text_regions'):
                            setattr(ctx, attr, None)
                    inflight.release()

        inpaint_task = asyncio.create_task(inpaint_stage())
        render_tasks = [asyncio.create_task(render_worker()) for _ in range(NUM_RENDER)]
        tl_tasks = []
        pre_tasks = []

        # ── S1: preprocess (decode + detect + OCR), PRE_WORKERS pages at a time ─────────
        # A page's host work (decode, bilateral filter, warps, box extraction — all on the
        # CPU pool since the models orchestrate their own placement) overlaps the previous
        # page's GPU kernels, keeping the GPU thread fed. Batches must still form in strict
        # page order (context mode, deterministic batching), so a sequencer consumes results
        # in order; pre_window bounds how far the workers run ahead of it.
        PRE_WORKERS = 3
        pre_results: dict[int, object] = {}
        pre_done = asyncio.Condition()
        pre_window = asyncio.Semaphore(PRE_WORKERS + 2)
        first_done = asyncio.Event()   # page 0 loads the models alone; later pages wait for it
        next_idx = 0
        pre_count = 0
        _CANCELLED = object()

        def _decode_page(i):
            raw = images[i]
            if isinstance(raw, (bytes, bytearray)):
                img = Image.open(_io.BytesIO(raw))
                img.load()   # force the actual pixel decode here, on the pool
                md5 = hashlib.md5(raw).hexdigest()[:8]
            else:
                img = raw
                md5 = f'{id(raw) & 0xffffffff:08x}'
            return img, md5

        async def preprocess_one(i):
            _dec_t0 = time.perf_counter()
            img, md5 = await run_cpu(_decode_page, i)
            self._accum_time('decode', time.perf_counter() - _dec_t0)
            # Per-page image context built locally — hashing the raw upload bytes instead of
            # re-encoding the decoded page to PNG (which cost a full page encode per page).
            # Only debug folder naming and the rendering_folder progress message consume it.
            ic = {
                'subfolder': f"{int(time.time() * 1000)}-{md5}-{config.detector.detection_size}-{config.translator.target_lang}-{config.translator.translator}",
                'file_md5': md5,
                'config': config,
            }
            self._current_image_context = ic
            if self.verbose:
                self._saved_image_contexts[md5] = dict(ic)
            # ic rides on the ctx from the start: 3 pre-workers run concurrently, so the verbose
            # stage dumps inside must never resolve through self._current_image_context.
            ctx = await self._translate_until_translation(img, config, image_context=ic)
            ctx.image_context = dict(ic)
            ctx._gallery = True
            ctx.verbose = self.verbose
            return ctx

        async def pre_worker():
            nonlocal next_idx, pre_count
            while True:
                if next_idx >= n:
                    return
                i = next_idx
                next_idx += 1
                await pre_window.acquire()
                if i > 0:
                    await first_done.wait()
                ctx = _CANCELLED
                try:
                    if not self._gallery_cancel:
                        try:
                            ctx = await preprocess_one(i)
                        except Exception as e:
                            logger.error(f'Gallery page {i + 1}/{n} pre-processing failed: {e}')
                            ctx = Context()
                            ctx.input = None
                            ctx.text_regions = []
                            ctx.result = None
                finally:
                    if i == 0:
                        first_done.set()
                if ctx is not _CANCELLED:
                    pre_count += 1
                    await self._report_progress(f'gallery-pre:{pre_count}/{n}')
                async with pre_done:
                    pre_results[i] = ctx
                    pre_done.notify_all()

        try:
            pre_tasks = [asyncio.create_task(pre_worker()) for _ in range(PRE_WORKERS)]
            batch = []
            batch_tokens = 0
            pages_batched = 0
            prev_task = None
            cancelled_at = None
            for i in range(n):
                async with pre_done:
                    while i not in pre_results:
                        await pre_done.wait()
                    ctx = pre_results.pop(i)
                pre_window.release()
                if ctx is _CANCELLED or self._gallery_cancel:
                    cancelled_at = i
                    break
                await inflight.acquire()   # bounds pages alive between here and emit
                batch.append((i, ctx))
                if adaptive:
                    batch_tokens += _est_page_tokens(ctx)
                    pages_left = n - (i + 1)   # gallery pages still to come after this one
                    if i == n - 1:
                        flush = True                        # end of gallery: flush the remainder
                    elif not tl_tasks:
                        # First request: fire on a fraction of the token target OR a few pages,
                        # whichever comes first, so translation (and the inpaint/render stages
                        # behind it) start early on both dense and sparse galleries.
                        flush = batch_tokens >= soft_tokens // 3 or len(batch) >= ADAPTIVE_FIRST_PAGES
                    else:
                        # Steady state: flush at the page ceiling (correctness + steady progress),
                        # or when the batch reaches the soft token target (dense pages split
                        # smaller), lowering the target near the end so the final request stays
                        # small (trims the pure-network tail). If every prior request has already
                        # returned, the LLM lanes are idle and OCR is pacing — flush a bit early to
                        # refill a lane, but never a trivially small request.
                        target = (soft_tokens // 3) if pages_left <= ADAPTIVE_TAIL_PAGES else soft_tokens
                        lanes_idle = all(t.done() for t in tl_tasks)
                        flush = (len(batch) >= ADAPTIVE_MAX_PAGES
                                 or batch_tokens >= target
                                 or (lanes_idle and batch_tokens >= soft_tokens // 3))
                else:
                    flush = len(batch) >= cap or i == n - 1
                if flush:
                    task = asyncio.create_task(run_batch(pages_batched, batch, prev_task if is_ctx_mode else None))
                    tl_tasks.append(task)
                    prev_task = task
                    pages_batched += len(batch)
                    batch = []
                    batch_tokens = 0
            if cancelled_at is not None:
                for idx, c in batch:
                    failed.add(idx)
                    inflight.release()     # these never reach a render worker
                failed.update(range(cancelled_at, n))
                batch = []
            for t in pre_tasks:
                t.cancel()                 # no-ops when already finished
            await asyncio.gather(*pre_tasks, return_exceptions=True)
            if tl_tasks:
                await asyncio.gather(*tl_tasks)
            await post_q.put(None)        # drain: inpaint_stage finishes, then fans None out to the render workers
            await inpaint_task
            await asyncio.gather(*render_tasks)
        except BaseException:
            for t in pre_tasks:
                t.cancel()
            for t in tl_tasks:
                t.cancel()
            inpaint_task.cancel()
            for t in render_tasks:
                t.cancel()
            raise
        finally:
            prof.stop()
            was_cancelled = self._gallery_cancel
            self.batch_concurrent = prev_concurrent
            self.all_page_translations = saved_pages
            self._original_page_texts = saved_originals
            self._gallery_cancel = False
            self._gallery_job_token = ""
            self._saved_image_contexts.clear()

        failed_list = sorted(failed)
        # Fold the models' host/kernel sub-splits into the stage summary (det_pre/gpu/post,
        # ocr_pre/gpu/post, inp_pre/gpu/post, mask_refine) — the split the wall numbers hide.
        for k, v in snapshot_substages().items():
            dv = v - _sub0.get(k, 0.0)
            if dv > 0.05:
                self._stage_times[k] = dv
        # This method runs once per SCHEDULER CHUNK, so these two summaries are chunk-scoped,
        # not whole-gallery — they're logged at DEBUG to avoid masquerading as a gallery result.
        # The single INFO-level gallery report is emitted by the job scheduler (gallery_jobs),
        # which folds the `telemetry` below across every chunk of the same job_token.
        # LLM request/token accounting for this chunk (empty for non-LLM translators).
        _llm = snapshot_llm_usage()
        llm_block = {}
        if _llm.get('requests'):
            from .translators.deepseek import estimate_cost_usd
            reqs = int(_llm.get('requests', 0))
            in_tok = int(_llm.get('prompt_tokens', 0))
            out_tok = int(_llm.get('completion_tokens', 0))
            hit = int(_llm.get('cache_hit', 0))
            miss = int(_llm.get('cache_miss', 0))
            llm_block = {
                'requests': reqs, 'in': in_tok, 'out': out_tok,
                'cache_hit': hit, 'cache_miss': miss,
                'max_wall': round(float(_llm.get('max_wall', 0.0)), 1),
                'sum_wall': round(float(_llm.get('sum_wall', 0.0)), 1),
                'cost': round(estimate_cost_usd(hit, miss, out_tok), 4),
            }

        # This method runs once per SCHEDULER CHUNK, so these two summaries are chunk-scoped,
        # not whole-gallery — they're logged at DEBUG to avoid masquerading as a gallery result.
        # The single INFO-level gallery report is emitted by the job scheduler (gallery_jobs),
        # which folds the `telemetry` below across every chunk of the same job_token.
        logger.debug('Gallery chunk ' + prof.summary(self._stage_times, n, emitted))
        if llm_block:
            logger.debug(
                f'Gallery chunk LLM: {llm_block["requests"]} requests, '
                f'in={llm_block["in"]} (cache hit={llm_block["cache_hit"]}) out={llm_block["out"]} tok, '
                f'slowest={llm_block["max_wall"]}s, ~${llm_block["cost"]:.4f} est')
        logger.debug(
            f'Gallery chunk completed: {emitted}/{n} pages emitted, {len(failed_list)} failed, '
            f'study_mode={study_mode} study_pages(meta={study_meta_pages}, image={study_image_pages}) '
            f'bubbles={bubbles_total}'
            + (' — CANCELLED (client cancel or liveness reaper)' if was_cancelled else ''))

        # Telemetry the scheduler folds into the one gallery-level summary.
        def _avg(xs):
            return (sum(xs) / len(xs)) if xs else 0.0
        _wall = (time.perf_counter() - prof.t0) if prof.t0 else 0.0
        telemetry = {
            'wall': round(_wall, 2),
            'emitted': emitted,
            'failed': len(failed_list),
            'study_meta': study_meta_pages,
            'study_image': study_image_pages,
            'bubbles': bubbles_total,
            'cancelled': bool(was_cancelled),
            'llm': llm_block,
            'stage_times': {k: round(v, 2) for k, v in self._stage_times.items()},
            'queue_wait': {k: round(v, 2) for k, v in prof.queue_wait.items()},
            'gpu_avg': round(_avg(prof.gpu), 1), 'gpu_max': round(max(prof.gpu), 1) if prof.gpu else 0.0,
            'vram_max': round(max(prof.vram_used)) if prof.vram_used else 0,
            'vram_total': round(prof.vram_total),
            'cpu_avg': round(_avg(prof.cpu), 1), 'cpu_max': round(max(prof.cpu), 1) if prof.cpu else 0.0,
        }
        return {"count": n, "failed": failed_list, "telemetry": telemetry}

    async def _translate_until_translation(self, image: Image.Image, config: Config, image_context: dict = None) -> Context:
        """
        执行翻译之前的所有步骤（彩色化、上采样、检测、OCR、文本行合并）

        `image_context` is this page's own debug-folder context. The gallery pipeline runs several
        of these concurrently, so every verbose dump below must resolve its folder from the ctx,
        never from the shared self._current_image_context.
        """
        ctx = Context()
        ctx.input = image
        ctx.result = None
        if image_context:
            ctx.image_context = dict(image_context)

        # 保存原始输入图片用于调试
        if self.verbose:
            try:
                input_img = np.array(image)
                if len(input_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                result_path = self._result_path('input.png', ctx)
                success = cv2.imwrite(result_path, input_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {result_path}")
            except Exception as e:
                logger.error(f"Error saving input.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if ( self.models_ttl == 0 ):
            logger.info('Loading models')
            if config.upscale.upscale_ratio:
                await prepare_upscaling(config.upscale.upscaler)
            await prepare_detection(config.detector.detector)
            await prepare_ocr(config.ocr.ocr, self.device)
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            await prepare_translation(config.translator.translator_gen)
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)

        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())

        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input
        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized
        else:
            ctx.upscaled = ctx.img_colorized

        # PIL→numpy of a full page is tens of ms — keep it off the orchestrating loop.
        ctx.img_rgb, ctx.img_alpha = await run_cpu(load_image, ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:
            logger.error(f"Error during detection:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.textlines = []
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
            cv2.imwrite(self._result_path('mask_raw.png', ctx), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png', ctx), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        # -- OCR
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as e:
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.textlines = []

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:  
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = []

        if self.verbose and ctx.text_regions:
            show_panels = not config.force_simple_sort  # 当不使用简单排序时显示panel
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions,
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl)
            cv2.imwrite(self._result_path('bboxes.png', ctx), bboxes)

        # Apply pre-dictionary after textline merge
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")

        # 保存当前图片上下文到ctx中，用于并发翻译时的路径管理
        # (only as a fallback — never overwrite the per-page context set at ctx creation,
        # self._current_image_context may belong to a different in-flight page by now)
        if self._current_image_context and not getattr(ctx, 'image_context', None):
            ctx.image_context = self._current_image_context.copy()

        return ctx

    async def _batch_translate_contexts(self, contexts_with_configs: List[tuple], batch_size: int) -> List[tuple]:
        """
        批量处理翻译步骤，防止内存溢出
        """
        results = []
        total_contexts = len(contexts_with_configs)
        
        # 按批次处理，防止内存溢出
        for i in range(0, total_contexts, batch_size):
            batch = contexts_with_configs[i:i + batch_size]
            logger.info(f'Processing translation batch {i//batch_size + 1}/{(total_contexts + batch_size - 1)//batch_size}')
            
            # 收集当前批次的所有文本
            all_texts = []
            batch_text_mapping = []  # 记录每个文本属于哪个context和region
            
            for ctx_idx, (ctx, config) in enumerate(batch):
                if not ctx.text_regions:
                    continue
                    
                region_start_idx = len(all_texts)
                for region_idx, region in enumerate(ctx.text_regions):
                    all_texts.append(region.text)
                    batch_text_mapping.append((ctx_idx, region_idx))
                
            if not all_texts:
                # 当前批次没有需要翻译的文本
                results.extend(batch)
                continue
                
            # 批量翻译
            try:
                await self._report_progress('translating')
                # 使用第一个配置进行翻译（假设批次内配置相同）
                sample_config = batch[0][1] if batch else None
                if sample_config:
                    # 支持批量翻译 - 传递所有批次上下文
                    batch_contexts = [ctx for ctx, config in batch]
                    translated_texts = await self._batch_translate_texts(all_texts, sample_config, batch[0][0], batch_contexts)
                else:
                    translated_texts = all_texts  # 无法翻译时保持原文
                    
                # 将翻译结果分配回各个context
                text_idx = 0
                for ctx_idx, (ctx, config) in enumerate(batch):
                    if not ctx.text_regions:  # 检查text_regions是否为None或空
                        continue
                    for region_idx, region in enumerate(ctx.text_regions):
                        if text_idx < len(translated_texts):
                            region.translation = translated_texts[text_idx]
                            region.target_lang = config.translator.target_lang
                            region._alignment = config.render.alignment
                            region._direction = config.render.direction
                            text_idx += 1
                        
                # 应用后处理逻辑（括号修正、过滤等）
                for ctx, config in batch:
                    if ctx.text_regions:
                        ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                        
                # 批次级别的目标语言检查
                if batch and batch[0][1].translator.enable_post_translation_check:
                    # 收集批次内所有页面的filtered regions
                    all_batch_regions = []
                    for ctx, config in batch:
                        if ctx.text_regions:
                            all_batch_regions.extend(ctx.text_regions)
                    
                    # 进行批次级别的目标语言检查
                    batch_lang_check_result = True
                    if all_batch_regions and len(all_batch_regions) > 10:
                        sample_config = batch[0][1]
                        logger.info(f"Starting batch-level target language check with {len(all_batch_regions)} regions...")
                        batch_lang_check_result = await self._check_target_language_ratio(
                            all_batch_regions,
                            sample_config.translator.target_lang,
                            min_ratio=0.5
                        )
                        
                        if not batch_lang_check_result:
                            logger.warning("Batch-level target language ratio check failed")
                            
                            # 批次重新翻译逻辑
                            max_batch_retry = sample_config.translator.post_check_max_retry_attempts
                            batch_retry_count = 0
                            
                            while batch_retry_count < max_batch_retry and not batch_lang_check_result:
                                batch_retry_count += 1
                                logger.warning(f"Starting batch retry {batch_retry_count}/{max_batch_retry}")
                                
                                # 重新翻译批次内所有区域
                                all_original_texts = []
                                region_mapping = []  # 记录每个text属于哪个ctx
                                
                                for ctx_idx, (ctx, config) in enumerate(batch):
                                    if ctx.text_regions:
                                        for region in ctx.text_regions:
                                            if hasattr(region, 'text') and region.text:
                                                all_original_texts.append(region.text)
                                                region_mapping.append((ctx_idx, region))
                                
                                if all_original_texts:
                                    try:
                                        # 重新批量翻译
                                        logger.info(f"Retrying translation for {len(all_original_texts)} regions...")
                                        new_translations = await self._batch_translate_texts(all_original_texts, sample_config, batch[0][0])
                                        
                                        # 更新翻译结果到各个region
                                        for i, (ctx_idx, region) in enumerate(region_mapping):
                                            if i < len(new_translations) and new_translations[i]:
                                                old_translation = region.translation
                                                region.translation = new_translations[i]
                                                logger.debug(f"Region {i+1} translation updated: '{old_translation}' -> '{new_translations[i]}'")
                                        
                                        # 重新收集所有regions并检查目标语言比例
                                        all_batch_regions = []
                                        for ctx, config in batch:
                                            if ctx.text_regions:
                                                all_batch_regions.extend(ctx.text_regions)
                                        
                                        logger.info(f"Re-checking batch-level target language ratio after batch retry {batch_retry_count}...")
                                        batch_lang_check_result = await self._check_target_language_ratio(
                                            all_batch_regions,
                                            sample_config.translator.target_lang,
                                            min_ratio=0.5
                                        )
                                        
                                        if batch_lang_check_result:
                                            logger.info(f"Batch-level target language check passed")
                                            break
                                        else:
                                            logger.warning(f"Batch-level target language check still failed")
                                            
                                    except Exception as e:
                                        logger.error(f"Error during batch retry {batch_retry_count}: {e}")
                                        break
                                else:
                                    logger.warning("No text found for batch retry")
                                    break
                            
                            if not batch_lang_check_result:
                                logger.error(f"Batch-level target language check failed after all {max_batch_retry} batch retries")
                    else:
                        logger.info(f"Skipping batch-level target language check: only {len(all_batch_regions)} regions (threshold: 10)")
                    
                    # 统一的成功信息
                    if batch_lang_check_result:
                        logger.info("All translation regions passed post-translation check.")
                    else:
                        logger.warning("Some translation regions failed post-translation check.")
                        
                # 过滤逻辑（简化版本，保留主要过滤条件）
                for ctx, config in batch:
                    if ctx.text_regions:
                        new_text_regions = []
                        for region in ctx.text_regions:
                            should_filter = False
                            filter_reason = ""

                            if not region.translation.strip():
                                should_filter = True
                                filter_reason = "Translation contain blank areas"
                            elif config.translator.translator != Translator.none:
                                if region.translation.isnumeric():
                                    should_filter = True
                                    filter_reason = "Numeric translation"
                                elif config.filter_text and re.search(config.re_filter_text, region.translation):
                                    should_filter = True
                                    filter_reason = f"Matched filter text: {config.filter_text}"
                                elif not config.translator.translator == Translator.original:
                                    text_equal = region.text.lower().strip() == region.translation.lower().strip()
                                    if text_equal:
                                        should_filter = True
                                        filter_reason = "Translation identical to original"

                            if should_filter:
                                if region.translation.strip():
                                    logger.info(f'Filtered out: {region.translation}')
                                    logger.info(f'Reason: {filter_reason}')
                            else:
                                new_text_regions.append(region)
                        ctx.text_regions = new_text_regions
                        
                results.extend(batch)
                
            except Exception as e:
                logger.error(f"Error in batch translation: {e}")
                if not self.ignore_errors:
                    raise
                # 错误时保持原文
                for ctx, config in batch:
                    if not ctx.text_regions:  # 检查text_regions是否为None或空
                        continue
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                results.extend(batch)
                
            # 强制垃圾回收以释放内存
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results

    async def _concurrent_translate_contexts(self, contexts_with_configs: List[tuple]) -> List[tuple]:
        """
        并发处理翻译步骤，为每个图片单独发送翻译请求，避免合并大批次
        """

        # 在并发模式下，先保存所有页面的原文用于上下文
        batch_original_texts = []  # 存储当前批次的原文
        if self.context_size > 0:
            for i, (ctx, config) in enumerate(contexts_with_configs):
                if ctx.text_regions:
                    # 保存当前页面的原文
                    page_texts = {}
                    for j, region in enumerate(ctx.text_regions):
                        page_texts[j] = region.text
                    batch_original_texts.append(page_texts)

                    # 确保 _original_page_texts 有足够的长度
                    while len(self._original_page_texts) <= len(self.all_page_translations) + i:
                        self._original_page_texts.append({})

                    self._original_page_texts[len(self.all_page_translations) + i] = page_texts
                else:
                    batch_original_texts.append({})

        async def translate_single_context(ctx_config_pair_with_index):
            """翻译单个context的异步函数"""
            ctx, config, page_index, batch_index = ctx_config_pair_with_index
            try:
                if not ctx.text_regions:
                    return ctx, config

                # 收集该context的所有文本
                texts = [region.text for region in ctx.text_regions]

                if not texts:
                    return ctx, config

                logger.debug(f'Translating {len(texts)} regions for single image in concurrent mode (page {page_index}, batch {batch_index})')

                # 单独翻译这一张图片的文本，传递页面索引和批次索引用于正确的上下文
                translated_texts = await self._batch_translate_texts(
                    texts, config, ctx,
                    page_index=page_index,
                    batch_index=batch_index,
                    batch_original_texts=batch_original_texts
                )

                # 将翻译结果分配回各个region
                for i, region in enumerate(ctx.text_regions):
                    if i < len(translated_texts):
                        region.translation = translated_texts[i]
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                
                # 应用后处理逻辑（括号修正、过滤等）
                if ctx.text_regions:
                    ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                
                # 单页目标语言检查（如果启用）
                if config.translator.enable_post_translation_check and ctx.text_regions:
                    page_lang_check_result = await self._check_target_language_ratio(
                        ctx.text_regions,
                        config.translator.target_lang,
                        min_ratio=0.3  # 对单页使用更宽松的阈值
                    )
                    
                    if not page_lang_check_result:
                        logger.warning(f"Page-level target language check failed for single image")
                        
                        # 单页重试逻辑
                        max_retry = config.translator.post_check_max_retry_attempts
                        retry_count = 0
                        
                        while retry_count < max_retry and not page_lang_check_result:
                            retry_count += 1
                            logger.info(f"Retrying single image translation {retry_count}/{max_retry}")
                            
                            # 重新翻译
                            original_texts = [region.text for region in ctx.text_regions if hasattr(region, 'text') and region.text]
                            if original_texts:
                                try:
                                    new_translations = await self._batch_translate_texts(original_texts, config, ctx)
                                    
                                    # 更新翻译结果
                                    text_idx = 0
                                    for region in ctx.text_regions:
                                        if hasattr(region, 'text') and region.text and text_idx < len(new_translations):
                                            old_translation = region.translation
                                            region.translation = new_translations[text_idx]
                                            logger.debug(f"Region translation updated: '{old_translation}' -> '{new_translations[text_idx]}'")
                                            text_idx += 1
                                    
                                    # 重新检查
                                    page_lang_check_result = await self._check_target_language_ratio(
                                        ctx.text_regions,
                                        config.translator.target_lang,
                                        min_ratio=0.3
                                    )
                                    
                                    if page_lang_check_result:
                                        logger.info(f"Single image target language check passed after retry {retry_count}")
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"Error during single image retry {retry_count}: {e}")
                                    break
                            else:
                                break
                        
                        if not page_lang_check_result:
                            logger.warning(f"Single image target language check failed after all {max_retry} retries")
                
                # 过滤逻辑
                if ctx.text_regions:
                    new_text_regions = []
                    for region in ctx.text_regions:
                        should_filter = False
                        filter_reason = ""

                        if not region.translation.strip():
                            should_filter = True
                            filter_reason = "Translation contain blank areas"
                        elif config.translator.translator != Translator.none:
                            if region.translation.isnumeric():
                                should_filter = True
                                filter_reason = "Numeric translation"
                            elif config.filter_text and re.search(config.re_filter_text, region.translation):
                                should_filter = True
                                filter_reason = f"Matched filter text: {config.filter_text}"
                            elif not config.translator.translator == Translator.original:
                                text_equal = region.text.lower().strip() == region.translation.lower().strip()
                                if text_equal:
                                    should_filter = True
                                    filter_reason = "Translation identical to original"

                        if should_filter:
                            if region.translation.strip():
                                logger.info(f'Filtered out: {region.translation}')
                                logger.info(f'Reason: {filter_reason}')
                        else:
                            new_text_regions.append(region)
                    ctx.text_regions = new_text_regions
                
                return ctx, config
                
            except Exception as e:
                logger.error(f"Error in concurrent translation for single image: {e}")
                if not self.ignore_errors:
                    raise
                # 错误时保持原文
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                return ctx, config
        
        # 创建并发任务，为每个任务添加页面索引和批次索引
        tasks = []
        for i, ctx_config_pair in enumerate(contexts_with_configs):
            # 计算当前页面在整个翻译序列中的索引
            page_index = len(self.all_page_translations) + i
            batch_index = i  # 在当前批次中的索引
            ctx_config_pair_with_index = (*ctx_config_pair, page_index, batch_index)
            task = asyncio.create_task(translate_single_context(ctx_config_pair_with_index))
            tasks.append(task)
        
        logger.info(f'Starting concurrent translation of {len(tasks)} images...')
        
        # 等待所有任务完成
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in concurrent translation gather: {e}")
            raise
        
        # 处理结果，检查是否有异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Image {i+1} concurrent translation failed: {result}")
                if not self.ignore_errors:
                    raise result
                # 创建失败的占位符
                ctx, config = contexts_with_configs[i]
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                final_results.append((ctx, config))
            else:
                final_results.append(result)
        
        logger.info(f'Concurrent translation completed: {len(final_results)} images processed')
        return final_results

    async def _batch_translate_texts(self, texts: List[str], config: Config, ctx: Context, batch_contexts: List[Context] = None, page_index: int = None, batch_index: int = None, batch_original_texts: List[dict] = None) -> List[str]:
        """
        批量翻译文本列表，使用现有的翻译器接口

        Args:
            texts: 要翻译的文本列表
            config: 配置对象
            ctx: 上下文对象
            batch_contexts: 批处理上下文列表
            page_index: 当前页面索引，用于并发模式下的上下文计算
            batch_index: 当前页面在批次中的索引
            batch_original_texts: 当前批次的原文数据
        """
        if config.translator.translator == Translator.none:
            return ["" for _ in texts]



        # 如果是ChatGPT翻译器（包括chatgpt和chatgpt_2stage），需要处理上下文
        if config.translator.translator in [Translator.chatgpt, Translator.chatgpt_2stage]:
            if config.translator.translator == Translator.chatgpt:
                from .translators.chatgpt import OpenAITranslator
                translator = OpenAITranslator()
            else:  # chatgpt_2stage
                from .translators.chatgpt_2stage import ChatGPT2StageTranslator
                translator = ChatGPT2StageTranslator()

            # 确定是否使用并发模式和原文上下文
            use_original_text = self.batch_concurrent and self.batch_size > 1

            done_pages = self.all_page_translations
            if self.context_size > 0 and done_pages:
                pages_expected = min(self.context_size, len(done_pages))
                non_empty_pages = [
                    page for page in done_pages
                    if any(sent.strip() for sent in page.values())
                ]
                pages_used = min(self.context_size, len(non_empty_pages))
                skipped = pages_expected - pages_used
            else:
                pages_used = skipped = 0

            if self.context_size > 0:
                context_type = "original text" if use_original_text else "translation results"
                logger.info(f"Context-aware translation enabled with {self.context_size} pages of history using {context_type}")

            translator.parse_args(config.translator)

            # 构建上下文 - 在并发模式下使用原文和页面索引
            prev_ctx = self._build_prev_context(
                use_original_text=use_original_text,
                current_page_index=page_index,
                batch_index=batch_index,
                batch_original_texts=batch_original_texts
            )
            translator.set_prev_context(prev_ctx)

            if pages_used > 0:
                context_count = prev_ctx.count("<|")
                logger.info(f"Carrying {pages_used} pages of context, {context_count} sentences as translation reference")
            if skipped > 0:
                logger.warning(f"Skipped {skipped} pages with no sentences")

            # ChatGPT2Stage需要特殊处理
            if config.translator.translator == Translator.chatgpt_2stage:
                # 为当前图片创建专用的result_path_callback，避免并发时路径错位
                current_image_context = getattr(ctx, 'image_context', None) or self._current_image_context

                def result_path_callback(path: str) -> str:
                    """为特定图片创建结果路径，使用保存的图片上下文"""
                    original_context = self._current_image_context
                    self._current_image_context = current_image_context
                    try:
                        return self._result_path(path)
                    finally:
                        self._current_image_context = original_context

                ctx.result_path_callback = result_path_callback

                # Check if batch processing is enabled and batch_contexts are provided
                if batch_contexts and len(batch_contexts) > 1 and not self.batch_concurrent:
                    # Enable batch processing for chatgpt_2stage
                    ctx.batch_contexts = batch_contexts
                    logger.info(f"Enabling batch processing for chatgpt_2stage with {len(batch_contexts)} images")

                    # Set result_path_callback for each context in the batch
                    for batch_ctx in batch_contexts:
                        if hasattr(batch_ctx, 'image_context'):
                            batch_image_context = batch_ctx.image_context
                        else:
                            batch_image_context = self._current_image_context

                        def create_result_path_callback(image_context):
                            def result_path_callback(path: str) -> str:
                                """为特定图片创建结果路径，使用保存的图片上下文"""
                                original_context = self._current_image_context
                                self._current_image_context = image_context
                                try:
                                    return self._result_path(path)
                                finally:
                                    self._current_image_context = original_context
                            return result_path_callback

                        batch_ctx.result_path_callback = create_result_path_callback(batch_image_context)

                # ChatGPT2Stage需要传递ctx参数
                return await translator._translate(
                    ctx.from_lang,
                    config.translator.target_lang,
                    texts,
                    ctx
                )
            else:
                # 普通ChatGPT不需要ctx参数
                return await translator._translate(
                    ctx.from_lang,
                    config.translator.target_lang,
                    texts
                )

        else:
            # 使用通用翻译调度器
            return await dispatch_translation(
                config.translator.translator_gen,
                texts,
                config.translator,
                self.use_mtpe,
                ctx,
                'cpu' if self._gpu_limited_memory else self.device
            )
            
    async def _apply_post_translation_processing(self, ctx: Context, config: Config) -> List:
        """
        应用翻译后处理逻辑（括号修正、过滤等）
        """
        # 检查text_regions是否为None或空
        if not ctx.text_regions:
            return []
            
        check_items = [
            # 圆括号处理
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],
            
            # 方括号处理
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],
            
            # 引号处理
            ["「", "“", "‘", "『", "【"],
            ["」", "”", "’", "』", "】"],
            ["『", "“", "‘", "「", "【"],
            ["』", "”", "’", "」", "】"],
            
            # 新增【】处理
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        replace_items = [
            ["「", "“"],
            ["「", "‘"],
            ["」", "”"],
            ["」", "’"],
            ["【", "["],  
            ["】", "]"],  
        ]

        for region in ctx.text_regions:
            if region.text and region.translation:
                # 引号处理逻辑
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text: 
                    quote_type = '【】'
                else:
                    quote_type = None
                
                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')
                    
                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):
                        
                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':  
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # 括号修正逻辑
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])
                    
                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # 强制替换规则
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

        # 注意：翻译结果的保存移动到了translate方法的最后，确保保存的是最终结果

        # 应用后字典
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:  
            original = region.translation  
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:  
            logger.info("Post-translation replacements:")  
            for replacement in post_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("No post-translation replacements made.")

        # 单个region幻觉检测
        failed_regions = []
        if config.translator.enable_post_translation_check:
            logger.info("Starting post-translation check...")
            
            # 单个region级别的幻觉检测
            for region in ctx.text_regions:
                if region.translation and region.translation.strip():
                    # 只检查重复内容幻觉
                    if await self._check_repetition_hallucination(
                        region.translation, 
                        config.translator.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)
            
            # 对失败的区域进行重试
            if failed_regions:
                logger.warning(f"Found {len(failed_regions)} regions that failed repetition check, starting retry...")
                for region in failed_regions:
                    try:
                        logger.info(f"Retrying translation for region with text: '{region.text}'")
                        new_translation = await self._retry_translation_with_validation(region, config, ctx)
                        if new_translation:
                            old_translation = region.translation
                            region.translation = new_translation
                            logger.info(f"Region retry successful: '{old_translation}' -> '{new_translation}'")
                        else:
                            logger.warning(f"Region retry failed, keeping original: '{region.translation}'")
                    except Exception as e:
                        logger.error(f"Error during region retry: {e}")

        return ctx.text_regions

    async def _complete_translation_pipeline(self, ctx: Context, config: Config) -> Context:
        """完成翻译后的处理步骤（掩码细化、修复、渲染）。

        Split into a GPU/per-image-context stage (`_inpaint_stage`) and a CPU stage
        (`_render_stage`) so the streaming gallery path can hold the GPU lock for the
        former while running the latter — rendering + final composition — outside the
        lock, overlapping it with the next page's detection/OCR. This wrapper preserves
        the original single-call behaviour for every other caller.
        """
        ctx = await self._inpaint_stage(ctx, config)
        if getattr(ctx, '_skip_render', False):
            return ctx
        return await self._render_stage(ctx, config)

    async def _inpaint_stage(self, ctx: Context, config: Config) -> Context:
        """GPU + per-image-context half of the post-translation pipeline: mask refinement
        and inpainting. On no-text / cancel it finalizes the result and sets `_skip_render`
        so the caller skips the render stage."""
        await self._report_progress('after-translating')

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            ctx._skip_render = True
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            ctx._skip_render = True
            return await self._revert_upscale(config, ctx)

        # -- Mask refinement
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:  
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0]

        if self.verbose and ctx.mask is not None:
            try:
                inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                              self.device, self.verbose)
                
                # 保存inpaint_input.png
                inpaint_input_path = self._result_path('inpaint_input.png', ctx)
                success1 = cv2.imwrite(inpaint_input_path, cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
                if not success1:
                    logger.warning(f"Failed to save debug image: {inpaint_input_path}")

                # 保存mask_final.png
                mask_final_path = self._result_path('mask_final.png', ctx)
                success2 = cv2.imwrite(mask_final_path, ctx.mask)
                if not success2:
                    logger.warning(f"Failed to save debug image: {mask_final_path}")
            except Exception as e:
                logger.error(f"Error saving debug images (inpaint_input.png, mask_final.png): {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # -- Inpainting
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)

        except Exception as e:  
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb
        if getattr(ctx, '_gallery', False):
            # Only the GIMP save format reads gimp_mask; the gallery path never does, and the
            # full-page dstack+cvtColor per page is pure loop-blocking waste there.
            ctx.gimp_mask = None
        else:
            ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png', ctx)
                success = cv2.imwrite(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))
                if not success:
                    logger.warning(f"Failed to save debug image: {inpainted_path}")
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        return ctx

    async def _render_stage(self, ctx: Context, config: Config) -> Context:
        """CPU half of the post-translation pipeline: text rendering + final composition.
        Reads only `ctx` (its per-page `image_context`), never the mutable
        `self._current_image_context`, so it is safe to run OUTSIDE the GPU lock and overlap
        with the next page's detection/OCR."""
        # -- Rendering
        await self._report_progress('rendering')

        # 在rendering状态后立即发送文件夹信息，用于前端精确检查final.png
        ic = getattr(ctx, 'image_context', None) or self._current_image_context
        if hasattr(self, '_progress_hooks') and ic:
            # 发送特殊格式的消息，前端可以解析
            await self._report_progress(f"rendering_folder:{ic['subfolder']}")

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted

        await self._report_progress('finished', True)
        ctx.result = await run_cpu(dump_image, ctx.input, ctx.img_rendered, ctx.img_alpha)

        # 保存debug文件夹信息到Context中（用于Web模式的缓存访问）
        if self.verbose:
            ctx.debug_folder = ic['subfolder'] if ic else ''

        return await self._revert_upscale(config, ctx)
    
    async def _check_repetition_hallucination(self, text: str, threshold: int = 5, silent: bool = False) -> bool:
        """
        检查文本是否包含重复内容（模型幻觉）
        Check if the text contains repetitive content (model hallucination)
        """
        if not text or len(text.strip()) < threshold:
            return False
            
        # 检查字符级重复
        consecutive_count = 1
        prev_char = None
        
        for char in text:
            if char == prev_char:
                consecutive_count += 1
                if consecutive_count >= threshold:
                    if not silent:
                        logger.warning(f'Detected character repetition hallucination: "{text}" - repeated character: "{char}", consecutive count: {consecutive_count}')
                    return True
            else:
                consecutive_count = 1
            prev_char = char
        
        # 检查词语级重复（按字符分割中文，按空格分割其他语言）
        segments = re.findall(r'[\u4e00-\u9fff]|\S+', text)
        
        if len(segments) >= threshold:
            consecutive_segments = 1
            prev_segment = None
            
            for segment in segments:
                if segment == prev_segment:
                    consecutive_segments += 1
                    if consecutive_segments >= threshold:
                        if not silent:
                            logger.warning(f'Detected word repetition hallucination: "{text}" - repeated segment: "{segment}", consecutive count: {consecutive_segments}')
                        return True
                else:
                    consecutive_segments = 1
                prev_segment = segment
        
        # 检查短语级重复
        words = text.split()
        if len(words) >= threshold * 2:
            for i in range(len(words) - threshold + 1):
                phrase = ' '.join(words[i:i + threshold//2])
                remaining_text = ' '.join(words[i + threshold//2:])
                if phrase in remaining_text:
                    phrase_count = text.count(phrase)
                    if phrase_count >= 3:  # 降低短语重复检测阈值
                        if not silent:
                            logger.warning(f'Detected phrase repetition hallucination: "{text}" - repeated phrase: "{phrase}", occurrence count: {phrase_count}')
                        return True
                        
        return False

    async def _check_target_language_ratio(self, text_regions: List, target_lang: str, min_ratio: float = 0.5) -> bool:
        """
        检查翻译结果中目标语言的占比是否达到要求
        使用py3langid进行语言检测
        Check if the target language ratio meets the requirement by detecting the merged translation text
        
        Args:
            text_regions: 文本区域列表
            target_lang: 目标语言代码
            min_ratio: 最小目标语言占比（此参数在新逻辑中不使用，保留为兼容性）
            
        Returns:
            bool: True表示通过检查，False表示未通过
        """
        if not text_regions or len(text_regions) <= 10:
            # 如果区域数量不超过10个，跳过此检查
            return True
            
        # 合并所有翻译文本
        all_translations = []
        for region in text_regions:
            translation = getattr(region, 'translation', '')
            if translation and translation.strip():
                all_translations.append(translation.strip())
        
        if not all_translations:
            logger.debug('No valid translation texts for language ratio check')
            return True
            
        # 将所有翻译合并为一个文本进行检测
        merged_text = ''.join(all_translations)
        
        # logger.info(f'Target language check - Merged text preview (first 200 chars): "{merged_text[:200]}"')
        # logger.info(f'Target language check - Total merged text length: {len(merged_text)} characters')
        # logger.info(f'Target language check - Number of regions: {len(all_translations)}')
        
        # 使用py3langid进行语言检测
        try:
            detected_lang, confidence = langid.classify(merged_text)
            detected_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
            if detected_language != 'UNKNOWN':
                detected_language = detected_language.upper()
            
            # logger.info(f'Target language check - py3langid result: "{detected_lang}" -> "{detected_language}" (confidence: {confidence:.3f})')
        except Exception as e:
            logger.debug(f'py3langid failed for merged text: {e}')
            detected_language = 'UNKNOWN'
            confidence = -9999
        
        # 检查检测出的语言是否为目标语言
        is_target_lang = (detected_language == target_lang.upper())
        
        # logger.info(f'Target language check: Detected language "{detected_language}" using py3langid (confidence: {confidence:.3f})')
        # logger.info(f'Target language check: Target is "{target_lang.upper()}"')
        # logger.info(f'Target language check result: {"PASSED" if is_target_lang else "FAILED"}')
        
        return is_target_lang

    async def _validate_translation(self, original_text: str, translation: str, target_lang: str, config, ctx: Context = None, silent: bool = False, page_lang_check_result: bool = None) -> bool:
        """
        验证翻译质量（包含目标语言比例检查和幻觉检测）
        Validate translation quality (includes target language ratio check and hallucination detection)
        
        Args:
            page_lang_check_result: 页面级目标语言检查结果，如果为None则进行检查，如果已有结果则直接使用
        """
        if not config.translator.enable_post_translation_check:
            return True
            
        if not translation or not translation.strip():
            return True
        
        # 1. 目标语言比例检查（页面级别）
        if page_lang_check_result is None and ctx and ctx.text_regions and len(ctx.text_regions) > 10:
            # 进行页面级目标语言检查
            page_lang_check_result = await self._check_target_language_ratio(
                ctx.text_regions,
                target_lang,
                min_ratio=0.5
            )
            
        # 如果页面级检查失败，直接返回失败
        if page_lang_check_result is False:
            if not silent:
                logger.debug("Target language ratio check failed for this region")
            return False
        
        # 2. 检查重复内容幻觉（region级别）
        if await self._check_repetition_hallucination(
            translation, 
            config.translator.post_check_repetition_threshold,
            silent
        ):
            return False
                
        return True

    async def _retry_translation_with_validation(self, region, config: Config, ctx: Context) -> str:
        """
        带验证的重试翻译
        Retry translation with validation
        """
        original_translation = region.translation
        max_attempts = config.translator.post_check_max_retry_attempts
        
        for attempt in range(max_attempts):
            # 验证当前翻译 - 在重试过程中只检查单个region（幻觉检测），不进行页面级检查
            is_valid = await self._validate_translation(
                region.text, 
                region.translation, 
                config.translator.target_lang,
                config,
                ctx=None,  # 不传ctx避免页面级检查
                silent=True,  # 重试过程中禁用日志输出
                page_lang_check_result=True  # 传入True跳过页面级检查，只做region级检查
            )
            
            if is_valid:
                if attempt > 0:
                    logger.info(f'Post-translation check passed (Attempt {attempt + 1}/{max_attempts}): "{region.translation}"')
                return region.translation
            
            # 如果不是最后一次尝试，进行重新翻译
            if attempt < max_attempts - 1:
                logger.warning(f'Post-translation check failed (Attempt {attempt + 1}/{max_attempts}), re-translating: "{region.text}"')
                
                try:
                    # 单独重新翻译这个文本区域
                    if config.translator.translator != Translator.none:
                        from .translators import dispatch
                        retranslated = await dispatch(
                            config.translator.translator_gen,
                            [region.text],
                            config.translator,
                            self.use_mtpe,
                            ctx,
                            'cpu' if self._gpu_limited_memory else self.device
                        )
                        if retranslated:
                            region.translation = retranslated[0]
                            
                            # 应用格式化处理
                            if config.render.uppercase:
                                region.translation = region.translation.upper()
                            elif config.render.lowercase:
                                region.translation = region.translation.lower()
                                
                            logger.info(f'Re-translation finished: "{region.text}" -> "{region.translation}"')
                        else:
                            logger.warning(f'Re-translation failed, keeping original translation: "{original_translation}"')
                            region.translation = original_translation
                            break
                    else:
                        logger.warning('Translator is none, cannot re-translate.')
                        break
                        
                except Exception as e:
                    logger.error(f'Error during re-translation: {e}')
                    region.translation = original_translation
                    break
            else:
                logger.warning(f'Post-translation check failed, maximum retry attempts ({max_attempts}) reached, keeping original translation: "{original_translation}"')
                region.translation = original_translation
        
        return region.translation
