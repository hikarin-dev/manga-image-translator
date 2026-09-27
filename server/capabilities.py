"""GET /capabilities: what this server can run, as the single source of truth for clients.

Built from the stage registry (implementation builds and versions) plus a curated label
registry below: which implementations and options are offered, their labels, recommended
defaults, presets and target languages. It is deliberately not a raw dump of the Config enums;
it lists what this server offers. Availability is computed cheaply, with no network probes:
model files present, a required API-key variable *present* (never its value), a native module
importable. External callers see no source paths, Git revisions or environment values; builds
are opaque digests.
"""
import copy
import importlib.util
import os
import sys
import threading
from pathlib import Path

from manga_translator import stages as stage_model
from manga_translator.config import Config
from manga_translator.stages import digest

API_VERSION = 2

# ── Curated registry ───────────────────────────────────────────────────────────────────────
# (id, label, short label or None). Order is display order.
IMPLEMENTATIONS = {
    'detect': [
        ('default', 'Default', None), ('ctd', 'CTD (comic text detector)', None), ('dbconvnext', 'DBConvNext', None),
        ('paddle', 'Paddle', None), ('craft', 'CRAFT (not for manga)', None),
        ('oneocr', 'OneOCR (detects + reads text)', None), ('none', 'None', None),
    ],
    'ocr': [
        ('48px', '48px (default, fast)', None), ('48px_exp', '48px (experimental)', '48px exp'),
        ('48px_ctc', '48px CTC', None), ('32px', '32px', None), ('mocr', 'manga-ocr (accurate, slower)', None),
        ('mocr_fast', 'manga-ocr (fast)', 'manga-ocr fast'), ('mocr_tflite', 'manga-ocr (TFLite, CPU)', 'manga-ocr TFLite'),
        ('hayai', 'Hayai OCR (Nova)', None), ('oneocr', 'OneOCR (system engine)', None),
    ],
    'translate': [
        ('sugoi', 'Sugoi — offline JA→EN, no key', None),
        ('qwen2_big', 'Qwen2-7B — local HuggingFace (fast, no Ollama needed)', None),
        ('chatgpt', 'Qwen2.5-7B + cross-page context — requires Ollama running', 'Qwen2.5-7B'),
        ('gemini', 'Gemini Flash — cloud, needs API key (smart, rate-limited)', None),
        ('deepseek', 'DeepSeek V4 Flash — cloud, needs API key', 'DeepSeek V4'),
        ('m2m100', 'M2M100 — offline, multilingual', None), ('nllb', 'NLLB — offline, multilingual', None),
        ('offline', 'Offline — auto-select', None), ('deepl', 'DeepL — needs API key on server', None),
        ('sakura', 'Sakura — local LLM endpoint', None), ('none', 'None — typeset only, no translation', None),
    ],
    'inpaint': [
        ('lama_large', 'Lama Large (best)', None), ('lama_mpe', 'Lama MPE', None), ('default', 'Default', None),
        ('sd', 'Stable Diffusion (slow)', None), ('original', 'Original (no neural)', None),
        ('none', "None (don't erase)", None),
    ],
    'render': [
        ('default', 'Default', None), ('manga2eng', 'manga2eng (English typesetting)', None),
        ('manga2eng_pillow', 'manga2eng (Pillow)', 'manga2eng Pillow'), ('shiori', 'shiori', None),
        ('shiori_v2', 'shiori hybrid (manga2eng colors & outlines)', None), ('none', 'None', None),
    ],
}
STAGE_INFO = {
    'detect': ('Text detection', 'detector.detector', 'default'),
    'ocr': ('Text recognition', 'ocr.ocr', '48px'),
    'translate': ('Translation', 'translator.translator', 'sugoi'),
    'inpaint': ('Inpainting & text removal', 'inpainter.inpainter', 'lama_large'),
    'render': ('Rendering', 'render.renderer', 'manga2eng'),
}
# Translators that send several pages per request. `user_tunable` caps are the client's choice.
BATCHING = {
    'gemini': {'mode': 'fixed', 'default': 8, 'user_tunable': True},
    'chatgpt': {'mode': 'fixed', 'default': 6, 'user_tunable': True},
    'deepseek': {'mode': 'adaptive', 'default': 10, 'user_tunable': False},
}
CONTEXT_TRANSLATORS = frozenset({'chatgpt'})
# Required environment variables (presence only) per remote translator.
TRANSLATOR_KEYS = {
    'deepl': ('DEEPL_AUTH_KEY',), 'chatgpt': ('OPENAI_API_KEY',), 'chatgpt_2stage': ('OPENAI_API_KEY',),
    'gemini': ('GEMINI_API_KEY',), 'gemini_2stage': ('GEMINI_API_KEY',), 'deepseek': ('DEEPSEEK_API_KEY',),
    'groq': ('GROQ_API_KEY',), 'custom_openai': ('CUSTOM_OPENAI_MODEL',), 'caiyun': ('CAIYUN_TOKEN',),
    'baidu': ('BAIDU_APP_ID', 'BAIDU_SECRET_KEY'), 'youdao': ('YOUDAO_APP_KEY', 'YOUDAO_SECRET_KEY'),
}
SCREEN_MODELS = [{'value': 'qwen2_big', 'label': 'Qwen2-7B (accurate)'}, {'value': 'qwen2', 'label': 'Qwen2-1.5B (lighter)'}]
SCREEN_FALLBACKS = [{'value': 'qwen2_big', 'label': 'Qwen2-7B'}, {'value': 'qwen2', 'label': 'Qwen2-1.5B'},
                    {'value': 'sugoi', 'label': 'Sugoi (JA→EN only)'}]
LANGUAGES = [
    ('ENG', 'English', 'en'), ('CHS', 'Chinese (Simplified)', 'zh'), ('CHT', 'Chinese (Traditional)', 'zh-TW'),
    ('JPN', 'Japanese', 'ja'), ('KOR', 'Korean', 'ko'), ('VIN', 'Vietnamese', 'vi'), ('FRA', 'French', 'fr'),
    ('DEU', 'German', 'de'), ('ESP', 'Spanish', 'es'), ('RUS', 'Russian', 'ru'), ('PTB', 'Portuguese (Brazil)', 'pt-BR'),
    ('IND', 'Indonesian', 'id'),
]
_SCREEN_ON = {'translator.content_screen_enabled': True}
# Parameters per stage. `primary` ones sit next to the implementation choice in clients;
# `group` gathers related options; `requires` lists values other params must have for this one
# to apply (it is omitted from a config otherwise); `omit_empty` drops an empty value and
# `omit_when` a value that means "off" (the server's own default).
PARAMS = {
    'detect': [
        {'key': 'detector.detection_size', 'label': 'Detection size', 'type': 'enum', 'default': 1536,
         'choices': [{'value': 1024, 'label': '1024 (fastest)'}, {'value': 1536, 'label': '1536'},
                     {'value': 2048, 'label': '2048'}, {'value': 2560, 'label': '2560 (most thorough)'}]},
        {'key': 'detector.text_threshold', 'label': 'Text threshold (0–1)', 'type': 'number', 'min': 0, 'max': 1,
         'step': 0.05, 'default': 0.5},
        {'key': 'detector.box_threshold', 'label': 'Box threshold (0–1)', 'type': 'number', 'min': 0, 'max': 1,
         'step': 0.05, 'default': 0.7},
        {'key': 'detector.unclip_ratio', 'label': 'Unclip ratio', 'type': 'number', 'min': 0, 'step': 0.1, 'default': 2.3},
    ],
    'ocr': [
        {'key': 'ocr.bubble_ocr', 'type': 'bool', 'default': False, 'omit_when': False,
         'label': 'Read whole bubbles (Hayai with shiori renderers) — one crop per bubble instead of line by line',
         'requires': {'ocr.ocr': 'hayai', 'render.renderer': ['shiori', 'shiori_v2']}},
        {'key': 'render.estimate_font_color', 'label': 'Text color from image (uniform per page)', 'type': 'bool',
         'default': False},
        {'key': 'render.estimate_outline_color', 'label': 'Outline color from image (uniform per page)', 'type': 'bool',
         'default': False},
    ],
    'translate': [
        {'key': 'translator.target_lang', 'label': 'Translate to', 'type': 'language', 'default': 'ENG', 'primary': True},
        {'key': 'translator.content_screen_enabled', 'group': 'screening', 'type': 'bool', 'default': False,
         'omit_when': False,
         'label': 'Pre-screen bubbles before cloud translator — routes explicit text to local model instead'},
        {'key': 'translator.content_screen_translator', 'group': 'screening', 'type': 'enum', 'label': 'Screen model',
         'choices': SCREEN_MODELS, 'default': 'qwen2_big', 'requires': _SCREEN_ON},
        {'key': 'translator.content_screen_fallback_translator', 'group': 'screening', 'type': 'enum',
         'label': 'Fallback model (for flagged bubbles)', 'choices': SCREEN_FALLBACKS, 'default': 'qwen2_big',
         'requires': _SCREEN_ON},
        {'key': 'translator.content_screen_prompt', 'group': 'screening', 'type': 'text', 'multiline': True,
         'label': 'Screen prompt', 'default': Config().translator.content_screen_prompt, 'requires': _SCREEN_ON,
         'omit_empty': True},
    ],
    'inpaint': [
        {'key': 'inpainter.inpainting_size', 'label': 'Inpainting size', 'type': 'enum', 'default': 1536,
         'choices': [{'value': 1024, 'label': '1024 (fastest)'}, {'value': 1536, 'label': '1536'},
                     {'value': 2048, 'label': '2048 (cleanest)'}]},
        {'key': 'inpainter.inpainting_precision', 'label': 'Precision', 'type': 'enum', 'default': 'bf16',
         'choices': [{'value': 'bf16', 'label': 'bf16'}, {'value': 'fp16', 'label': 'fp16'}, {'value': 'fp32', 'label': 'fp32'}]},
        {'key': 'mask_dilation_offset', 'label': 'Mask dilation (clear residue)', 'type': 'number', 'min': 0, 'step': 1,
         'default': 30, 'integer': True},
        {'key': 'kernel_size', 'label': 'Cleanup kernel (odd)', 'type': 'number', 'min': 1, 'step': 2, 'default': 5,
         'integer': True},
    ],
    'render': [
        {'key': 'render.direction', 'label': 'Direction', 'type': 'enum', 'default': 'auto',
         'choices': [{'value': 'auto', 'label': 'Auto'}, {'value': 'horizontal', 'label': 'Horizontal'},
                     {'value': 'vertical', 'label': 'Vertical'}]},
        {'key': 'render.alignment', 'label': 'Alignment', 'type': 'enum', 'default': 'auto',
         'choices': [{'value': 'auto', 'label': 'Auto'}, {'value': 'left', 'label': 'Left'},
                     {'value': 'center', 'label': 'Center'}, {'value': 'right', 'label': 'Right'}]},
        {'key': 'render.font_size_offset', 'label': 'Font size offset', 'type': 'number', 'step': 1, 'default': 0,
         'integer': True},
        {'key': 'render.font_color', 'label': 'Font color (hex, blank = auto)', 'type': 'text', 'default': '',
         'placeholder': 'auto', 'omit_empty': True},
        {'key': 'render.uppercase', 'label': 'Force uppercase', 'type': 'bool', 'default': False},
        {'key': 'render.no_hyphenation', 'label': 'No hyphenation', 'type': 'bool', 'default': False},
    ],
}
PRESETS = [
    {'id': 'fast', 'label': 'Fast — quickest, may leave residue on busy art', 'stage': 'inpaint',
     'values': {'inpainter.inpainting_size': 1024, 'mask_dilation_offset': 20, 'kernel_size': 3}},
    {'id': 'balanced', 'label': 'Balanced — recommended', 'stage': 'inpaint',
     'values': {'inpainter.inpainting_size': 1536, 'mask_dilation_offset': 30, 'kernel_size': 5}},
    {'id': 'thorough', 'label': 'Thorough — best clearing, slowest', 'stage': 'inpaint',
     'values': {'inpainter.inpainting_size': 2048, 'mask_dilation_offset': 40, 'kernel_size': 7}},
]
_IMPL_FIELD = {'detect': ('detector', 'detector'), 'ocr': ('ocr', 'ocr'), 'translate': ('translator', 'translator'),
               'inpaint': ('inpainter', 'inpainter'), 'render': ('render', 'renderer')}


# ── Availability ───────────────────────────────────────────────────────────────────────────
def _model_files_present(cls):
    from manga_translator.utils.inference import ModelWrapper, get_filename_from_url
    mapping = getattr(cls, '_MODEL_MAPPING', None) or {}
    model_dir = Path(ModelWrapper._MODEL_DIR) / getattr(cls, '_MODEL_SUB_DIR', '')
    for key, entry in mapping.items():
        if 'archive' in entry:
            for orig, dest in entry['archive'].items():
                if os.path.basename(dest) in ('', '.'):
                    dest = os.path.join(dest, os.path.basename(orig[:-1] if orig.endswith('/') else orig))
                if not (model_dir / dest).exists():
                    return False
        else:
            path = entry.get('file', '.')
            if os.path.basename(path) in ('.', ''):
                path = os.path.join(path, get_filename_from_url(entry['url'], key))
            if not (model_dir / path).exists():
                return False
    return True


def _availability(stage, key):
    """(available, reason, downloaded). Never probes a network or reads a secret's value."""
    kind = {'detect': 'detector', 'ocr': 'ocr', 'translate': 'translator', 'inpaint': 'inpainter'}.get(stage)
    if stage == 'render':
        if key in stage_model.TEXT_MASK_RENDERERS and importlib.util.find_spec('shiori_renderer') is None:
            return False, 'native renderer not installed', None
        return True, None, None
    if key == 'oneocr' and sys.platform != 'win32':
        return False, 'requires Windows', None
    cls = stage_model.component_class(kind, key)
    if cls is None:
        return False, 'not installed on this server', None
    for name in TRANSLATOR_KEYS.get(key, ()) if stage == 'translate' else ():
        if not os.environ.get(name):
            return False, 'API key not configured on this server', None
    downloaded = _model_files_present(cls) if getattr(cls, '_MODEL_MAPPING', None) else None
    return True, None, downloaded


def _version(stage, key, build):
    if stage == 'translate' and key in stage_model.TRANSLATOR_MODEL_ENV:
        from manga_translator.translators import keys
        name = stage_model.TRANSLATOR_MODEL_ENV[key]
        model = getattr(keys, name, None) or os.environ.get(name)
        if model:
            return str(model)
    return f'r{stage_model.REVISIONS[stage]} · {build[:7]}'


# ── Document ───────────────────────────────────────────────────────────────────────────────
_static = None
_static_lock = threading.Lock()


def _builds():
    """Implementation builds are frozen per process, so compute them once."""
    global _static
    if _static is None:
        with _static_lock:
            if _static is None:
                table = {}
                for stage, rows in IMPLEMENTATIONS.items():
                    section, field = _IMPL_FIELD[stage]
                    for key, _, _ in rows:
                        config = Config()
                        try:
                            setattr(getattr(config, section), field, key)
                            table[(stage, key)] = stage_model.build(stage, config)
                        except Exception:
                            table[(stage, key)] = None
                _static = table
    return _static


def document():
    builds = _builds()
    stages = []
    for stage in ('detect', 'ocr', 'translate', 'inpaint', 'render'):
        label, param, default = STAGE_INFO[stage]
        implementations = []
        for key, impl_label, short in IMPLEMENTATIONS[stage]:
            build = builds.get((stage, key))
            available, reason, downloaded = _availability(stage, key)
            if build is None:
                available, reason = False, reason or 'not installed on this server'
            item = {'id': key, 'label': impl_label, 'version': _version(stage, key, build or ''),
                    'build': build, 'available': available, 'unavailable_reason': reason,
                    'default': key == default}
            if short:
                item['short'] = short
            if downloaded is not None:
                item['downloaded'] = downloaded
            if stage == 'translate':
                item['roles'] = ['translate']
                if key in BATCHING:
                    item['batching'] = dict(BATCHING[key])
                item['context'] = key in CONTEXT_TRANSLATORS
            implementations.append(item)
        stages.append({'id': stage, 'label': label, 'implementation_param': param,
                       'implementations': implementations, 'params': copy.deepcopy(PARAMS[stage])})
    body = {
        'api': 'shiori-translate', 'api_version': API_VERSION,
        # Pages come back with their pipeline data and may be sent with it (see page_data).
        'features': {'pipeline_data': True},
        'stages': stages,
        'languages': [{'id': code, 'label': label, 'bcp47': tag} for code, label, tag in LANGUAGES],
        'presets': copy.deepcopy(PRESETS),
    }
    body['etag'] = digest(body)[:32]
    return body
