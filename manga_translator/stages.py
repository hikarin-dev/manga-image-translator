"""Pipeline stage model: which config fields each stage reads, and what code produces it.

Clients keep a page's stage outputs (see page_data) and decide what a later run can skip. They
compare, per stage, the config fields the stage reads (FIELDS) and its build token:

* a build describes what runs: a hand revision, the source of the modules and orchestration
  callables the stage executes, model identity, native binaries, relevant packages, and the
  worker parameters/environment that change the stage's output (RUNTIME);
* builds describe the code this process LOADED: computed once per process (warm() at start-up)
  and never invalidated. An edit without a restart must not relabel old code as new.

What a build hashes is listed below (CALLABLES, MODULES, the renderer/model modules), and
test/test_stage_coverage.py keeps the list whole: it traces a gallery run and fails on any
pipeline function that is neither hashed into some stage's build nor declared unable to change
output (NOT_OUTPUT). So when code a stage runs is added or moved, register it here — never hand-edit
a build. When unsure, hash it: a needless rerun is cheap, reusing stale output is silent.

Revision checklist — bump REVISIONS[stage] when shared orchestration changes what a stage
produces without touching any file or callable listed for it below.
"""
import ast
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import threading
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from .config import Config

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent

STAGES = ('prepare', 'detect', 'ocr', 'merge', 'translate', 'mask', 'inpaint', 'bubbles', 'render')
ORDER = {stage: index for index, stage in enumerate(STAGES)}

# Renderers that segment balloons, and the ones that read the detector's raw text mask.
BUBBLE_RENDERERS = frozenset({'manga2eng', 'shiori', 'shiori_v2'})
TEXT_MASK_RENDERERS = frozenset({'shiori', 'shiori_v2'})
# Translators that read the page image/regions beyond their text: their output can't be keyed
# on texts, so a translation made by them is never reused.
IMAGE_TRANSLATORS = frozenset({'chatgpt_2stage', 'gemini_2stage'})

REVISIONS = {stage: 1 for stage in STAGES}

# ── Config ownership audit (gallery path) ─────────────────────────────────────────────────
# Every Config leaf maps to the stages whose output it can change, or to () with the reason it
# has none. test_stages enforces full coverage, so a new Config field must be classified here.
FIELDS = {
    # Root
    'filter_text': ('translate',),                 # survivor filter in _batch_translate_contexts
    'study_mode_generation': ('render',),          # study payload is rebuilt by every render
    'force_simple_sort': ('merge',),               # sort_regions in _run_textline_merge
    'kernel_size': (),                             # unused: mask refinement reads the worker's --kernel-size (runtime.kernel_size)
    'mask_dilation_offset': ('mask',),             # _run_mask_refinement
    # render
    'render.renderer': ('render',),                # _run_text_rendering dispatch
    'render.alignment': ('render',),               # copied onto regions during translation, read only by renderers
    'render.disable_font_border': ('render',),     # shiori paint + study hints
    'render.font_size_offset': ('render',),        # default renderer
    'render.font_size_minimum': ('render',),       # default renderer
    'render.direction': ('render',),               # copied onto regions during translation, read only by renderers
    'render.uppercase': ('translate',),            # _retry_translation_with_validation (the gallery path's only reader)
    'render.lowercase': ('translate',),            # _retry_translation_with_validation
    'render.gimp_font': (),                        # unused: GIMP save format only
    'render.no_hyphenation': ('render',),          # default renderer
    'render.font_color': ('ocr', 'merge', 'render'),  # _run_ocr colour override; merge adjust_bg_color; study hints
    'render.estimate_font_color': ('ocr',),        # _run_ocr apply_estimated_colors
    'render.estimate_outline_color': ('ocr',),     # _run_ocr apply_estimated_colors
    'render.line_spacing': ('render',),            # every text renderer
    'render.font_size': ('render',),               # default renderer
    'render.rtl': ('merge',),                      # sort_regions reading order
    # upscale
    'upscale.upscaler': ('prepare',),              # _run_upscaling
    'upscale.revert_upscaling': ('render',),       # _revert_upscale of the final composition
    'upscale.upscale_ratio': ('prepare',),         # _run_upscaling
    # translator
    'translator.translator': ('translate',),
    'translator.target_lang': ('merge', 'translate'),   # merge language filter; translation target
    'translator.no_text_lang_skip': ('merge',),    # _run_textline_merge language filter
    'translator.skip_lang': ('merge',),            # _run_textline_merge language filter
    'translator.gpt_config': ('translate',),       # hashed by file content, never by path
    'translator.translator_chain': ('translate',),
    'translator.selective_translation': ('translate',),
    'translator.content_screen_enabled': (),       # unused: only the single-image _run_text_translation screens
    'translator.content_screen_translator': (),    # unused: see content_screen_enabled
    'translator.content_screen_fallback_translator': (),  # unused: see content_screen_enabled
    'translator.content_screen_prompt': (),        # unused: see content_screen_enabled
    'translator.enable_post_translation_check': ('translate',),
    'translator.post_check_max_retry_attempts': ('translate',),
    'translator.post_check_repetition_threshold': ('translate',),
    'translator.post_check_target_lang_threshold': (),  # unused: the checks use fixed ratios
    # detector
    'detector.detector': ('detect',),
    'detector.detection_size': ('detect',),
    'detector.text_threshold': ('detect',),
    'detector.det_rotate': ('detect',),
    'detector.det_auto_rotate': ('detect',),
    'detector.det_invert': ('detect',),
    'detector.det_gamma_correct': ('detect',),
    'detector.box_threshold': ('detect',),
    'detector.unclip_ratio': ('detect',),
    # colorizer
    'colorizer.colorization_size': ('prepare',),
    'colorizer.denoise_sigma': ('prepare',),
    'colorizer.colorizer': ('prepare',),
    # inpainter
    'inpainter.inpainter': ('inpaint',),
    'inpainter.inpainting_size': ('inpaint',),
    'inpainter.inpainting_precision': ('inpaint',),
    # ocr
    'ocr.use_mocr_merge': ('ocr',),                # manga-ocr bbox merge
    'ocr.bubble_ocr': ('ocr',),                    # region crops for region-reading OCR models
    'ocr.ocr': ('ocr',),
    'ocr.min_text_length': ('merge',),             # _run_textline_merge length filter
    'ocr.ignore_bubble': ('ocr', 'mask'),          # 32px/48px_ctc OCR; mask refinement
    'ocr.prob': ('ocr',),                          # recognition threshold
}

# Worker parameters and environment that change output (not part of Config).
RUNTIME = {
    'kernel_size': ('mask',),                      # --kernel-size, read by _run_mask_refinement
    'pre_dict': ('merge',),                        # --pre-dict, applied after merging (content digest)
    'post_dict': ('translate',),                   # --post-dict (content digest)
    'context_size': ('translate',),                # --context-size, cross-page context translators
    'use_mtpe': ('translate',),
    'prep_manual': ('translate',),
    'font_path': ('render',),                      # --font-path
    'manga2eng_safe_layout': ('render',),          # MT_MANGA2ENG_SAFE_LAYOUT
    'bubble_seg': ('bubbles',),                    # MT_BUBBLE_SEG
}

# Stage → orchestration callables it executes ('module:qualname'), and extra module trees.
_MT = 'manga_translator.manga_translator:'
_SL = 'manga_translator.study_layers:'
CALLABLES = {
    # The page runner and the gallery orchestrator shape what every later stage receives (and what
    # is restored), so any edit to them reruns everything.
    'prepare': (_MT + 'MangaTranslator._run_colorizer', _MT + 'MangaTranslator._run_upscaling',
                'manga_translator.utils.generic:load_image', _MT + 'MangaTranslator._translate_until_translation',
                _MT + 'MangaTranslator.translate_gallery_stream', _MT + 'MangaTranslator._translate_gallery_run',
                _MT + 'GalleryRun.__init__', _MT + '_run_attribute',
                'manga_translator.utils.inference:ModelWrapper.load'),
    'detect': (_MT + 'MangaTranslator._run_detection', 'manga_translator.detection:dispatch'),
    'ocr': (_MT + 'MangaTranslator._run_ocr', 'manga_translator.ocr:dispatch'),
    'merge': (_MT + 'MangaTranslator._run_textline_merge', _MT + 'load_dictionary', _MT + 'apply_dictionary',
              'manga_translator.utils.generic2:is_valuable_text'),
    'translate': (_MT + 'MangaTranslator._batch_translate_contexts', _MT + 'MangaTranslator._batch_translate_texts',
                  _MT + 'MangaTranslator._apply_post_translation_processing',
                  _MT + 'MangaTranslator._check_repetition_hallucination',
                  _MT + 'MangaTranslator._check_target_language_ratio', _MT + 'MangaTranslator._validate_translation',
                  _MT + 'MangaTranslator._retry_translation_with_validation', _MT + 'MangaTranslator._build_prev_context',
                  _MT + 'MangaTranslator._drop_untranslated', _MT + 'MangaTranslator._page_stream',
                  _MT + 'apply_dictionary', 'manga_translator.translators:dispatch',
                  'manga_translator.config:TranslatorConfig.translator_gen', 'manga_translator.config:TranslatorChain',
                  'manga_translator.config:Translator.__str__'),
    'mask': (_MT + 'MangaTranslator._run_mask_refinement', 'manga_translator.utils.bubble:is_ignore'),
    'inpaint': (_MT + 'MangaTranslator._run_inpainting', 'manga_translator.inpainting:dispatch'),
    'bubbles': (),
    'render': (_MT + 'MangaTranslator._run_text_rendering', _MT + 'MangaTranslator._render_stage',
               _MT + 'MangaTranslator._revert_upscale', _MT + 'MangaTranslator._build_bubble_overlays',
               _SL + '_build_page_layers_job', _SL + '_study_meta_bubble', _SL + '_furi_lines',
               _SL + '_study_bg', _SL + '_study_img_data_url', _SL + '_study_norm', _SL + '_furi_seg_append',
               _SL + '_has_kanji', 'manga_translator.utils.generic:dump_image',
               'manga_translator.config:RenderConfig.font_color_fg', 'manga_translator.config:RenderConfig.font_color_bg',
               'manga_translator.stages:renderer_uses_bubbles'),
}
MODULES = {
    'prepare': ('manga_translator.page_data', 'manga_translator.page_reuse'),   # what saved data restores
    'ocr': ('manga_translator.ocr.colors',),
    'merge': ('manga_translator.textline_merge', 'manga_translator.utils.sort', 'manga_translator.utils.textblock'),
    'translate': ('manga_translator.translators.common',),
    'mask': ('manga_translator.mask_refinement',),
    'bubbles': ('manga_translator.rendering.bubble_seg',),
}
# Pipeline code that runs but cannot change what any stage produces: 'module' or 'module:qualname'
# (a qualname covers what is nested in it). test_stage_coverage.py fails on any function a gallery
# run executes that is neither here nor part of some stage's build above.
NOT_OUTPUT = (
    'manga_translator.utils.profiling',                                     # timing and usage telemetry
    _MT + 'MangaTranslator.__init__', _MT + 'MangaTranslator.parse_init_params',   # set-up (output-shaping
                                                                            # parameters are RUNTIME)
    _MT + 'MangaTranslator.using_gpu',                                      # device placement
    _MT + 'MangaTranslator._accum_time', _MT + 'MangaTranslator._report_progress',
    _MT + 'MangaTranslator._add_logger_hook', _MT + 'MangaTranslator.add_progress_hook',
    _MT + 'MangaTranslator.add_page_result_hook', _MT + 'MangaTranslator.add_page_data_hook',
    _MT + 'MangaTranslator.add_page_bubbles_hook',                          # progress and hooks
    _MT + 'MangaTranslator._emit_page_result', _MT + 'MangaTranslator._emit_page_data',
    _MT + 'MangaTranslator._emit_page_bubbles',                             # hand finished pages out
    _MT + 'MangaTranslator._detector_cleanup_job',                          # unloads idle models
    'manga_translator.stages:plain', 'manga_translator.stages:translator_keys',   # read the config
)
RENDERER_MODULES = {
    'default': ('manga_translator.rendering.text_render',),
    'manga2eng': ('manga_translator.rendering.text_render_eng', 'manga_translator.rendering.text_render_pillow_eng'),
    'manga2eng_pillow': ('manga_translator.rendering.text_render_pillow_eng',),
    'shiori': ('manga_translator.rendering.shiori_render',),
    'shiori_v2': ('manga_translator.rendering.shiori_render',),
    'none': (),
}
RENDERER_CALLABLES = {
    'default': ('manga_translator.rendering:dispatch',),
    'manga2eng': ('manga_translator.rendering:dispatch_eng_render', 'manga_translator.rendering:dispatch_eng_render_pillow'),
    'manga2eng_pillow': ('manga_translator.rendering:dispatch_eng_render_pillow',),
    'shiori': (), 'shiori_v2': (), 'none': (),
}
PACKAGES = {
    'prepare': ('numpy', 'Pillow', 'opencv-python', 'torch'),
    'detect': ('numpy', 'Pillow', 'opencv-python', 'torch', 'onnxruntime'),
    'ocr': ('numpy', 'Pillow', 'opencv-python', 'torch', 'transformers', 'onnxruntime', 'ai-edge-litert'),
    'merge': ('numpy', 'opencv-python', 'shapely', 'networkx', 'py3langid', 'langcodes'),
    'translate': ('py3langid', 'langcodes'),
    'mask': ('numpy', 'opencv-python', 'shapely'),
    'inpaint': ('numpy', 'opencv-python', 'torch'),
    'bubbles': ('numpy', 'opencv-python', 'onnxruntime'),
    'render': ('numpy', 'Pillow', 'opencv-python', 'freetype-py', 'pyphen', 'PyHyphen', 'pykakasi', 'shiori-renderer'),
}
OFFLINE_TRANSLATE_PACKAGES = ('torch', 'ctranslate2', 'sentencepiece', 'transformers')
# Non-secret environment identity of remote translators. '@NAME' hashes the file NAME points to.
TRANSLATOR_ENV = {
    'chatgpt': ('OPENAI_MODEL', 'OPENAI_API_BASE', 'OPENAI_FALLBACK_MODEL', '@OPENAI_GLOSSARY_PATH'),
    'chatgpt_2stage': ('OPENAI_MODEL', 'OPENAI_API_BASE', 'OPENAI_STAGE1_MODEL', 'OPENAI_STAGE2_MODEL'),
    'deepseek': ('DEEPSEEK_MODEL', 'DEEPSEEK_API_BASE'),
    'gemini': ('GEMINI_MODEL',),
    'gemini_2stage': ('GEMINI_MODEL',),
    'groq': ('GROQ_MODEL', 'CONTEXT_RETENTION', 'CONTEXT_LENGTH'),
    'custom_openai': ('CUSTOM_OPENAI_MODEL', 'CUSTOM_OPENAI_API_BASE', 'CUSTOM_OPENAI_MODEL_CONF'),
    'sakura': ('SAKURA_API_BASE', 'SAKURA_VERSION', '@SAKURA_DICT_PATH'),
}
# The environment variable that names a remote translator's model (shown as its version).
TRANSLATOR_MODEL_ENV = {'chatgpt': 'OPENAI_MODEL', 'chatgpt_2stage': 'OPENAI_MODEL', 'deepseek': 'DEEPSEEK_MODEL',
                        'gemini': 'GEMINI_MODEL', 'gemini_2stage': 'GEMINI_MODEL', 'groq': 'GROQ_MODEL',
                        'custom_openai': 'CUSTOM_OPENAI_MODEL'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def canon(value):
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return sha256(canon(value))


# ── Config helpers ─────────────────────────────────────────────────────────────────────────
def config_fields(model=Config, prefix=''):
    """Every leaf field of Config as a dotted path."""
    fields = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            fields.extend(config_fields(annotation, prefix + name + '.'))
        else:
            fields.append(prefix + name)
    return fields


def field_value(config, path):
    target = config
    for part in path.split('.'):
        target = getattr(target, part)
    return target


@functools.lru_cache(maxsize=1024)
def _file_digest(path, size, mtime_ns):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _file_value(path):
    """Content digest of a file (None when absent), cached by size and modification time."""
    if not path or not Path(path).is_file():
        return None
    stat = Path(path).stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns)


def runtime_values(params=None):
    """Normalise worker parameters/environment into the RUNTIME identity."""
    params = params or {}
    return {
        'kernel_size': int(params.get('kernel_size') or 3),
        'pre_dict': _file_value(params.get('pre_dict')),
        'post_dict': _file_value(params.get('post_dict')),
        'context_size': int(params.get('context_size') or 0),
        'use_mtpe': bool(params.get('use_mtpe')),
        'prep_manual': bool(params.get('prep_manual')),
        'font_path': _file_value(params.get('font_path')) if params.get('font_path') else None,
        'manga2eng_safe_layout': os.environ.get('MT_MANGA2ENG_SAFE_LAYOUT', '1') != '0',
        'bubble_seg': os.environ.get('MT_BUBBLE_SEG', '1') != '0',
    }


def runtime_slice(stage, runtime):
    return {name: runtime.get(name) for name, owners in RUNTIME.items() if stage in owners}


def translator_keys(config):
    translator = config.translator
    chain = translator.selective_translation or translator.translator_chain
    if chain:
        return [part.split(':')[0] for part in chain.split(';') if part]
    return [plain(translator.translator)]


def implementation(stage, config):
    """The implementation id a stage runs for this config (display + registry key)."""
    if stage == 'prepare':
        parts = []
        if plain(config.colorizer.colorizer) != 'none':
            parts.append('colorizer:' + plain(config.colorizer.colorizer))
        if config.upscale.upscale_ratio:
            parts.append('upscaler:' + plain(config.upscale.upscaler))
        return '+'.join(parts) or 'default'
    if stage == 'detect':
        return plain(config.detector.detector)
    if stage == 'ocr':
        return plain(config.ocr.ocr)
    if stage == 'translate':
        return '+'.join(translator_keys(config))
    if stage == 'inpaint':
        return plain(config.inpainter.inpainter)
    if stage == 'render':
        return plain(config.render.renderer)
    return 'default'


def renderer_uses_bubbles(config):
    return plain(config.render.renderer) in BUBBLE_RENDERERS


# ── Builds ─────────────────────────────────────────────────────────────────────────────────
def _registry(kind):
    if kind == 'detector':
        from .detection import DETECTORS as table
    elif kind == 'ocr':
        from .ocr import OCRS as table
    elif kind == 'translator':
        from .translators import TRANSLATORS as table
    elif kind == 'inpainter':
        from .inpainting import INPAINTERS as table
    elif kind == 'upscaler':
        from .upscaling import UPSCALERS as table
    elif kind == 'colorizer':
        from .colorization import COLORIZERS as table
    else:
        raise KeyError(kind)
    return {plain(key): value for key, value in table.items()}


def component_class(kind, key):
    return _registry(kind).get(key)


def components(stage, config):
    """(kind, key) pairs whose implementation classes run in this stage."""
    if stage == 'prepare':
        found = []
        if plain(config.colorizer.colorizer) != 'none':
            found.append(('colorizer', plain(config.colorizer.colorizer)))
        if config.upscale.upscale_ratio:
            found.append(('upscaler', plain(config.upscale.upscaler)))
        return found
    if stage == 'detect':
        return [('detector', plain(config.detector.detector))]
    if stage == 'ocr':
        return [('ocr', plain(config.ocr.ocr))]
    if stage == 'translate':
        return [('translator', key) for key in translator_keys(config)]
    if stage == 'inpaint':
        return [('inpainter', plain(config.inpainter.inpainter))]
    if stage == 'render':
        return [('renderer', plain(config.render.renderer))]
    return []


def _module_file(name):
    parts = name.split('.')
    base = ROOT.joinpath(*parts)
    if (base / '__init__.py').is_file():
        return base / '__init__.py'
    if base.with_suffix('.py').is_file():
        return base.with_suffix('.py')
    return None


def _resolve_relative(module, node, is_package):
    if node.level == 0:
        return node.module
    parts = module.split('.')
    base = parts if is_package else parts[:-1]
    if node.level > 1:
        base = base[:-(node.level - 1)]
    return '.'.join(base + ([node.module] if node.module else []))


def module_closure(name):
    """A module plus every module it imports from inside the same `manga_translator.<sub>`
    package (e.g. a detector's helpers), found statically. Imports leaving that package
    (shared utils) are not followed: those are listed explicitly where they matter."""
    parts = name.split('.')
    scope = '.'.join(parts[:2])
    seen, stack = {}, [name]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        path = _module_file(current)
        if path is None:
            continue
        seen[current] = path
        is_package = path.name == '__init__.py'
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom):
                base = _resolve_relative(current, node, is_package)
                if base:
                    targets.append(base)
                    targets.extend(base + '.' + alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            for target in targets:
                if (target == scope or target.startswith(scope + '.')) and _module_file(target):
                    stack.append(target)
    return seen


def _callable(spec):
    module_name, _, qualname = spec.partition(':')
    target = importlib.import_module(module_name)
    for part in qualname.split('.'):
        target = getattr(target, part)
    return target.fget if isinstance(target, property) else target


def _callable_digest(spec):
    return sha256(inspect.getsource(_callable(spec)).encode('utf-8'))


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _env_identity(names):
    from .translators import keys
    identity = {}
    for name in names:
        if name.startswith('@'):
            name = name[1:]
            value = getattr(keys, name, None) or os.environ.get(name)
            identity[name] = _file_value(value) if value and Path(value).is_file() else None
        else:
            identity[name] = sha256((str(getattr(keys, name, None) or os.environ.get(name) or '')).encode('utf-8'))
    return identity


def model_identity(cls):
    """Declared model identity: the class's download mapping (URL + pinned hash), which the
    downloader verifies, so it identifies the weights without reading gigabytes."""
    mapping = getattr(cls, '_MODEL_MAPPING', None) or {}
    return plain({key: {k: v for k, v in entry.items() if k in ('url', 'hash', 'file', 'archive')}
                  for key, entry in mapping.items()})


def _native_binary():
    try:
        import shiori_renderer
    except Exception:
        return None
    module = getattr(shiori_renderer, 'shiori_renderer', shiori_renderer)
    path = getattr(module, '__file__', None)
    return {'file': _file_value(path), 'upstream': getattr(shiori_renderer, 'UPSTREAM_REVISION', None)}


def _fonts():
    folder = ROOT / 'fonts'
    return {p.relative_to(ROOT).as_posix(): _file_value(str(p)) for p in sorted(folder.rglob('*')) if p.is_file()}


def build_spec(stage, config):
    """The ingredients of a stage's build for this config (unhashed)."""
    spec = {'stage': stage, 'revision': REVISIONS[stage], 'callables': list(CALLABLES.get(stage, ())),
            'modules': list(MODULES.get(stage, ())), 'models': {}, 'env': [], 'files': {},
            'packages': list(PACKAGES.get(stage, ()))}
    for kind, key in components(stage, config):
        if kind == 'renderer':
            spec['modules'].extend(RENDERER_MODULES.get(key, ()))
            spec['callables'].extend(RENDERER_CALLABLES.get(key, ()))
            if key in TEXT_MASK_RENDERERS:
                spec['files']['native'] = 'shiori_renderer'
            if key != 'none':
                spec['files']['fonts'] = 'fonts'
            continue
        cls = component_class(kind, key)
        if cls is None:
            spec['models'][f'{kind}:{key}'] = 'unknown'
            continue
        spec['modules'].append(cls.__module__)
        spec['models'][f'{kind}:{key}'] = cls.__module__ + '.' + cls.__qualname__
        if kind == 'translator':
            spec['env'].extend(TRANSLATOR_ENV.get(key, ()))
            from .translators import OFFLINE_TRANSLATORS
            if key in {plain(k) for k in OFFLINE_TRANSLATORS}:
                spec['packages'].extend(OFFLINE_TRANSLATE_PACKAGES)
    if stage == 'bubbles':
        spec['files']['bubble_model'] = 'bubble_seg'
    return spec


def compute_build(spec):
    """Hash a build spec. Pure given the files it reads (tests call it directly)."""
    parts = {'stage': spec['stage'], 'revision': spec['revision'], 'callables': {}, 'modules': {},
             'models': {}, 'env': {}, 'files': {}, 'packages': {}}
    for name in spec['callables']:
        parts['callables'][name] = _callable_digest(name)
    for module in spec['modules']:
        for name, path in module_closure(module).items():
            parts['modules'][name] = _file_value(str(path))
    for key, qualified in spec['models'].items():
        if qualified == 'unknown':
            parts['models'][key] = None
            continue
        module_name, _, qualname = qualified.rpartition('.')
        cls = getattr(importlib.import_module(module_name), qualname, None)
        parts['models'][key] = model_identity(cls) if cls is not None else None
    parts['env'] = _env_identity(spec['env'])
    for key, what in spec['files'].items():
        if what == 'shiori_renderer':
            parts['files'][key] = _native_binary()
        elif what == 'fonts':
            parts['files'][key] = _fonts()
        elif what == 'bubble_seg':
            from .rendering.bubble_seg import MODEL_PATH
            parts['files'][key] = _file_value(MODEL_PATH)
    for name in sorted(set(spec['packages'])):
        parts['packages'][name] = _package_version(name)
    return digest(parts), parts


_builds = {}
_builds_lock = threading.Lock()


def build(stage, config):
    """Frozen per process: the first computation describes the code this process runs."""
    spec = build_spec(stage, config)
    key = canon(spec)
    cached = _builds.get(key)
    if cached is None:
        with _builds_lock:
            cached = _builds.get(key)
            if cached is None:
                cached = _builds[key] = compute_build(spec)[0]
    return cached


def applies(stage, config):
    return stage != 'bubbles' or renderer_uses_bubbles(config)


def stage_builds(config, runtime):
    """Build token per applicable stage: 12 hex of the build plus the worker parameters it reads."""
    return {stage: digest({'build': build(stage, config), 'runtime': runtime_slice(stage, runtime)})[:12]
            for stage in STAGES if applies(stage, config)}


def signature(builds):
    return digest(builds)[:16]


def owned_config(config):
    """The effective config restricted to fields some stage reads, nested like Config."""
    out = {}
    for path, owners in FIELDS.items():
        if not owners:
            continue
        value = field_value(config, path)
        if path == 'translator.gpt_config':
            value = _file_value(value)
        target = out
        *parents, leaf = path.split('.')
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = plain(value)
    return out


def stage_fields():
    return {stage: [path for path, owners in FIELDS.items() if stage in owners] for stage in STAGES}


def resolve(config, runtime):
    """What a client needs to decide which stages a run can skip (stateless). `signature` is
    what the client sends back with the job, so a server that changed since refuses it."""
    builds = stage_builds(config, runtime)
    return {'config': owned_config(config), 'builds': builds, 'fields': stage_fields(), 'signature': signature(builds)}


def warm():
    """Compute every implementation's builds now, so they describe the code loaded at start-up.
    Call it in a background thread when a worker or server starts."""
    from .config import Detector, Inpainter, Ocr, Renderer, Translator
    choices = {'detect': ('detector', 'detector', Detector), 'ocr': ('ocr', 'ocr', Ocr),
               'translate': ('translator', 'translator', Translator), 'inpaint': ('inpainter', 'inpainter', Inpainter),
               'render': ('render', 'renderer', Renderer)}
    base = Config()
    for stage in STAGES:
        try:
            build(stage, base)
        except Exception:
            pass
    for stage, (section, field, choice) in choices.items():
        for member in choice:
            config = Config()
            try:
                setattr(getattr(config, section), field, member)
                build(stage, config)
            except Exception:
                pass
