"""Stage model: config ownership, builds, and the resolve document clients plan against."""
import importlib
import json
import sys
import time

import pytest

from manga_translator import stages
from manga_translator.config import Config, Detector, Inpainter, Ocr, Renderer, Translator


def resolve(config=None, runtime=None):
    return stages.resolve(config or Config(), stages.runtime_values(runtime or {}))


def _value(doc, path):
    for part in path.split('.'):
        doc = doc[part]
    return doc


def changed_stages(old, new):
    """What a client concludes from two resolve documents: a stage changed when its build or
    any field it reads differs (the app's rule)."""
    return {stage for stage in stages.STAGES
            if old['builds'].get(stage) != new['builds'].get(stage)
            or any(_value(old['config'], f) != _value(new['config'], f) for f in new['fields'][stage])}


def test_same_config_same_document():
    assert resolve() == resolve()


def test_explicit_default_resolves_like_omitted_value():
    explicit = Config.parse_raw('{"detector": {"detection_size": 2048}, "render": {"renderer": "default"}}')
    assert resolve(explicit) == resolve(Config())


def test_a_resolved_config_can_be_sent_back_as_a_job_config():
    """A page that keeps its own settings is translated again with the config its translation
    recorded, so that document must parse back to the same effective config."""
    config = Config.parse_raw('{"translator": {"translator": "deepseek", "target_lang": "CHS"}, "filter_text": "x+", '
                              '"render": {"renderer": "manga2eng", "font_color": "FFFFFF"}, "mask_dilation_offset": 40, '
                              '"detector": {"detection_size": 1536}, "inpainter": {"inpainting_size": 1024}}')
    doc = resolve(config)
    assert resolve(Config.parse_raw(json.dumps(doc['config']))) == doc


def test_document_is_compact():
    doc = resolve()
    assert all(len(token) == 12 for token in doc['builds'].values())
    assert 'kernel_size' not in doc['config'], 'fields no stage reads are left out'
    assert 'bubbles' not in doc['builds'], 'the default renderer does not segment balloons'


@pytest.mark.parametrize('path, value, changed', [
    ('render.renderer', 'manga2eng', {'render'}),
    ('render.alignment', 'left', {'render'}),
    ('detector.detection_size', 1536, {'detect'}),
    ('inpainter.inpainting_size', 1024, {'inpaint'}),
    ('mask_dilation_offset', 40, {'mask'}),
    ('ocr.ocr', 'mocr', {'ocr'}),
    ('ocr.ignore_bubble', 5, {'ocr', 'mask'}),
    ('ocr.min_text_length', 2, {'merge'}),
    ('render.font_color', 'FFFFFF', {'ocr', 'merge', 'render'}),
    ('render.estimate_font_color', True, {'ocr'}),
    ('render.rtl', False, {'merge'}),
    ('translator.target_lang', 'CHS', {'merge', 'translate'}),
    ('translator.translator', 'deepseek', {'translate'}),
    ('filter_text', 'x+', {'translate'}),
    ('study_mode_generation', 'text_only', {'render'}),
    ('upscale.revert_upscaling', True, {'render'}),
    ('kernel_size', 9, set()),
    ('translator.content_screen_enabled', True, set()),
])
def test_only_owning_stages_change(path, value, changed):
    config = Config()
    target = config
    *parents, leaf = path.split('.')
    for part in parents:
        target = getattr(target, part)
    setattr(target, leaf, value)
    after = resolve(config)
    if path == 'render.renderer':
        changed = changed | {'bubbles'}   # manga2eng segments balloons; the default renderer does not
    assert changed_stages(resolve(), after) == changed


def test_runtime_parameters_join_their_stages(tmp_path):
    dictionary = tmp_path / 'pre.txt'
    dictionary.write_text('a b\n', encoding='utf-8')
    base, other = resolve(), resolve(runtime={'kernel_size': 5, 'pre_dict': str(dictionary)})
    assert changed_stages(base, other) == {'mask', 'merge'}
    dictionary.write_text('a c\n', encoding='utf-8')
    import os
    os.utime(dictionary, ns=(time.time_ns(), time.time_ns() + 10_000_000))
    edited = resolve(runtime={'kernel_size': 5, 'pre_dict': str(dictionary)})
    assert edited['builds']['merge'] != other['builds']['merge'], 'dictionary content, not path, is identified'


def test_every_config_field_is_classified():
    fields = stages.config_fields()
    assert set(fields) == set(stages.FIELDS), (set(fields) ^ set(stages.FIELDS))
    for path, owners in stages.FIELDS.items():
        assert set(owners) <= set(stages.STAGES), path


def _all_configs():
    """One config per selectable implementation of every stage."""
    for detector in Detector:
        config = Config()
        config.detector.detector = detector
        yield 'detect', config
    for ocr in Ocr:
        config = Config()
        config.ocr.ocr = ocr
        yield 'ocr', config
    for translator in Translator:
        config = Config()
        config.translator.translator = translator
        yield 'translate', config
    for inpainter in Inpainter:
        config = Config()
        config.inpainter.inpainter = inpainter
        yield 'inpaint', config
    for renderer in Renderer:
        config = Config()
        config.render.renderer = renderer
        yield 'render', config
    for stage in ('prepare', 'merge', 'mask', 'bubbles'):
        yield stage, Config()


def test_every_declared_source_and_callable_exists():
    for stage, config in _all_configs():
        spec = stages.build_spec(stage, config)
        for name in spec['callables']:
            assert callable(stages._callable(name)), name
        for module in spec['modules']:
            assert stages._module_file(module) is not None, module
        for kind, key in stages.components(stage, config):
            if kind != 'renderer':
                assert stages.component_class(kind, key) is not None, (kind, key)


def test_edits_to_declared_sources_change_the_build(tmp_path, monkeypatch):
    package = tmp_path / 'fakepkg'
    (package / 'stagehelp').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'stagehelp' / '__init__.py').write_text('')
    (package / 'stagehelp' / 'impl.py').write_text('from . import helper\n\ndef run():\n    return helper.value()\n')
    (package / 'stagehelp' / 'helper.py').write_text('def value():\n    return 1\n')
    (package / 'stagehelp' / 'unrelated.py').write_text('def other():\n    return 1\n')
    (package / 'orchestration.py').write_text(
        'def declared():\n    return 1\n\n\ndef undeclared():\n    return 1\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(stages, 'ROOT', tmp_path)

    def build():
        # A real edit is followed by a restart; emulate it by dropping this process's caches.
        import linecache
        stages._file_digest.cache_clear()
        linecache.clearcache()
        importlib.invalidate_caches()
        sys.modules.pop('fakepkg.orchestration', None)
        spec = {'stage': 'render', 'revision': 1, 'callables': ['fakepkg.orchestration:declared'],
                'modules': ['fakepkg.stagehelp.impl'], 'models': {}, 'env': [], 'files': {}, 'packages': []}
        return stages.compute_build(spec)[0]

    first = build()
    (package / 'stagehelp' / 'unrelated.py').write_text('def other():\n    return 2\n')
    source = (package / 'orchestration.py').read_text()
    (package / 'orchestration.py').write_text(source.replace('def undeclared():\n    return 1', 'def undeclared():\n    return 2'))
    assert build() == first, 'undeclared function and unimported module do not affect the build'
    (package / 'stagehelp' / 'helper.py').write_text('def value():\n    return 2\n')
    second = build()
    assert second != first, 'a module the implementation imports is part of its build'
    source = (package / 'orchestration.py').read_text()
    (package / 'orchestration.py').write_text(source.replace('def declared():\n    return 1', 'def declared():\n    return 3'))
    assert build() != second, 'a declared callable is part of the build'


def test_build_is_cheap_after_warmup():
    config = Config()
    for stage in stages.STAGES:
        stages.build(stage, config)
    start = time.perf_counter()
    for _ in range(200):
        stages.build('render', config)
    assert (time.perf_counter() - start) / 200 < 0.001


def test_signature_follows_every_build():
    config = Config()
    builds = stages.stage_builds(config, stages.runtime_values({}))
    assert stages.signature(builds) == stages.signature(dict(builds))
    assert stages.signature({**builds, 'render': '0' * 12}) != stages.signature(builds)
