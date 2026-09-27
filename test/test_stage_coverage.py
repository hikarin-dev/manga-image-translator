"""Build tokens are computed from the code itself (stages.py), so they are only as good as the
list of what each stage runs. This test keeps that list honest: it traces a real gallery run and
fails on any pipeline function that is neither part of some stage's build (CALLABLES, MODULES,
renderer and model modules) nor declared unable to change what a stage produces
(stages.NOT_OUTPUT). New code a stage runs must be registered where its build sees it — or the
reuse of saved page data would keep serving output the new code no longer produces.
"""
import inspect
import sys
from pathlib import Path

import pytest

import manga_translator.manga_translator as pipeline
from manga_translator import stages
from manga_translator.config import Renderer
from test_pipeline_reuse import Harness, RAWS, config_for

PACKAGE = Path(stages.__file__).resolve().parent


def _code(fn):
    fn = inspect.unwrap(getattr(fn, '__func__', fn))
    return getattr(fn, '__code__', None)


def _declared(configs):
    """Files hashed into some stage's build, and hashed callables as (file, qualname)."""
    files, callables = set(), set()
    for config in configs:
        for stage in stages.STAGES:
            if not stages.applies(stage, config):
                continue
            spec = stages.build_spec(stage, config)
            for module in spec['modules']:
                files.update(str(p.resolve()) for p in stages.module_closure(module).values())
            for name in spec['callables']:
                target = stages._callable(name)
                if inspect.isclass(target):   # a class is hashed whole: its methods with it
                    callables.add((str(Path(inspect.getsourcefile(target)).resolve()), target.__qualname__))
                    continue
                code = _code(target)
                if code is not None:
                    callables.add((str(Path(code.co_filename).resolve()), code.co_qualname))
    for name in stages.NOT_OUTPUT:
        module_name, _, qualname = name.partition(':')
        path = stages._module_file(module_name)
        if path is not None and not qualname:
            files.add(str(path.resolve()))
        elif path is not None:
            callables.add((str(path.resolve()), qualname))
    return files, callables


def _covered(fn, files, callables):
    path, qualname = fn
    if path in files:
        return True
    return any(p == path and (qualname == q or qualname.startswith(q + '.')) for p, q in callables)


@pytest.fixture
def traced(monkeypatch):
    async def inline_cpu(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'run_cpu', inline_cpu)
    monkeypatch.setattr(pipeline, 'submit_gpu', lambda coro, lane=0: coro)
    seen = set()

    def profile(frame, event, arg):
        if event == 'call':
            path = frame.f_code.co_filename
            if path.startswith(str(PACKAGE)):
                seen.add((str(Path(path).resolve()), frame.f_code.co_qualname))
    return seen, profile


def test_every_function_a_stage_runs_is_in_its_build_or_declared_outside_it(monkeypatch, traced):
    seen, profile = traced
    harness = Harness(monkeypatch)
    configs = [config_for(study_mode_generation='text_and_image'),
               config_for(study_mode_generation='text_only', **{'render.renderer': Renderer.shiori})]
    sys.setprofile(profile)
    try:
        for config in configs:
            harness.run(config, RAWS)
    finally:
        sys.setprofile(None)
    files, callables = _declared(configs)
    missing = sorted(f'{Path(p).relative_to(PACKAGE.parent).as_posix()}:{q}' for p, q in seen if not _covered((p, q), files, callables))
    assert not missing, ('These pipeline functions ran but no stage build includes them. Add each to the '
                         'CALLABLES/MODULES of the stage whose output it shapes, or to stages.NOT_OUTPUT if it '
                         'cannot change any output:\n  ' + '\n  '.join(missing))
