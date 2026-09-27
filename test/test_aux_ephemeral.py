"""An aux node writes nothing of the pages it translates to its disk."""
import io
import subprocess
from types import SimpleNamespace

import manga_translator.manga_translator as pipeline
from manga_translator.manga_translator import MangaTranslator
from server import aux_agent


def test_the_node_worker_is_ephemeral_and_its_output_stays_in_memory(monkeypatch):
    started = {}

    class FakeProc:
        def __init__(self, cmds, **kwargs):
            started.update(kwargs, cmds=cmds)
            self.stdout = io.BytesIO(b'loading models\nready\n')
    monkeypatch.setattr(aux_agent.subprocess, 'Popen', FakeProc)
    proc = aux_agent.spawn_worker(5099, SimpleNamespace(verbose=False, use_gpu=True))
    assert started['env']['MT_EPHEMERAL'] == '1'
    assert started['stdout'] is subprocess.PIPE and started['stderr'] is subprocess.STDOUT
    assert '--verbose' not in started['cmds']
    aux_agent._keep_tail(proc.stdout, proc._aux_tail)
    assert list(proc._aux_tail) == ['loading models', 'ready']


def test_only_the_last_lines_are_kept():
    tail = aux_agent.collections.deque(maxlen=aux_agent.WORKER_TAIL_LINES)
    aux_agent._keep_tail(io.BytesIO(b''.join(b'line %d\n' % i for i in range(500))), tail)
    assert len(tail) == aux_agent.WORKER_TAIL_LINES and tail[-1] == 'line 499'


def test_an_ephemeral_worker_keeps_no_log_file_and_no_stage_images(monkeypatch):
    monkeypatch.setattr(pipeline, 'EPHEMERAL', True)
    monkeypatch.setattr(MangaTranslator, '_setup_log_file', lambda self: (_ for _ in ()).throw(AssertionError('log file')))
    mt = MangaTranslator({'verbose': True, 'kernel_size': 3, 'models_ttl': 60})
    assert mt.verbose is False
