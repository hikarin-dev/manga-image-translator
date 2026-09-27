"""An aux node keeps the pages it translates private: nothing of them on its disk or its screen."""
import asyncio
import io
import json
import logging
import pickle
import subprocess
from types import SimpleNamespace

import manga_translator.manga_translator as pipeline
from manga_translator.manga_translator import MangaTranslator
from server import aux_agent


def spawn(monkeypatch, **args):
    started = {}

    class FakeProc:
        def __init__(self, cmds, **kwargs):
            started.update(kwargs, cmds=cmds)
            self.stdout = io.BytesIO(b'loading models\n')
    monkeypatch.setattr(aux_agent.subprocess, 'Popen', FakeProc)
    monkeypatch.setattr(aux_agent.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    return aux_agent.spawn_worker(5099, SimpleNamespace(use_gpu=True, **args)), started


def test_the_worker_is_ephemeral_and_never_writes_to_this_console(monkeypatch):
    for verbose in (False, True):
        _, started = spawn(monkeypatch, verbose=verbose)
        assert started['env']['MT_EPHEMERAL'] == '1'
        assert started['stdout'] is subprocess.PIPE and started['stderr'] is subprocess.STDOUT
        assert '--verbose' not in started['cmds']


def test_worker_output_is_kept_only_while_it_starts(monkeypatch):
    proc, _ = spawn(monkeypatch, verbose=False)
    aux_agent._keep_tail(io.BytesIO(b'loading models\n'), proc._aux_tail, proc._aux_starting)
    assert list(proc._aux_tail) == ['loading models']
    aux_agent.worker_started(proc)
    aux_agent._keep_tail(io.BytesIO('原文 => translation\n'.encode()), proc._aux_tail, proc._aux_starting)
    assert list(proc._aux_tail) == []


def test_a_failed_chunk_reports_its_error_to_the_server_but_not_to_this_console(monkeypatch):
    lines = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    aux_agent.logger.addHandler(handler)
    sent = []

    class Ws:
        async def send(self, message):
            sent.append(message)

    async def fetch(*args, **kwargs):
        raise RuntimeError('OCR failed on 秘密のセリフ')
    import server.sent_data_internal as sdi
    monkeypatch.setattr(sdi, 'fetch_gallery_stream', fetch)
    try:
        relay = aux_agent._Relay(Ws(), 'http://127.0.0.1:1')
        payload = pickle.dumps({'images': [b'page'], 'config': None, 'job_token': 't'})
        asyncio.run(relay._run(1, payload))
    finally:
        aux_agent.logger.removeHandler(handler)
    assert lines == ['chunk 1: received', 'chunk 1: failed']
    assert '秘密のセリフ' in json.loads(sent[-1])['error']


def test_an_ephemeral_worker_keeps_no_log_file_and_no_stage_images(monkeypatch):
    monkeypatch.setattr(pipeline, 'EPHEMERAL', True)
    monkeypatch.setattr(MangaTranslator, '_setup_log_file', lambda self: (_ for _ in ()).throw(AssertionError('log file')))
    mt = MangaTranslator({'verbose': True, 'kernel_size': 3, 'models_ttl': 60})
    assert mt.verbose is False
