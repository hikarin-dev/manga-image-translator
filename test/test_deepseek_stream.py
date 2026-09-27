import asyncio
import json
from unittest.mock import Mock

import httpx
import openai
import pytest

from manga_translator.translators import deepseek
from manga_translator.translators.common import TRANSLATION_SINK


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def translator(monkeypatch):
    monkeypatch.setattr(deepseek, 'deepseekTokenCounter', lambda: Mock(count_tokens=len))
    monkeypatch.setattr(deepseek, 'DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setattr(openai, 'api_key', None)
    return deepseek.DeepseekTranslator()


RESPONSE = '<|1|>Hello there.\n<|2|>|Thank you!!\n<|3|>Good-bye'


def test_finished_lines_are_what_the_whole_response_parses_to():
    whole = deepseek.finished_lines(RESPONSE + '<|4|>', 3)
    assert whole == ['Hello there.', 'Thank you!!', 'Good-bye']
    for k in range(len(RESPONSE) + 1):
        partial = deepseek.finished_lines(RESPONSE[:k], 3)
        assert partial == whole[:len(partial)], RESPONSE[:k]
    assert deepseek.finished_lines('<|1|>Hello there.\n<|2', 3) == []
    assert deepseek.finished_lines('<|1|>Hello there.\n<|2|>Th', 3) == ['Hello there.']


def sse(chunks, gate):
    """A streamed completion whose last chunk waits for `gate`."""
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i, text in enumerate(chunks):
                if i == len(chunks) - 1:
                    await asyncio.wait_for(gate.wait(), 2)
                yield b'data: ' + json.dumps({
                    'id': 't', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm',
                    'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}).encode() + b'\n\n'
            yield b'data: ' + json.dumps({
                'id': 't', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'm', 'choices': [],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 9, 'total_tokens': 16}}).encode() + b'\n\n'
            yield b'data: [DONE]\n\n'
    return Stream()


@pytest.mark.anyio
async def test_lines_go_out_while_the_response_streams(translator):
    first_line = asyncio.Event()
    got = {}

    def sink(i, text):
        got[i] = text
        first_line.set()

    async def respond(request):
        assert json.loads(request.content)['stream'] is True
        chunks = ['<|1|>Hello', ' there.\n<|2|>', 'Thank you!!\n<|3|>', 'Good-bye']
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=sse(chunks, first_line))

    queries = ['こんにちは', 'ありがとう', 'さようなら']
    await translator.client.close()
    async with translator.client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            translator.client._client = http
            token = TRANSLATION_SINK.set(sink)
            try:
                result = await translator.translate('JPN', 'ENG', queries)
            finally:
                TRANSLATION_SINK.reset(token)
    # the stream only finished because line 1 was released before its last chunk
    assert result[:2] == [got[0], got[1]] and 2 not in got
    assert translator.token_count == 16


@pytest.mark.anyio
async def test_without_a_sink_nothing_streams(translator):
    async def respond(request):
        assert not json.loads(request.content).get('stream')
        return httpx.Response(200, json={
            'id': 't', 'object': 'chat.completion', 'created': 0, 'model': 'm',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': RESPONSE}}],
            'usage': {'prompt_tokens': 7, 'completion_tokens': 9, 'total_tokens': 16}})

    await translator.client.close()
    async with translator.client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            translator.client._client = http
            assert len(await translator.translate('JPN', 'ENG', ['こんにちは', 'ありがとう', 'さようなら'])) == 3
