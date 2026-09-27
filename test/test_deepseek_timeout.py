import asyncio
import json
from unittest.mock import Mock

import httpx
import openai
import pytest

from manga_translator.translators import deepseek


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def translator(monkeypatch):
    monkeypatch.setattr(deepseek, 'deepseekTokenCounter', lambda: Mock(count_tokens=len))
    monkeypatch.setattr(deepseek, 'DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setattr(openai, 'api_key', None)
    return deepseek.DeepseekTranslator()


def completion():
    return {
        'id': 'test', 'object': 'chat.completion', 'created': 0, 'model': 'test',
        'choices': [{'index': 0, 'finish_reason': 'stop',
                     'message': {'role': 'assistant', 'content': '<|1|>Hello'}}],
        'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
    }


@pytest.mark.anyio
async def test_keep_alive_response_can_outlast_timeout(translator):
    class KeepAliveStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(12):
                await asyncio.sleep(0.02)
                yield b'\n'
            yield json.dumps(completion()).encode()

    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={'content-type': 'application/json'},
                              stream=KeepAliveStream())

    translator._TIMEOUT = 0.05
    translator._RETRY_ATTEMPTS = 1
    await translator.client.close()
    async with translator.client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            translator.client._client = http
            assert await translator._translate('Japanese', 'English', ['Hello']) == ['Hello']
    assert len(requests) == 1
    assert requests[0].extensions['timeout']['read'] == 0.05
    assert translator.token_count == 5


@pytest.mark.anyio
async def test_cancelling_translation_cancels_request(translator, monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()
    request_tasks = []

    async def pending_request(*args):
        request_tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(translator, '_request_translation', pending_request)
    async with translator.client:
        task = asyncio.create_task(translator._translate('Japanese', 'English', ['Hello']))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stopped.is_set(), 'Cancelled translation left its API request running'
        finally:
            for request in request_tasks:
                request.cancel()
            await asyncio.gather(*request_tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_total_deadline_cleans_up_request(translator, monkeypatch):
    stopped = asyncio.Event()

    async def pending_request(*args):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(translator, '_request_translation', pending_request)
    translator._REQUEST_TIMEOUT = 0.05
    translator._RETRY_ATTEMPTS = 1
    async with translator.client:
        with pytest.raises(asyncio.TimeoutError):
            await translator._translate('Japanese', 'English', ['Hello'])
    assert stopped.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize('recover', [False, True])
async def test_network_timeout_retries_are_bounded(translator, recover):
    requests = []

    async def respond(request):
        requests.append(request)
        if recover and len(requests) == 2:
            return httpx.Response(200, json=completion())
        raise httpx.ReadTimeout('No response data', request=request)

    await translator.client.close()
    async with translator.client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            translator.client._client = http
            if recover:
                assert await translator._translate('Japanese', 'English', ['Hello']) == ['Hello']
            else:
                with pytest.raises(openai.APITimeoutError):
                    await translator._translate('Japanese', 'English', ['Hello'])
    assert len(requests) == (2 if recover else translator._RETRY_ATTEMPTS)
