"""ModelWrapper.load: pages preprocessed concurrently must share one load of a model."""
import asyncio
import tempfile

from manga_translator.utils.inference import ModelWrapper


class _Model(ModelWrapper):
    _MODEL_DIR = tempfile.gettempdir()   # never touch the real models folder
    _MODEL_SUB_DIR = 'test-model-load'

    def __init__(self, fail_first=False):
        super().__init__()
        self.loads = 0
        self.fail_first = fail_first

    async def _load(self, device):
        self.loads += 1
        await asyncio.sleep(0.05)   # the window in which other pages used to start their own load
        if self.fail_first and self.loads == 1:
            raise RuntimeError('load failed')

    async def _unload(self):
        pass

    async def _infer(self):
        return 'ok'


def test_concurrent_loads_share_one():
    model = _Model()

    async def scenario():
        await asyncio.gather(*(model.load('cpu') for _ in range(3)))
        return await model.infer()

    assert asyncio.run(scenario()) == 'ok'
    assert model.loads == 1 and model.is_loaded()


def test_failed_load_is_retried():
    model = _Model(fail_first=True)

    async def scenario():
        results = await asyncio.gather(*(model.load('cpu') for _ in range(2)), return_exceptions=True)
        return results

    results = asyncio.run(scenario())
    assert isinstance(results[0], RuntimeError)
    assert results[1] is None, 'the waiting caller loads it after the first attempt failed'
    assert model.loads == 2 and model.is_loaded()
