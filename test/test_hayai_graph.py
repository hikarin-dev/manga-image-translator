"""Hayai's CUDA-graph decoding: graphs are reused, and a failed capture decodes that batch eagerly."""
import numpy as np
import pytest
import torch

from manga_translator.ocr.hayai_nova import HayaiModel, _DecodeGraph, preprocess

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def model_and_inputs():
    torch.manual_seed(0)
    model = HayaiModel(vocab_size=1000, n_layers=2).cuda().eval()
    images = [np.random.RandomState(i).randint(0, 255, (64, 32 + 16 * i, 3), np.uint8) for i in range(3)]
    pixel_values, pixel_mask, shapes = preprocess(images, 64)
    return model, (pixel_values.cuda(), pixel_mask.cuda(), shapes.cuda(), 1, 2, 0, 8)


def eager(model, args):
    model.graphed_decode = False
    try:
        return model.generate(*args)
    finally:
        model.graphed_decode = True


def test_a_graph_is_captured_once_per_shape_and_reused():
    model, args = model_and_inputs()
    with torch.inference_mode():
        first = model.generate(*args)
        again = model.generate(*args)
    assert first == again and len(model._graphs) == 1
    assert next(iter(model._graphs.values())).graph is not None


def test_a_failed_capture_decodes_eagerly_and_is_retried_later(monkeypatch):
    model, args = model_and_inputs()
    with torch.inference_mode():
        expected = eager(model, args)
        monkeypatch.setattr(_DecodeGraph, 'start', lambda *a: (_ for _ in ()).throw(RuntimeError('no capture')))
        assert model.generate(*args) == expected
        assert model.graphed_decode and not model._graphs, 'one failure only drops that graph'
        for _ in range(HayaiModel.CAPTURE_TRIES - 1):
            assert model.generate(*args) == expected
        assert model.graphed_decode is False, 'failures in a row turn graphs off'


def test_a_capture_interrupted_mid_way_leaves_this_thread_able_to_decode(monkeypatch):
    """The failure that lost a page: an invalidated capture leaves its CUDA error pending on the
    thread (and torch.cuda.graph() would leave the capture stream current), so the eager
    fallback's first launch failed too. A device sync mid-capture invalidates it for real."""
    model, args = model_and_inputs()
    step = _DecodeGraph._step

    def interrupted(self):
        if torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()
        return step(self)
    with torch.inference_mode():
        expected = eager(model, args)
        monkeypatch.setattr(_DecodeGraph, '_step', interrupted)
        assert model.generate(*args) == expected
        monkeypatch.setattr(_DecodeGraph, '_step', step)
        assert model.generate(*args) is not None and len(model._graphs) == 1
