"""Hayai's CUDA-graph decoding: graphs are reused, and a failed capture falls back to the eager loop."""
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


def test_a_graph_is_captured_once_per_shape_and_reused():
    model, args = model_and_inputs()
    with torch.inference_mode():
        first = model.generate(*args)
        again = model.generate(*args)
    assert first == again and len(model._graphs) == 1
    assert next(iter(model._graphs.values())).graph is not None


def test_a_failed_capture_falls_back_to_the_eager_loop(monkeypatch):
    model, args = model_and_inputs()
    with torch.inference_mode():
        model.graphed_decode = False
        eager = model.generate(*args)
        model.graphed_decode = True
        monkeypatch.setattr(_DecodeGraph, 'start', lambda *a: (_ for _ in ()).throw(RuntimeError('no capture')))
        with pytest.warns(UserWarning):
            fallback = model.generate(*args)
    assert fallback == eager and model.graphed_decode is False
