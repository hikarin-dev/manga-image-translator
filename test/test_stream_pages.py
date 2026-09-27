"""_batch_translate_contexts with a translator that streams: a page goes out as soon as its lines
are final, and the batch's result is the same as without streaming."""
import asyncio
from types import SimpleNamespace

from manga_translator.config import Config, Translator
from manga_translator.manga_translator import MangaTranslator
from manga_translator.translators.common import TRANSLATION_SINK


def run(texts_per_page, translations, wait_for_first_page=False):
    """Translate with a fake translator that streams every line but the last one tick at a time.
    With `wait_for_first_page` its response only ends once page 1 went out, so page 1 provably
    left while the response streamed."""
    config = Config()
    config.translator.translator = Translator.deepseek
    config.translator.target_lang = 'ENG'
    pairs = [(SimpleNamespace(text_regions=[SimpleNamespace(text=t, translation='') for t in texts]), config)
             for texts in texts_per_page]
    first_page_out = asyncio.Event()
    out = []

    async def page_done(ctx):
        out.append(next(i for i, (c, _) in enumerate(pairs) if c is ctx))
        first_page_out.set()

    async def translate_texts(texts, config, ctx, batch_contexts=None):
        sink = TRANSLATION_SINK.get()
        if sink is None:   # a retry of the whole batch
            return list(translations)
        for i in range(len(texts) - 1):
            sink(i, translations[i])
            await asyncio.sleep(0)
            if wait_for_first_page and i == len(texts_per_page[0]) - 1:
                await asyncio.wait_for(first_page_out.wait(), 2)
        for _ in range(5):
            await asyncio.sleep(0)
        return list(translations)

    async def go():
        mt = MangaTranslator({'models_ttl': 60, 'kernel_size': 3})
        mt._batch_translate_texts = translate_texts
        mt._report_progress = lambda *a, **k: asyncio.sleep(0)
        await mt._batch_translate_contexts(pairs, len(pairs), page_done=page_done)
        return [[r.translation for r in ctx.text_regions] for ctx, _ in pairs], out
    return asyncio.run(go())


def test_a_page_goes_out_while_its_batch_streams():
    texts = [['一', '二'], ['三', '四'], ['五']]
    translations = ['One.', 'Two.', 'Three.', 'Four.', 'Five.']
    result, out = run(texts, translations, wait_for_first_page=True)
    assert result == [['One.', 'Two.'], ['Three.', 'Four.'], ['Five.']]
    assert out == [0, 1]    # the last page's last line is final only when the response ends


def test_large_batches_go_out_once_the_first_lines_pass_the_language_check():
    texts = [['一', '二'], ['三'] * 12]
    translations = ['One.', 'Two.'] + [f'This is line number {i} of the second page.' for i in range(12)]
    result, out = run(texts, translations)
    assert out == [0]
    assert result == [translations[:2], translations[2:]]


def test_lines_in_the_wrong_language_hold_every_page_for_the_batch():
    texts = [['一', '二'], ['三'] * 12]
    translations = ['一つ目の文です。', '二つ目の文です。'] + ['これはまだ日本語の文章のままです。'] * 12
    _, out = run(texts, translations)
    assert out == []
