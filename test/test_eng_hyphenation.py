import json

import pytest
from hyphen import dictools

from manga_translator.rendering import eng_hyphenation as hyphenation


@pytest.fixture(autouse=True)
def offline_dictionary(tmp_path, monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail('Rendering must not download a hyphenation dictionary')

    monkeypatch.setattr(dictools, 'DEFAULT_DICT_PATH', str(tmp_path))
    monkeypatch.setattr(dictools.requests, 'get', unexpected_download)
    loader = hyphenation._english_hyphenator
    loader.cache_clear()
    yield tmp_path
    loader.cache_clear()


def install_test_dictionary(directory):
    # A tiny prepared pattern fixture exercises the actual libhyphen API offline.
    (directory / 'hyph_en.dic').write_text(
        'UTF-8\nLEFTHYPHENMIN 3\nRIGHTHYPHENMIN 3\n.beau1ti1ful.\n', encoding='utf-8')
    (directory / 'dictionaries.json').write_text(
        json.dumps({'en_GB': {'file': 'hyph_en.dic'}}), encoding='utf-8')


def test_cached_dictionary_only_offers_syllabic_positions(offline_dictionary):
    install_test_dictionary(offline_dictionary)
    assert hyphenation.syllable_break_positions('BEAUTIFUL') == (4, 6)
    assert hyphenation._english_hyphenator().language == 'en_GB'


def test_positions_preserve_surrounding_punctuation(offline_dictionary):
    install_test_dictionary(offline_dictionary)
    word = '"BEAUTIFUL...!"'
    positions = hyphenation.syllable_break_positions(word)
    assert positions == (5, 7)
    assert [(word[:pos] + '-', word[pos:]) for pos in positions] == [
        ('"BEAU-', 'TIFUL...!"'), ('"BEAUTI-', 'FUL...!"')]


@pytest.mark.parametrize('word', [
    'WHAT', 'WORST', 'PEOPLE', 'MERCY', 'HIMESAKI-SAN', "SUPERIORITY'S", '123BEAUTIFUL',
])
def test_short_or_mixed_tokens_have_no_added_breaks(offline_dictionary, word):
    install_test_dictionary(offline_dictionary)
    assert hyphenation.syllable_break_positions(word) == ()


def test_missing_dictionary_fails_closed_without_network():
    assert hyphenation.syllable_break_positions('BEAUTIFUL') == ()


def test_stale_dictionary_manifest_fails_closed_without_network(offline_dictionary):
    (offline_dictionary / 'dictionaries.json').write_text(
        json.dumps({'en_US': {'file': 'missing.dic'}}), encoding='utf-8')
    assert hyphenation.syllable_break_positions('BEAUTIFUL') == ()


def test_nonstandard_or_tiny_fragments_are_rejected(monkeypatch):
    class InvalidPairs:
        def pairs(self, word):
            return [('beau', 'tiful'), ('be', 'autiful'), ('beautifu', 'l'), ('beauty', 'ful')]

    monkeypatch.setattr(hyphenation, '_english_hyphenator', lambda: InvalidPairs())
    assert hyphenation.syllable_break_positions('BEAUTIFUL') == (4,)


def test_original_case_and_name_components_are_protected():
    assert hyphenation.protected_words('Himesaki-san met Midorimine and teacher.') == {
        'himesaki', 'san', 'midorimine'}
    assert hyphenation.protected_words('WHAT A BEAUTIFUL DAY!') == {'what', 'a', 'beautiful', 'day'}
    assert hyphenation.protected_words('himesaki-san met runa\u2011chan') == {
        'himesaki', 'san', 'runa', 'chan'}
