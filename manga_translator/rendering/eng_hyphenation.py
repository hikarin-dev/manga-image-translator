"""Conservative English break opportunities for the manga2eng safe layout."""

from functools import lru_cache
from pathlib import Path
import re

from hyphen import Hyphenator, dictools


@lru_cache(maxsize=1)
def _english_hyphenator():
    # Hyphenator downloads missing dictionaries, so only open a verified cache.
    try:
        dictionaries = dictools.Dictionaries()
        for language in ('en_US', 'en_GB', 'en_AU', 'en'):
            path = Path(dictionaries.filepath(language)) if dictionaries.is_installed(language) else None
            if path is not None and path.is_file():
                hyphenator = Hyphenator(language, directory=dictionaries.directory,
                                       lmin=3, rmin=3, compound_lmin=3, compound_rmin=3)
                return hyphenator
    except (OSError, ValueError, KeyError, RuntimeError):
        pass
    return None


def protected_words(text):
    """Retain original case clues before rendering uppercases the translation."""
    words = re.findall(r'[A-Za-z]+', text)
    protected = {word.lower() for word in words if not word.islower()}
    for compound in re.findall(r'[A-Za-z]+(?:[-\u2010\u2011][A-Za-z]+)+', text):
        protected.update(word.lower() for word in re.findall(r'[A-Za-z]+', compound))
    return frozenset(protected)


def syllable_break_positions(word):
    """Return original-token indices; callers decide if a last-resort break is needed."""
    match = re.fullmatch(r'([^\w\s]*)([A-Za-z]{8,})([^\w\s]*)', word)
    if match is None:
        return ()
    hyphenator = _english_hyphenator()
    if hyphenator is None:
        return ()
    core = match[2].lower()
    return tuple(sorted({
        len(match[1]) + len(left)
        for left, right in hyphenator.pairs(core)
        if len(left) >= 3 and len(right) >= 3 and left + right == core
    }))
