import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manga_translator import *  # noqa: F401,F403  (resolved lazily below)

import colorama
from dotenv import load_dotenv

colorama.init(autoreset=True)
load_dotenv()


def __getattr__(name):
    # The package used to `from .manga_translator import *` here, which loads torch, CUDA and every
    # model module (~2.6 GB of commit) for ANY import of the package - including the process-pool
    # workers, which only need numpy/PIL helpers. Names still resolve exactly as that import made
    # them, just on first use.
    if name.startswith('__'):
        raise AttributeError(name)
    # import_module, not `from . import`: that looks the name up on this package first,
    # which would land right back here.
    _mt = importlib.import_module(__name__ + '.manga_translator')
    try:
        value = getattr(_mt, name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
    globals()[name] = value
    return value
