import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # What the package used to import eagerly. Never runs; kept for editors and for
    # stages.module_closure, which reads these statements to hash the utils modules.
    from .sort import *  # noqa: F401,F403
    from .bubble import is_ignore  # noqa: F401
    from .generic import *  # noqa: F401,F403
    from .inference import *  # noqa: F401,F403
    from .log import *  # noqa: F401,F403
    from .textblock import *  # noqa: F401,F403
    from .threading import *  # noqa: F401,F403

# Resolved on first use instead of star-importing every submodule up front: `inference` pulls in
# torch, which the process-pool workers never need. No two submodules export different objects
# under one name, so the search order below changes nothing but what gets loaded; torch-bearing
# `inference` comes last so a name found anywhere else never loads it.
_SUBMODULES = ('sort', 'bubble', 'generic', 'log', 'textblock', 'threading', 'inference')


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(name)
    for sub in _SUBMODULES:
        module = importlib.import_module(f'{__name__}.{sub}')
        exported = getattr(module, '__all__', None)
        if (name in exported) if exported is not None else hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
