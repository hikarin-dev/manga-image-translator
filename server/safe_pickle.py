"""Restricted unpickling for data arriving FROM an executor.

The local worker is trusted — we spawned it. An aux node is someone else's machine, and
`pickle.loads` on what it sends back would hand it code execution here. Everything an
executor returns on the gallery path is a plain summary dict ({count, failed, telemetry}),
so nothing outside a handful of container/scalar types ever needs to be reconstructed.

Note that plain dicts/lists/ints/strings are pickle *opcodes* and never reach find_class —
this allowlist exists to make the dangerous cases (os.system, subprocess.Popen, and the
various __reduce__ gadgets) fail loudly instead of executing.
"""
import io
import pickle

_ALLOWED = {
    'builtins': frozenset({'dict', 'list', 'tuple', 'set', 'frozenset',
                           'int', 'float', 'complex', 'bool', 'str', 'bytes'}),
    'collections': frozenset({'OrderedDict', 'defaultdict'}),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name in _ALLOWED.get(module, ()):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f'Deserialization of {module}.{name} is not allowed')


def loads(data: bytes):
    return _RestrictedUnpickler(io.BytesIO(data)).load()
