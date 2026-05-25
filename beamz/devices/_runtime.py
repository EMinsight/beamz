"""Small helpers for device objects with mutable compiled runtime state."""


class RuntimeStateProxy:
    """Proxy selected runtime attributes to ``self._state`` after preparation."""

    _RUNTIME_ATTRS: set[str] | frozenset[str] = frozenset()

    def __getattr__(self, name):
        if name in self._RUNTIME_ATTRS and "_state" in self.__dict__:
            return getattr(self._state, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def _set_runtime_attr(self, name, value) -> bool:
        if name in self._RUNTIME_ATTRS and "_state" in self.__dict__:
            setattr(self._state, name, value)
            return True
        return False

    def __setattr__(self, name, value):
        if self._set_runtime_attr(name, value):
            return
        object.__setattr__(self, name, value)
