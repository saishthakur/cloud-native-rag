import threading


class PicklableLocked:
    """Mixin: objects guarded by ``self.lock`` that can still be snapshotted with pickle."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.RLock()
