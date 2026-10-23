"""Durable same-directory file replacement, shared by local stores."""
import os
import math
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import time


_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_HELD_PATHS = threading.local()


@contextmanager
def file_lock(path: Path, *, timeout: float = 10.0):
    """Serialize a file transaction across threads, instances and processes.

    Lock a stable sidecar, not the replaced data file. The OS releases the lock
    if a worker exits; the sidecar must therefore never be unlinked on release.
    """
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("file lock timeout must be finite and non-negative")
    deadline = time.monotonic() + timeout
    path = Path(path).resolve()
    key = os.path.normcase(str(path))
    with _PATH_LOCKS_GUARD:
        local_lock = _PATH_LOCKS.setdefault(key, threading.RLock())
    if not local_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise TimeoutError("账户或日志正在写入，请稍后重试")
    try:
        held = getattr(_HELD_PATHS, "paths", None)
        if held is None:
            held = _HELD_PATHS.paths = set()
        if key in held:
            # RLock is reentrant, while locking the same sidecar through a
            # second OS handle is not. Reuse this thread's outer transaction.
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_name(path.name + ".lock").open("a+b") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("账户或日志正在写入，请稍后重试") from exc
                    time.sleep(0.025)
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        local_lock.release()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + "-", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
