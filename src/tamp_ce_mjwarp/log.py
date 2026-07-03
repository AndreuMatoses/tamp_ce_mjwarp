"""Minimal status logging so a run shows what it's doing (compile / CE / plot / video)."""

import time
from contextlib import contextmanager

from tqdm import tqdm


def log(msg):
    """Print a status line that plays nicely with an active tqdm bar."""
    tqdm.write(f"[tamp] {msg}")


@contextmanager
def timed(msg):
    """Log `msg ...` then `msg — done in Xs`, timing the wrapped block."""
    log(f"{msg} ...")
    t = time.perf_counter()
    yield
    log(f"{msg} — done in {time.perf_counter() - t:.1f}s")
