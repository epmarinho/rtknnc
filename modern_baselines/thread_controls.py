"""Required process-level thread declarations, checked before numerical imports."""

import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require_thread_controls():
    controls = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if any(value != "1" for value in controls.values()):
        raise RuntimeError(f"All thread controls must be explicitly set to '1': {controls}")
    return controls
