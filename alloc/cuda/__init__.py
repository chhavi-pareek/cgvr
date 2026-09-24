"""CUDA allocator (Numba-CUDA).

Without an NVIDIA driver (this Mac) the module falls back to numba's CUDA
simulator, which runs the same kernel source on the CPU for correctness only.
Set NUMBA_ENABLE_CUDASIM=0 explicitly to force the real driver.
"""
import os
import shutil
import sys

if "NUMBA_ENABLE_CUDASIM" not in os.environ:
    if sys.platform == "darwin" or shutil.which("nvidia-smi") is None:
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
        if "numba" in sys.modules and "numba.cuda" not in sys.modules:
            import numba
            numba.config.reload_config()

from alloc.cuda.allocator import CudaAllocator, SIMULATED, cuda, occupancy_report  # noqa: E402

# `cuda` is re-exported so callers can synchronise against whichever numba CUDA module is
# live (real driver or simulator) without importing numba themselves; bench/speedup.py
# relies on it, and on the Mac that path is unreachable because SIMULATED short-circuits it.
__all__ = ["CudaAllocator", "SIMULATED", "cuda", "occupancy_report"]
