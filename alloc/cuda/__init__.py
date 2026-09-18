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

from alloc.cuda.allocator import CudaAllocator, SIMULATED, occupancy_report  # noqa: E402

__all__ = ["CudaAllocator", "SIMULATED", "occupancy_report"]
