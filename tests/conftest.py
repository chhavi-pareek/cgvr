import os
import shutil
import sys

if "NUMBA_ENABLE_CUDASIM" not in os.environ:
    if sys.platform == "darwin" or shutil.which("nvidia-smi") is None:
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
