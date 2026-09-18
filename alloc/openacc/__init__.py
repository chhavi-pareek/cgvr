"""ctypes wrapper for the OpenACC allocator (alloc_acc.c).

`make serial` (any C compiler) builds a host-only library for oracle checks;
`make acc` (nvc) builds the T4 offload version. `backend()` reports which.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

from alloc.common import finish, prepare

_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_DIR, "libparity_acc." + ("dylib" if sys.platform == "darwin" else "so"))
_lib = None


def build(target="serial"):
    subprocess.run(["make", "-s", "-C", _DIR, target], check=True)
    global _lib
    _lib = None
    return load()


def available():
    return os.path.exists(_LIB)


def load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_LIB)
        dp = ctypes.POINTER(ctypes.c_double)
        ip = ctypes.POINTER(ctypes.c_int32)
        lib.parity_alloc_acc.restype = ctypes.c_int
        lib.parity_alloc_acc.argtypes = [
            ctypes.c_int, ctypes.c_int, dp, dp, dp, dp, dp,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ip, dp]
        lib.parity_acc_backend.restype = ctypes.c_int
        _lib = lib
    return _lib


def backend():
    return "openacc" if load().parity_acc_backend() else "serial-c"


class OpenACCAllocator:
    def __init__(self, table, rtol=1e-4, btol=1e-3, max_iter=64, fill=True):
        self.table = table
        self.rtol = rtol
        self.btol = btol
        self.max_iter = max_iter
        self.fill = fill
        self.lam = None
        self.lib = load()

    def allocate(self, salience, cost, budget, headroom=None):
        s, q, c, e, h, lam_max = prepare(self.table, salience, cost, headroom)
        n, m = len(s), len(q)
        a = np.empty(n, np.int32)
        stats = np.zeros(5, np.float64)
        dp = ctypes.POINTER(ctypes.c_double)
        rc = self.lib.parity_alloc_acc(
            n, m, s.ctypes.data_as(dp), q.ctypes.data_as(dp), c.ctypes.data_as(dp),
            e.ctypes.data_as(dp), h.ctypes.data_as(dp), float(budget),
            -1.0 if self.lam is None else float(self.lam), float(lam_max),
            self.rtol, self.btol, self.max_iter, int(self.fill),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), stats.ctypes.data_as(dp))
        if rc != 0:
            raise RuntimeError("parity_alloc_acc failed")
        lam, t, evals, infeasible, steps = stats
        infeasible = bool(infeasible)
        if not (infeasible and lam_max < 0):
            self.lam = float(lam)
        return finish(s, q, c, a, float(lam), int(evals), infeasible, int(steps))
