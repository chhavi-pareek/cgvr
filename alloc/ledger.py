"""Per-agent accumulated-divergence ledger (invariant 3).

D_i accumulates the error rate of the configuration the allocator chose each frame and is
reset only by reconciliation. headroom_i = cap - D_i enters the allocator's feasibility mask,
which the allocator never violates (it overrides the budget), so an agent at the cap is
restored regardless of camera. Error-rate 0 rows (behaviour tier 0 with the reference
process) are always feasible, so the mask can never be empty.
"""
import numpy as np


class ErrorLedger:
    def __init__(self, n, cap, rng=None):
        self.cap = float(cap)
        # desynchronise initial ledgers so restorations do not arrive as one wave
        self.D = np.zeros(n) if rng is None else rng.uniform(0.0, 0.5 * cap, n)
        self.restorations = 0

    def headroom(self):
        return np.maximum(self.cap - self.D, 0.0)

    def accrue(self, err_rate):
        self.D += np.asarray(err_rate, np.float64)
        return self.D

    def reset(self, idx):
        idx = np.atleast_1d(idx)
        self.restorations += idx.size
        self.D[idx] = 0.0

    def check(self, table, assign):
        """True if every chosen row's rate fits inside the headroom (called before accrue)."""
        return bool(np.all(table.err[assign] <= self.headroom() + 1e-12))
