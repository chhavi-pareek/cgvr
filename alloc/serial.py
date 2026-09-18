"""Serial Lagrangian-bisection allocator with warm-started multiplier.

For multiplier lam >= 0 each agent independently picks
    argmax_c  salience_i * quality_c - lam * cost_c
over rows whose error rate fits its headroom, ties broken toward lower cost.
Total cost is non-increasing in lam. The returned assignment is always the one
evaluated at the feasible end of the bracket, so its cost was computed and
compared against the budget directly.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    assign: np.ndarray
    cost: float
    utility: float
    lam: float
    evals: int
    infeasible: bool
    fill_steps: int = 0


class SerialAllocator:
    def __init__(self, table, rtol=1e-4, btol=1e-3, max_iter=64, fill=True):
        self.table = table
        self.rtol = rtol
        self.btol = btol
        self.max_iter = max_iter
        self.fill = fill
        self.lam = None

    # -- one Lagrangian evaluation -------------------------------------------
    @staticmethod
    def _select(U, cost, lam):
        score = U - lam * cost[None, :]
        return np.argmax(score, axis=1)  # first max: rows are sorted by cost asc

    def allocate(self, salience, cost, budget, headroom=None):
        tab = self.table
        s = np.asarray(salience, np.float64)
        cost = np.asarray(cost, np.float64)
        n = len(s)
        # order rows by (cost asc, quality desc) so argmax tie-break prefers cheap
        order = np.lexsort((-tab.quality, cost))
        q, c, e = tab.quality[order], cost[order], tab.err[order]
        U = s[:, None] * q[None, :]
        if headroom is not None:
            feas = e[None, :] <= np.asarray(headroom, np.float64)[:, None]
            if not feas.any(1).all():
                raise ValueError("agent with no feasible configuration")
            U = np.where(feas, U, -np.inf)
        evals = 0

        def T(lam):
            nonlocal evals
            evals += 1
            a = self._select(U, c, lam)
            return a, float(c[a].sum())

        a0, t0 = T(0.0)
        if t0 <= budget:
            self.lam = 0.0
            return self._finish(a0, t0, U, c, budget, 0.0, evals, False, order)

        dc = np.diff(np.unique(c))
        dc = dc[dc > 0]
        if len(dc) == 0:  # all rows cost the same and T(0) > budget
            return self._finish(a0, t0, U, c, budget, 0.0, evals, True, order, fill=False)
        span = np.where(np.isfinite(U), U, np.nan)
        lam_max = float(np.nanmax(span) - min(0.0, np.nanmin(span))) / float(dc.min())
        lam_max = max(lam_max, 1e-12)
        a_hi, t_hi = T(lam_max)
        if t_hi > budget:
            self.lam = lam_max
            return self._finish(a_hi, t_hi, U, c, budget, lam_max, evals, True, order, fill=False)

        # bracket from warm start
        if self.lam is None or self.lam <= 0.0:
            lo, hi = 0.0, lam_max
            a_lo, t_lo = a0, t0
        else:
            lo = min(self.lam / 2, lam_max)
            hi = min(self.lam * 2, lam_max)
            a_lo, t_lo = T(lo)
            a_hi, t_hi = T(hi)
            while t_hi > budget:
                lo, a_lo, t_lo = hi, a_hi, t_hi
                hi = min(hi * 2, lam_max)
                a_hi, t_hi = T(hi)
            while t_lo <= budget:
                hi, a_hi, t_hi = lo, a_lo, t_lo
                lo = lo / 2 if lo > 1e-12 else 0.0
                a_lo, t_lo = T(lo)
                if lo == 0.0:
                    break
        # invariant: t_lo > budget >= t_hi
        for _ in range(self.max_iter):
            if hi - lo <= self.rtol * hi or budget - t_hi <= self.btol * budget:
                break
            mid = 0.5 * (lo + hi)
            a_m, t_m = T(mid)
            if t_m <= budget:
                hi, a_hi, t_hi = mid, a_m, t_m
            else:
                lo, a_lo, t_lo = mid, a_m, t_m
        self.lam = hi
        return self._finish(a_hi, t_hi, U, c, budget, hi, evals, False, order)

    # -- greedy fill of the integrality gap -----------------------------------
    def _finish(self, a, t, U, c, budget, lam, evals, infeasible, order, fill=None):
        fill = self.fill if fill is None else fill
        n = len(a)
        steps = 0
        if fill and not infeasible:
            slack = budget - t
            cur_u = U[np.arange(n), a]
            cur_c = c[a]
            for _ in range(n):
                du = U - cur_u[:, None]
                dcost = c[None, :] - cur_c[:, None]
                ok = (du > 0) & (dcost > 0) & (dcost <= slack)
                if not ok.any():
                    break
                ratio = np.where(ok, du / np.where(dcost > 0, dcost, 1.0), -np.inf)
                i = int(np.argmax(ratio.max(1)))
                j = int(np.argmax(ratio[i]))
                a[i] = j
                slack -= dcost[i, j]
                cur_u[i] = U[i, j]
                cur_c[i] = c[j]
                steps += 1
            t = float(c[a].sum())
        util = float(U[np.arange(n), a].sum())
        return Result(order[a], t, util, lam, evals, infeasible, steps)
