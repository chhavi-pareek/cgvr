"""Two genuine multipliers: frame time AND aggregate crowd divergence.

Why this exists. The sequential ablation (alloc/sequential.py, STATE.md) showed that PARITY's
"dual-budget joint" formulation is not joint at all: the error budget enters alloc/serial.py as
a hard PER-AGENT mask, so it couples nothing, and a two-stage solve reaches the joint solver's
exact allocation. One multiplier, one mask.

What that leaves unguarded is the crowd. A per-agent bound says nothing about aggregate error:
if every agent sits at 99% of its cap, every agent is individually within guarantee and the
simulation as a whole is as wrong as it can be. Anything that reads a crowd-level statistic --
flow rate, egress time, a density field -- cares about the aggregate, not the worst individual.

Invariant 4 makes that aggregate well defined. The view-invariant core is simulated exactly for
every agent at every tier, so agent states are conditionally independent given the core, and
the KL divergence of the whole crowd's joint state from the reference is the SUM of the
per-agent divergences. An additive quantity over agents is a knapsack resource. So:

    maximise   sum_i salience_i * quality(c_i)
    subject to sum_i time(c_i)  <= T          (frame budget)
               sum_i rate(c_i)  <= R          (aggregate divergence budget, NEW)
               rate(c_i) <= headroom_i        (per-agent cap, invariant 3, unchanged)

Now both budgets are shared resources, both get a multiplier, and the agents genuinely couple
through both. This is the form in which "dual-budget joint allocation" has teeth -- and it is
testable: if a two-stage solve still matches, the claim is dead for good.

The per-agent cap is kept as a hard mask. It is a safety property (no agent is ever arbitrarily
wrong) and the aggregate budget does not imply it: a shared budget alone would happily let one
agent absorb the entire allowance.
"""
import numpy as np

from alloc.serial import Result


class DualAllocator:
    """Lagrangian relaxation over two additive constraints, solved by nested bisection.

    The dual is convex in (lam, mu); total time is non-increasing in lam and total rate is
    non-increasing in mu. Outer bisection on mu, inner on lam, then a feasibility check on the
    returned point -- if the inner solve moved the rate back over budget the outer loop sees it.
    """

    def __init__(self, table, rtol=1e-4, btol=1e-3, max_iter=48, outer_iter=40, fill=True):
        self.table = table
        self.rtol, self.btol = rtol, btol
        self.max_iter, self.outer_iter = max_iter, outer_iter
        self.fill = fill
        self.lam = None
        self.mu = None

    # -- one evaluation: every agent picks independently given (lam, mu) -------------------
    @staticmethod
    def _select(U, c, e, lam, mu):
        return np.argmax(U - lam * c[None, :] - mu * e[None, :], axis=1)

    def _prepare(self, salience, cost, headroom):
        tab = self.table
        s = np.asarray(salience, np.float64)
        cost = np.asarray(cost, np.float64)
        order = np.lexsort((-tab.quality, cost))
        q, c, e = tab.quality[order], cost[order], tab.err[order]
        U = s[:, None] * q[None, :]
        if headroom is not None:
            feas = e[None, :] <= np.asarray(headroom, np.float64)[:, None]
            if not feas.any(1).all():
                raise ValueError("agent with no feasible configuration")
            U = np.where(feas, U, -np.inf)
        return U, q, c, e, order

    def _lam_max(self, U, c):
        dc = np.diff(np.unique(c))
        dc = dc[dc > 0]
        if len(dc) == 0:
            return 0.0
        span = np.where(np.isfinite(U), U, np.nan)
        return max(float(np.nanmax(span) - min(0.0, np.nanmin(span))) / float(dc.min()), 1e-12)

    def _solve_lam(self, U, c, e, mu, budget, evals):
        """Inner: smallest lam >= 0 meeting the time budget at this mu."""
        a = self._select(U, c, e, 0.0, mu); evals[0] += 1
        if float(c[a].sum()) <= budget:
            return 0.0, a
        hi = self._lam_max(U, c)
        a_hi = self._select(U, c, e, hi, mu); evals[0] += 1
        if float(c[a_hi].sum()) > budget:
            return hi, a_hi                      # infeasible even at the floor
        lo = 0.0
        for _ in range(self.max_iter):
            if hi - lo <= self.rtol * max(hi, 1e-12):
                break
            mid = 0.5 * (lo + hi)
            a_m = self._select(U, c, e, mid, mu); evals[0] += 1
            if float(c[a_m].sum()) <= budget:
                hi, a_hi = mid, a_m
            else:
                lo = mid
        return hi, a_hi

    def allocate(self, salience, cost, budget, rate_budget=np.inf, headroom=None):
        U, q, c, e, order = self._prepare(salience, cost, headroom)
        n = len(U)
        evals = [0]

        lam, a = self._solve_lam(U, c, e, 0.0, budget, evals)
        if float(e[a].sum()) <= rate_budget or not np.isfinite(rate_budget):
            self.lam, self.mu = lam, 0.0
            return self._finish(a, U, c, e, budget, rate_budget, lam, 0.0, evals[0], order)

        # the aggregate divergence budget binds: raise mu until it is met
        de = np.diff(np.unique(e))
        de = de[de > 0]
        span = np.where(np.isfinite(U), U, np.nan)
        mu_hi = max(float(np.nanmax(span)) / float(de.min()), 1e-12) if len(de) else 1e-12
        lo, hi = 0.0, mu_hi
        best = None
        for _ in range(self.outer_iter):
            mid = 0.5 * (lo + hi)
            lam_m, a_m = self._solve_lam(U, c, e, mid, budget, evals)
            if float(e[a_m].sum()) <= rate_budget:
                hi, best = mid, (lam_m, a_m)
            else:
                lo = mid
            if hi - lo <= self.rtol * max(hi, 1e-12):
                break
        if best is None:
            lam_m, a_m = self._solve_lam(U, c, e, mu_hi, budget, evals)
            best = (mu_hi, (lam_m, a_m))[1]
            hi = mu_hi
        lam, a = best
        self.lam, self.mu = lam, hi
        return self._finish(a, U, c, e, budget, rate_budget, lam, hi, evals[0], order)

    def _finish(self, a, U, c, e, budget, rate_budget, lam, mu, evals, order):
        n = len(a)
        idx = np.arange(n)
        steps = 0
        t = float(c[a].sum())
        r = float(e[a].sum())
        infeasible = t > budget * (1 + 1e-9) or r > rate_budget * (1 + 1e-9)
        if self.fill and not infeasible:
            # greedy fill of the integrality gap in BOTH resources at once
            cur_u, cur_c, cur_e = U[idx, a], c[a], e[a]
            for _ in range(n):
                du = U - cur_u[:, None]
                dc = c[None, :] - cur_c[:, None]
                dr = e[None, :] - cur_e[:, None]
                ok = (du > 0) & (dc + cur_c[:, None] >= 0)
                ok &= (dc <= budget - t) & (dr <= rate_budget - r)
                ok &= (dc > 0) | (dr > 0)
                if not ok.any():
                    break
                # rank by utility gain per unit of the scarcer resource actually consumed
                denom = np.where(dc > 0, dc, 0.0) * max(lam, 1e-12) + np.where(dr > 0, dr, 0.0) * max(mu, 1e-12)
                ratio = np.where(ok & (denom > 0), du / np.where(denom > 0, denom, 1.0), -np.inf)
                i = int(np.argmax(ratio.max(1)))
                j = int(np.argmax(ratio[i]))
                if not np.isfinite(ratio[i, j]):
                    break
                t += c[j] - cur_c[i]; r += e[j] - cur_e[i]
                cur_u[i], cur_c[i], cur_e[i] = U[i, j], c[j], e[j]
                a[i] = j
                steps += 1
        util = float(U[idx, a].sum())
        res = Result(order[a], t, util, lam, evals, infeasible, steps)
        res.rate = r
        res.mu = mu
        return res
