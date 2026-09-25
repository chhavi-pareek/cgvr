"""Ablation of the joint dual-budget solve: the two budgets resolved in sequence.

Two stage-1 policies, because the first one turned out not to constrain stage 2 at all.

`mode="cheapest"` -- every agent independently takes the cheapest configuration whose
error rate fits its own headroom. That is the floor of the error-feasible set, so stage 2
starts with the whole time budget unspent and the error decision restricts nothing.

`mode="maximal"` -- every agent independently takes the *highest-utility* configuration
whose error rate fits its headroom, i.e. it spends its error budget as hard as it can
without ever looking at the frame-time budget or at the rest of the crowd. Stage 2 then
has to claw the time back: repeatedly apply the downgrade with the smallest utility loss
per unit of time saved until the frame budget is met. This is the sharper opponent -- the
error-side commitment is made first and stage 2 can only react to it.

Stage 2 in both modes finishes with the serial allocator's own greedy fill
(`SerialAllocator._finish`) run to exhaustion, so any slack left over is spent. Table,
costs, budget, headroom mask and fill step are identical to SerialAllocator; the only
difference is joint Lagrangian relaxation over both budgets versus this two-stage
resolution. Infeasible rows carry -inf utility, so neither stage can leave the
error-feasible set, and the result is feasible for both budgets by construction.
"""
import numpy as np

from alloc.serial import SerialAllocator

MODES = ("cheapest", "maximal")


class SequentialAllocator(SerialAllocator):
    """Same interface as SerialAllocator; `lam` is nan because none is solved for."""

    max_fill_rounds = 256

    def __init__(self, table, *args, mode="cheapest", **kw):
        super().__init__(table, *args, **kw)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode

    def _stage1(self, U, feas, n):
        if self.mode == "maximal":
            return np.argmax(U, axis=1).astype(np.int64)  # ties to the cheapest such row
        if feas is None:
            return np.zeros(n, np.int64)  # rows are cost-ascending: row 0 is the cheapest
        return np.argmax(feas, axis=1).astype(np.int64)  # first (= cheapest) feasible row

    def _downgrade(self, a, t, U, c, budget):
        """Claw back time: while over budget, apply the single move with the smallest
        utility loss per unit of time saved. A cheaper row that is also better carries a
        negative loss and so is taken first. Returns (a, cost, steps, infeasible)."""
        idx = np.arange(len(a))
        steps = 0
        limit = len(a) * len(c)  # each move takes one agent strictly cheaper
        while t > budget and steps < limit:
            dc = c[a][:, None] - c[None, :]
            du = U[idx, a][:, None] - U  # +inf against an error-infeasible row
            ok = dc > 0
            ratio = np.where(ok, du / np.where(ok, dc, 1.0), np.inf)
            i = int(np.argmin(ratio.min(1)))
            j = int(np.argmin(ratio[i]))
            if not np.isfinite(ratio[i, j]):
                break  # every agent already sits on its cheapest error-feasible row
            a[i] = j
            t -= dc[i, j]
            steps += 1
        t = float(c[a].sum())
        return a, t, steps, t > budget

    def allocate(self, salience, cost, budget, headroom=None):
        tab = self.table
        s = np.asarray(salience, np.float64)
        cost = np.asarray(cost, np.float64)
        n = len(s)
        order = np.lexsort((-tab.quality, cost))
        q, c, e = tab.quality[order], cost[order], tab.err[order]
        U = s[:, None] * q[None, :]
        feas = None
        if headroom is not None:
            feas = e[None, :] <= np.asarray(headroom, np.float64)[:, None]
            if not feas.any(1).all():
                raise ValueError("agent with no feasible configuration")
            U = np.where(feas, U, -np.inf)
        a = self._stage1(U, feas, n)
        t = float(c[a].sum())
        self.lam = None
        nan = float("nan")

        down = 0
        if t > budget:
            a, t, down, infeasible = self._downgrade(a, t, U, c, budget)
            if infeasible:  # cannot be paid for at any error-feasible assignment
                return self._finish(a, t, U, c, budget, nan, 1, True, order, fill=False)
        # _finish takes at most one upgrade per agent per call, which is enough to close the
        # joint solver's integrality gap but not to spend a whole budget, so run it to
        # exhaustion. Identity order keeps `a` in sorted-row space between rounds.
        steps = 0
        if self.fill:
            idm = np.arange(len(c))
            for _ in range(self.max_fill_rounds):
                r = self._finish(a, t, U, c, budget, nan, 1, False, idm)
                a, t = np.asarray(r.assign), r.cost
                if r.fill_steps == 0:
                    break
                steps += r.fill_steps
        res = self._finish(a, t, U, c, budget, nan, 1, False, order, fill=False)
        res.fill_steps = steps + down
        return res


class MaximalSequentialAllocator(SequentialAllocator):
    """`mode="maximal"` bound into the constructor, so the sweep can pass it as alloc_cls."""

    def __init__(self, table, *args, **kw):
        super().__init__(table, *args, mode="maximal", **kw)
