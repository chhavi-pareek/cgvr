"""Ablation of the joint dual-budget solve: the two budgets resolved in sequence.

Stage 1 (error budget, per-agent, local). Every agent independently takes the
cheapest configuration whose error rate fits its own headroom -- ties to higher
quality, which is the first row of the (cost asc, quality desc) ordering the
serial allocator already uses. There is no multiplier and no coupling between
agents: the per-agent cap is discharged without ever looking at the frame-time
budget or at what the rest of the crowd is doing.

Stage 2 (time budget, greedy). Whatever time stage 1 left over is spent by the
serial allocator's own greedy fill (`SerialAllocator._finish`): repeatedly take
the single upgrade with the largest utility gain per unit of extra cost that
still fits the remaining slack. Infeasible rows carry -inf utility, so no
upgrade can ever leave the error-feasible set.

Table, costs, budget, headroom mask and fill step are identical to
SerialAllocator; the only difference is joint Lagrangian relaxation over both
budgets versus this two-stage resolution. The result is feasible for both
budgets by construction, and its utility is a lower bound on what a solver that
couples the two budgets can reach.
"""
import numpy as np

from alloc.serial import SerialAllocator


class SequentialAllocator(SerialAllocator):
    """Same interface as SerialAllocator; `lam` is nan because none is solved for."""

    max_fill_rounds = 256

    def allocate(self, salience, cost, budget, headroom=None):
        tab = self.table
        s = np.asarray(salience, np.float64)
        cost = np.asarray(cost, np.float64)
        n = len(s)
        order = np.lexsort((-tab.quality, cost))
        q, c, e = tab.quality[order], cost[order], tab.err[order]
        U = s[:, None] * q[None, :]
        if headroom is None:
            a = np.zeros(n, np.int64)  # rows are cost-ascending: row 0 is the cheapest
        else:
            feas = e[None, :] <= np.asarray(headroom, np.float64)[:, None]
            if not feas.any(1).all():
                raise ValueError("agent with no feasible configuration")
            U = np.where(feas, U, -np.inf)
            a = np.argmax(feas, axis=1).astype(np.int64)  # first (= cheapest) feasible row
        t = float(c[a].sum())
        self.lam = None
        nan = float("nan")
        if t > budget:  # stage 1 alone already overspends; nothing to greedily add
            return self._finish(a, t, U, c, budget, nan, 1, True, order, fill=False)
        # stage 2: _finish takes at most one upgrade per agent per call, which is enough to
        # close the joint solver's integrality gap but not to spend a whole budget, so run it
        # to exhaustion. Identity order keeps `a` in sorted-row space between rounds.
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
        res.fill_steps = steps
        return res
