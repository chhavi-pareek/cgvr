"""Allocator whose per-frame work is a sort, not a scan over the whole table.

alloc/serial.py evaluates the Lagrangian by scanning every agent against every row, ~16 times
per frame to bisect lambda. At N = 2000 with the pruned table that is the dominant cost of the
whole frame (42-70% measured in the engine, bench/engine_bench.csv).

It does not have to be. Every agent chooses from the SAME menu at the SAME prices, so

    argmax_j [ s_i q_j - lam c_j ]  =  argmax_j [ q_j - (lam / s_i) c_j ]

and the choice depends on the single scalar theta_i = lam / s_i. As theta grows from 0 the
argmax walks the **upper concave hull** of the (cost, quality) points, from the highest-quality
vertex down to the cheapest one, switching at the hull's slopes. Rows off the hull are chosen
by no agent at any lambda -- they are the LP-dominated ones.

So: build the hull once (the menu is shared and static), sort the agents by salience once, and
the assignment is a step function -- the most salient block takes the best vertex, the next
block the next, and so on. Block boundaries are a binary search per hull edge, and the total
cost is a prefix sum. Cost per lambda evaluation is O(K log n) for K hull vertices instead of
O(n m), and the sort amortises because the salience order barely changes between frames.

Prior art: this is the classic greedy/hull solution of the multiple-choice knapsack
(Sinha & Zoltners 1979; Dudzinski & Walukiewicz 1987), and LOD selection as a knapsack is
Funkhouser & Sequin 1993. What is specific here is that a *shared* menu collapses the search
to a sort by salience.

Caveat, stated because it is load-bearing: this requires every agent to rank the rows
identically, which holds only while quality is agent-independent. It is today. Per-agent
quality (phase 6's INT8 scoring) would break it, and the fallback is alloc/serial.py.
"""
import numpy as np

from alloc.serial import Result


def upper_hull(cost, quality):
    """Indices of the upper concave hull of (cost, quality), ordered by increasing cost.

    These are exactly the rows some agent picks at some lambda; everything else is dominated
    either outright or by a convex combination of two hull vertices."""
    cost = np.asarray(cost, np.float64)
    quality = np.asarray(quality, np.float64)
    order = np.lexsort((-quality, cost))          # cost asc, then quality desc
    hull = []
    for j in order:
        if hull and quality[j] <= quality[hull[-1]]:
            continue                               # costs more, no better: dominated
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # drop b if it lies below the chord a->j (i.e. the turn is not concave)
            lhs = (quality[b] - quality[a]) * (cost[j] - cost[a])
            rhs = (quality[j] - quality[a]) * (cost[b] - cost[a])
            if lhs <= rhs:
                hull.pop()
            else:
                break
        hull.append(j)
    return np.array(hull, np.int64)


def hull_thetas(cost, quality, hull):
    """theta at which the choice switches from hull[k+1] to hull[k], for k = 0..K-2.

    Decreasing in k: a large theta (a low-salience agent, or a high lambda) means the cheap
    end of the hull, a small theta means the expensive end."""
    c, q = np.asarray(cost)[hull], np.asarray(quality)[hull]
    return (q[1:] - q[:-1]) / (c[1:] - c[:-1])


class HullAllocator:
    """Same interface as SerialAllocator. `fill` is accepted and ignored: the hull solution is
    the exact LP optimum, so there is no integrality gap to fill beyond the single agent that
    can straddle a breakpoint."""

    def __init__(self, table, rtol=1e-4, btol=1e-3, max_iter=64, fill=True):
        self.table = table
        self.rtol, self.btol, self.max_iter = rtol, btol, max_iter
        self.fill = fill
        self.lam = None
        self._cache = None

    def _hull_for(self, cost, mask=None):
        q = self.table.quality
        if mask is None:
            idx = np.arange(len(q))
        else:
            idx = np.flatnonzero(mask)
        h = upper_hull(cost[idx], q[idx])
        rows = idx[h]
        return rows, hull_thetas(cost, q, rows)

    @staticmethod
    def _blocks(lam, s_desc, thetas):
        """Block boundaries over agents sorted by salience DESCENDING.

        Agent i takes hull vertex v_i = #{k : lam / thetas[k] <= s_i}. thetas is decreasing, so
        the thresholds S_k = lam / thetas[k] are increasing and each contributes one contiguous
        prefix of the sorted agents. The result is a step function: K blocks, boundaries found
        by one binary search per hull edge, total cost by prefix counts. No per-agent scan."""
        n = len(s_desc)
        K = len(thetas) + 1
        if K == 1 or n == 0:
            return np.array([0, n], np.int64), np.array([0], np.int64)
        S = lam / np.maximum(thetas, 1e-300)                 # ascending
        # count of agents with s_i >= S_k, for each k; descending in k
        cnt = np.searchsorted(-s_desc, -S, side="right").astype(np.int64)
        bounds = np.concatenate([[0], cnt[::-1], [n]])
        bounds = np.maximum.accumulate(np.minimum(bounds, n))
        verts = np.arange(K - 1, -1, -1, dtype=np.int64)     # K-1 down to 0
        return bounds, verts

    def allocate(self, salience, cost, budget, headroom=None):
        tab = self.table
        s = np.asarray(salience, np.float64)
        cost = np.asarray(cost, np.float64)
        n = len(s)
        e = tab.err

        # Group by FEASIBLE SET, not by headroom value. Two agents with different headroom
        # but the same set of affordable rows share a hull and a sort, so the number of groups
        # is bounded by the number of distinct error levels (two, for the phase-7 column) and
        # not by the number of agents. Grouping on the raw headroom instead produces one group
        # per agent and makes this allocator 25x SLOWER than the one it replaces.
        if headroom is None:
            groups = [(np.arange(n), None)]
        else:
            hr = np.asarray(headroom, np.float64)
            levels = np.unique(e)
            g = np.searchsorted(levels, hr, side="right")     # affordable levels per agent
            if (g == 0).any():
                raise ValueError("agent with no feasible configuration")
            groups = []
            for gv in np.unique(g):
                sel = np.flatnonzero(g == gv)
                groups.append((sel, e <= levels[gv - 1] + 1e-12))

        prepared = []
        for idx, mask in groups:
            rows, thetas = self._hull_for(cost, mask)
            order = idx[np.argsort(-s[idx], kind="stable")]
            prepared.append((order, s[order], rows, thetas))

        def total(lam):
            t = 0.0
            for order, ss, rows, thetas in prepared:
                bounds, verts = self._blocks(lam, ss, thetas)
                sizes = np.diff(bounds)
                t += float((sizes * cost[rows[verts]]).sum())
            return t

        t0 = total(0.0)
        if t0 <= budget:
            lam = 0.0
        else:
            hi = 1.0
            while total(hi) > budget and hi < 1e18:
                hi *= 4.0
            lo_l = 0.0
            for _ in range(self.max_iter):
                if hi - lo_l <= self.rtol * max(hi, 1e-12):
                    break
                mid = 0.5 * (lo_l + hi)
                if total(mid) <= budget:
                    hi = mid
                else:
                    lo_l = mid
            lam = hi
        self.lam = lam

        assign = np.empty(n, np.int64)
        for order, ss, rows, thetas in prepared:
            bounds, verts = self._blocks(lam, ss, thetas)
            for bi in range(len(verts)):
                lo_b, hi_b = int(bounds[bi]), int(bounds[bi + 1])
                if hi_b > lo_b:
                    assign[order[lo_b:hi_b]] = rows[verts[bi]]
        tot = float(cost[assign].sum())
        util = float((s * tab.quality[assign]).sum())
        return Result(assign, tot, util, lam, self.max_iter, tot > budget * (1 + 1e-9), 0)
