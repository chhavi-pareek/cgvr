"""Exact solvers for the multiple-choice knapsack, for validating the allocator.

brute_force: enumerates all m^N assignments (N <= 8).
dp_exact:    dynamic program over integer-tick costs, exact for the quantised
             problem, usable at N = 200.
"""
import itertools

import numpy as np


def _utility(table, salience, headroom):
    U = np.asarray(salience, np.float64)[:, None] * table.quality[None, :]
    if headroom is not None:
        feas = table.err[None, :] <= np.asarray(headroom, np.float64)[:, None]
        U = np.where(feas, U, -np.inf)
    return U


def brute_force(table, salience, cost, budget, headroom=None):
    U = _utility(table, salience, headroom)
    cost = np.asarray(cost, np.float64)
    n = len(salience)
    choices = [np.flatnonzero(np.isfinite(U[i])) for i in range(n)]
    best, best_a = -np.inf, None
    for a in itertools.product(*choices):
        a = np.array(a)
        if cost[a].sum() <= budget:
            u = U[np.arange(n), a].sum()
            if u > best:
                best, best_a = u, a
    return best_a, best


def dp_exact(table, salience, cost_ticks, budget_ticks, headroom=None):
    """cost_ticks: non-negative ints per row. budget_ticks: int."""
    U = _utility(table, salience, headroom)
    ct = np.asarray(cost_ticks, np.int64)
    B = int(budget_ticks)
    n, m = U.shape
    f = np.zeros(B + 1)  # zero agents placed: utility 0 at any budget
    choice = np.full((n, B + 1), -1, np.int16)
    for i in range(n):
        g = np.full(B + 1, -np.inf)
        for c in range(m):
            if not np.isfinite(U[i, c]) or ct[c] > B:
                continue
            cand = f[: B + 1 - ct[c]] + U[i, c]
            seg = g[ct[c]:]
            better = cand > seg
            seg[better] = cand[better]
            choice[i, ct[c]:][better] = c
        f = g
    if not np.isfinite(f[B]):
        return None, -np.inf
    a = np.zeros(n, np.int64)
    b = B
    for i in range(n - 1, -1, -1):
        c = int(choice[i, b])
        a[i] = c
        b -= int(ct[c])
    return a, float(f[B])
