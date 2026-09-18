"""Multithreaded CPU allocator (shared-memory reference point).

Agents are split across threads with numba's prange (OpenMP-style fork/join).
Each Lagrangian evaluation is one parallel region: every thread scans the
table for its agents and writes the choice; the aggregate cost is a numpy sum
over the selected costs in agent order, which is the oracle's own summation
order, so totals match bit for bit even for float costs. The bisection loop runs on the host, as in the serial
allocator. Fill steps are one parallel region (per-agent best upgrade) plus a
serial first-max scan over agents.
"""
import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from alloc.common import drive, finish, prepare


@njit(parallel=True, cache=True)
def _select(s, q, c, e, h, lam, a):
    n = s.shape[0]
    m = q.shape[0]
    for i in prange(n):
        si = s[i]
        hi_ = h[i]
        best = -np.inf
        bj = -1
        bc = np.inf
        bq = -np.inf
        for j in range(m):
            if e[j] <= hi_:
                cj = c[j]
                qj = q[j]
                u = si * qj
                sc = u - lam * cj
                if sc > best or (sc == best and (cj < bc or (cj == bc and qj > bq))):
                    best = sc
                    bj = j
                    bc = cj
                    bq = qj
        a[i] = bj


@njit(parallel=True, cache=True)
def _fill_step(s, q, c, e, h, a, slack, br, bj):
    n = s.shape[0]
    m = q.shape[0]
    for i in prange(n):
        si = s[i]
        hi_ = h[i]
        ai = a[i]
        cu = si * q[ai]
        cc = c[ai]
        best = -np.inf
        bidx = -1
        bc = np.inf
        bq = -np.inf
        for j in range(m):
            if e[j] <= hi_:
                cj = c[j]
                qj = q[j]
                du = si * qj - cu
                dc = cj - cc
                if du > 0.0 and dc > 0.0 and dc <= slack:
                    r = du / dc
                    if r > best or (r == best and (cj < bc or (cj == bc and qj > bq))):
                        best = r
                        bidx = j
                        bc = cj
                        bq = qj
        br[i] = best
        bj[i] = bidx
    # first agent with the maximum ratio (numpy argmax semantics)
    best = -np.inf
    bi = -1
    for i in range(n):
        if br[i] > best:
            best = br[i]
            bi = i
    if bi < 0:
        return -1.0
    j = bj[bi]
    d = c[j] - c[a[bi]]
    a[bi] = j
    return d


class ThreadedAllocator:
    def __init__(self, table, rtol=1e-4, btol=1e-3, max_iter=64, fill=True, threads=None):
        self.table = table
        self.rtol = rtol
        self.btol = btol
        self.max_iter = max_iter
        self.fill = fill
        self.threads = threads
        self.lam = None

    def allocate(self, salience, cost, budget, headroom=None):
        if self.threads is not None:
            set_num_threads(self.threads)
        s, q, c, e, h, lam_max = prepare(self.table, salience, cost, headroom)
        n = len(s)

        def T(lam):
            a = np.empty(n, np.int32)
            _select(s, q, c, e, h, float(lam), a)
            return a, float(c[a].sum())  # same summation order as the oracle

        a, t, lam, evals, infeasible = drive(T, budget, self.lam, lam_max,
                                             self.rtol, self.btol, self.max_iter)
        if not (infeasible and lam_max < 0):  # serial keeps its warm lambda in that case
            self.lam = lam
        steps = 0
        if self.fill and not infeasible:
            slack = budget - t
            br = np.empty(n, np.float64)
            bj = np.empty(n, np.int32)
            for _ in range(n):
                d = _fill_step(s, q, c, e, h, a, slack, br, bj)
                if d < 0.0:
                    break
                slack -= d
                steps += 1
        return finish(s, q, c, a, lam, evals, infeasible, steps)


def num_threads():
    return get_num_threads()
