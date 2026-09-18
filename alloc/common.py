"""Host-side pieces shared by the parallel allocators.

The serial allocator (alloc/serial.py) is the oracle and is not touched. The
parallel versions reproduce its arithmetic exactly:

* selection is argmax over the *unsorted* table with an explicit tie-break
  (higher score, then lower cost, then higher quality, then lower row index),
  which equals first-argmax over rows lexsorted by (cost asc, quality desc);
* lambda bracketing and bisection follow the same sequence of evaluations;
* lam_max, utility and cost are computed from the same fp64 products in the
  same order.
"""
import numpy as np

from alloc.serial import Result


def prepare(table, salience, cost, headroom):
    s = np.ascontiguousarray(salience, np.float64)
    c = np.ascontiguousarray(cost, np.float64)
    q = np.ascontiguousarray(table.quality, np.float64)
    e = np.ascontiguousarray(table.err, np.float64)
    if (s < 0).any():
        raise ValueError("salience must be non-negative")
    if headroom is None:
        h = np.full(len(s), np.inf)
        qmax = np.full(len(s), q.max())
        qmin = np.full(len(s), q.min())
    else:
        h = np.ascontiguousarray(headroom, np.float64)
        if (h < e.min()).any():
            raise ValueError("agent with no feasible configuration")
        oe = np.argsort(e, kind="stable")
        es = e[oe]
        k = np.searchsorted(es, h, side="right") - 1
        qmax = np.maximum.accumulate(q[oe])[k]
        qmin = np.minimum.accumulate(q[oe])[k]
    umax = float((s * qmax).max())
    umin = float((s * qmin).min())
    dc = np.diff(np.unique(c))
    dc = dc[dc > 0]
    if len(dc) == 0:
        lam_max = -1.0  # sentinel: every row costs the same
    else:
        lam_max = max(float(umax - min(0.0, umin)) / float(dc.min()), 1e-12)
    return s, q, c, e, h, lam_max


def drive(T, budget, lam_warm, lam_max, rtol, btol, max_iter):
    """Mirror of SerialAllocator.allocate's bracketing and bisection.

    T(lam) -> (assign, total). Returns (assign, total, lam, evals, infeasible).
    """
    evals = 1
    a0, t0 = T(0.0)
    if t0 <= budget:
        return a0, t0, 0.0, evals, False
    if lam_max < 0:
        return a0, t0, 0.0, evals, True
    evals += 1
    a_hi, t_hi = T(lam_max)
    if t_hi > budget:
        return a_hi, t_hi, lam_max, evals, True
    if lam_warm is None or lam_warm <= 0.0:
        lo, hi = 0.0, lam_max
        a_lo, t_lo = a0, t0
    else:
        lo = min(lam_warm / 2, lam_max)
        hi = min(lam_warm * 2, lam_max)
        a_lo, t_lo = T(lo)
        a_hi, t_hi = T(hi)
        evals += 2
        while t_hi > budget:
            lo, a_lo, t_lo = hi, a_hi, t_hi
            hi = min(hi * 2, lam_max)
            a_hi, t_hi = T(hi)
            evals += 1
        while t_lo <= budget:
            hi, a_hi, t_hi = lo, a_lo, t_lo
            lo = lo / 2 if lo > 1e-12 else 0.0
            a_lo, t_lo = T(lo)
            evals += 1
            if lo == 0.0:
                break
    for _ in range(max_iter):
        if hi - lo <= rtol * hi or budget - t_hi <= btol * budget:
            break
        mid = 0.5 * (lo + hi)
        a_m, t_m = T(mid)
        evals += 1
        if t_m <= budget:
            hi, a_hi, t_hi = mid, a_m, t_m
        else:
            lo, a_lo, t_lo = mid, a_m, t_m
    return a_hi, t_hi, hi, evals, False


def finish(s, q, c, a, lam, evals, infeasible, fill_steps):
    a = np.asarray(a)
    cost = float(c[a].sum())
    util = float((s * q[a]).sum())
    return Result(a, cost, util, lam, evals, infeasible, fill_steps)
