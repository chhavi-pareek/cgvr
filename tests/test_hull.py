"""The hull allocator: same allocation, per-frame work of a sort instead of a table scan."""
import time

import numpy as np
import pytest

from alloc.costmodel import row_costs_from_theta
from alloc.hull import HullAllocator, hull_thetas, upper_hull
from alloc.serial import SerialAllocator
from sim.tiered import phase7_table

TABLE = phase7_table(0.06934)


def _instance(seed, n, mask=True):
    rng = np.random.default_rng(seed)
    s = rng.lognormal(0.0, 0.8, n)
    th = np.zeros((4, 4))
    for ax in range(4):
        th[ax, :3] = np.sort(rng.uniform(2, 40, 3))[::-1]
    cost = row_costs_from_theta(TABLE, 0.0, th) * 1e-3
    hr = rng.uniform(0.0, 0.15, n) if mask else None
    return s, cost, hr


def test_hull_is_concave_and_undominated():
    _, cost, _ = _instance(0, 10)
    h = upper_hull(cost, TABLE.quality)
    c, q = cost[h], TABLE.quality[h]
    assert np.all(np.diff(c) > 0), "hull must be strictly increasing in cost"
    assert np.all(np.diff(q) > 0), "a costlier hull vertex must be better, or it is dominated"
    th = hull_thetas(cost, TABLE.quality, h)
    assert np.all(np.diff(th) < 0), "switch thetas must decrease: that IS concavity"


@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("frac", [0.2, 0.5, 0.8])
def test_matches_the_lagrangian_and_respects_both_constraints(mask, frac):
    for seed in (0, 1):
        s, cost, hr = _instance(seed, 400, mask)
        B = len(s) * (cost.min() + frac * (cost.max() - cost.min()))
        rh = HullAllocator(TABLE).allocate(s, cost, B, headroom=hr)
        rs = SerialAllocator(TABLE).allocate(s, cost, B, headroom=hr)
        assert float(cost[rh.assign].sum()) <= B * (1 + 1e-6), "budget overrun"
        if hr is not None:
            assert np.all(TABLE.err[rh.assign] <= hr + 1e-12), "error mask violated"
        # the hull gives the exact LP optimum; the Lagrangian adds a greedy rounding, so it can
        # be a shade ahead. Anything past ~0.5% would mean the hull is solving a different problem.
        gap = (rs.utility - rh.utility) / abs(rs.utility)
        assert gap < 5e-3, f"utility gap {gap * 100:.4f}% is too large to be the integrality gap"


def test_groups_by_feasible_set_not_by_headroom_value():
    """Regression. Grouping on the raw headroom builds one hull and one sort PER AGENT when
    headroom is continuous, which made this allocator 25x slower than the one it replaces.
    The error column has two distinct levels, so there must be at most two groups however many
    distinct headroom values appear."""
    s, cost, hr = _instance(3, 800, mask=True)
    assert len(np.unique(hr)) > 100, "test needs continuous headroom to be meaningful"
    B = len(s) * (cost.min() + 0.4 * (cost.max() - cost.min()))
    A = HullAllocator(TABLE)
    t = time.perf_counter()
    A.allocate(s, cost, B, headroom=hr)
    hull_ms = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    SerialAllocator(TABLE).allocate(s, cost, B, headroom=hr)
    serial_ms = (time.perf_counter() - t) * 1e3
    assert hull_ms < serial_ms, (
        f"hull {hull_ms:.2f} ms is not faster than serial {serial_ms:.2f} ms at N=800; "
        f"the feasible-set grouping has probably regressed to per-agent groups")


def test_advantage_grows_with_crowd_size():
    """The point of the rewrite: serial is O(n m evals), the hull is O(n log n). If the ratio
    does not grow with n, the hull is not doing what it claims."""
    def ratio(n):
        s, cost, hr = _instance(0, n)
        B = n * (cost.min() + 0.4 * (cost.max() - cost.min()))
        A, H = SerialAllocator(TABLE), HullAllocator(TABLE)
        A.allocate(s, cost, B, headroom=hr); H.allocate(s, cost, B, headroom=hr)
        t = time.perf_counter(); A.allocate(s, cost, B, headroom=hr)
        a = time.perf_counter() - t
        t = time.perf_counter(); H.allocate(s, cost, B, headroom=hr)
        return a / (time.perf_counter() - t)
    small, large = ratio(300), ratio(3000)
    assert large > small, f"speedup did not grow with n: {small:.1f}x at 300, {large:.1f}x at 3000"
    assert large > 5.0, f"only {large:.1f}x at n=3000"
