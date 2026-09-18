import numpy as np
import pytest

from alloc.config import N_AXES, N_TIERS, build_table
from alloc.costmodel import RLSCostModel, row_costs_from_theta
from alloc.oracle import brute_force, dp_exact
from alloc.serial import SerialAllocator
from bench.telemetry import Workloads, random_counts

TABLE = build_table()


def _theta(rng, core=0.0):
    # incremental us/agent per axis-tier, decreasing with tier, tier 3 = 0
    th = np.zeros((N_AXES, N_TIERS))
    for ax in range(N_AXES):
        th[ax, :3] = np.sort(rng.uniform(2, 40, 3))[::-1]
    return core, th


def _ticks(cost):
    return np.round(cost).astype(np.int64)


def _instance(seed, n, core=0.0):
    rng = np.random.default_rng(seed)
    s = rng.lognormal(0.0, 0.8, n)
    core, th = _theta(rng, core)
    ticks = _ticks(row_costs_from_theta(TABLE, core, th))
    return s, ticks


def test_table_couplings():
    t = TABLE.tiers
    assert not np.any((t[:, 3] == 3) & (t[:, 2] == 0))
    assert not np.any(np.isin(t[:, 1], (2, 3)) & np.isin(t[:, 0], (0, 1)))
    assert TABLE.m == 180


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dp_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    sub = TABLE.subset(rng.choice(TABLE.m, 5, replace=False))
    n = 6
    s = rng.lognormal(0, 0.8, n)
    ticks = rng.integers(0, 12, sub.m)
    lo, hi = ticks.min() * n, ticks.max() * n
    for frac in (0.2, 0.5, 0.8):
        B = int(lo + frac * (hi - lo))
        _, ub = brute_force(sub, s, ticks.astype(float), B)
        a, ud = dp_exact(sub, s, ticks, B)
        assert ud == pytest.approx(ub, abs=1e-9)
        assert ticks[a].sum() <= B


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("frac", [0.15, 0.4, 0.7])
def test_lagrangian_within_2pct_of_optimal(seed, frac):
    n = 200
    s, ticks = _instance(seed, n)
    cost = ticks.astype(np.float64)
    lo, hi = cost.min() * n, cost.max() * n
    B = int(lo + frac * (hi - lo))
    r = SerialAllocator(TABLE).allocate(s, cost, float(B))
    _, opt = dp_exact(TABLE, s, ticks, B)
    assert not r.infeasible
    assert r.cost <= B
    assert r.utility >= 0.98 * opt


def test_budget_never_exceeded_any_setting():
    n = 200
    s, ticks = _instance(11, n)  # core = 0, so the all-lowest row costs 0
    cost = ticks.astype(np.float64)
    alloc = SerialAllocator(TABLE)
    t0 = alloc.allocate(s, cost, np.inf).cost
    for B in np.linspace(0.0, 1.2 * t0, 60):
        r = alloc.allocate(s, cost, B)  # warm-started across the sweep
        assert not r.infeasible
        assert float(cost[r.assign].sum()) <= B
        assert r.cost == float(cost[r.assign].sum())


def test_budget_below_core_floor_is_flagged():
    n = 50
    s, ticks = _instance(3, n, core=5.0)
    cost = ticks.astype(np.float64)
    floor = cost.min() * n
    r = SerialAllocator(TABLE).allocate(s, cost, floor * 0.5)
    assert r.infeasible
    assert r.cost == pytest.approx(floor)
    r = SerialAllocator(TABLE).allocate(s, cost, floor)
    assert not r.infeasible and r.cost <= floor


def test_error_headroom_mask_respected():
    n = 120
    rng = np.random.default_rng(5)
    s, ticks = _instance(7, n)
    cost = ticks.astype(np.float64)
    headroom = rng.uniform(0.15, 1.5, n)
    B = 0.4 * cost.max() * n
    r = SerialAllocator(TABLE).allocate(s, cost, B, headroom=headroom)
    assert r.cost <= B
    assert np.all(TABLE.err[r.assign] <= headroom)


def test_warm_start_reduces_evaluations():
    n = 200
    s, ticks = _instance(2, n)
    cost = ticks.astype(np.float64)
    B = 0.4 * cost.max() * n
    alloc = SerialAllocator(TABLE)
    cold = alloc.allocate(s, cost, B)
    warm = alloc.allocate(s, cost * 1.02, B)
    assert warm.evals < cold.evals
    assert warm.cost <= B


def test_rls_recovers_linear_generator():
    rng = np.random.default_rng(0)
    core, th = _theta(rng, core=3.0)
    model = RLSCostModel()
    for _ in range(400):
        h = random_counts(rng, 200)
        y = core * 200 + (th * h).sum()
        model.update(h, y + rng.normal(0, 0.5))
    assert model.theta_core == pytest.approx(core, abs=0.3)
    assert np.allclose(model.theta_axis[:, :3], th[:, :3], atol=0.3)
    h = random_counts(rng, 200)
    assert model.predict(h) == pytest.approx(core * 200 + (th * h).sum(), rel=0.02)


def test_rls_predicts_measured_frame_time():
    rng = np.random.default_rng(1)
    n = 200
    w = Workloads(n, 1)
    model = RLSCostModel()
    for _ in range(3):
        w.frame(random_counts(rng, n))
    for _ in range(120):
        h = random_counts(rng, n)
        total, _ = w.frame(h)
        model.update(h, total)
    errs = []
    for _ in range(20):
        h = random_counts(rng, n)
        total, _ = w.frame(h)
        errs.append((model.predict(h) - total) / total)
    assert np.sqrt(np.mean(np.square(errs))) < 0.3
    assert np.all(model.row_costs(TABLE) >= 0)
