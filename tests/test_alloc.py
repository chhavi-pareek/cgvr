import numpy as np
import pytest

from alloc.config import N_AXES, N_TIERS, build_table
from alloc.costmodel import RLSCostModel, row_costs_from_theta
from alloc.oracle import brute_force, dp_exact
from alloc.sequential import SequentialAllocator
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


# -- sequential (two-stage) ablation of the joint dual-budget solve --------------------


def test_sequential_budget_never_exceeded_any_setting():
    n = 120
    s, ticks = _instance(11, n)
    cost = ticks.astype(np.float64)
    alloc = SequentialAllocator(TABLE)
    t0 = SerialAllocator(TABLE).allocate(s, cost, np.inf).cost
    for B in np.linspace(0.0, 1.2 * t0, 24):
        r = alloc.allocate(s, cost, B)
        assert not r.infeasible
        assert float(cost[r.assign].sum()) <= B
        assert r.cost == float(cost[r.assign].sum())


def test_sequential_budget_never_exceeded_under_headroom():
    n = 120
    rng = np.random.default_rng(21)
    s, ticks = _instance(13, n)
    cost = ticks.astype(np.float64)
    headroom = rng.uniform(0.15, 1.5, n)
    alloc = SequentialAllocator(TABLE)
    t0 = SerialAllocator(TABLE).allocate(s, cost, np.inf, headroom=headroom).cost
    for B in np.linspace(0.0, 1.2 * t0, 24):
        r = alloc.allocate(s, cost, B, headroom=headroom)
        assert np.all(TABLE.err[r.assign] <= headroom)  # stage 1 mask holds even when infeasible
        if not r.infeasible:
            assert float(cost[r.assign].sum()) <= B


def test_sequential_error_headroom_mask_respected():
    n = 120
    rng = np.random.default_rng(5)
    s, ticks = _instance(7, n)
    cost = ticks.astype(np.float64)
    headroom = rng.uniform(0.15, 1.5, n)
    B = 0.4 * cost.max() * n
    r = SequentialAllocator(TABLE).allocate(s, cost, B, headroom=headroom)
    assert not r.infeasible
    assert r.cost <= B
    assert np.all(TABLE.err[r.assign] <= headroom)


def test_sequential_below_floor_is_flagged_like_serial():
    n = 50
    s, ticks = _instance(3, n, core=5.0)
    cost = ticks.astype(np.float64)
    floor = cost.min() * n
    r = SequentialAllocator(TABLE).allocate(s, cost, floor * 0.5)
    assert r.infeasible
    assert r.cost == pytest.approx(floor)
    r = SequentialAllocator(TABLE).allocate(s, cost, floor)
    assert not r.infeasible and r.cost <= floor


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("frac", [0.15, 0.4, 0.7])
def test_sequential_utility_never_above_joint(seed, frac):
    """The two-stage solve searches the same feasible set with a weaker procedure, so its
    utility must be <= the joint Lagrangian's. Measured: on this table it is *equal*, not
    worse -- the joint formulation buys allocator time, not utility. If this assertion ever
    fires with sequential strictly ahead, that is a real result, not a flaky test."""
    n = 120
    rng = np.random.default_rng(100 + seed)
    s, ticks = _instance(seed, n)
    cost = ticks.astype(np.float64)
    headroom = rng.uniform(0.15, 1.5, n)
    B = float(cost.min() * n + frac * (cost.max() - cost.min()) * n)
    seq = SequentialAllocator(TABLE).allocate(s, cost, B, headroom=headroom)
    joint = SerialAllocator(TABLE).allocate(s, cost, B, headroom=headroom)
    assert np.all(TABLE.err[seq.assign] <= headroom)
    if not seq.infeasible:
        assert seq.cost <= B
    assert seq.utility <= joint.utility * (1 + 1e-9) + 1e-9


# -- sharper strawman: stage 1 spends the error budget maximally, stage 2 downgrades ----


def test_sequential_mode_is_validated():
    with pytest.raises(ValueError):
        SequentialAllocator(TABLE, mode="greedy")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("frac", [0.1, 0.35, 0.7, 0.95])
def test_sequential_maximal_matches_joint(seed, frac):
    """`mode="maximal"` commits each agent to the best configuration its error headroom
    allows before the frame budget is even looked at, then claws time back by downgrading.
    It is the opponent the cheapest-first strawman is not: its stage-1 choice is the one
    stage 2 has to live with. It still lands on the joint solver's exact assignment."""
    n = 120
    rng = np.random.default_rng(200 + seed)
    s, ticks = _instance(seed, n)
    cost = ticks.astype(np.float64)
    headroom = rng.uniform(0.15, 1.5, n)
    B = float(cost.min() * n + frac * (cost.max() - cost.min()) * n)
    seq = SequentialAllocator(TABLE, mode="maximal").allocate(s, cost, B, headroom=headroom)
    joint = SerialAllocator(TABLE).allocate(s, cost, B, headroom=headroom)
    assert np.all(TABLE.err[seq.assign] <= headroom)
    assert seq.infeasible == joint.infeasible
    if not seq.infeasible:
        assert seq.cost <= B
        assert np.array_equal(seq.assign, joint.assign)


def test_greedy_fill_is_what_closes_the_gap_not_the_joint_solve():
    """With the fill disabled the three solves separate, and not in the expected direction:
    cheapest-first collapses (it never spends the time budget at all), while maximal-first
    is level with the joint Lagrangian or slightly ahead of it, because the bisection stops
    at a bracket end below the budget and leaves the integrality gap unspent. So the fill,
    not the joint relaxation, is what makes the allocation exact; the joint relaxation is
    what makes it fast. Re-enabling the fill collapses all three onto one assignment."""
    n = 120
    rng = np.random.default_rng(7)
    s, ticks = _instance(4, n)
    cost = ticks.astype(np.float64)
    headroom = rng.uniform(0.15, 1.5, n)
    B = float(cost.min() * n + 0.4 * (cost.max() - cost.min()) * n)
    kw = dict(headroom=headroom)
    joint = SerialAllocator(TABLE, fill=False).allocate(s, cost, B, **kw)
    cheap = SequentialAllocator(TABLE, fill=False, mode="cheapest").allocate(s, cost, B, **kw)
    maxim = SequentialAllocator(TABLE, fill=False, mode="maximal").allocate(s, cost, B, **kw)
    assert cheap.utility < 0.9 * joint.utility  # measured ~0.15x: stage 1 is the floor
    assert maxim.utility >= joint.utility * (1 - 1e-9)
    assert maxim.cost <= B and cheap.cost <= B
