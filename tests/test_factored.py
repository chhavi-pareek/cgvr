"""The two-salience allocator: factorisation, exactness, and why it needs the full table."""
import numpy as np
import pytest

from alloc.config import prune_dominated
from alloc.costmodel import row_costs_from_theta
from alloc.factored import Factorisation, FactoredAllocator, split_quality, view_salience
from alloc.hull import HullAllocator
from sim.tiered import phase7_table

E_MAX = 0.05716
T = phase7_table(E_MAX)


def costs(rng, geo_heavy=False):
    th = np.zeros((4, 4))
    for ax in range(4):
        th[ax, :3] = np.sort(rng.uniform(2, 40, 3))[::-1]
    if geo_heavy:  # the engine's measured shape: behaviour ~0.15 us, a detailed mesh ~6-15 us
        th[0, :3] = (0.15, 0.14, 0.14)
        th[3, :3] = (14.8, 3.9, 0.6)
    return row_costs_from_theta(T, 5.0, th) * 1e-3


def instance(seed, n=600, mask=True):
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0.0, 0.8, n)
    b = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.05, 1.0, n))   # 30% out of view
    hr = rng.uniform(0.15, 1.5, n) * E_MAX if mask else None
    return a, b, hr, costs(rng)


def brute_best(a, b, q_s, q_v, c, feas, lam):
    """Per-agent maximum of a q_s + b q_v - lam c over the full table."""
    val = a[:, None] * q_s[None, :] + b[:, None] * q_v[None, :] - lam * c[None, :]
    return np.where(feas, val, -np.inf).max(1)


def test_table_is_the_product_of_state_and_view_pairs():
    F = Factorisation(T, costs(np.random.default_rng(0)))
    assert (len(F.skeys), len(F.vkeys)) == (12, 15)
    assert sorted(F.row.ravel().tolist()) == list(range(T.m))
    q_s, q_v = split_quality(T)
    assert np.allclose(q_s + q_v, T.quality)


def test_pruned_table_and_non_additive_cost_are_refused():
    c = costs(np.random.default_rng(1))
    with pytest.raises(ValueError):
        Factorisation(T.subset(prune_dominated(T, c)), c[prune_dominated(T, c)])
    bad = c.copy()
    bad[7] += 1e-3
    with pytest.raises(ValueError):
        Factorisation(T, bad)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mask", [False, True])
def test_one_salience_is_the_hull_solution(seed, mask):
    a, _, hr, c = instance(seed, mask=mask)
    t0 = float(c[HullAllocator(T).allocate(a, c, np.inf, headroom=hr).assign].sum())
    for frac in (0.15, 0.4, 0.8):
        rh = HullAllocator(T).allocate(a, c, frac * t0, headroom=hr)
        rf = FactoredAllocator(T).allocate(a, c, frac * t0, headroom=hr)
        assert rf.utility == pytest.approx(rh.utility, rel=1e-12)
        assert rf.cost == pytest.approx(rh.cost, rel=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_every_agent_takes_its_exact_best_row_at_lambda(seed):
    a, b, hr, c = instance(seed)
    q_s, q_v = split_quality(T)
    feas = T.err[None, :] <= hr[:, None] + 1e-12
    t_top = float(c[FactoredAllocator(T).allocate(a, c, np.inf, headroom=hr, view_salience=b).assign].sum())
    for frac in (0.1, 0.3, 0.6, 0.9):
        B = frac * t_top
        r = FactoredAllocator(T).allocate(a, c, B, headroom=hr, view_salience=b)
        got = a * q_s[r.assign] + b * q_v[r.assign] - r.lam * c[r.assign]
        assert np.allclose(got, brute_best(a, b, q_s, q_v, c, feas, r.lam), atol=1e-12)
        assert r.infeasible or r.cost <= B * (1 + 1e-9)
        assert (T.err[r.assign] <= hr + 1e-12).all()


def test_view_detail_follows_view_salience_and_nothing_is_spent_off_screen():
    a, b, hr, c = instance(3)
    q_s, q_v = split_quality(T)
    top = float(c[FactoredAllocator(T).allocate(a, c, np.inf, headroom=hr, view_salience=b).assign].sum())
    r = FactoredAllocator(T).allocate(a, c, 0.3 * top, headroom=hr, view_salience=b)
    assert r.lam > 0
    order = np.argsort(b, kind="stable")
    assert (np.diff(q_v[r.assign][order]) >= -1e-12).all()        # more visible, never less detail
    F = Factorisation(T, c)
    view_pair = np.empty(T.m, np.int64)
    view_pair[F.row] = np.arange(F.row.shape[1])[None, :]
    off = b == 0
    assert np.allclose(F.c_view[view_pair[r.assign[off]]], F.c_view.min())


def test_pruning_on_combined_quality_removes_rows_the_two_salience_optimum_needs():
    rng = np.random.default_rng(5)
    c = costs(rng, geo_heavy=True)
    n = 800
    a = rng.lognormal(0.0, 0.5, n)
    b = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.2, 1.0, n))
    top = float(c[FactoredAllocator(T).allocate(a, c, np.inf, view_salience=b).assign].sum())
    r = FactoredAllocator(T).allocate(a, c, 0.35 * top, view_salience=b)
    kept = set(prune_dominated(T, c).tolist())
    lost = sorted(set(r.assign.tolist()) - kept)
    assert lost, "expected the two-salience optimum to use rows the combined prune drops"


def test_view_salience_is_projected_area():
    v = view_salience([2.0, 10.0, 40.0, 5.0], [True, True, True, False], d0=10.0)
    assert np.allclose(v, [1.0, 1.0, 0.0625, 0.0])
