"""Several viewers on one simulation, and the pop ledger's bound."""
import itertools

import numpy as np
import pytest

from alloc.costmodel import row_costs_from_theta
from alloc.factored import Factorisation, FactoredAllocator
from alloc.pops import PopLedger
from sim.tiered import phase7_table

E_MAX = 0.05716
T = phase7_table(E_MAX)


def instance(seed, n=300, V=2):
    rng = np.random.default_rng(seed)
    th = np.zeros((4, 4))
    for ax in range(4):
        th[ax, :3] = np.sort(rng.uniform(2, 40, 3))[::-1]
    c = row_costs_from_theta(T, 5.0, th) * 1e-3
    a = rng.lognormal(0.0, 0.8, n)
    B = np.where(rng.random((V, n)) < 0.35, 0.0, rng.uniform(0.05, 1.0, (V, n)))
    hr = rng.uniform(0.15, 1.5, n) * E_MAX
    return a, B, hr, c


def top(a, B, hr, c):
    return FactoredAllocator(T).allocate(a, c, np.inf, headroom=hr, view_salience=B).cost


@pytest.mark.parametrize("seed", range(3))
def test_two_viewers_exact_against_enumeration(seed):
    a, B, hr, c = instance(seed, n=40)
    F = Factorisation(T, c)
    alloc = FactoredAllocator(T)
    r = alloc.allocate(a, c, 0.45 * top(a, B, hr, c), headroom=hr, view_salience=B)
    S, Vn = F.row.shape
    for i in range(len(a)):
        best = -np.inf
        for s, v0, v1 in itertools.product(range(S), range(Vn), range(Vn)):
            if F.e_state[s] > hr[i] + 1e-12:
                continue
            val = (a[i] * F.q_state[s] + B[0, i] * F.q_view[v0] + B[1, i] * F.q_view[v1]
                   - r.lam * (F.c_state[s] + F.c_view[v0] + F.c_view[v1]))
            best = max(best, val)
        s, v0, v1 = alloc.state_pair[i], alloc.view_pair[0, i], alloc.view_pair[1, i]
        got = (a[i] * F.q_state[s] + B[0, i] * F.q_view[v0] + B[1, i] * F.q_view[v1]
               - r.lam * (F.c_state[s] + F.c_view[v0] + F.c_view[v1]))
        assert got == pytest.approx(best, abs=1e-12)
    # one behaviour per agent: both viewers' rows carry the same state pair
    assert (F.state_of[r.assign[0]] == F.state_of[r.assign[1]]).all()


def test_every_viewer_pays_a_non_negative_view_cost():
    a, B, hr, c = instance(2)
    F = Factorisation(T, c)
    assert F.c_view.min() == 0.0 and (F.c_view >= 0).all()
    assert np.allclose(F.c_state[F.state_of] + F.c_view[F.view_of], c)
    alloc = FactoredAllocator(T)
    r = alloc.allocate(a, c, np.inf, headroom=hr, view_salience=B)
    # two viewers cost one full row each for the first view, plus the second view's increment
    assert r.cost == pytest.approx(float(c[r.assign[0]].sum() + F.c_view[alloc.view_pair[1]].sum()))
    assert r.cost >= float(c[r.assign[0]].sum())


def test_one_viewer_as_a_matrix_is_the_single_view_problem():
    a, B, hr, c = instance(4)
    b = B[0]
    budget = 0.4 * FactoredAllocator(T).allocate(a, c, np.inf, headroom=hr, view_salience=b).cost
    r1 = FactoredAllocator(T).allocate(a, c, budget, headroom=hr, view_salience=b)
    r2 = FactoredAllocator(T).allocate(a, c, budget, headroom=hr, view_salience=b[None, :])
    assert (r2.assign[0] == r1.assign).all() and r2.utility == pytest.approx(r1.utility, rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_per_viewer_detail_beats_one_detail_for_every_viewer(seed):
    """The naive way to serve two viewers from one allocator: one view pair per agent, chosen
    for the viewer who sees it best, and drawn at that detail in both views. Same budget."""
    a, B, hr, c = instance(seed)
    V = B.shape[0]
    F = Factorisation(T, c)
    budget = 0.4 * top(a, B, hr, c)
    multi = FactoredAllocator(T).allocate(a, c, budget, headroom=hr, view_salience=B)
    merged_cost = F.c_state[F.state_of] + V * F.c_view[F.view_of]     # every view draws the pair
    alloc = FactoredAllocator(T)
    m = alloc.allocate(a, merged_cost, budget, headroom=hr, view_salience=B.max(0))
    u_merged = float((a * F.q_state[alloc.state_pair]).sum() + (B * F.q_view[alloc.view_pair[0]][None, :]).sum())
    assert m.cost <= budget * (1 + 1e-9) and multi.cost <= budget * (1 + 1e-9)
    assert multi.utility > u_merged


def test_holds_are_kept_and_released_only_to_meet_the_budget():
    a, B, hr, c = instance(7)
    F = Factorisation(T, c)
    b = B[0]
    n = len(a)
    rng = np.random.default_rng(0)
    lock = np.where(rng.random(n) < 0.4, rng.integers(0, len(F.vkeys), n), -1)
    budget = 0.5 * FactoredAllocator(T).allocate(a, c, np.inf, headroom=hr, view_salience=b).cost
    alloc = FactoredAllocator(T)
    r = alloc.allocate(a, c, budget, headroom=hr, view_salience=b, view_lock=lock)
    assert alloc.released == 0
    held = lock >= 0
    assert (F.view_of[r.assign[held]] == lock[held]).all()
    assert r.cost <= budget * (1 + 1e-9)

    # hold everyone at the most expensive view pair, then ask for a budget only the cheapest fits
    lock = np.full(n, int(np.argmax(F.c_view)))
    floor = F.c_state.min() * n + F.c_view.min() * n
    r = alloc.allocate(a, c, floor * 1.02, view_salience=b, view_lock=lock)
    assert alloc.released > 0 and not r.infeasible and r.cost <= floor * 1.02 * (1 + 1e-9)


def test_pop_ledger_bounds_visible_pops_in_every_window():
    """Adversarial driver: salience redrawn every frame, so the unconstrained allocator changes
    view pair constantly. With the ledger, no agent pops more than C + r * W in any W frames."""
    rng = np.random.default_rng(3)
    a, B, hr, c = instance(3, n=120, V=1)
    n, frames, C, r_ = 120, 900, 2.0, 1.0 / 60.0
    budget = 0.4 * top(a, B, hr, c)
    for bounded in (False, True):
        led = PopLedger((1, n), C, r_)
        alloc = FactoredAllocator(T)
        hist = np.zeros((frames, n), bool)
        released = 0
        for f in range(frames):
            b = rng.uniform(0.0, 1.0, (1, n))
            vis = rng.random((1, n)) < 0.8
            lock = led.holds(vis) if bounded else None
            res = alloc.allocate(a, c, budget, headroom=hr, view_salience=b, view_lock=lock)
            released += alloc.released
            hist[f] = led.update(alloc.view_pair, vis)[0]
        worst = 0
        for W in (30, 120, 600):
            cs = np.cumsum(np.vstack([np.zeros((1, n), int), hist.astype(int)]), 0)
            win = (cs[W:] - cs[:-W]).max()
            worst = max(worst, win - (C + r_ * W))
        if bounded:
            assert released == 0 and worst <= 1e-9
        else:
            assert worst > 5          # without the ledger the same driver blows straight through


def test_pop_ledger_ignores_changes_nobody_sees():
    led = PopLedger((1, 4), 1.0, 0.0)
    led.update(np.array([[0, 0, 0, 0]]), np.array([[True] * 4]))
    popped = led.update(np.array([[1, 1, 0, 0]]), np.array([[True, False, True, False]]))
    assert popped.tolist() == [[True, False, False, False]]
    assert led.holds(np.array([[True] * 4])).tolist() == [[1, -1, -1, -1]]
