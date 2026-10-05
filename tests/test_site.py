"""Palace site: geometry, gate fields, routes, the large-crowd neighbour kernels against their
all-pairs oracles, and the palace scene end to end at N = 200."""
import functools

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from sim.neighbours import resolve, separation
from sim.run import _separation, _step
from sim.scenes import Palace
from sim.site import RES, Site
from sim.state import Agents

FORECOURT = np.array([-60.0, 30.0])  # site frame, open ground in front of the palace


@functools.lru_cache(maxsize=None)
def _site():
    return Site()


def test_open_gates_are_the_public_ones():
    st = _site()
    assert {st.gates[k]["key"] for k in st.open} == {g["key"] for g in st.raw["gates"] if g["public"]}


@pytest.mark.parametrize("k", [0, 2, 3, 4, 5])
def test_every_open_gate_is_reachable_from_the_forecourt(k):
    st = _site()
    if k not in st.open:
        pytest.skip("gate closed")
    p = FORECOURT + st.shift
    assert np.isfinite(st.distance(p, k))
    r = st.route(p, k)
    assert np.allclose(r[0], p)
    assert st.sink[k][tuple(st.cell(r[-1]))]
    assert all(st.visible(a, b, st.walk) for a, b in zip(r[:-1], r[1:]))
    # the string-pulled route is no longer than the 8-connected field distance
    assert np.linalg.norm(np.diff(r, axis=0), axis=1).sum() <= st.distance(p, k) + RES


def test_buildings_are_not_walkable_and_gates_cut_through_gatehouses():
    st = _site()
    assert not (st.walk & st.building & ~st.passage).any()
    palace = [b for b in st.raw["buildings"] if b["name"] == "Mysore Palace"][0]
    centre = np.mean(palace["rings"][0], 0) + st.shift
    assert not st.walkable(centre)
    for k in st.open:
        assert st.gates[k]["depth"] < 30.0


def test_constrain_never_leaves_walkable_ground():
    st = _site()
    rng = np.random.default_rng(0)
    prev = FORECOURT + st.shift + rng.uniform(-30, 30, (2000, 2))
    prev = prev[st.walkable(prev)]
    pos = prev + rng.normal(0, 3.0, prev.shape)
    out = st.constrain(pos.copy(), prev)
    assert st.walkable(out).all()


def test_cell_list_separation_matches_all_pairs():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 15, (600, 2)).astype(np.float32)
    assert np.abs(separation(p, 1.0) - _separation(p, 1.0)).max() < 1e-5
    act = rng.random(600) < 0.7
    assert np.abs(separation(p, 1.0, act)[act] - _separation(p[act], 1.0)).max() < 1e-5


def test_resolve_removes_overlap_and_keeps_fixed_agents():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 20, (400, 2))  # 1 person per m2: overlap is fully removable
    q = resolve(p, 0.25, iters=60)
    assert pdist(q).min() > 0.5 - 1e-3
    fixed = np.zeros(400, bool)
    fixed[::4] = True
    q = resolve(p, 0.25, fixed=fixed, iters=60)
    assert np.array_equal(q[fixed], p[fixed])


def _palace(n, seed):
    sc, a = Palace(site=_site()), Agents(n)
    rng = np.random.default_rng(seed)
    sc.spawn(a, rng)
    traj = []
    for f in range(20000):
        _step(a, sc, f, rng)
        live = (sc.k > 0) & ~sc.done
        assert _site().walkable(a.pos[live]).all()
        if f % 300 == 0:
            traj.append(a.pos.copy())
        if sc.done.all():
            break
    return sc, a, np.array(traj)


def test_palace_scene_everyone_leaves_and_is_deterministic():
    sc, a, t1 = _palace(200, seed=3)
    assert sc.done.all()
    assert len(sc.left) == 200 and len({i for i, _, _ in sc.left}) == 200
    assert {g for _, g, _ in sc.left} <= set(_site().open)
    _, _, t2 = _palace(200, seed=3)
    assert np.array_equal(t1, t2)


def test_palace_seats_follow_the_layout_assumptions():
    sc, a = Palace(site=_site()), Agents(1600)
    sc.spawn(a, np.random.default_rng(0))
    assert len(np.unique(sc.block)) == 2
    assert pdist(a.pos).min() >= Palace.SEAT - 1e-6
    st = _site()
    assert st.walkable(a.pos, st.clear).all()


def test_measured_plan_evens_out_the_estimated_finish():
    sc = Palace(site=_site(), plan="measured", rates={0: 350.0, 2: 122.0, 3: 132.0, 4: 278.0, 5: 84.0})
    corners = sc.blocks(60)
    bw, bh = Palace.ROWS * Palace.ROW, Palace.COLS * Palace.SEAT
    starts = np.concatenate([corners + (bw / 2, -Palace.AISLE / 2), corners + (bw / 2, bh + Palace.AISLE / 2)])
    gate = sc._by_rate(starts, len(corners))
    seats = np.array([(gate == k).sum() * 400 / sc.rates[k] for k in sc.rates])
    assert set(np.unique(gate)) <= set(sc.rates)
    assert seats.max() / seats.min() < 1.3  # queue minutes per gate within 30% of each other
