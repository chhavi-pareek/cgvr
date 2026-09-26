import numpy as np

from sim import orca


def test_alone_takes_the_preferred_velocity_clamped_to_max_speed():
    p, v = np.zeros(2), np.zeros(2)
    none = np.zeros((0, 2))
    assert np.allclose(orca.new_velocity(p, v, np.array([0.5, 0.2]), 1.3, none, none), [0.5, 0.2])
    out = orca.new_velocity(p, v, np.array([3.0, 4.0]), 1.0, none, none)
    assert np.allclose(out, [0.6, 0.8])


def test_a_feasible_result_satisfies_every_half_plane():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(200):
        p = np.zeros(2)
        v = rng.normal(size=2)
        pn = rng.uniform(-2, 2, size=(6, 2))
        pn = pn[np.hypot(pn[:, 0], pn[:, 1]) > 0.7]
        vn = rng.normal(size=(len(pn), 2))
        lines = orca.half_planes(p, v, pn, vn)
        fail, res = orca._lp2(lines, 2.0, rng.normal(size=2), False)
        if fail < len(lines):
            continue
        checked += 1
        for ln in lines:
            assert orca._det(ln.direction, ln.point - res) <= 1e-9
    assert checked > 50


def test_neighbours_are_the_k_nearest_within_reach_ties_by_index():
    pos = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [2.4, 0], [3.0, 0]], float)
    assert list(orca.neighbours(pos, 0, 3)) == [1, 2, 3]
    assert list(orca.neighbours(pos, 0, 10)) == [1, 2, 3, 4]      # 5 is out of reach
    assert list(orca.neighbours(pos, 0, 2)) == [1, 2]


def test_mirror_symmetric_agents_get_mirrored_velocities():
    p = np.array([[-1.0, 0.0], [1.0, 0.0]])
    v = np.array([[1.0, 0.1], [-1.0, -0.1]])
    out = orca.step(p, v, v.copy(), np.full(2, 1.5), k=1)
    assert np.allclose(out[0], -out[1], atol=1e-12)


def test_agents_swapping_across_a_circle_never_overlap_and_arrive():
    # Speeds differ, as in any crowd. With identical speeds the swap is a stable equilibrium of
    # ORCA -- a symmetric ring of reciprocal agents at rest, a known deadlock -- that tiny
    # perturbations of the preferred velocity do not break at this density and horizon.
    n, R, dt = 12, 4.0, 1.0 / 30.0
    ang = 2 * np.pi * np.arange(n) / n
    pos = R * np.stack([np.cos(ang), np.sin(ang)], 1)
    goal = -pos
    vel = np.zeros_like(pos)
    speed = np.random.default_rng(1).uniform(1.0, 1.5, n)
    closest = np.inf
    for _ in range(720):
        d = goal - pos
        dist = np.hypot(d[:, 0], d[:, 1])
        pref = d / np.maximum(dist, 1e-9)[:, None] * np.minimum(speed, dist / dt)[:, None]
        vel = orca.step(pos, vel, pref, speed, k=10, dt=dt, reach=6.0)
        pos = pos + vel * dt
        diff = pos[:, None] - pos[None]
        pd = np.hypot(diff[..., 0], diff[..., 1]) + np.eye(n) * 1e9
        closest = min(closest, pd.min())
    assert closest > 2 * orca.RADIUS * 0.95
    assert np.max(np.hypot(*(goal - pos).T)) < 0.05
