"""Ordering study: key properties, kernel oracle agreement, counters vs host model."""
import numpy as np
import pytest

from order import adaptive, density, kernel, key, sweep


@pytest.fixture(scope="module")
def frame():
    return next(density.frames("mixed", 200, seed=0, warm=30))


def _sorted(fr, w, blind=False):
    p = key.order(fr.pos, fr.tier, fr.cls, w, fr.size, blind)
    return fr.pos[p], fr.vel[p], fr.cls[p], fr.tier[p]


def test_morton_locality():
    size = (64.0, 64.0)
    pos = np.array([[1, 1], [1.05, 1.05], [40, 40], [1.1, 1.0]], np.float32)
    m = key.morton(pos, size)
    assert abs(int(m[0]) - int(m[1])) < abs(int(m[0]) - int(m[2]))
    assert key.morton(np.array([[64.0, 64.0]], np.float32), size)[0] == (1 << key.BITS) - 1


def test_key_endpoints(frame):
    fr = frame
    p1 = key.order(fr.pos, fr.tier, fr.cls, 1.0, fr.size)
    assert np.all(np.diff(key.morton(fr.pos[p1], fr.size).astype(np.int64)) >= 0)
    p0 = key.order(fr.pos, fr.tier, fr.cls, 0.0, fr.size)
    state = fr.tier.astype(int) * key.NCLS + fr.cls
    assert np.all(np.diff(state[p0]) >= 0)
    for s in np.unique(state):
        sel = p0[state[p0] == s]
        assert np.all(np.diff(key.morton(fr.pos[sel], fr.size).astype(np.int64)) >= 0)
    pb = key.order(fr.pos, fr.tier, fr.cls, 0.0, fr.size, blind=True)
    assert np.all(np.diff(state[pb]) >= 0)
    for s in np.unique(state):
        assert np.all(np.diff(pb[state[pb] == s]) > 0)  # id tie-break, no spatial order


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
def test_kernel_matches_reference_and_model(frame, w):
    pos, vel, cls, tier = _sorted(frame, w)
    a = kernel.analyse(pos, vel, cls, tier, frame.size)
    out_r, tr_r = kernel.run_ref(pos, vel, cls, tier, frame.size)
    out_d, tr_d, ws = kernel.run_instr(pos, vel, cls, tier, frame.size)
    assert np.array_equal(tr_r, a["trips"])
    assert np.array_equal(tr_d, a["trips"])
    assert np.array_equal(ws.astype(np.int64), a["wstats"])
    assert kernel.stats_to_wee(ws, len(pos), cls, tier)["wee"] == pytest.approx(a["wee"])
    np.testing.assert_allclose(out_d, out_r, rtol=1e-5, atol=1e-6)
    assert 0 < a["wee"] <= 1


def test_state_order_is_body_coherent(frame):
    """At w = 0 no full warp mixes class or tier bodies; at w = 1 some do."""
    for w, mixed_expected in ((0.0, False), (1.0, True)):
        pos, vel, cls, tier = _sorted(frame, w)
        ws = kernel.analyse(pos, vel, cls, tier, frame.size)["wstats"]
        full = np.arange(len(ws)) < len(pos) // kernel.WARP
        popc = lambda m: np.array([bin(int(x)).count("1") for x in m])
        boundaries = len(np.unique(tier.astype(int) * key.NCLS + cls)) - 1
        mixed = ((popc(ws[full, 2]) > 1) | (popc(ws[full, 3]) > 1)).sum()
        assert (mixed <= boundaries) if not mixed_expected else (mixed > boundaries)


def test_blind_order_moves_more_bytes(frame):
    b = [kernel.analyse(*_sorted(frame, 0.0, blind=bl), frame.size)["bytes_per_agent"] for bl in (False, True)]
    assert b[1] > b[0]


def test_sweep_row_and_policy(frame):
    fr = density.tile(frame, 2)
    assert fr.n == 4 * frame.n and fr.size == (2 * frame.size[0], 2 * frame.size[1])
    rows = []
    for w in (0.0, 0.5, 1.0, sweep.BLIND):
        r = sweep.measure(fr, w)
        r["seed"] = 0
        rows.append(r)
    assert all(np.isfinite(r["t_us"]) and r["bw_gbs"] > 0 for r in rows)
    pol = adaptive.Policy().fit(rows)
    assert pol.choose(rows[0]["rho"], rows[0]["geff"]) in (0.0, 0.5, 1.0)
    _, ok = adaptive.evaluate(pol, rows)
    assert ok
