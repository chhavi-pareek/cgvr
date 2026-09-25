import os
import tempfile

import numpy as np
import pytest

from alloc.ledger import ErrorLedger
from bench.camerapaths import camera
from sim.invariant import BAND, partition
from sim.reconcile import promote
from sim.surrogate import Surrogate, kl_rows, mh_correct
from sim.tiered import Run, calibrate

N, FRAMES = 80, 400


@pytest.fixture(scope="module")
def calib():
    d = tempfile.mkdtemp()
    import sim.tiered as t

    orig = t._logs
    t._logs = lambda scene, tag: os.path.join(d, f"{scene}_{tag}")
    try:
        yield calibrate("corridor", n=N, frames=900, seed=0, force=True)
    finally:
        t._logs = orig


def test_mh_correction_exact_stationary():
    rng = np.random.default_rng(0)
    Q = rng.random((16, 16)) + 0.05
    Q /= Q.sum(1, keepdims=True)
    pi = rng.random(16)
    pi /= pi.sum()
    P = mh_correct(Q, pi)
    assert np.allclose(P.sum(1), 1.0) and (P >= 0).all()
    assert np.abs(pi @ P - pi).max() < 1e-12


def test_kl_rows_zero_on_identical():
    rng = np.random.default_rng(1)
    P = rng.random((4, 8, 8)) + 0.01
    P /= P.sum(-1, keepdims=True)
    assert np.abs(kl_rows(P, P)).max() < 1e-15
    assert (kl_rows(P, np.roll(P, 1, -1)) > 0).all()


def test_surrogate_fit_matches_reference_occupancy(calib):
    s = Surrogate(Run("corridor", 8, "calib").proc).load(calib["surrogate"])
    for c in range(4):
        assert np.abs(s.pi_c[c] @ s.P_sur[c] - s.pi_c[c]).max() < 1e-12
    assert (s.kl_coarse >= 0).all()
    if hasattr(s, "kl_coarse_ucb"):
        # v2: noise inflation removed, which can take a pure-noise frozen cell to 0; the ledger's
        # bound sits on or above the estimate that is plotted
        assert (s.kl_frozen >= 0).all() and (s.kl_frozen > 0).mean() > 0.9
        assert (s.kl_coarse_ucb >= s.kl_coarse - 1e-12).all()
    else:
        assert (s.kl_frozen > 0).all()
    assert s.e_rate[s.ctx_freq > 0.01].min() > 0


def test_ledger_cap_is_never_exceeded(calib):
    r = Run("corridor", N, "parity", cam=camera("corridor", "orbit", FRAMES), calib=calib).run(FRAMES)
    assert r.ledger.restorations > 0 and r.rows[-1]["t3"] > 0  # surrogate tier was used and restored
    assert np.all(r.ledger.D <= r.cap + 1e-9)
    assert r.infeasible == 0
    assert r.band_max <= BAND + 0.15


def test_ledger_headroom_blocks_capped_rows():
    from sim.tiered import phase7_table

    tab = phase7_table(0.1)
    led = ErrorLedger(3, cap=1.0)
    led.D[:] = (0.0, 0.95, 1.0)
    h = led.headroom()
    rows_sur = np.flatnonzero(tab.tiers[:, 0] == 3)
    assert tab.err[rows_sur[0]] <= h[0] and tab.err[rows_sur[0]] > h[1] and h[2] == 0.0
    assert led.check(tab, np.array([rows_sur[0], 0, 0]))
    assert not led.check(tab, np.array([0, rows_sur[0], 0]))


def test_reconcile_is_consistent_with_surrogate(calib):
    r = Run("corridor", N, "parity", cam=camera("corridor", "orbit", FRAMES), calib=calib)
    for f in range(60):
        r.step(f)
    idx = np.flatnonzero(r.tier == 3)[:5]
    assert idx.size
    old_R = r.region[idx].copy()
    r.a.pos[idx] += 50.0  # push the fine state far off so reconciliation must reposition it
    promote(idx, r.d, r.region, r.sur, r.core, r.a, r.phase, r.dist_at_demote, r.brng)
    assert np.all(r.sur.region_of(r.d[idx]) == old_R)
    assert np.all(np.abs(r.core.project(r.a.pos[idx], idx) - r.core.s[idx]) <= 1e-6)
    assert np.all((r.phase[idx] >= 0) & (r.phase[idx] < 1))
    assert np.all(np.linalg.norm(r.a.vel[idx], axis=1) > 0)


def test_core_egress_is_camera_invariant(calib):
    f = 1600  # first corridor egress arrives after ~1300 frames
    runs = [Run("corridor", N, "parity", cam=camera("corridor", c, f), calib=calib).run(f)
            for c in ("orbit", "static_wide")]
    assert len(runs[0].world.egress) > 0
    assert runs[0].world.egress == runs[1].world.egress
    assert np.array_equal(runs[0].core.s, runs[1].core.s)
    assert not np.array_equal(runs[0].tier, runs[1].tier)  # the cameras did allocate differently


def test_partition_names():
    core, view = partition()
    assert "s" in core and "queued" in core and "pos" in view and "d" in view and not set(core) & set(view)
