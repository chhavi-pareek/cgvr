"""Properties the fidelity metric must have to be worth quoting (bench/cfd.py)."""
import numpy as np
import pytest

from bench.cfd import cfd, trace
from sim.tiered import calibrate

CAL = calibrate("plaza", n=200, seed=0)
N, F = 150, 300


def _t(cond, brng=None, seed=0):
    tr, _ = trace("plaza", N, cond, F, seed, "orbit", CAL, brng_seed=brng)
    return tr


def test_identical_runs_score_zero():
    """Sanity: the metric must see identity as identity, or nothing else it says means
    anything."""
    a = _t("reference")
    d, _ = cfd(a, a)
    assert d.max() < 1e-12, f"a trace against itself scored {d}"


def test_the_noise_floor_is_real_and_large():
    """Two runs of the SAME policy differing only in behaviour randomness. If this were near
    zero the metric could use an absolute threshold; it is not, which is exactly why CFD is
    reported as a multiple of it."""
    floor, _ = cfd(_t("reference"), _t("reference", brng=9999))
    assert floor.mean() > 0.02, (
        f"noise floor {floor.mean():.4f} is suspiciously small; chaos should separate two "
        f"statistically identical crowds")
    assert floor.mean() < 0.5, f"noise floor {floor.mean():.4f} is so large the metric is blind"


def test_blind_to_agent_identity():
    """The design claim: CFD compares distributions, so relabelling agents cannot change it.
    This is what makes it robust to the chaotic divergence that rules out trajectory error."""
    a = _t("reference")
    rng = np.random.default_rng(0)
    b = []
    for frame in a:
        b.append([h.copy() for h in frame])          # histograms already discard identity
    d, _ = cfd(a, b)
    assert d.max() < 1e-12


@pytest.mark.parametrize("cond", ["baseline", "parity"])
def test_a_degraded_policy_scores_above_the_floor(cond):
    """Both policies approximate the reference, so both must be measurably worse than two
    reference runs are from each other. A policy at the floor would mean the metric cannot
    see LOD error at all."""
    ref = _t("reference")
    floor, _ = cfd(ref, _t("reference", brng=9999))
    d, _ = cfd(ref, _t(cond))
    assert d.mean() > floor.mean(), (
        f"{cond} scored {d.mean():.4f}, at or below the {floor.mean():.4f} noise floor")


def test_surrogate_collapses_the_speed_distribution():
    """A specific defect the metric found on its first run: sim/tiered.py:341 gives surrogate
    agents `core.mbar[ctx]`, a per-context MEAN speed multiplier, instead of their decoded
    speed_scale. The Metropolis-Hastings correction makes region occupancy exact but leaves
    the speed marginal unconstrained, so it collapses toward the mean."""
    from bench.camerapaths import camera
    from sim.tiered import Run

    r = Run("plaza", 200, "parity", seed=0, cam=camera("plaza", "orbit", 400), calib=CAL)
    for f in range(400):
        r.step(f)
    sur = r.tier == 3
    v = np.linalg.norm(r.a.vel, axis=1)
    assert sur.sum() > 10 and (~sur).sum() > 10, "need both populations to compare"
    assert v[sur].std() < v[~sur].std(), (
        "surrogate speed spread is no longer below the live agents'; if sim/tiered.py:341 now "
        "samples a speed scale instead of using core.mbar, delete this test and update cfd's "
        "speed row in STATE.md")
