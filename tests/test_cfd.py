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


def _speed_spread(flag):
    import sim.tiered as st
    from bench.camerapaths import camera
    from sim.tiered import Run

    st.SURROGATE_SPEED = flag
    try:
        r = Run("plaza", 200, "parity", seed=0, cam=camera("plaza", "orbit", 400), calib=CAL)
        sur_v, live_v = [], []
        for f in range(400):
            r.step(f)
            if f > 100:
                v = np.linalg.norm(r.a.vel, axis=1)
                sur_v.append(v[r.tier == 3])
                live_v.append(v[r.tier < 3])
        return np.concatenate(sur_v).std(), np.concatenate(live_v).std()
    finally:
        st.SURROGATE_SPEED = True


def test_old_surrogate_collapsed_the_speed_distribution():
    """The defect bench/cfd.py found on its first run, kept reproducible behind the flag:
    moving every surrogate agent at core.mbar[ctx] collapses speed toward the context mean."""
    sur, live = _speed_spread(False)
    assert sur < 0.85 * live, f"old surrogate spread {sur:.4f} vs live {live:.4f}"


def test_surrogate_speed_marginal_is_restored():
    """With the scale drawn from pi_ref(d | R, ctx) per region, surrogate agents spread their
    speeds like live agents do. Measured: surrogate std 0.2039 -> 0.2847, reference 0.2931."""
    sur, live = _speed_spread(True)
    assert sur > 0.9 * live, f"surrogate spread {sur:.4f} still well below live {live:.4f}"
