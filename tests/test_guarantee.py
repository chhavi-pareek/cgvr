"""Machine-checked verification of invariant 3's bound (see alloc/guarantee.py).

The point of these tests is that the bound is claimed for EVERY selection rule the mask
admits, not for the one the allocator happens to implement. So the pickers here are
adversarial: maximally aggressive, random, and worst-context-always. If the theorem is right
none of them can breach the cap; if it is wrong, an adversary finds it faster than a simulator
does. test_verification_has_power is the negative control -- break assumption (A1) and the
harness must catch it, otherwise the passing tests above mean nothing.
"""
import numpy as np
import pytest

from alloc.guarantee import check_assumptions, headroom, verify_schedule
from sim.tiered import phase7_table

E_RATE = np.array([0.00393, 0.01033, 0.05716, 0.00393])  # plaza, bench/logs/phase7_plaza_calib.npz
E_ADMIT = float(E_RATE.max())
E_SUR = 0.01323
CAP = 300.0 * E_SUR


def _table():
    t = phase7_table(E_ADMIT)
    rate = np.where((t.tiers[:, 0] == 3)[:, None], E_RATE[None, :], 0.0)
    return t, rate


def _greedy(t, rate):
    """The adversary: always take the most divergent configuration the mask still allows."""
    order = np.argsort(-t.err)

    def pick(_t, mask, _D, _h):
        n = mask.shape[0]
        out = np.empty(n, np.int64)
        for i in range(n):
            for j in order:
                if mask[i, j]:
                    out[i] = j
                    break
        return out
    return pick


def test_assumptions_hold_on_the_real_table():
    t, rate = _table()
    ok, msg = check_assumptions(t.err, rate)
    assert ok, msg
    assert (t.err <= 0).sum() > 0                 # (A2): rate-0 rows exist
    assert np.isclose(t.err.max(), E_ADMIT)       # admission is the worst context's rate


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cap_holds_under_the_greedy_adversary(seed):
    t, rate = _table()
    n, T = 64, 900
    rng = np.random.default_rng(seed)
    ctx = rng.integers(0, 4, T)
    D0 = rng.uniform(0.0, CAP, n)                 # start anywhere legal, including at the cap
    ok, worst, frame = verify_schedule(t.err, rate, CAP, ctx, _greedy(t, rate), D0)
    assert ok, f"cap breached at frame {frame}, max D {worst}"
    assert worst <= CAP + 1e-9


def test_cap_holds_when_every_frame_is_the_worst_context():
    """The adversary also controls the context, so e_admit is charged every single frame."""
    t, rate = _table()
    n, T = 48, 1200
    ctx = np.full(T, int(np.argmax(E_RATE)))
    D0 = np.full(n, CAP)                          # every agent starts exactly at the cap
    ok, worst, frame = verify_schedule(t.err, rate, CAP, ctx, _greedy(t, rate), D0)
    assert ok, f"cap breached at frame {frame}"
    # starting at the cap, headroom is 0, so only rate-0 rows are admissible and D cannot move
    assert worst <= CAP + 1e-9


@pytest.mark.parametrize("seed", [3, 4])
def test_cap_holds_under_random_admissible_choices(seed):
    t, rate = _table()
    rng = np.random.default_rng(seed)
    n, T = 50, 600

    def pick(_t, mask, _D, _h):
        out = np.empty(mask.shape[0], np.int64)
        for i in range(mask.shape[0]):
            legal = np.flatnonzero(mask[i])
            out[i] = legal[rng.integers(len(legal))]
        return out

    ctx = rng.integers(0, 4, T)
    ok, worst, frame = verify_schedule(t.err, rate, CAP, ctx, pick, rng.uniform(0, CAP, n))
    assert ok, f"cap breached at frame {frame}"


@pytest.mark.parametrize("latency", [1, 2, 4])
def test_deferred_application_holds_with_the_margin(latency):
    """The corollary: a decision applied L frames late still cannot breach, because the
    headroom reserved L * e_admit up front. unity/PORT_SPEC.md section 4."""
    t, rate = _table()
    n, T = 40, 700
    rng = np.random.default_rng(11)
    ctx = rng.integers(0, 4, T)
    D0 = rng.uniform(0, CAP - latency * E_ADMIT, n)
    pick = _greedy(t, rate)

    # apply each decision for `latency` extra frames, which is the situation the margin covers
    held = None
    D = D0.copy()
    worst = 0.0
    for f in range(T):
        if f % latency == 0:
            h = headroom(D, CAP, latency, E_ADMIT)
            mask = t.err[None, :] <= h[:, None]
            assert mask.any(1).all(), "mask emptied under the latency margin"
            held = pick(f, mask, D, h)
        D = D + rate[held, ctx[f]]
        worst = max(worst, float(D.max()))
    assert worst <= CAP + 1e-9, f"deferred application breached: {worst} > {CAP}"


def test_verification_has_power():
    """Negative control. Break (A1) -- charge more than the mask admitted -- and the harness
    must find the breach. Without this the passing tests above prove nothing."""
    t, _ = _table()
    broken = np.where((t.tiers[:, 0] == 3)[:, None], (E_RATE * 3.0)[None, :], 0.0)
    ok, msg = check_assumptions(t.err, broken)
    assert not ok and "(A1)" in msg

    n, T = 32, 400
    rng = np.random.default_rng(5)
    ctx = np.full(T, int(np.argmax(E_RATE)))
    ok, worst, frame = verify_schedule(t.err, broken, CAP, ctx, _greedy(t, broken),
                                       rng.uniform(0, CAP, n))
    assert not ok, "the harness failed to detect a genuine cap breach"
    assert worst > CAP


def test_bound_is_tight():
    """No constant below the cap works: an agent admitted at exactly its headroom, whose
    realised rate equals the admission rate, lands on the cap exactly."""
    err = np.array([0.0, E_ADMIT])
    rate = np.array([[0.0], [E_ADMIT]])
    D0 = np.array([CAP - E_ADMIT])
    ok, worst, _ = verify_schedule(err, rate, CAP, np.zeros(1, int),
                                   lambda *_a: np.array([1]), D0)
    assert ok
    assert np.isclose(worst, CAP, atol=1e-9), f"expected to land exactly on the cap, got {worst}"


# -- invariant 4's positional bound, which invariant 3 does not imply ----------------------


@pytest.mark.parametrize("scene", ["plaza", "corridor"])
def test_core_band_bounds_the_demotion_snap(scene):
    """Demotion replaces the fine position with the core point, so the snap is exactly the
    accumulated fine-vs-core drift. Core.bind holds that at BAND; without it a chokepoint
    scene reaches 20 m. See alloc/guarantee.py and bench/invariant4.py."""
    from bench.invariant4 import play
    from sim.invariant import BAND
    from sim.tiered import calibrate

    cal = calibrate(scene, n=200, seed=0)
    # long enough for the drift to actually reach the band; at 300 frames it does not, bind
    # never engages, and the ablation is a no-op that proves nothing
    on = play(scene, 200, 600, 0, "orbit", cal, True)
    off = play(scene, 200, 600, 0, "orbit", cal, False)

    LATERAL_CLIP = 1.8
    # bind is exact on a single-segment route and leaves a per-corner residual otherwise; see
    # the theorem in alloc/guarantee.py. Measured: plaza and hub 0%, corridor 5.9%.
    EPS = 0.0 if scene in ("plaza", "hub") else 0.08
    TOL = 1e-4          # 0.1 mm: projection arithmetic, not an overshoot
    assert on["band"].max() <= BAND * (1 + EPS) + TOL, (
        f"bind did not hold the band: {on['band'].max():.4f} > {BAND * (1 + EPS):.4f}")
    assert on["dem"].max() <= BAND * (1 + EPS) + LATERAL_CLIP + TOL, (
        f"demote snap {on['dem'].max():.4f} exceeded BAND + lateral clip")
    # the clamp has to be binding for the ablation to mean anything -- state that rather than
    # let a scene where it never engages pass silently
    assert on["band"].max() >= BAND - 1e-2, (
        f"bind never engaged ({on['band'].max():.4f} < {BAND}); this scene cannot test it")
    assert off["band"].max() > on["band"].max() + 1e-6, "ablating bind changed nothing"
    # the two guarantees are independent: invariant 3 survives invariant 4 being ablated
    assert off["dmax"] <= off["cap"] + 1e-6, "ledger bound broke when the core band was removed"
