"""Do the guarantees still hold when the mechanisms are switched on together?

Every bound in this project has been verified in isolation, and every mechanism added this
session changes something the others depend on:

  coupled reconciliation  changes what promote() does to position and latent
  priority aging          changes the allocation, so it changes who is degraded and when
  aggregate budget        adds a second multiplier, so it changes the feasible set

An ensemble that is individually correct and jointly broken is the obvious failure mode, and
nothing was checking for it. These tests run the mechanisms together and assert every bound at
once: the ledger cap (invariant 3), the core band and the demotion snap (invariant 4), bounded
continuous degradation (the corollary), and the frame budget.
"""
import numpy as np
import pytest

from bench.camerapaths import camera
from bench.fairness import AgedRun
from sim import invariant, reconcile
from sim.invariant import BAND
from sim.tiered import calibrate

CAL = calibrate("plaza", n=200, seed=0)
LATERAL_CLIP = 1.8


def _run(frames=600, seed=0, couple=False, alpha=0.0, n=200):
    reconcile.COUPLE = couple
    try:
        r = AgedRun("plaza", n, "parity", seed=seed, cam=camera("plaza", "orbit", frames), calib=CAL)
        r.alpha = alpha
        band, snap, psnap, spend = [], [], [], []
        at_sur = np.zeros(n)
        longest = np.zeros(n)
        cur = np.zeros(n)
        for f in range(frames):
            prev_tier, prev_pos = r.tier.copy(), r.a.pos.copy()
            r.step(f)
            dem = np.flatnonzero((prev_tier < 3) & (r.tier == 3))
            if dem.size:
                snap.append(np.linalg.norm(r.a.pos[dem] - prev_pos[dem], axis=1))
            pro = np.flatnonzero((prev_tier == 3) & (r.tier < 3))
            if pro.size:
                psnap.append(np.linalg.norm(r.a.pos[pro] - prev_pos[pro], axis=1))
            band.append(float(np.abs(r.core.project(r.a.pos) - r.core.s).max()))
            spend.append(float(r.cost[r.assign].sum()))
            s = r.tier == 3
            at_sur += s
            cur = np.where(s, cur + 1, 0)
            longest = np.maximum(longest, cur)
        return dict(r=r, band=np.array(band),
                    snap=np.concatenate(snap) if snap else np.zeros(1),
                    psnap=np.concatenate(psnap) if psnap else np.zeros(1),
                    spend=np.array(spend), frac=at_sur / frames, longest=longest)
    finally:
        reconcile.COUPLE = False


def _assert_all_bounds(o):
    r = o["r"]
    e_min = float(np.min(r.sur.e_rate[r.sur.ctx_freq > 0.01]))
    assert r.ledger.D.max() <= r.cap + 1e-9, "invariant 3: ledger passed the cap"
    assert o["band"].max() <= BAND + 1e-4, "invariant 4: fine position left the core band"
    assert o["snap"].max() <= BAND + LATERAL_CLIP + 1e-4, "demotion snap exceeded its bound"
    assert o["longest"].max() <= r.cap / e_min + 1e-6, "continuous degradation exceeded cap/rate"
    assert int((o["spend"] > r.budget * (1 + 1e-6)).sum()) == 0, "frame budget overrun"


@pytest.mark.parametrize("couple,alpha", [(False, 0.0), (True, 0.0), (False, 4.0), (True, 4.0)])
def test_every_bound_holds_for_each_combination(couple, alpha):
    _assert_all_bounds(_run(couple=couple, alpha=alpha))


def test_aging_still_removes_starvation_with_coupling_on():
    """Aging was measured alone. Coupled reconciliation changes which agents are restored and
    when, so the fairness result has to be re-checked in its presence rather than assumed."""
    off = _run(frames=900, couple=True, alpha=0.0)
    on = _run(frames=900, couple=True, alpha=4.0)
    starved_off = int((off["frac"] > 0.99).sum())
    starved_on = int((on["frac"] > 0.99).sum())
    assert starved_on <= starved_off, (
        f"aging made starvation worse with coupling on: {starved_off} -> {starved_on}")
    assert np.percentile(on["frac"], 99) <= np.percentile(off["frac"], 99) + 1e-9


def test_coupling_still_shrinks_the_snap_with_aging_on():
    """And the converse: aging changes the reconciliation schedule, so the coupling result has
    to survive it too. Coupling acts on PROMOTION; an earlier version of this test compared
    demotion snaps, which coupling never touches, and passed by schedule noise."""
    plain = _run(couple=False, alpha=4.0)
    cpl = _run(couple=True, alpha=4.0)
    assert np.percentile(cpl["psnap"], 95) < np.percentile(plain["psnap"], 95), (
        "coupling stopped shrinking the promotion snap once aging was enabled")


def test_band_ablation_does_not_break_the_ledger_under_the_full_ensemble():
    """invariant 4 removed, everything else on: invariant 3 must be untouched. The two bounds
    are over different quantities and neither implies the other."""
    invariant.BIND = False
    try:
        o = _run(couple=True, alpha=4.0)
        r = o["r"]
        assert r.ledger.D.max() <= r.cap + 1e-9, "ledger broke when the core band was removed"
    finally:
        invariant.BIND = True
