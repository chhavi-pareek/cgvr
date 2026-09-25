"""Coupled reconciliation must change the tie-break and nothing else.

The claim is that sim/reconcile.COUPLE reduces the observable discontinuity while leaving the
reference law intact. The second half is the part worth testing: a coupling that quietly biased
the latent occupancy would look like a win on the jump statistics and be wrong.
"""
import numpy as np
import pytest

from bench.camerapaths import camera
from sim import reconcile
from sim.tiered import Run, calibrate

CAL = calibrate("plaza", n=200, seed=0)


def _run(seed, couple, frames=360):
    keep = reconcile.COUPLE
    reconcile.COUPLE = couple
    try:
        r = Run("plaza", 200, "parity", seed=seed, cam=camera("plaza", "orbit", frames), calib=CAL)
        jumps, occ = [], np.zeros(len(r.sur.coarse), np.int64)
        for f in range(frames):
            prev_tier, prev_pos = r.tier.copy(), r.a.pos.copy()
            r.step(f)
            prom = np.flatnonzero((prev_tier == 3) & (r.tier < 3))
            if prom.size:
                jumps.append(np.linalg.norm(r.a.pos[prom] - prev_pos[prom], axis=1))
            np.add.at(occ, r.d, 1)          # latent occupancy over the whole run
        return np.concatenate(jumps), occ, r
    finally:
        reconcile.COUPLE = keep


@pytest.mark.parametrize("seed", [0, 1])
def test_coupling_shrinks_the_snap(seed):
    jb, _, rb = _run(seed, False)
    jc, _, rc = _run(seed, True)
    # the p95 snap should fall to roughly one ordinary frame of walking (~0.07 m)
    assert np.percentile(jc, 95) < 0.35 * np.percentile(jb, 95), (
        f"p95 jump only went {np.percentile(jb,95):.4f} -> {np.percentile(jc,95):.4f} m")
    assert jc.mean() < jb.mean()


def _region_occupancy(seed, couple):
    _, occ, r = _run(seed, couple)
    p = occ / occ.sum()
    return np.bincount(r.sur.coarse, weights=p, minlength=r.sur.coarse.max() + 1)


def test_coupling_does_not_move_the_law():
    """The correctness claim, tested against a control rather than an invented threshold.

    Retaining d when its coarse region already matches is exact because the region is set FROM
    the live latent at demotion, so a retained d is already a pi_ref(. | R, ctx) draw. The
    residual approximation is that the context may have changed during the surrogate period;
    in plaza 94% of agents sit in one context, so it rarely bites.

    Measure the shift the way an experiment should be measured: compare it to how much two
    DEFAULT runs at different seeds already differ. Use the 16 coarse regions, not the 1000
    corpus points -- at 1000 bins the histogram is so sparse that two identical policies differ
    by TV 0.67, which is why an absolute threshold on the fine latent tests nothing."""
    base = {s: _region_occupancy(s, False) for s in (0, 1, 2)}
    treat = [0.5 * np.abs(base[s] - _region_occupancy(s, True)).sum() for s in (0, 1, 2)]
    ctl = [0.5 * np.abs(base[a] - base[b]).sum() for a, b in ((0, 1), (0, 2), (1, 2))]
    assert np.mean(treat) <= max(ctl), (
        f"coupling shifted region occupancy by TV {np.mean(treat):.4f}, above the "
        f"seed-to-seed noise floor {max(ctl):.4f} -- the tie-break is not free")


@pytest.mark.parametrize("seed", [0, 1])
def test_coupling_leaves_the_guarantee_alone(seed):
    _, _, rb = _run(seed, False)
    _, _, rc = _run(seed, True)
    assert rc.ledger.D.max() <= rc.cap + 1e-9
    assert rb.ledger.D.max() <= rb.cap + 1e-9
    # the allocation itself is untouched: same budget, same tier mix shape
    mb = np.bincount(np.asarray(rb.tier, np.int64), minlength=4)
    mc = np.bincount(np.asarray(rc.tier, np.int64), minlength=4)
    assert abs(int(mb[3]) - int(mc[3])) <= 0.05 * rb.n, f"surrogate population moved {mb} -> {mc}"
