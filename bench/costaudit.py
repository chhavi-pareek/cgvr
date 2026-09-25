"""Is the cost model complete? Regress measured frame time on it and look at what is left.

    python -m bench.costaudit

Invariant 1 says tier costs are estimated from measured telemetry rather than authored. That
was checked for the tiers it knows about, and nothing checked whether it knows about every
cost. It did not: reconciliation costs 16-39 us per agent and was never charged at all, which
silently flattered every configuration that churns.

The general form of that bug is a cost model whose residual correlates with an observable it
does not track, so that is what this measures. Regress the measured fine-step time on the
model's own prediction, then correlate the residual against counts the model ignores. A
complete model leaves a residual that is noise; a correlated residual names the missing term.

This is a diagnostic, not a benchmark -- the absolute times are Python and mean nothing. What
means something is which observable the residual points at.
"""
import time

import numpy as np

from bench.camerapaths import camera
from sim.tiered import Run, calibrate


def audit(scene="plaza", n=200, frames=600, seed=0, cam="orbit"):
    cal = calibrate(scene, n=200, seed=0)
    r = Run(scene, n, "parity", seed=seed, cam=camera(scene, cam, frames), calib=cal)
    # Time BOTH the fine step and the allocate step. Reconciliation runs inside _allocate,
    # not _fine_core, so instrumenting only the fine step hides exactly the cost this audit
    # exists to find -- the first version of this script did that and reported a clean bill.
    fine, alloc = r._fine_core, r._allocate
    timing = {"fine": 0.0, "alloc": 0.0}

    def timed_fine():
        t = time.perf_counter()
        fine()
        timing["fine"] = (time.perf_counter() - t) * 1e6

    def timed_alloc(frame):
        t = time.perf_counter()
        alloc(frame)
        timing["alloc"] = (time.perf_counter() - t) * 1e6

    r._fine_core = timed_fine
    r._allocate = timed_alloc

    pred, meas, promo, demo, surro = [], [], [], [], []
    for f in range(frames):
        prev = r.tier.copy()
        r.step(f)
        pred.append(float(r.cost[r.assign].sum()) * 1e3)      # ms -> us
        meas.append(timing["fine"] + timing["alloc"])
        promo.append(int(((prev == 3) & (r.tier < 3)).sum()))
        demo.append(int(((prev < 3) & (r.tier == 3)).sum()))
        surro.append(int((r.tier == 3).sum()))
    return (np.array(pred), np.array(meas), np.array(promo, float),
            np.array(demo, float), np.array(surro, float))


def residual_correlation(promo_only=True):
    """Correlation of the cost model's residual with the reconciliation count. Used by
    tests/test_costaudit.py as a regression guard on invariant 1's completeness."""
    pred, meas, promo, _, _ = audit(frames=420)
    w = slice(60, None)
    pred, meas, promo = pred[w], meas[w], promo[w]
    A = np.stack([np.ones_like(pred), pred], 1)
    coef, *_ = np.linalg.lstsq(A, meas, rcond=None)
    resid = meas - A @ coef
    return float(np.corrcoef(resid, promo)[0, 1]) if promo.std() > 0 else 0.0


def main():
    pred, meas, promo, demo, surro = audit()
    warm = slice(60, None)                     # drop JIT / cache warm-up
    pred, meas, promo, demo, surro = (v[warm] for v in (pred, meas, promo, demo, surro))

    A = np.stack([np.ones_like(pred), pred], 1)
    coef, *_ = np.linalg.lstsq(A, meas, rcond=None)
    resid = meas - A @ coef
    print(f"measured crowd time (fine step + allocate) vs the model's prediction, "
          f"{len(pred)} frames")
    print(f"  fit: measured = {coef[0]:.1f} us + {coef[1]:.3f} * predicted")
    print(f"  slope should be ~1 if the model prices the tiers correctly")
    print(f"  residual std {resid.std():.1f} us on a mean of {meas.mean():.1f} us\n")

    print(f"{'observable the model ignores':34} {'corr with residual':>19}")
    flagged = []
    for name, v in (("promotions (reconciliations)", promo), ("demotions", demo),
                    ("agents on the surrogate", surro)):
        if v.std() < 1e-9:
            continue
        c = float(np.corrcoef(resid, v)[0, 1])
        mark = "   <- uncharged cost" if abs(c) > 0.2 else ""
        if abs(c) > 0.2:
            flagged.append((name, c))
        print(f"{name:34} {c:>19.3f}{mark}")

    if flagged:
        print(f"\nThe residual is correlated with {len(flagged)} observable(s) the cost model does")
        print("not price. Each is a term that should be in it; see STATE.md on R * e / cap.")
    else:
        print("\nResidual uncorrelated with every tracked observable: no missing term detected.")
    return resid, promo


if __name__ == "__main__":
    main()
