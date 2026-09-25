"""What the ledger should charge: the fitted surrogate's in-sample KL rate, or a valid upper bound.

    python -m bench.ratebound --scene plaza

The rate the ledger accrues is an estimate. Estimated in-sample, on the run the surrogate was
fitted to, it is inflated by the reference kernel's estimation noise (KL is convex in the kernel,
so by Jensen the plug-in overshoots) -- and it is not a bound on anything. Sim/surrogate.py's
rate_bounds() gives an upper confidence bound on the TRUE rate from an independent run: valid at
level 1 - delta (tests/test_rate_bound.py measures 60/60 coverage at nominal 0.95), and 10-25%
tighter than the plug-in. Same cap in nats for both; a lower charged rate means the ledger fills
more slowly, so fewer forced restorations for the same guarantee.
"""
import argparse

import numpy as np

from bench.aggregation import counts, play
from sim.tiered import calibrate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="plaza")
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--play", type=int, default=1800)
    ap.add_argument("--agents", type=int, default=200)
    a = ap.parse_args()

    _, A = counts(a.scene, 0, a.frames)
    _, B = counts(a.scene, 1, a.frames)
    b = A.rate_bounds(B.Cf, delta=0.05, draws=200)
    f = A.ctx_freq
    busy = f > 0.01
    plug_in = A.e_rate.copy()
    print(f"{a.scene}: surrogate fitted on run A; rates per busy context {np.flatnonzero(busy)}")
    print(f"  in-sample plug-in (what the ledger charges today) {np.round(plug_in[busy], 5)}")
    print(f"  held-out plug-in                                  {np.round(b['plug'][busy], 5)}")
    print(f"  held-out, noise inflation removed                 {np.round(b['corrected'][busy], 5)}")
    print(f"  held-out upper bound, 95% simultaneous            {np.round(b['upper'][busy], 5)}")

    cal = calibrate(a.scene, n=200, seed=0)
    cap = 300.0 * float(plug_in @ f)
    kc0 = A.kl_coarse.copy()
    res = {}
    for name, e in (("in-sample plug-in", plug_in), ("held-out upper bound", b["upper"])):
        A.e_rate = e
        A.kl_coarse = kc0 * (e / np.maximum(plug_in, 1e-12))[:, None]   # per-context rescale
        res[name] = play(a.scene, A, cap, a.play, cal, a.agents)
    print(f"\n  PARITY, N={a.agents}, {a.play} frames, both at cap {cap:.2f} nats")
    print(f"                             restorations   mean surrogates   utility   worst ledger")
    for name, m in res.items():
        print(f"  {name:26s} {m['rest']:12d} {m['surro']:17.1f} {m['util']:9.2f} {m['dmax']:8.3f} / {cap:.2f}")
    k, o = res["in-sample plug-in"], res["held-out upper bound"]
    print(f"  restorations {100 * (o['rest'] / max(k['rest'], 1) - 1):+.1f}%, utility {100 * (o['util'] / k['util'] - 1):+.2f}%")


if __name__ == "__main__":
    main()
