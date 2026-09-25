"""The surrogate's regions, chosen for its drift rate instead of for latent proximity.

    python -m bench.aggregation --scene plaza

The surrogate lumps 64 fine latent regions into 16 coarse ones by k-means, i.e. by where the
latents ARE. What the ledger pays for is how well the lumped chain reproduces where they GO,
and that is computable exactly from the calibration's transition counts for any grouping
(sim/surrogate.py::optimise_partition, optimal KL aggregation of a Markov chain). Three
questions, each answered on data the grouping was not fitted to:

1. Rate. Fit on calibration run A, measure the KL rate against the reference kernel of an
   independent run B.
2. Does it buy anything? PARITY run with each surrogate at the SAME cap in nats, both charged
   their held-out rates: a lower rate means the ledger fills more slowly, so fewer forced
   restorations for the same bound.
3. Is the bound still honest? Worst ledger against the cap.
"""
import argparse
import time

import numpy as np

from bench.camerapaths import camera
from sim.surrogate import Surrogate, objective, optimise_partition, _tables
from sim.tiered import Run, calibrate


def counts(scene, seed, frames):
    r = Run(scene, 200, "calib", seed=seed).run(frames, log=False)
    ctx, d = np.stack(r.ctx_log)[300:], np.stack(r.d_log)[300:]
    return r.proc, Surrogate(r.proc).fit(ctx, d)


def held_out(sur, other):
    e, kc = sur.rates_against(other.Cf)
    f = other.ctx_freq
    return e, kc, float(e @ f), float(e[f > 0.01].max())


def play(scene, sur, cap, frames, cal, n):
    r = Run(scene, n, "parity", seed=3, cam=camera(scene, "orbit", frames),
            calib={"surrogate": sur, "kappa": cal["kappa"], "theta_beh": cal["theta_beh"]}, cap=cap)
    util, surro = [], []
    for f in range(frames):
        r.step(f)
        s = 1.0 / (1.0 + r.a.sig / 20.0)
        util.append(float((s * r.table.quality[r.assign]).sum()))
        surro.append(int((r.tier == 3).sum()))
    return dict(rest=r.ledger.restorations, util=float(np.mean(util)), surro=float(np.mean(surro)),
                dmax=float(r.ledger.D.max()), kl=float(r.D_meas.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="plaza")
    ap.add_argument("--frames", type=int, default=6000, help="calibration frames per run")
    ap.add_argument("--play", type=int, default=1800)
    ap.add_argument("--agents", type=int, default=200)
    ap.add_argument("--fit-runs", type=int, default=2, help="calibration runs pooled for the fit")
    a = ap.parse_args()

    # fit on the pooled counts of the first fit-runs seeds, hold out the next seed entirely
    runs = [counts(a.scene, sd, a.frames) for sd in range(a.fit_runs + 1)]
    proc, B = runs[0][0], runs[-1][1]
    A = runs[0][1]
    for _, extra in runs[1:-1]:
        A.Cf = A.Cf + extra.Cf
        A.occ_f_n = A.occ_f_n + extra.occ_f_n
        A.occ_d_n = A.occ_d_n + extra.occ_d_n
    A.set_partition(A.part)                              # refit the k-means surrogate on the pool
    t0 = time.time()
    part, hist = optimise_partition(A.Cf, A.occ_f_n, A.occ_d_n, A.part)
    secs = time.time() - t0
    opt = Surrogate(proc).fit(np.zeros((2, 1), int), np.zeros((2, 1), int))   # replaced below
    opt.Cf, opt.occ_f_n, opt.occ_d_n = A.Cf, A.occ_f_n, A.occ_d_n
    opt.set_partition(part)
    moved = int((part != A.part).sum())

    print(f"{a.scene}: fit on {a.fit_runs} pooled calibration runs, held out on another, "
          f"{a.frames} frames x 200 agents each")
    print(f"  search: {len(hist) - 1} sweeps, {secs:.1f} s, {moved} of 64 fine regions regrouped")
    print(f"                              e_sur fit    e_sur held-out    e_max held-out")
    rows = {}
    for name, sur in (("k-means regions", A), ("KL-optimal regions", opt)):
        e, kc, es, em = held_out(sur, B)
        rows[name] = (sur, e, kc, es, em)
        print(f"  {name:26s} {objective(_tables(sur.Cf, sur.occ_f_n, sur.occ_d_n, sur.part)):11.5f}"
              f" {es:15.5f} {em:17.5f}")
    km, op = rows["k-means regions"], rows["KL-optimal regions"]
    print(f"  held-out mean rate {100 * (op[3] / km[3] - 1):+.1f}%, worst-context rate {100 * (op[4] / km[4] - 1):+.1f}%")

    # both surrogates charged their HELD-OUT rates, both held to the k-means cap in nats
    cal = calibrate(a.scene, n=200, seed=0)
    cap = 300.0 * km[3]
    print(f"\n  PARITY, N={a.agents}, {a.play} frames, both at cap {cap:.2f} nats, held-out rates")
    print(f"                              restorations   mean surrogates   utility    worst ledger")
    res = {}
    for name, (sur, e, kc, es, em) in rows.items():
        sur.e_rate, sur.kl_coarse = e, kc
        res[name] = play(a.scene, sur, cap, a.play, cal, a.agents)
        m = res[name]
        print(f"  {name:26s} {m['rest']:12d} {m['surro']:17.1f} {m['util']:10.2f} {m['dmax']:9.3f} / {cap:.2f}")
    k, o = res["k-means regions"], res["KL-optimal regions"]
    print(f"  restorations {100 * (o['rest'] / max(k['rest'], 1) - 1):+.1f}%, utility {100 * (o['util'] / k['util'] - 1):+.2f}%")


if __name__ == "__main__":
    main()
