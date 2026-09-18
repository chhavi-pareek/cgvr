"""Sweep the key weight w over three densities; Pareto front of coalescing vs coherence.

python -m order.sweep --agents 2000 --tile 5 --frames 3          # main sweep, N = 50k (~2 min)
python -m order.sweep --agents 200 --densities mixed --frames 1  # development
python -m order.sweep --agents 500 1000 2000 --seed 1 --out bench/logs/order_sweep_test.csv

Per configuration: warp execution efficiency (analytic counters, and on hardware
the instrumented kernel's counters, asserted equal), bytes moved per agent per
frame (sector model), kernel time (CUDA events on hardware; modelled under the
simulator, column `source` says which) and achieved bandwidth = bytes / time.
"""
import argparse
import csv
import math
import os

import numpy as np

from . import density, kernel, key

LOG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench", "logs", "order_sweep.csv")
FIELDS = ["density", "n", "seed", "frame", "w", "rho", "geff", "issued", "active", "wee", "wee_meas",
          "sectors", "bytes_per_agent", "t_model_us", "t_meas_us", "t_us", "bw_gbs", "source"]
W_GRID = [round(0.1 * i, 1) for i in range(11)]


def geff(tier, cls):
    h = np.bincount(tier.astype(np.int64) * key.NCLS + cls, minlength=4 * key.NCLS).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


BLIND = -1.0  # w label of the spatially blind (state, id) reference ordering


def measure(fr, w, reps=20):
    p = key.order(fr.pos, fr.tier, fr.cls, max(w, 0.0), fr.size, blind=(w == BLIND))
    pos, vel, cls, tier = fr.pos[p], fr.vel[p], fr.cls[p], fr.tier[p]
    a = kernel.analyse(pos, vel, cls, tier, fr.size)
    row = dict(density=fr.density, n=fr.n, frame=fr.index, w=w, rho=a["rho"], geff=geff(tier, cls),
               issued=a["issued"], active=a["active"], wee=a["wee"], sectors=a["sectors"],
               bytes_per_agent=a["bytes_per_agent"], t_model_us=a["t_model_us"], wee_meas=math.nan,
               t_meas_us=math.nan)
    if not kernel.SIMULATED:
        _, trips, ws = kernel.run_instr(pos, vel, cls, tier, fr.size)
        m = kernel.stats_to_wee(ws, fr.n, cls, tier)
        row["wee_meas"] = m["wee"]
        if not np.array_equal(ws.astype(np.int64), a["wstats"]):
            raise RuntimeError("instrumented warp counters differ from the host model")
        row["t_meas_us"] = kernel.time_kernel_us(pos, vel, cls, tier, fr.size, reps)
    row["source"] = "model" if kernel.SIMULATED else "events"
    row["t_us"] = row["t_model_us"] if kernel.SIMULATED else row["t_meas_us"]
    row["bw_gbs"] = a["bytes"] / row["t_us"] * 1e-3
    return row


def pareto(points):
    """Indices not dominated in (bytes low, wee high)."""
    keep = []
    for i, (b, e) in enumerate(points):
        dom = any((b2 <= b and e2 >= e) and (b2 < b or e2 > e) for j, (b2, e2) in enumerate(points) if j != i)
        if not dom:
            keep.append(i)
    return keep


def summarise(rows, ws):
    out = []
    groups = sorted({(r["density"], r["n"]) for r in rows}, key=lambda g: (g[1], g[0]))
    dominated = {0.0: [], 1.0: []}
    blind = BLIND in ws
    ws = [w for w in ws if w != BLIND]
    for d, n in groups:
        sub = [r for r in rows if r["density"] == d and r["n"] == n]
        mean = lambda w, k: float(np.mean([r[k] for r in sub if r["w"] == w]))
        pts = [(mean(w, "bytes_per_agent"), mean(w, "wee")) for w in ws]
        front = pareto(pts)
        best = min(ws, key=lambda w: mean(w, "t_us"))
        for w0 in (0.0, 1.0):
            if w0 in ws and ws.index(w0) not in front:
                dominated[w0].append(f"{d}/{n}")
        out.append(f"{d:6s} n={n:5d} rho={mean(1.0, 'rho'):5.1f} geff={mean(1.0, 'geff'):4.1f} | "
                   f"w=0: {pts[ws.index(0.0)][0]:5.0f} B/agent wee={pts[ws.index(0.0)][1]:.3f} | "
                   f"w=1: {pts[ws.index(1.0)][0]:5.0f} B/agent wee={pts[ws.index(1.0)][1]:.3f} | "
                   f"front w={[ws[i] for i in front]} best-time w={best} ({mean(best, 't_us'):.1f} us)"
                   + (f" | blind: {mean(BLIND, 'bytes_per_agent'):5.0f} B/agent wee={mean(BLIND, 'wee'):.3f}" if blind else ""))
    out.append(f"w=0 (pure state) dominated at: {dominated[0.0] or 'nowhere'}")
    out.append(f"w=1 (pure Morton) dominated at: {dominated[1.0] or 'nowhere'}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", nargs="+", default=list(density.DENSITIES))
    ap.add_argument("--agents", nargs="+", type=int, default=[2000])
    ap.add_argument("--w", nargs="+", type=float, default=W_GRID + [BLIND])
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--tile", type=int, default=1, help="k x k copies of each frame (N = k^2 * agents)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=LOG)
    args = ap.parse_args()
    rows = []
    for d in args.densities:
        for n in args.agents:
            for fr in density.frames(d, n, seed=args.seed, count=args.frames):
                fr = density.tile(fr, args.tile, args.seed)
                for w in args.w:
                    r = measure(fr, w, args.reps)
                    r["seed"] = args.seed
                    rows.append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} rows -> {args.out} (time source: {rows[0]['source']})")
    print(summarise(rows, list(args.w)))


if __name__ == "__main__":
    main()
