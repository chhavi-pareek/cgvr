"""Phase 7 acceptance harness: (a) bounded accumulated KL, (b) camera-invariant egress.

python -m bench.guarantees --scenes hub corridor --agents 200 --frames 18000
Writes bench/logs/phase7_kl.csv, phase7_egress.csv, phase7_summary.txt.
"""
import argparse
import csv
import os
import time

import numpy as np

from sim.invariant import BAND
from sim.tiered import DT, Run, calibrate

from .camerapaths import NAMES, camera

LOGS = os.path.join(os.path.dirname(__file__), "logs")
KL_FIELDS = ["scene", "cond", "cam", "cap_scale", "cap", "frame", "t0", "t1", "t2", "t3", "kl_mean", "kl_p95", "kl_max",
             "kl_reset_max", "egress", "ledger_mean", "ledger_max", "restorations", "infeasible", "choke_waits", "band_max"]


def run_all(scenes, n, frames, cams, cap_scales, seed, calib_frames):
    kl_rows, eg_rows = [], []
    for scene in scenes:
        t = time.perf_counter()
        cal = calibrate(scene, n=n, frames=calib_frames, seed=seed)
        print(f"{scene}: calibrated in {time.perf_counter() - t:.0f}s, kappa={cal['kappa']:.3f}, "
              f"theta_beh={np.round(cal['theta_beh'], 2)}", flush=True)
        ref = Run(scene, n, "reference", seed=seed, cam=camera(scene, cams[0], frames), calib=cal).run(frames)
        _collect(ref, scene, "reference", "none", 0.0, kl_rows, eg_rows)
        for cam in cams:
            for cs in cap_scales:
                r = Run(scene, n, "parity", seed=seed, cam=camera(scene, cam, frames), calib=cal, cap_scale=cs).run(frames)
                _collect(r, scene, "parity", cam, cs, kl_rows, eg_rows)
                print(f"  parity {cam} cap x{cs}: {r.wall:.0f}s kl_max={r.rows[-1]['kl_max']:.2f} cap={r.cap:.2f} "
                      f"restorations={r.ledger.restorations} egress={len(r.world.egress)}", flush=True)
            b = Run(scene, n, "baseline", seed=seed, cam=camera(scene, cam, frames), calib=cal).run(frames)
            _collect(b, scene, "baseline", cam, 0.0, kl_rows, eg_rows)
            print(f"  baseline {cam}: {b.wall:.0f}s kl_max={b.rows[-1]['kl_max']:.2f} egress={len(b.world.egress)}", flush=True)
    return kl_rows, eg_rows


def _collect(r, scene, cond, cam, cs, kl_rows, eg_rows):
    cap = getattr(r, "cap", 0.0)
    for row in r.rows:
        d = {k: "" for k in KL_FIELDS}
        d.update(row, scene=scene, cond=cond, cam=cam, cap_scale=cs, cap=cap)
        d.pop("cond_", None)
        kl_rows.append(d)
    for agent, trip, frame in r.world.egress:
        eg_rows.append({"scene": scene, "cond": cond, "cam": cam, "cap_scale": cs, "agent": agent, "trip": trip, "frame": frame})


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _slope(frames, y):
    """Least-squares slope over the second half of the run, per 1000 frames."""
    h = len(frames) // 2
    x, yy = np.asarray(frames[h:], float), np.asarray(y[h:], float)
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, yy, 1)[0] * 1000)


def summarise(kl_rows, eg_rows, scenes, cams, cap_scales, frames):
    out = []
    for scene in scenes:
        out.append(f"== {scene} ==")
        out.append("(a) accumulated KL(cond || ref), nats, since last reconciliation (parity) / since t=0 (baseline)")
        out.append("cond      cam           cap    end p95   end max  slope max/1k  peak max  restor.  band")
        for cond in ("parity", "baseline"):
            for cam in cams:
                for cs in (cap_scales if cond == "parity" else [0.0]):
                    rows = [r for r in kl_rows if r["scene"] == scene and r["cond"] == cond and r["cam"] == cam
                            and r["cap_scale"] == cs]
                    if not rows:
                        continue
                    fr = [r["frame"] for r in rows]
                    mx = [r["kl_max"] for r in rows]
                    last = rows[-1]
                    cap = last["cap"] if cond == "parity" else 0.0
                    restor = last.get("restorations", "")
                    band = f"{last['band_max']:.2f}" if cond == "parity" else ""
                    extra = f"  reset-variant end max {last['kl_reset_max']:.2f}" if cond == "baseline" else ""
                    out.append(f"{cond:9s} {cam:13s} {cap:6.2f} {last['kl_p95']:8.2f} {last['kl_max']:9.2f} "
                               f"{_slope(fr, mx):13.3f} {max(mx):9.2f} {restor!s:>8s}  {band}{extra}")
        out.append("(b) egress time per (agent, trip) across the five camera paths")
        out.append("cond      cap   trips(all cams)  max |delta| s   W1 vs cam0 (max over cams) s   pass(<= tol)")
        tol_w1, tol_max = 1.0, BAND / 1.0
        for cond in ("parity", "baseline"):
            for cs in (cap_scales if cond == "parity" else [0.0]):
                per_cam = {}
                for r in eg_rows:
                    if r["scene"] == scene and r["cond"] == cond and r["cap_scale"] == cs:
                        per_cam.setdefault(r["cam"], {})[(r["agent"], r["trip"])] = r["frame"]
                if len(per_cam) < 2:
                    continue
                common = set.intersection(*[set(v) for v in per_cam.values()])
                if common:
                    deltas = [max(per_cam[c][k] for c in per_cam) - min(per_cam[c][k] for c in per_cam) for k in common]
                    dmax = max(deltas) * DT
                else:
                    dmax = float("nan")
                c0 = cams[0]
                w1 = 0.0
                for c in per_cam:
                    if c == c0:
                        continue
                    a, b = np.sort(list(per_cam[c0].values())), np.sort(list(per_cam[c].values()))
                    q = np.linspace(0, 1, 200)
                    w1 = max(w1, float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean()) * DT)
                counts = "/".join(str(len(v)) for v in per_cam.values())
                ok = (dmax <= tol_max) and (w1 <= tol_w1)
                out.append(f"{cond:9s} {cs:4.1f}  {counts:16s} {dmax:12.2f}   {w1:28.2f}   {ok}")
        out.append(f"tolerances: max |delta| <= {tol_max:.1f} s (BAND {BAND} m / 1 m/s), W1 <= {tol_w1:.1f} s")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["hub", "corridor"])
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--frames", type=int, default=18000)
    p.add_argument("--calib-frames", type=int, default=6000)
    p.add_argument("--cams", nargs="+", default=list(NAMES))
    p.add_argument("--caps", nargs="+", type=float, default=[1.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    kl, eg = run_all(args.scenes, args.agents, args.frames, args.cams, args.caps, args.seed, args.calib_frames)
    os.makedirs(LOGS, exist_ok=True)
    _write(os.path.join(LOGS, f"phase7_kl{args.tag}.csv"), kl, KL_FIELDS)
    _write(os.path.join(LOGS, f"phase7_egress{args.tag}.csv"), eg, ["scene", "cond", "cam", "cap_scale", "agent", "trip", "frame"])
    s = summarise(kl, eg, args.scenes, args.cams, args.caps, args.frames)
    with open(os.path.join(LOGS, f"phase7_summary{args.tag}.txt"), "w") as f:
        f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
