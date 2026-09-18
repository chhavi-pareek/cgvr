"""Phase 8 sweep: 5 conditions x 3 scenes x 6 agent counts (plus a camera-path sub-sweep).

Only on an explicit RUN SWEEP. Development: --scenes hub --agents 200 --frames 300 --limit 2.

    python -m bench.sweep --dry-run                 # list cells, no runs
    python -m bench.sweep                           # full grid, resumable
    python -m bench.sweep --cam-sweep               # + all five camera paths for parity/baseline

Conditions
  reference         full fidelity + invariant core (quality ceiling, cost ceiling; KL = 0)
  baseline          UE5 MassLOD threshold-and-cap (phase 1 assigner, phase 7 tick semantics)
  parity            allocator + ledger + surrogate + reconciliation + core
  parity_nocap      ablation of invariant 3: cap = inf, restoration is never forced
  parity_authored   ablation of invariant 2: hand-weighted per-axis quality ramps instead of the
                    latent-derived table (only the quality column changes; costs and rates do not)

Outputs (bench/logs/, gitignored), one row appended per finished cell so the sweep resumes:
  sweep_cells{tag}.csv   one row per (scene, cond, n, cam, seed): frame-time stats, KL end state,
                         allocator time, tier mix
  sweep_kl{tag}.csv      every LOG_EVERY frames: the phase 7 KL_FIELDS plus n and seed
  sweep_egress{tag}.csv  (agent, trip, frame) per completed trip, for the camera-variance figure
"""
import argparse
import csv
import itertools
import os
import time

import numpy as np

from alloc.serial import SerialAllocator
from sim.tiered import DT, LOG_EVERY, Run, calibrate

from .camerapaths import NAMES, camera
from .guarantees import KL_FIELDS

LOGS = os.path.join(os.path.dirname(__file__), "logs")

SCENES = ("plaza", "hub", "corridor")
AGENTS = (100, 200, 500, 1000, 2000, 5000)
CONDITIONS = ("reference", "baseline", "parity", "parity_nocap", "parity_authored")

# invariant-2 ablation: authored per-axis ramps and hand weights (behaviour, nav, anim, geo)
AUTHORED_RAMP = np.array([1.0, 0.75, 0.5, 0.0])
AUTHORED_W = np.array([0.4, 0.2, 0.2, 0.2])

NOCAP = 1e9  # nats; finite so the ledger's initial desynchronisation draw stays valid
WARMUP = 30  # frames excluded from the frame-time statistics

CELL_FIELDS = ["scene", "cond", "n", "cam", "seed", "frames", "wall_s", "ft_mean_ms", "ft_p50_ms", "ft_p95_ms",
               "ft_p99_ms", "ft_low1_ms", "ft_max_ms", "alloc_ms_mean", "alloc_ms_p99", "alloc_share",
               "kl_p95_end", "kl_max_end", "kl_max_peak", "kl_slope_per_1k", "cap", "restorations", "infeasible",
               "egress", "t0_frac", "t1_frac", "t2_frac", "t3_frac", "band_max", "allocator"]
KL_SWEEP_FIELDS = ["n", "seed"] + KL_FIELDS
EGRESS_FIELDS = ["scene", "cond", "n", "cam", "seed", "agent", "trip", "frame"]


def _authored_quality(table):
    return AUTHORED_RAMP[table.tiers] @ AUTHORED_W


def make_run(scene, n, cond, seed, cam, cal, allocator="serial"):
    if allocator == "threaded":
        from alloc.threaded import ThreadedAllocator as alloc_cls
    else:
        alloc_cls = SerialAllocator
    kw = dict(seed=seed, cam=camera(scene, cam, 1), calib=cal)
    if cond == "reference" or cond == "baseline":
        return Run(scene, n, cond, **kw)
    if cond == "parity":
        return Run(scene, n, "parity", alloc_cls=alloc_cls, **kw)
    if cond == "parity_nocap":
        return Run(scene, n, "parity", alloc_cls=alloc_cls, cap=NOCAP, **kw)
    if cond == "parity_authored":
        r = Run(scene, n, "parity", alloc_cls=alloc_cls, **kw)
        r.table.quality = _authored_quality(r.table)
        return r
    raise ValueError(cond)


def run_cell(scene, n, cond, seed, cam, frames, cal, allocator):
    r = make_run(scene, n, cond, seed, cam, cal, allocator)
    r.cam = camera(scene, cam, frames)
    alloc_t = []
    if hasattr(r, "alloc"):
        inner = r._allocate

        def timed(frame):
            t = time.perf_counter()
            inner(frame)
            alloc_t.append(time.perf_counter() - t)

        r._allocate = timed
    ft = np.empty(frames)
    tiers = np.zeros(4)
    t_wall = time.perf_counter()
    for f in range(frames):
        t = time.perf_counter()
        r.step(f)
        ft[f] = time.perf_counter() - t
        tiers += np.bincount(r.tier.astype(np.int64), minlength=4)
        if f % LOG_EVERY == LOG_EVERY - 1:
            r.rows.append(r._row(f))
    r.wall = time.perf_counter() - t_wall
    ft = ft[WARMUP:] * 1e3 if frames > 2 * WARMUP else ft * 1e3
    k = max(1, int(round(0.01 * len(ft))))
    low1 = float(np.sort(ft)[-k:].mean())
    kl_max = np.array([row["kl_max"] for row in r.rows])
    fr = np.array([row["frame"] for row in r.rows])
    half = len(fr) // 2
    slope = float(np.polyfit(fr[half:], kl_max[half:], 1)[0] * 1000) if len(fr) - half >= 3 else float("nan")
    at = np.array(alloc_t[WARMUP:] if len(alloc_t) > 2 * WARMUP else alloc_t) * 1e3
    cell = dict(
        scene=scene, cond=cond, n=n, cam=cam, seed=seed, frames=frames, wall_s=round(r.wall, 2),
        ft_mean_ms=float(ft.mean()), ft_p50_ms=float(np.percentile(ft, 50)), ft_p95_ms=float(np.percentile(ft, 95)),
        ft_p99_ms=float(np.percentile(ft, 99)), ft_low1_ms=low1, ft_max_ms=float(ft.max()),
        alloc_ms_mean=float(at.mean()) if at.size else 0.0, alloc_ms_p99=float(np.percentile(at, 99)) if at.size else 0.0,
        alloc_share=float(at.sum() / (ft.sum())) if at.size else 0.0,
        kl_p95_end=r.rows[-1]["kl_p95"], kl_max_end=r.rows[-1]["kl_max"], kl_max_peak=float(kl_max.max()),
        kl_slope_per_1k=slope, cap=getattr(r, "cap", 0.0), restorations=getattr(getattr(r, "ledger", None), "restorations", 0),
        infeasible=getattr(r, "infeasible", 0), egress=len(r.world.egress), band_max=r.band_max, allocator=allocator,
    )
    tf = tiers / tiers.sum()
    cell.update(t0_frac=tf[0], t1_frac=tf[1], t2_frac=tf[2], t3_frac=tf[3])
    kl_rows = []
    for row in r.rows:
        d = {k_: "" for k_ in KL_SWEEP_FIELDS}
        d.update(row, scene=scene, cond=cond, cam=cam, n=n, seed=seed, cap_scale=1.0, cap=cell["cap"])
        kl_rows.append(d)
    eg_rows = [dict(scene=scene, cond=cond, n=n, cam=cam, seed=seed, agent=a, trip=t, frame=f)
               for a, t, f in r.world.egress]
    return cell, kl_rows, eg_rows


def _append(path, rows, fields):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["scene"], r["cond"], int(r["n"]), r["cam"], int(r["seed"])) for r in csv.DictReader(f)}


def cells(args):
    out = []
    for scene, cond, n, seed in itertools.product(args.scenes, args.conditions, args.agents, args.seeds):
        for cam in args.cams:
            out.append((scene, cond, n, cam, seed))
    if args.cam_sweep:
        for scene, cond, n, seed in itertools.product(args.scenes, ("parity", "baseline"), args.cam_agents, args.seeds):
            for cam in NAMES:
                c = (scene, cond, n, cam, seed)
                if c not in out and cond in args.conditions:
                    out.append(c)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=list(SCENES), choices=SCENES)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--agents", nargs="+", type=int, default=list(AGENTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--cams", nargs="+", default=["orbit"], choices=NAMES)
    p.add_argument("--cam-sweep", action="store_true", help="all five camera paths for parity and baseline")
    p.add_argument("--cam-agents", nargs="+", type=int, default=[200])
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--frames-big", type=int, default=600, help="frames for cells with n >= --big")
    p.add_argument("--big", type=int, default=2000)
    p.add_argument("--calib-frames", type=int, default=6000)
    p.add_argument("--allocator", choices=["serial", "threaded"], default="serial")
    p.add_argument("--limit", type=int, default=0, help="run at most this many cells (0 = all)")
    p.add_argument("--tag", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    cells_path = os.path.join(LOGS, f"sweep_cells{args.tag}.csv")
    kl_path = os.path.join(LOGS, f"sweep_kl{args.tag}.csv")
    eg_path = os.path.join(LOGS, f"sweep_egress{args.tag}.csv")
    todo = [c for c in cells(args) if c not in _done(cells_path)]
    if args.limit:
        todo = todo[: args.limit]
    frames_of = lambda n: args.frames_big if n >= args.big else args.frames  # noqa: E731
    total_frames = sum(frames_of(c[2]) * c[2] for c in todo)
    print(f"{len(todo)} cells to run ({len(cells(args)) - len(todo)} done), "
          f"{total_frames / 1e6:.1f} M agent-frames -> {cells_path}", flush=True)
    if args.dry_run:
        for c in todo[:12]:
            print("  ", *c, frames_of(c[2]))
        if len(todo) > 12:
            print(f"   ... {len(todo) - 12} more")
        return
    cal = {}
    t_all = time.perf_counter()
    for scene, cond, n, cam, seed in todo:
        if scene not in cal:
            cal[scene] = calibrate(scene, n=200, frames=args.calib_frames, seed=0)
        frames = frames_of(n)
        cell, kl_rows, eg_rows = run_cell(scene, n, cond, seed, cam, frames, cal[scene], args.allocator)
        _append(kl_path, kl_rows, KL_SWEEP_FIELDS)
        _append(eg_path, eg_rows, EGRESS_FIELDS)
        _append(cells_path, [cell], CELL_FIELDS)
        print(f"{scene:8s} {cond:16s} n={n:5d} {cam:12s} s{seed}: {cell['wall_s']:7.1f}s "
              f"ft {cell['ft_mean_ms']:.2f}/{cell['ft_low1_ms']:.2f} ms  kl_max {cell['kl_max_end']:.2f} "
              f"egress {cell['egress']}", flush=True)
    print(f"done in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
