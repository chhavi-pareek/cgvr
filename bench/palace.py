"""Jamboo Savari egress from the palace forecourt, full fidelity for every agent.

    python -m bench.palace --seats 48000 --plan nearest
    python -m bench.palace --seats 4000 --minutes 20 --tag flowcheck

Motion is sim.run's model (goal-seek plus separation, SEP_R, SEP_K, DT) with the cell-list
separation, plus bodies of radius BODY that may not overlap (position projection; velocity is
then the displacement actually made), and desired speed scaled by Weidmann's (1993) speed-density
relation on the local walking density (LIT), along each agent's route (sim/scenes.Palace). The
soft model alone lets a dense crowd compress to 19 persons/m2 and pass a gate at 590 persons/m/min;
bodies alone still allow 354, because a packed column keeps walking at free speed. Seated agents are obstacles until
released; agents that reach a gate's sink have left. Writes bench/logs/palace/<tag>.txt (summary),
<tag>_exits.csv (agent, gate, frame, block, speed) and <tag>_density.npz (per 2 m cell, the
peak density over the run, from 1 s snapshots of the walking crowd).

Results are comparative under the stated assumptions (docs/site/mysore_palace.md), not
predictions for the real site.
"""
import argparse
import pathlib
import time

import numpy as np

from sim.neighbours import local_density, resolve, separation, weidmann
from sim.run import DT, SEP_K, SEP_R
from sim.scenes import Palace
from sim.site import Site
from sim.state import Agents

OUT = pathlib.Path("bench/logs/palace")
CELL = 2.0  # metres, density cells
BODY = 0.25  # metres, body radius (ASSUMED: shoulder breadth about 0.45-0.5 m); bodies never overlap
RHO_R = 1.0  # metres, radius of the disc the local walking density is measured in
FRUIN_E, FRUIN_F = 1 / 0.93, 1 / 0.46  # persons per m2 above which walkway LoS is E, F


def step(a, sc, frame, active):
    prev = a.pos.copy()
    mv = np.flatnonzero((sc.k > 0) & ~sc.done)
    if mv.size:
        sep = separation(a.pos, SEP_R, active)[mv]
        d = a.goal[mv] - a.pos[mv]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        walking = np.zeros(len(a.pos), bool)
        walking[mv] = True
        slow = weidmann(local_density(a.pos, RHO_R, walking)[mv])
        desired = d / np.maximum(dist, 1e-6)[:, None] * (a.speed[mv] * slow)[:, None]
        v = a.vel[mv] + 0.2 * (desired + SEP_K * sep - a.vel[mv])
        sp = np.sqrt(np.einsum("ij,ij->i", v, v))
        v *= np.minimum(1.0, 1.5 * a.speed[mv] * slow / np.maximum(sp, 1e-6))[:, None]
        a.pos[mv] += v * DT
        a.pos[:] = resolve(a.pos, BODY, active, sc.k == 0, iters=6)
    sc.constrain(a, prev)
    a.vel[mv] = (a.pos[mv] - prev[mv]) / DT
    sc.after_step(a, frame, None)


def run(seats, plan="nearest", waves=0.0, width_scale=1.0, gardens=True, minutes=90.0, tag=None, seed=0):
    rng = np.random.default_rng(seed)
    site0 = Site(gardens_walkable=gardens)
    widths = {g["key"]: g["width"] * width_scale for g in site0.gates}
    site = site0 if width_scale == 1.0 else Site(gardens_walkable=gardens, widths=widths)
    sc = Palace(site=site, plan=plan, waves=waves)
    a = Agents(seats)
    t0 = time.perf_counter()
    sc.spawn(a, rng)
    nx, ny = int(np.ceil(site.size[0] / CELL)), int(np.ceil(site.size[1] / CELL))
    peak = np.zeros((nx, ny))
    over_e = over_f = 0.0  # cell-seconds above Fruin E / F
    frames = int(minutes * 60 / DT)
    f = 0
    for f in range(frames):
        active = ~sc.done
        step(a, sc, f, active)
        if f % 30 == 0:
            walk = np.flatnonzero((sc.k > 0) & ~sc.done)
            if walk.size:
                ij = np.floor(a.pos[walk] / CELL).astype(np.int64)
                ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] < ny)
                h = np.zeros((nx, ny))
                np.add.at(h, (ij[ok, 0], ij[ok, 1]), 1.0 / CELL**2)
                np.maximum(peak, h, out=peak)
                over_e += float((h > FRUIN_E).sum())
                over_f += float((h > FRUIN_F).sum())
        if sc.done.all():
            break
    wall = time.perf_counter() - t0
    ex = np.array(sc.left, np.int64).reshape(-1, 3)
    tag = tag or f"s{seats}_{plan}_w{waves:g}_x{width_scale:g}" + ("" if gardens else "_nolawn")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / f"{tag}_exits.csv", np.column_stack([ex, sc.block[ex[:, 0]] if len(ex) else ex[:, :0],
                                                          a.speed[ex[:, 0]] if len(ex) else ex[:, :0]]),
               fmt=["%d", "%d", "%d", "%d", "%.3f"], delimiter=",", header="agent,gate,frame,block,speed", comments="")
    np.savez_compressed(OUT / f"{tag}_density.npz", peak=peak, cell=CELL)
    keys = [g["key"] for g in site.gates]
    t_rel = sc.t_release.min()
    mins = (ex[:, 2] - t_rel) * DT / 60 if len(ex) else np.zeros(0)
    lines = [f"palace egress {tag}: seats {seats}, plan {plan}, waves {waves:g} s, gate widths x{width_scale:g}, "
             f"lawns {'walkable' if gardens else 'blocked'}, seed {seed}; {f + 1} frames in {wall:.0f} s wall",
             f"left {len(ex)}/{seats}" + ("" if len(ex) == seats else f" (stopped at {minutes:g} min)")]
    if len(ex):
        q = np.percentile(mins, [50, 95, 100])
        lines.append(f"egress minutes after release: T50 {q[0]:.1f}  T95 {q[1]:.1f}  T100 {q[2]:.1f}  (8-minute guide)")
    for k in site.open:
        g = site.gates[k]
        m = ex[:, 1] == k
        if not m.any():
            lines.append(f"  {keys[k]:7s} width {g['width']:4.1f} m: unused")
            continue
        fr = ex[m, 2]
        per_min = np.bincount((fr - fr.min()) // int(60 / DT))
        lines.append(f"  {keys[k]:7s} width {g['width']:4.1f} m: {m.sum():6d} out, last at {mins[m].max():5.1f} min, "
                     f"peak {per_min.max() / g['width']:5.1f} persons/m/min (Green Guide 82 level)")
    slow = a.speed < Palace.SPEED * Palace.ELDERLY_SCALE
    if len(ex) and slow[ex[:, 0]].any():
        lines.append(f"slowest walkers (< {Palace.SPEED * Palace.ELDERLY_SCALE:.2f} m/s) median exit {np.median(mins[slow[ex[:, 0]]]):.1f} min, "
                     f"others {np.median(mins[~slow[ex[:, 0]]]):.1f} min")
    left_in = np.flatnonzero(~sc.done)
    if left_in.size:
        dg = np.linalg.norm(a.goal[left_in] - a.pos[left_in], axis=1)
        lines.append(f"still inside: {left_in.size}, at waypoint {np.bincount(sc.k[left_in]).tolist()} (index: count), "
                     f"median {np.median(dg):.1f} m from it; e.g. agent {left_in[0]} at {np.round(a.pos[left_in[0]] - site.shift, 1).tolist()} "
                     f"(site frame) heading {np.round(a.goal[left_in[0]] - site.shift, 1).tolist()}")
    lines.append(f"walking crowd: peak {peak.max():.2f} persons/m2 in a {CELL:g} m cell; "
                 f"{over_e / 1e3:.1f}k cell-s above Fruin E ({FRUIN_E:.2f}/m2), {over_f / 1e3:.1f}k above F ({FRUIN_F:.2f}/m2)")
    (OUT / f"{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seats", type=int, default=48000)
    p.add_argument("--plan", choices=["nearest", "balanced"], default="nearest")
    p.add_argument("--waves", type=float, default=0.0, help="seconds between block releases (0: all at once)")
    p.add_argument("--width-scale", type=float, default=1.0)
    p.add_argument("--no-lawns", action="store_true")
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag")
    a = p.parse_args()
    run(a.seats, a.plan, a.waves, a.width_scale, not a.no_lawns, a.minutes, a.tag, a.seed)


if __name__ == "__main__":
    main()
