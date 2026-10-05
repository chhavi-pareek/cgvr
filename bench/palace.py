"""Jamboo Savari egress from the palace forecourt, full fidelity for every agent.

    python -m bench.palace --seats 48000 --plan nearest
    python -m bench.palace --seats 4000 --minutes 20 --tag flowcheck
    python -m bench.palace --mode entry --seats 48000 --service 600   # arrival through screening

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
from sim.site import APPROACH
from sim.site import RES as RES_
from sim.site import Site
from sim.state import Agents

OUT = pathlib.Path("bench/logs/palace")
CELL = 2.0  # metres, density cells
BODY = 0.25  # metres, body radius (ASSUMED: shoulder breadth about 0.45-0.5 m); bodies never overlap
RHO_R = 1.0  # metres, radius of the disc the local walking density is measured in
FRUIN_E, FRUIN_F = 1 / 0.93, 1 / 0.46  # persons per m2 above which walkway LoS is E, F


def step(a, sc, frame, active, rng=None, dt=DT):
    prev = a.pos.copy()
    mv = np.flatnonzero(sc.moving())
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
        a.pos[mv] += v * dt
        a.pos[:] = resolve(a.pos, BODY, active, sc.fixed(), iters=6)
    sc.constrain(a, prev)
    a.vel[mv] = (a.pos[mv] - prev[mv]) / dt
    sc.after_step(a, frame, rng)


def measured_rates(tag):
    """Sustained persons per minute per gate in an earlier run: the median over the minutes
    in which the gate passed anyone."""
    ex = np.loadtxt(OUT / f"{tag}_exits.csv", delimiter=",", skiprows=1)
    out = {}
    for g in np.unique(ex[:, 1]).astype(int):
        per = np.bincount((ex[ex[:, 1] == g, 2] // int(60 / DT)).astype(int))
        out[g] = float(np.median(per[per > 0]))
    return out


def run(seats, plan="nearest", waves=0.0, width_scale=1.0, gardens=True, minutes=90.0, tag=None, seed=0,
        rates_from=None, open_gates=None):
    rng = np.random.default_rng(seed)
    site0 = Site(gardens_walkable=gardens, open_gates=open_gates)
    widths = {g["key"]: g["width"] * width_scale for g in site0.gates}
    site = site0 if width_scale == 1.0 else Site(gardens_walkable=gardens, widths=widths, open_gates=open_gates)
    rates = measured_rates(rates_from) if plan == "measured" else None
    sc = Palace(site=site, plan=plan, waves=waves, rates=rates)
    a = Agents(seats)
    t0 = time.perf_counter()
    sc.spawn(a, rng)
    nx, ny = int(np.ceil(site.size[0] / CELL)), int(np.ceil(site.size[1] / CELL))
    peak = np.zeros((nx, ny))
    over_e = over_f = 0.0  # cell-seconds above Fruin E / F
    frames = int(minutes * 60 / DT)
    f = 0
    for f in range(frames):
        step(a, sc, f, sc.bodies())
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
    keys = [g["key"] for g in site.gates]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / f"{tag}_exits.csv", np.column_stack([ex, sc.block[ex[:, 0]] if len(ex) else ex[:, :0],
                                                          a.speed[ex[:, 0]] if len(ex) else ex[:, :0]]),
               fmt=["%d", "%d", "%d", "%d", "%.3f"], delimiter=",", header="agent,gate,frame,block,speed", comments="")
    np.savez_compressed(OUT / f"{tag}_density.npz", peak=peak, cell=CELL)
    t_rel = sc.t_release.min()
    mins = (ex[:, 2] - t_rel) * DT / 60 if len(ex) else np.zeros(0)
    lines = [f"palace egress {tag}: seats {seats}, plan {plan}"
             + (f" (rates from {rates_from}: {', '.join(f'{keys[k]} {r:.0f}/min' for k, r in rates.items())})" if rates else "")
             + f", waves {waves:g} s, gate widths x{width_scale:g}, "
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


def run_entry(seats, plan="nearest", window=120.0, peak=45.0, service=600.0, lanes_per_m=1 / 1.5,
              width_scale=1.0, open_gates=None, extra=60.0, fps=15, tag=None, seed=0, rates_from=None):
    """Pass holders arriving through screening to their seats; frame 0 is the gates opening."""
    rng = np.random.default_rng(seed)
    site0 = Site(open_gates=open_gates)
    site = site0 if width_scale == 1.0 else Site(open_gates=open_gates,
                                                  widths={g["key"]: g["width"] * width_scale for g in site0.gates})
    rates = measured_rates(rates_from) if plan == "measured" else None
    if plan == "capacity":
        # pass gates shared by screening capacity (lanes x service), walking distance included
        rates = {k: max(1, int(round(site.gates[k]["width"] * lanes_per_m))) * service / 60.0 for k in site.open}
    sc = Palace(site=site, plan="measured" if plan == "capacity" else plan, rates=rates, mode="entry", window=window,
                peak=peak, service=service, lanes_per_m=lanes_per_m, fps=fps)
    a = Agents(seats)
    t0 = time.perf_counter()
    sc.spawn(a, rng)
    dt = 1.0 / fps
    nx, ny = int(np.ceil(site.size[0] / CELL)), int(np.ceil(site.size[1] / CELL))
    zone_cell = np.zeros((nx, ny), bool)
    zc = np.argwhere(site.zone)
    zone_cell[np.minimum(zc[:, 0] * RES_ // CELL, nx - 1).astype(int), np.minimum(zc[:, 1] * RES_ // CELL, ny - 1).astype(int)] = True
    peak_d = np.zeros((nx, ny))
    over_e = over_f = 0.0
    qmax = {k: 0 for k in site.open}   # held in the gate passage, waiting for a lane
    wmax = {k: 0 for k in site.open}   # still outside, in the gate's approach zone
    frames = int((window + extra) * 60 * fps)
    srng = np.random.default_rng(seed + 1)
    f = 0
    seated_at_start = None
    for f in range(frames):
        step(a, sc, f, sc.bodies(), srng, dt)
        if f % fps == 0:
            outside = np.flatnonzero(sc.phase == 1)
            in_zone = outside[site.walkable(a.pos[outside], site.zone)] if outside.size else outside
            for k in site.open:
                qmax[k] = max(qmax[k], len(sc.queue[k]))
                wmax[k] = max(wmax[k], int((sc.gate[in_zone] == k).sum()))
            out = np.flatnonzero((sc.phase >= 1) & (sc.phase <= 3))
            if out.size:
                ij = np.floor(a.pos[out] / CELL).astype(np.int64)
                ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] < ny)
                h = np.zeros((nx, ny))
                np.add.at(h, (ij[ok, 0], ij[ok, 1]), 1.0 / CELL**2)
                np.maximum(peak_d, h, out=peak_d)
                over_e += float(((h > FRUIN_E) & zone_cell).sum())
                over_f += float(((h > FRUIN_F) & zone_cell).sum())
        if f == sc.t_start:
            seated_at_start = int((sc.phase == 4).sum())
        if (sc.phase == 4).all() and f >= sc.t_start:
            break
    wall = time.perf_counter() - t0
    if seated_at_start is None:
        seated_at_start = int((sc.phase == 4).sum())
    keys = [g["key"] for g in site.gates]
    tag = tag or f"entry_s{seats}_{plan}_svc{service:g}_x{width_scale:g}"
    OUT.mkdir(parents=True, exist_ok=True)
    scr = np.array(sc.screened, np.int64).reshape(-1, 4)
    np.savetxt(OUT / f"{tag}_screened.csv", scr, fmt="%d", delimiter=",", header="agent,gate,joined,released", comments="")
    np.savez_compressed(OUT / f"{tag}_density.npz", peak=peak_d, cell=CELL)
    to_min = 1.0 / (60.0 * fps)
    lines = [f"palace entry {tag}: {seats} pass holders, plan {plan}, arrivals over {window:g} min (mode {peak:g} min "
             f"before the start), screening {service:g}/h per lane, {1 / lanes_per_m:.1f} m of gate per lane, gate widths "
             f"x{width_scale:g}, seed {seed}; {f + 1} frames at {fps} fps in {wall:.0f} s wall",
             f"seated when the procession starts: {seated_at_start}/{seats} ({100 * seated_at_start / seats:.1f}%); "
             f"all seated {'%.1f min after the start' % ((max(t for _, t in sc.seated_at) - sc.t_start) * to_min) if (sc.phase == 4).all() else 'NOT within the run'}"]
    for k in site.open:
        g = site.gates[k]
        m = scr[:, 1] == k
        if not m.any():
            lines.append(f"  {keys[k]:7s} {g['width']:4.1f} m, {sc.lanes[k]} lanes: nobody assigned")
            continue
        wait = (scr[m, 3] - sc.t_arrive[scr[m, 0]]) * to_min
        lines.append(f"  {keys[k]:7s} {g['width']:4.1f} m, {sc.lanes[k]} lanes ({sc.lanes[k] * service / 60:.0f}/min): "
                     f"{m.sum():6d} screened; at most {qmax[k]:5d} at the gate line and {wmax[k]:5d} more in the approach zone; "
                     f"arrival to screened median {np.median(wait):5.1f} p95 {np.percentile(wait, 95):5.1f} max {wait.max():5.1f} min")
    lines.append(f"queue spilling beyond the {APPROACH:g} m approach zones (waiting in the street): at most {sc.spill_max}")
    lines.append(f"walkers put back on their route after a 5 s stall: {sc.snaps}")
    lines.append(f"outside the gates (approach zones): peak {peak_d[zone_cell].max():.2f} persons/m2 in a {CELL:g} m cell; "
                 f"{over_e / 1e3:.1f}k cell-s above Fruin E, {over_f / 1e3:.1f}k above F")
    (OUT / f"{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return tag


def capacity_table(seats=48000, peak=45.0, walk=8.0):
    """Fluid model of entry (DER from the stated assumptions, no crowd simulation): arrivals
    triangular over the window with the mode `peak` minutes before the start, served at the total
    screening capacity, then `walk` minutes to a seat. Share seated at the start, the largest
    number waiting at once, and when the last person clears screening."""
    st = Site()
    width = sum(st.gates[k]["width"] for k in st.open)
    rows = []
    for per_m in (1 / 1.5, 2 / 1.5):
        lanes = sum(max(1, int(round(st.gates[k]["width"] * per_m))) for k in st.open)
        for service in (300.0, 600.0, 1200.0):
            for window in (120.0, 180.0, 240.0):
                t = np.linspace(-window, 120.0, int((window + 120.0) * 60) + 1)  # one-second steps
                a, b, c = -window, -peak, 0.0
                cdf = np.where(t < b, (t - a) ** 2 / ((c - a) * (b - a)),
                               1 - (c - t) ** 2 / ((c - a) * (c - b)))
                arrived = seats * np.clip(np.where(t <= a, 0.0, np.where(t >= c, 1.0, cdf)), 0, 1)
                cap = lanes * service / 3600.0
                served = np.zeros_like(arrived)
                for i in range(1, len(t)):
                    served[i] = min(arrived[i], served[i - 1] + cap)
                at0 = np.searchsorted(t, -walk)
                done = t[np.searchsorted(served, seats - 0.5)] if served[-1] >= seats - 0.5 else np.inf
                rows.append((lanes, service, window, 100 * served[at0] / seats, int((arrived - served).max()), done))
    lines = [f"entry fluid model: {seats} pass holders, {len(st.open)} spectator gates ({width:g} m of clear width), "
             f"arrivals triangular with the mode {peak:g} min before the start, {walk:g} min from screening to a seat",
             "lanes  per lane/h  window min  seated at start  most waiting at once  last screened (min after start)"]
    for lanes, service, window, pct, qmax, done in rows:
        lines.append(f"{lanes:5d}  {service:10.0f}  {window:10.0f}  {pct:14.1f}%  {qmax:20d}  "
                     f"{('%.0f' % done) if np.isfinite(done) and done > 0 else ('before the start' if np.isfinite(done) else 'after +120')}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "entry_capacity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["egress", "entry", "capacity"], default="egress")
    p.add_argument("--seats", type=int, default=48000)
    p.add_argument("--service", type=float, default=600.0, help="entry: persons per hour per screening lane")
    p.add_argument("--lane-m", type=float, default=1.5, help="entry: metres of gate clear width per lane")
    p.add_argument("--window", type=float, default=120.0, help="entry: minutes of arrivals before the start")
    p.add_argument("--peak", type=float, default=45.0, help="entry: most arrivals this many minutes before the start")
    p.add_argument("--open", help="comma list of open gate keys (default: the spectator gates)")
    p.add_argument("--plan", choices=["nearest", "balanced", "measured", "capacity"], default="nearest",
                   help="capacity (entry): pass gates shared by screening lanes x service rate")
    p.add_argument("--rates-from", help="tag of an earlier run whose sustained gate rates the 'measured' plan uses")
    p.add_argument("--waves", type=float, default=0.0, help="seconds between block releases (0: all at once)")
    p.add_argument("--width-scale", type=float, default=1.0)
    p.add_argument("--no-lawns", action="store_true")
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag")
    a = p.parse_args()
    if a.plan == "measured" and not a.rates_from:
        p.error("--plan measured needs --rates-from")
    if a.mode == "capacity":
        capacity_table(a.seats, a.peak)
        return
    if a.mode == "entry":
        run_entry(a.seats, a.plan, a.window, a.peak, a.service, 1.0 / a.lane_m, a.width_scale,
                  a.open.split(",") if a.open else None, tag=a.tag, seed=a.seed, rates_from=a.rates_from)
        return
    run(a.seats, a.plan, a.waves, a.width_scale, not a.no_lawns, a.minutes, a.tag, a.seed, a.rates_from,
        a.open.split(",") if a.open else None)


if __name__ == "__main__":
    main()
