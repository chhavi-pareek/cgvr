"""The view side, measured on the simulator's own trajectories.

    python -m bench.views --scene plaza --agents 400 --frames 1200

Positions come from the reference simulation; costs and the geometry quality column are the
engine's measured ones (plaza, shadows on: geometry 14.8 / 3.8 / 1.0 us over the impostor,
quality 1.000 / 0.891 / 0.545 / 0.457 from the pixel judge), so the view axes are priced the
way the renderer actually prices them. Two questions:

1. Pops. How often does an on-screen agent visibly change detail, and how badly can one agent
   flicker? Three policies: MassLOD's bands with its own hysteresis (10% distance margin,
   15-frame visibility hold), the two-salience allocator unconstrained, and the same with the
   pop ledger (alloc/pops.py). Hysteresis reduces the average; only the ledger bounds the worst.

2. Two viewers on one simulation. The per-viewer allocator (one behaviour per agent, one view
   pair per agent per viewer) against the obvious shortcut at the same budget: one view pair,
   chosen for whichever viewer sees the agent best, drawn in both views.
"""
import argparse

import numpy as np

from alloc.config import AXIS_QUALITY
from alloc.costmodel import row_costs_from_theta
from alloc.factored import Factorisation, FactoredAllocator
from alloc.pops import PopLedger
from bench.camerapaths import camera
from sim.assign_threshold import MassLODConfig, ThresholdAssigner
from sim.tiered import Run, calibrate, phase7_table

GEO_US = (14.8, 3.8, 1.0)
ANIM_US = (0.167, 0.126, 0.062)
BEH_US = (0.150, 0.138, 0.138)
GEO_Q = (1.000, 0.891, 0.545, 0.457)
D0 = 10.0


def engine_costs(table, beh_us=BEH_US):
    th = np.zeros((4, 4))
    th[0, :3], th[2, :3], th[3, :3] = beh_us, ANIM_US, GEO_US
    return row_costs_from_theta(table, 5.0, th) * 1e-3


def seen_and_dist(pos, cam_p, yaw, c=MassLODConfig()):
    rel = pos - cam_p
    d2 = np.einsum("ij,ij->i", rel, rel)
    d = np.sqrt(d2 + c.cam_h ** 2)
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    cosang = (rel @ fwd) / np.maximum(np.sqrt(d2), 1e-6)
    return (cosang >= np.cos(c.fov / 2 + c.view_margin)) & (d <= c.far), d


def budget_for(F, n, V, frac):
    lo = F.c_state.min() + V * F.c_view.min()
    hi = F.c_state.max() + V * F.c_view.max()
    return n * (lo + frac * (hi - lo))


def worst_window(hist, W):
    W = min(W, hist.shape[0])
    cs = np.cumsum(np.vstack([np.zeros((1, hist.shape[1]), int), hist.astype(int)]), 0)
    return (cs[W:] - cs[:-W]).max(0)


def run(scene, n, frames, seed, frac, cam2, C, refill, beh_us=BEH_US):
    cal = calibrate(scene, n=200, seed=0)
    table = phase7_table(0.05716)
    aq = AXIS_QUALITY.copy()
    aq[3] = GEO_Q
    cost = engine_costs(table, beh_us)
    F = Factorisation(table, cost, axis_quality=aq)
    cams = [camera(scene, "orbit", frames), camera(scene, cam2, frames)]
    r = Run(scene, n, "reference", seed=seed, cam=cams[0], calib=cal)
    asg = ThresholdAssigner()

    free, held = FactoredAllocator(table, axis_quality=aq), FactoredAllocator(table, axis_quality=aq)
    led = PopLedger((1, n), C, refill)
    prev_free = np.full(n, -1)
    prev_mass = None
    pops = {"MassLOD": [], "free": [], "ledger": []}
    util_free, util_held, released = [], [], 0
    multi, merged = [], []
    B1 = budget_for(F, n, 1, frac)
    B2 = budget_for(F, n, 2, frac)
    merged_cost = F.c_state[F.state_of] + 2 * F.c_view[F.view_of]

    for f in range(frames):
        r.step(f)
        views = [seen_and_dist(r.a.pos, *c.at(f)) for c in cams]
        (s1, d1), (s2, d2) = views
        sig = np.minimum(d1 * np.where(s1, 1.0, 2.0), d2 * np.where(s2, 1.0, 2.0))
        a = 1.0 / (1.0 + sig / 20.0)
        b1 = np.where(s1, np.minimum(1.0, (D0 / d1) ** 2), 0.0)
        b2 = np.where(s2, np.minimum(1.0, (D0 / d2) ** 2), 0.0)

        # -- pops, viewer 1 --------------------------------------------------------
        asg.step(r.a, *cams[0].at(f))
        mass = r.a.vis_tier.copy()
        if prev_mass is not None:
            pops["MassLOD"].append((mass != prev_mass) & s1)
        prev_mass = mass
        rf = free.allocate(a, cost, B1, view_salience=b1)
        if f:
            pops["free"].append((free.view_pair[0] != prev_free) & s1)
        prev_free = free.view_pair[0].copy()
        rh = held.allocate(a, cost, B1, view_salience=b1, view_lock=led.holds(s1[None, :]))
        released += held.released
        popped = led.update(held.view_pair, s1[None, :])[0]
        if f:
            pops["ledger"].append(popped)
        util_free.append(rf.utility)
        util_held.append(rh.utility)

        # -- two viewers -------------------------------------------------------------
        B = np.vstack([b1, b2])
        m = FactoredAllocator(table, axis_quality=aq)
        rm = m.allocate(a, cost, B2, view_salience=B)
        g = FactoredAllocator(table, axis_quality=aq)
        g.allocate(a, merged_cost, B2, view_salience=B.max(0))
        u_merged = float((a * F.q_state[g.state_pair]).sum() + (B * F.q_view[g.view_pair[0]][None, :]).sum())
        multi.append(rm.utility)
        merged.append(u_merged)

    secs = frames / 60.0
    print(f"{scene}: N={n}, {frames} frames ({secs:.0f} s), budget {100*frac:.0f}% of the priced range, "
          f"viewer 2 = '{cam2}', behaviour {beh_us} us")
    print("\nvisible detail changes (viewer 1)        per agent-min   worst agent in any 2 s   any 10 s")
    for k in ("MassLOD", "free", "ledger"):
        h = np.array(pops[k])
        rate = h.sum() / n / (h.shape[0] / 3600.0)
        w2, w10 = worst_window(h, 120), worst_window(h, 600)
        name = {"MassLOD": "MassLOD bands + hysteresis", "free": "two-salience, no pop limit",
                "ledger": f"two-salience + pop ledger"}[k]
        print(f"  {name:36s} {rate:10.2f} {w2.max():16d} {w10.max():12d}")
    print(f"  ledger bound C + r*T: {C + refill * 120:.1f} in 2 s, {C + refill * 600:.1f} in 10 s;"
          f"  holds released for the budget: {released}")
    print(f"  utility cost of the pop ledger: {100 * (1 - np.mean(util_held) / np.mean(util_free)):.3f}%")
    print("\ntwo viewers on one simulation, same budget")
    print(f"  per-viewer detail vs one detail drawn in both views: utility +{100 * (np.mean(multi) / np.mean(merged) - 1):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="plaza")
    ap.add_argument("--agents", type=int, default=400)
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--cam2", default="sweep")
    ap.add_argument("--capacity", type=float, default=2.0)
    ap.add_argument("--refill", type=float, default=1.0 / 300.0)
    ap.add_argument("--heavy-behaviour", action="store_true",
                    help="behaviour at the Python model's measured 6.1 / 6.1 / 5.7 us over the surrogate")
    a = ap.parse_args()
    beh = (6.08, 6.09, 5.68) if a.heavy_behaviour else BEH_US
    run(a.scene, a.agents, a.frames, a.seed, a.frac, a.cam2, a.capacity, a.refill, beh)


if __name__ == "__main__":
    main()
