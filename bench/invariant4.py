"""What the view-invariant core actually buys: a bound on the reconciliation snap.

    python -m bench.invariant4

Invariant 4 says a view-invariant core is simulated at fixed cost for every agent and is never
allocated. `Core.bind` is the half of it that is easy to overlook: it clamps the *fine*
simulation of agents at EVERY tier to within BAND = 2 m of core progress, so full-fidelity
agents are constrained too, not just surrogates.

That clamp is what makes every other guarantee in the system land somewhere finite. A
surrogate's position is read straight off the core, so when an agent is demoted its fine
position is replaced by the core point -- and the size of that replacement is exactly how far
the fine simulation had been allowed to drift. With the band it is at most BAND. Without it,
the drift accumulates for as long as the agent stays live and the snap grows with it.

The error ledger bounds behavioural divergence; the core bounds positional discontinuity.
Different mechanisms, different quantities, and the ledger's bound says nothing about the
second -- which is why ablating the core is the way to see it.
"""
import argparse

import numpy as np

from bench.camerapaths import camera
from sim import invariant
from sim.tiered import Run, calibrate


def play(scene, n, frames, seed, cam, cal, bind):
    invariant.BIND = bind
    try:
        r = Run(scene, n, "parity", seed=seed, cam=camera(scene, cam, frames), calib=cal)
        demote_jump, promote_jump, band = [], [], []
        for f in range(frames):
            prev_tier, prev_pos, prev_s = r.tier.copy(), r.a.pos.copy(), r.core.s.copy()
            r.step(f)
            # an agent whose trip ended this frame was respawned at the route start, core and
            # all -- a new trip, not a reconciliation snap -- so it is not counted
            same_trip = r.core.s >= prev_s
            dem = np.flatnonzero((prev_tier < 3) & (r.tier == 3) & same_trip)
            pro = np.flatnonzero((prev_tier == 3) & (r.tier < 3) & same_trip)
            if dem.size:
                demote_jump.append(np.linalg.norm(r.a.pos[dem] - prev_pos[dem], axis=1))
            if pro.size:
                promote_jump.append(np.linalg.norm(r.a.pos[pro] - prev_pos[pro], axis=1))
            band.append(float(np.abs(r.core.project(r.a.pos) - r.core.s).max()))
        return dict(dem=np.concatenate(demote_jump) if demote_jump else np.zeros(1),
                    pro=np.concatenate(promote_jump) if promote_jump else np.zeros(1),
                    band=np.array(band), egress=len(r.world.egress),
                    kl=float(r.D_meas.max()), dmax=float(r.ledger.D.max()), cap=float(r.cap))
    finally:
        invariant.BIND = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="plaza")
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--cam", default="orbit")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = p.parse_args()
    cal = calibrate(a.scene, n=200, seed=0)

    for seed in a.seeds:
        on = play(a.scene, a.agents, a.frames, seed, a.cam, cal, True)
        off = play(a.scene, a.agents, a.frames, seed, a.cam, cal, False)
        print(f"\n=== seed {seed} ===    BAND = {invariant.BAND} m")
        print(f"{'':28} {'core bound (invariant 4)':>26} {'ablated':>14}")
        for lbl, f in (("fine-vs-core drift max (m)", lambda r: r["band"].max()),
                       ("fine-vs-core drift end (m)", lambda r: r["band"][-1]),
                       ("demote snap mean (m)", lambda r: r["dem"].mean()),
                       ("demote snap p95 (m)", lambda r: np.percentile(r["dem"], 95)),
                       ("demote snap max (m)", lambda r: r["dem"].max()),
                       ("promote snap max (m)", lambda r: r["pro"].max()),
                       ("trips completed", lambda r: r["egress"]),
                       ("ledger max (nats)", lambda r: r["dmax"]),
                       ("measured KL max (nats)", lambda r: r["kl"])):
            b, c = f(on), f(off)
            print(f"{lbl:28} {b:>26.4f} {c:>14.4f}")
        print(f"  the ledger's bound is unaffected ({on['dmax']:.3f} vs {off['dmax']:.3f}, cap "
              f"{on['cap']:.3f}): it bounds behaviour, not position.")


if __name__ == "__main__":
    main()
