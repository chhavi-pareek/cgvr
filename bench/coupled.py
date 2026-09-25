"""Coupled reconciliation: same law, smaller snap.

    python -m bench.coupled --agents 200 --frames 600

sim/reconcile.py is free to pick any outcome consistent with the reference law. The default
picks one independently of where the agent already is, so the restoration is a teleport --
and bench/anticipate.py shows 83.6% of them land on camera. Coupling changes only the
tie-break: the lateral slot search starts from the agent's current offset, and the latent is
retained when its coarse region already matches. Neither alters the distribution.

That last point is the one that has to be checked rather than asserted, so this script reports
the divergence and tier statistics alongside the jump: if coupling biased the law, the KL would
move.
"""
import argparse

import numpy as np

from bench.camerapaths import camera
from sim import reconcile
from sim.tiered import Run, calibrate


def play(scene, n, frames, seed, cam, cal, couple):
    keep = reconcile.COUPLE
    reconcile.COUPLE = couple
    r = Run(scene, n, "parity", seed=seed, cam=camera(scene, cam, frames), calib=cal)
    jumps, changed, steps = [], [], []
    seen = total = 0
    util = []
    for f in range(frames):
        prev_tier, prev_pos, prev_d = r.tier.copy(), r.a.pos.copy(), r.d.copy()
        r.step(f)
        prom = np.flatnonzero((prev_tier == 3) & (r.tier < 3))
        stay = np.flatnonzero((prev_tier < 3) & (r.tier < 3))
        if prom.size:
            jumps.append(np.linalg.norm(r.a.pos[prom] - prev_pos[prom], axis=1))
            changed.append((r.d[prom] != prev_d[prom]).astype(float))
            total += prom.size
            seen += int(r.a.in_view[prom].sum())
        if stay.size:
            steps.append(np.linalg.norm(r.a.pos[stay] - prev_pos[stay], axis=1))
        s = 1.0 / (1.0 + r.a.sig / 20.0)
        util.append(float((s * r.table.quality[r.assign]).sum()))
    reconcile.COUPLE = keep
    j = np.concatenate(jumps)
    return dict(j=j, step=np.concatenate(steps), changed=float(np.concatenate(changed).mean()),
                total=total, seen=seen, util=float(np.mean(util)),
                kl=float(r.D_meas.max()), klmean=float(r.D_meas.mean()),
                dmax=float(r.ledger.D.max()), cap=float(r.cap),
                mix=np.bincount(np.asarray(r.tier, np.int64), minlength=4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="plaza")
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--cam", default="orbit")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()

    cal = calibrate(a.scene, n=a.agents if a.agents <= 200 else 200, seed=0)
    for seed in a.seeds:
        base = play(a.scene, a.agents, a.frames, seed, a.cam, cal, False)
        cpl = play(a.scene, a.agents, a.frames, seed, a.cam, cal, True)
        st = base["step"]
        print(f"\n=== seed {seed} ===   ordinary per-frame movement: "
              f"mean {st.mean():.4f}  p95 {np.percentile(st,95):.4f} m")
        print(f"{'':16} {'default':>28} {'coupled':>28}")
        for lbl, f in (("jump mean (m)", lambda r: r["j"].mean()),
                       ("jump p95 (m)", lambda r: np.percentile(r["j"], 95)),
                       ("jump max (m)", lambda r: r["j"].max()),
                       ("latent redrawn", lambda r: r["changed"]),
                       ("reconciliations", lambda r: r["total"]),
                       ("on camera", lambda r: r["seen"] / max(r["total"], 1)),
                       ("utility", lambda r: r["util"]),
                       ("KL mean (nats)", lambda r: r["klmean"]),
                       ("KL max (nats)", lambda r: r["kl"]),
                       ("ledger max", lambda r: r["dmax"])):
            b, c = f(base), f(cpl)
            delta = "" if b == 0 else f"   ({100*(c/b-1):+6.1f}%)"
            print(f"{lbl:16} {b:>28.4f} {c:>22.4f}{delta}")
        print(f"{'tier mix':16} {str(base['mix']):>28} {str(cpl['mix']):>28}")
        assert cpl["dmax"] <= cpl["cap"] + 1e-9, "coupling broke the cap"


if __name__ == "__main__":
    main()
