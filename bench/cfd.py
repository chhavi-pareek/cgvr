"""Crowd Fidelity Deviation: one metric both policies can be scored on without either's
machinery, reported against the noise floor of the simulator itself.

    python -m bench.cfd --agents 200 --frames 900

WHY A NEW METRIC. Every comparison in this repo so far is open to a fair objection:

  utility     = sum s_i * quality(c_i) is PARITY's OBJECTIVE. MassLOD is not trying to
                maximise it, so beating MassLOD on it says nothing.
  divergence  = the KL ledger, computed from PARITY's own surrogate model. MassLOD has no
                such model; the number is defined by the thing being evaluated.

Neither is neutral. The neutral ground truth is the `reference` condition -- full fidelity,
every agent, no LOD -- which both policies are approximations of.

WHY IT IS DISTRIBUTIONAL. Per-agent trajectory error against the reference does not work: a
crowd simulation is chaotic, so two runs of the SAME policy with different behaviour
randomness separate within seconds. Trajectory error would measure chaos, not fidelity. What
survives is the crowd's statistics, which is also what a practitioner actually consumes --
density, flow, speed, spacing. So CFD compares DISTRIBUTIONS, and is blind to agent identity.

WHY THE NOISE FLOOR IS PART OF THE METRIC. A raw TV distance is unreadable on its own. Running
the reference against ITSELF with different behaviour randomness gives the deviation that two
statistically identical crowds already show. CFD is reported as a multiple of that floor:
1.0 means indistinguishable from the reference at this sample size, and a difference below it
is not a difference. Two earlier results in this project were only interpretable once the same
control was added, and one absolute threshold had to be thrown away for lacking it.

The observables are read off positions and velocities only. Nothing in the metric knows which
policy produced them, what a tier is, or that an error ledger exists.
"""
import argparse

import numpy as np

from bench.camerapaths import camera
from sim.tiered import Run, calibrate

GRID = 12          # density field resolution per axis
SPEED_BINS = np.linspace(0.0, 2.2, 17)
SPACE_BINS = np.linspace(0.0, 6.0, 17)


def _hists(r, size, view_weighted):
    """Density, speed and nearest-neighbour spacing, as normalised histograms."""
    pos, vel = r.a.pos, r.a.vel
    w = r.a.in_view.astype(np.float64) if view_weighted else None
    if w is not None and w.sum() < 1.0:
        w = None
    gx = np.clip((pos[:, 0] / size[0] * GRID).astype(int), 0, GRID - 1)
    gy = np.clip((pos[:, 1] / size[1] * GRID).astype(int), 0, GRID - 1)
    dens = np.bincount(gy * GRID + gx, weights=w, minlength=GRID * GRID).astype(np.float64)
    spd = np.linalg.norm(vel, axis=1)
    sh, _ = np.histogram(spd, SPEED_BINS, weights=w)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    ph, _ = np.histogram(nn, SPACE_BINS, weights=w)
    out = []
    for h in (dens, sh.astype(np.float64), ph.astype(np.float64)):
        t = h.sum()
        out.append(h / t if t > 0 else h)
    return out


def _tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def trace(scene, n, cond, frames, seed, cam, cal, brng_seed=None, view_weighted=False,
          budget=None):
    r = Run(scene, n, cond, seed=seed, cam=camera(scene, cam, frames), calib=cal)
    if brng_seed is not None:
        r.brng = np.random.default_rng(brng_seed)     # same scene, different behaviour draw
    if budget is not None and hasattr(r, "budget"):
        r.budget = float(budget)
    size = np.asarray(r.world.sc.size, np.float64)
    out = []
    spend = []
    for f in range(frames):
        r.step(f)
        out.append(_hists(r, size, view_weighted))
        if hasattr(r, "cost") and getattr(r, "assign", None) is not None:
            spend.append(float(r.cost[r.assign].sum()))
    return out, (float(np.mean(spend)) if spend else float("nan"))


def cfd(a, b, stride=5):
    """Mean TV between two traces, per observable. stride subsamples frames; the traces are
    already heavily autocorrelated so every frame adds little."""
    names = ("density", "speed", "spacing")
    per = np.zeros(3)
    k = 0
    for f in range(0, min(len(a), len(b)), stride):
        for j in range(3):
            per[j] += _tv(a[f][j], b[f][j])
        k += 1
    return per / max(k, 1), names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="plaza")
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--cam", default="orbit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view-weighted", action="store_true")
    a = p.parse_args()
    cal = calibrate(a.scene, n=200, seed=0)
    kw = dict(view_weighted=a.view_weighted)

    ref, _ = trace(a.scene, a.agents, "reference", a.frames, a.seed, a.cam, cal, **kw)
    # the floor: the same scene and the same policy, a different behaviour realisation
    ref2, _ = trace(a.scene, a.agents, "reference", a.frames, a.seed, a.cam, cal,
                    brng_seed=a.seed + 4242, **kw)
    floor, names = cfd(ref, ref2)

    base, base_spend = trace(a.scene, a.agents, "baseline", a.frames, a.seed, a.cam, cal, **kw)
    par, par_spend = trace(a.scene, a.agents, "parity", a.frames, a.seed, a.cam, cal, **kw)
    d_base, _ = cfd(ref, base)
    d_par, _ = cfd(ref, par)

    print(f"{a.scene}  N={a.agents}  {a.frames} frames  cam={a.cam}"
          f"{'  (view-weighted)' if a.view_weighted else ''}")
    print(f"\nCrowd Fidelity Deviation from the reference, mean TV per frame")
    print(f"{'observable':12} {'noise floor':>12} {'MassLOD':>10} {'PARITY':>9} "
          f"{'MassLOD/floor':>14} {'PARITY/floor':>13}")
    for j, nm in enumerate(names):
        print(f"{nm:12} {floor[j]:>12.4f} {d_base[j]:>10.4f} {d_par[j]:>9.4f} "
              f"{d_base[j] / max(floor[j], 1e-9):>14.2f} {d_par[j] / max(floor[j], 1e-9):>13.2f}")
    fb, fp = d_base.mean() / max(floor.mean(), 1e-9), d_par.mean() / max(floor.mean(), 1e-9)
    print(f"{'ALL':12} {floor.mean():>12.4f} {d_base.mean():>10.4f} {d_par.mean():>9.4f} "
          f"{fb:>14.2f} {fp:>13.2f}")
    print(f"\n1.00 = indistinguishable from the reference at this sample size.")
    print(f"PARITY predicted spend {par_spend:.4f} ms; the baseline has no allocator and spends")
    print(f"whatever its bands dictate, so a cost-matched run is bench/costmatch.py.")


if __name__ == "__main__":
    main()
