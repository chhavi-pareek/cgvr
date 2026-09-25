"""Anticipatory reconciliation: restore agents before they are seen, not because they are seen.

    python -m bench.anticipate --agents 200 --frames 600

The error ledger forces a restoration when an agent reaches its cap. Nothing says the
restoration has to happen *then* -- restoring earlier only lowers D, so the bound is
untouched. What the current policy actually does is worse than arbitrary: an agent enters
view, its salience rises, the allocator promotes it off the surrogate, and the reconciliation
fires. The snap therefore happens on camera by construction. Measured: 83.6% of
reconciliations occur while the agent is in view.

Invariant 4 is what makes this fixable. The view-invariant core is simulated exactly for every
agent at every tier, so the agent's future position is known regardless of its fidelity, and
the camera path is known too. Predict which agents enter the frustum within a lookahead
window, zero their headroom, and the mask forces them live -- and therefore reconciled --
while they are still off screen.

The cost is time budget: those agents now occupy high-fidelity rows slightly earlier than they
otherwise would. The allocator pays for it by demoting agents that are not about to be seen,
which is exactly the trade salience is supposed to make.
"""
import argparse

import numpy as np

from alloc.ledger import ErrorLedger
from bench.camerapaths import camera
from sim.assign_threshold import MassLODConfig
from sim.tiered import DT, Run, calibrate


def predicted_sig(r, frame, k):
    """Significance the agent WILL have in k frames, from the view-invariant core.

    The core is simulated exactly for every agent at every tier (invariant 4), so this is
    available even for an agent currently running the surrogate -- which is precisely the
    agent whose future matters. Same distance/frustum/penalty form as the live assigner, so
    the two are on one scale."""
    c = MassLODConfig()
    cam_p, cam_yaw = r.cam.at(frame + k)
    s_ahead = np.minimum(r.core.s + r.a.speed * k * DT, r.core.L)
    pos = r.core.point(s_ahead, np.arange(r.n))
    rel = pos - cam_p
    d2 = np.einsum("ij,ij->i", rel, rel)
    d = np.sqrt(d2 + c.cam_h ** 2)
    fwd = np.array([np.cos(cam_yaw), np.sin(cam_yaw)], np.float32)
    cosang = (rel @ fwd) / np.maximum(np.sqrt(d2), 1e-6)
    seen = (cosang >= np.cos(c.fov / 2 + c.view_margin)) & (d <= c.far)
    return d * np.where(seen, 1.0, c.oov_penalty)


class PredictiveRun(Run):
    """Salience from where the camera WILL be, not only where it is.

    The allocator maximises sum s_i * quality(c_i), so raising an agent's salience before it
    becomes visible makes the allocator upgrade it early -- and an upgrade off the surrogate is
    a reconciliation. The snap therefore lands while the agent is still off screen, and because
    this is a preference inside the objective rather than a constraint on the mask, it cannot
    push the allocation over the frame budget the way zeroing headroom does.
    """

    lookahead = 0

    def _salience(self, frame):
        s = super()._salience(frame)
        if self.lookahead <= 0:
            return s
        sig = np.minimum(self.a.sig, predicted_sig(self, frame, self.lookahead))
        return 1.0 / (1.0 + sig / 20.0)


class AnticipatoryLedger(ErrorLedger):
    """The naive version, kept because it is instructive: zeroing the headroom of soon-visible
    agents forces them live, but the mask overrides the frame budget, so it simply overspends
    -- measured at 1.92 ms against a 1.275 ms budget on 574 of 600 frames."""

    def __init__(self, run, cap, rng, lookahead):
        super().__init__(run.n, cap, rng)
        self.run, self.lookahead = run, lookahead
        self.frame = 0
        self.forced = 0

    def _will_be_seen(self):
        r = self.run
        c = MassLODConfig()
        k = self.lookahead
        cam_p, cam_yaw = r.cam.at(min(self.frame + k, 10 ** 9))
        # the core is authoritative and tier-independent, so this prediction is available for
        # every agent including the ones currently running the surrogate (invariant 4)
        s_ahead = r.core.s + r.a.speed * k * DT
        pos = r.core.point(np.minimum(s_ahead, r.core.L), np.arange(r.n))
        rel = pos - cam_p
        d2 = np.einsum("ij,ij->i", rel, rel)
        d = np.sqrt(d2 + c.cam_h ** 2)
        fwd = np.array([np.cos(cam_yaw), np.sin(cam_yaw)], np.float32)
        cosang = (rel @ fwd) / np.maximum(np.sqrt(d2), 1e-6)
        return (cosang >= np.cos(c.fov / 2 + c.view_margin)) & (d <= c.far)

    def headroom(self):
        h = super().headroom()
        if self.lookahead <= 0:
            return h
        soon = self._will_be_seen()
        self.forced = int(soon.sum())
        return np.where(soon, 0.0, h)


def play(scene, n, frames, seed, cam, cal, lookahead, hard=False):
    cls = PredictiveRun if not hard else Run
    r = cls(scene, n, "parity", seed=seed, cam=camera(scene, cam, frames), calib=cal)
    if hard:
        led = AnticipatoryLedger(r, r.cap, np.random.default_rng(seed + 7), lookahead)
        led.D = r.ledger.D.copy()
        r.ledger = led
        r._led = led
    else:
        r.lookahead = lookahead
    seen = total = 0
    util, surro, spend = [], [], []
    for f in range(frames):
        if hard:
            r._led.frame = f
        prev = r.tier.copy()
        r.step(f)
        prom = np.flatnonzero((prev == 3) & (r.tier < 3))
        total += prom.size
        seen += int(r.a.in_view[prom].sum())
        s = 1.0 / (1.0 + r.a.sig / 20.0)
        util.append(float((s * r.table.quality[r.assign]).sum()))
        surro.append(int((r.tier == 3).sum()))
        spend.append(float(r.cost[r.assign].sum()))
    spend = np.array(spend)
    return dict(total=total, seen=seen, util=float(np.mean(util)),
                surro=float(np.mean(surro)), kl=float(r.D_meas.max()),
                cap=float(r.cap), dmax=float(r.ledger.D.max()),
                spend=float(spend.mean()), budget=float(r.budget),
                over=int((spend > r.budget * (1 + 1e-9)).sum()),
                infeasible=int(r.infeasible), frames=frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="plaza")
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--cam", default="orbit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookaheads", type=int, nargs="+", default=[0, 10, 20, 40, 80])
    p.add_argument("--hard", action="store_true", help="the naive headroom-zeroing version")
    a = p.parse_args()

    cal = calibrate(a.scene, n=200, seed=0)
    print(f"{a.scene}  N={a.agents}  {a.frames} frames  cam={a.cam}\n")
    print(f"{'lookahead':>10} {'reconcil':>9} {'on camera':>12} {'utility':>8} "
          f"{'surrog':>7} {'spend/budget':>13} {'over budget':>12} {'ledger max':>11}")
    base_u = None
    for k in a.lookaheads:
        r = play(a.scene, a.agents, a.frames, a.seed, a.cam, cal, k, hard=a.hard)
        if base_u is None:
            base_u = r["util"]
        pct = 100 * r["seen"] / max(r["total"], 1)
        print(f"{k:>10} {r['total']:>9} {r['seen']:>6} ({pct:>4.1f}%) "
              f"{r['util']:>8.2f} {r['surro']:>7.1f} "
              f"{r['spend']:>6.3f}/{r['budget']:<6.3f} "
              f"{r['over']:>5}/{r['frames']:<6} {r['dmax']:>11.3f}"
              f"{'  CAP BREACH' if r['dmax'] > r['cap'] + 1e-6 else ''}")
    print(f"\nlookahead 0 is the current policy. Restoring early only lowers D, so the bound is")
    print(f"untouched at every lookahead -- no schedule can breach the cap by restoring sooner.")


if __name__ == "__main__":
    main()
