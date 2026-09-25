"""An ensemble of three disciplines, aimed at the one quantity still unbounded.

    python -m bench.fairness --agents 200 --frames 1800

What PARITY already is, read across fields:

  information theory   the ledger accumulates KL and caps it -- invariant 3
  network QoS          that ledger IS a token bucket: accumulate, cap, drain on service. A
                       bucket gives a DEADLINE (serve before overflow) and nothing else.
  optimization         a Lagrangian MCKP maximises salience-weighted quality under the budget

QoS theory is explicit that a deadline alone does not prevent starvation: a flow can be served
exactly at its deadline, forever, and receive an arbitrarily small share. That is precisely
what happens here. Measured over 1800 frames of plaza, the ledger does bound the longest
unbroken degradation at cap / e_rate ~ 384 frames, but 15 of 200 agents are on the surrogate
more than 99% of the time and the median agent is there 77.6%. Instantaneous divergence is
bounded; long-run fidelity share is not.

The fourth discipline supplies the missing half. OS schedulers solve exactly this with
PRIORITY AGING: a task's effective priority grows with time-since-service, so a starved task
eventually outranks a favoured one without any hard reservation. Applied here, the agent's
salience -- which is what the allocator ranks by -- is aged by how long it has gone without
full fidelity:

    s_eff = s * (1 + alpha * age / age_scale)

This deliberately enters the OBJECTIVE, not the mask. A mask overrides the frame budget (see
bench/anticipate.py, where forcing headroom to zero put 574 of 600 frames over budget); an
objective term cannot, because the allocator still solves under the same constraint. The cost
is paid in utility, where it is visible and measurable, rather than in silent overspend.
"""
import argparse

import numpy as np

from bench.camerapaths import camera
from sim.tiered import Run, calibrate


class AgedRun(Run):
    """Salience aged by time-since-full-fidelity. alpha = 0 is the stock policy."""

    alpha = 0.0
    age_scale = 300.0

    def _salience(self, frame):
        s = super()._salience(frame)
        if self.alpha <= 0.0:
            return s
        if not hasattr(self, "_age"):
            self._age = np.zeros(self.n)
        return s * (1.0 + self.alpha * self._age / self.age_scale)

    def step(self, frame):
        super().step(frame)
        if self.alpha > 0.0:
            if not hasattr(self, "_age"):
                self._age = np.zeros(self.n)
            served = self.tier == 0
            self._age = np.where(served, 0.0, self._age + 1.0)


def play(n, frames, seed, cal, alpha, scene="plaza", cam="orbit"):
    r = AgedRun(scene, n, "parity", seed=seed, cam=camera(scene, cam, frames), calib=cal)
    r.alpha = alpha
    at_sur = np.zeros(r.n)
    longest = np.zeros(r.n)
    cur = np.zeros(r.n)
    util, spend = [], []
    for f in range(frames):
        r.step(f)
        s = r.tier == 3
        at_sur += s
        cur = np.where(s, cur + 1, 0)
        longest = np.maximum(longest, cur)
        sal = 1.0 / (1.0 + r.a.sig / 20.0)          # TRUE salience, not the aged one
        util.append(float((sal * r.table.quality[r.assign]).sum()))
        spend.append(float(r.cost[r.assign].sum()))
    frac = at_sur / frames
    spend = np.array(spend)
    return dict(frac=frac, longest=longest, util=float(np.mean(util)),
                over=int((spend > r.budget * (1 + 1e-9)).sum()), frames=frames,
                kl=float(r.D_meas.max()), dmax=float(r.ledger.D.max()), cap=float(r.cap))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--frames", type=int, default=1800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    a = p.parse_args()
    cal = calibrate("plaza", n=200, seed=0)

    print(f"plaza N={a.agents} {a.frames} frames. Utility is scored against TRUE salience, so")
    print(f"aging can only ever cost utility -- the question is how much fairness it buys.\n")
    print(f"{'alpha':>6} {'surrogate p50':>14} {'p99':>7} {'>99% starved':>13} "
          f"{'worst run (s)':>14} {'utility':>9} {'over budget':>12} {'ledger max':>11}")
    for al in a.alphas:
        r = play(a.agents, a.frames, a.seed, cal, al)
        print(f"{al:>6.2f} {np.percentile(r['frac'],50):>14.3f} "
              f"{np.percentile(r['frac'],99):>7.3f} {int((r['frac']>0.99).sum()):>13} "
              f"{r['longest'].max()/60:>14.1f} {r['util']:>9.2f} "
              f"{r['over']:>5}/{r['frames']:<6} {r['dmax']:>11.3f}")


if __name__ == "__main__":
    main()
