"""Cost-matched comparison: give PARITY exactly the frame budget MassLOD spends, then ask
which one diverges.

    python -m bench.costmatch --agents 200 --frames 900

Every other comparison in this repo lets the two policies spend whatever they spend, so a
reader can fairly object that PARITY simply bought more fidelity. Here the baseline runs
first, its per-frame cost is measured with the SAME calibrated cost vector, and PARITY is then
handed that number as its budget. Neither policy is allowed to outspend the other.

Phase 7 executes only the behaviour axis, so quality and cost are both read off that axis for
both policies, which is the only apples-to-apples comparison available.
"""
import argparse
import numpy as np

from alloc.config import AXIS_QUALITY
from bench.camerapaths import camera
from alloc.ledger import ErrorLedger
from alloc.guarantee import safe_admission_rate
from sim.tiered import Run, calibrate


class MeasuredLedger(ErrorLedger):
    """A ledger that charges the quantity it claims to bound.

    alloc/ledger.py is handed e_rate[ctx] -- the region-AVERAGED surrogate rate -- while
    sim/tiered.py charges D_meas the agent's own kl_coarse[ctx, region]. Two different
    accumulators, so raising the admission rate bounds the proxy harder and does nothing for
    the measured divergence. Charging kl_coarse here makes D == D_meas, and then the mask's
    bound transfers to the quantity the figures actually plot.
    """

    def __init__(self, run, cap, rng):
        super().__init__(run.n, cap, rng)
        self.run = run

    def accrue(self, _rate_ignored):
        r = self.run
        sur = r.tier == 3
        self.D += np.where(sur, r.sur.kl_coarse[r.core.ctx, r.region], 0.0)
        return self.D


def behaviour_tier_cost(run):
    """[4] ms per agent per frame at each behaviour tier, from the run's own cost vector.

    In phase 7 the other three axes contribute nothing, so every row sharing a behaviour tier
    costs the same; assert that rather than assume it."""
    t = run.table.tiers[:, 0]
    out = np.zeros(4)
    for b in range(4):
        c = run.cost[t == b]
        assert np.ptp(c) < 1e-12, f"behaviour tier {b} rows differ in cost by {np.ptp(c):.3g}"
        out[b] = c[0]
    return out


def play(scene, n, cond, frames, seed, cam, cal, tier_cost, budget=None, safe_admit=False):
    r = Run(scene, n, cond, seed=seed, cam=camera(scene, cam, frames), calib=cal)
    if budget is not None:
        r.budget = float(budget)
    if safe_admit and hasattr(r, "table"):
        # (A1) for the MEASURED divergence: admit against the worst region an agent can be in
        # rather than the context average, AND charge the ledger the same quantity D_meas
        # accrues. Either alone is not enough. alloc/guarantee.py.
        a = safe_admission_rate({"kl_coarse": r.sur.kl_coarse, "pi_c": r.sur.pi_c})
        r.table.err = np.where(r.table.tiers[:, 0] == 3, a, 0.0)
        r.ledger = MeasuredLedger(r, r.cap, np.random.default_rng(seed + 7))
    spend, util, kl = [], [], []
    for f in range(frames):
        r.step(f)
        s = 1.0 / (1.0 + r.a.sig / 20.0)
        spend.append(float(tier_cost[r.tier].sum()))
        util.append(float((s * AXIS_QUALITY[0][r.tier]).sum()))
        kl.append(float(r.D_meas.max()))
    mix = np.bincount(np.asarray(r.tier, np.int64), minlength=4)
    return dict(spend=np.array(spend), util=np.array(util), kl=np.array(kl), mix=mix,
                cap=float(getattr(r, "cap", np.nan)), budget=float(getattr(r, "budget", np.nan)),
                restorations=int(getattr(getattr(r, "ledger", None), "restorations", 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="plaza")
    p.add_argument("--agents", type=int, default=200)
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--cam", default="orbit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--safe-admission", action="store_true",
                   help="admit against max_region kl_coarse so the cap bounds D_meas, not just the ledger")
    a = p.parse_args()

    cal = calibrate(a.scene, n=200, seed=0)
    # only the parity branch of Run builds a table and a cost vector, so price both policies
    # from one parity instance -- that is also what makes the match exact
    probe = Run(a.scene, a.agents, "parity", seed=a.seed, cam=camera(a.scene, a.cam, 1), calib=cal)
    tier_cost = behaviour_tier_cost(probe)
    print(f"behaviour-tier cost per agent (ms): "
          f"{'  '.join(f'{v*1e3:.3f}us' for v in tier_cost)}")

    base = play(a.scene, a.agents, "baseline", a.frames, a.seed, a.cam, cal, tier_cost)
    b_ms = float(base["spend"].mean())
    par = play(a.scene, a.agents, "parity", a.frames, a.seed, a.cam, cal, tier_cost,
               budget=b_ms, safe_admit=a.safe_admission)

    print(f"scene {a.scene}  N={a.agents}  {a.frames} frames  cap {par['cap']:.3f} nats")
    print(f"budget handed to PARITY = the baseline's own mean spend = {b_ms:.4f} ms\n")
    print(f"{'':22} {'baseline':>12} {'PARITY':>12}")
    print(f"{'mean spend (ms)':22} {base['spend'].mean():>12.4f} {par['spend'].mean():>12.4f}")
    print(f"{'p95 spend (ms)':22} {np.percentile(base['spend'],95):>12.4f} "
          f"{np.percentile(par['spend'],95):>12.4f}")
    print(f"{'delivered utility':22} {base['util'].mean():>12.3f} {par['util'].mean():>12.3f}"
          f"   ({100*(par['util'].mean()/base['util'].mean()-1):+.1f}%)")
    print(f"{'worst-agent KL end':22} {base['kl'][-1]:>12.3f} {par['kl'][-1]:>12.3f}")
    print(f"{'worst-agent KL max':22} {base['kl'].max():>12.3f} {par['kl'].max():>12.3f}")
    print(f"{'restorations':22} {base['restorations']:>12d} {par['restorations']:>12d}")
    print(f"{'behaviour mix t0/1/2/3':22} {'/'.join(map(str, base['mix'])):>12} "
          f"{'/'.join(map(str, par['mix'])):>12}")

    over = par["spend"] > b_ms * 1.0001
    print(f"\nPARITY frames over the matched budget: {int(over.sum())} of {a.frames}")
    print(f"PARITY divergence stayed under the cap: {bool(par['kl'].max() <= par['cap'] + 1e-6)}")
    ratio = base["kl"][-1] / max(par["kl"][-1], 1e-9)
    print(f"\nAt equal spend the baseline ends {ratio:.1f}x more diverged"
          f" ({base['kl'][-1]:.2f} vs {par['kl'][-1]:.2f} nats).")


if __name__ == "__main__":
    main()
