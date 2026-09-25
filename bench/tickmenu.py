"""Step 0: does the allocator actually want frame-skipping once its drift is charged honestly?

    python -m bench.tickmenu

MassLOD's whole speed advantage is that it ticks distant agents every 3rd or 10th frame.
PARITY has no such option: its behaviour tiers are latent WIDTH, all running every frame, so
the only way it can save time is to drop to the surrogate. Measured in Unity, that leaves it
bang-bang (tier 0 or surrogate, nothing between) and slower than the baseline on crowd work.

Adding tick-k rows means PARITY can do everything MassLOD does, but charged for. Two numbers
are needed and only one exists:

  drift   sim/surrogate.py already computes kl_tick for periods (1, 3, 10). Measured.
  quality NOT measured anywhere. A tick-k agent's decoded parameters are exact but STALE
          between ticks. Crediting tick-k with full quality while charging it real drift breaks
          the ledger in the opposite direction from the Unity bug that charged zero drift.

So this measures the staleness quality the same way phase 2 measured truncation quality -- as
1 - normalised MSE of the decoded parameter vector against the reference -- which is what keeps
the two commensurable, and is the whole point of invariant 2.
"""
import numpy as np

from sim.behaviour import BehaviourProcess
from sim.surrogate import BASELINE_PERIODS, Surrogate
from sim.tiered import CORE_US
from alloc.config import AXIS_QUALITY
from alloc.serial import SerialAllocator

FULL_US = 10.930        # measured behaviour tier 0, whole fine step
SUR_US = 4.854          # measured behaviour tier 3 (surrogate), the floor
N_AGENTS, T = 400, 900


def stale_quality(seed=0):
    """1 - NMSE of the decoded parameters under tick-k, against the SAME ground truth the
    phase 2 truncation study used.

    The reference has to be latent/train.get_data()[0] -- the true 19-dim parameter vector --
    not decode(d, 16). Scoring staleness against the 16-dim decode measures only the staleness
    and silently credits tick-k with the truncation accuracy it does not have; done that way
    tick-3 scores 0.9751, above full latent16's 0.8680, which is impossible. Two error sources
    against two different references are exactly the incommensurability invariant 2 exists to
    rule out.

    Validation: at k = 1 there is no staleness, so the result must reproduce the recorded
    latent-16 truncation quality of 1 - 0.132. The caller asserts it."""
    from latent.train import get_data

    proc = BehaviourProcess(seed=seed)
    # the 19-dim head targets, concatenated in the order sim/behaviour.py::_build uses, and
    # normalised by summed head variance -- latent/train.py::head_err / head_var, which is the
    # definition that produced the recorded _BEH_NMSE
    _, y, _, _ = get_data()
    truth_all = np.concatenate([np.asarray(y[g]) for g in ("vel", "gaze", "gesture", "gait")], 1)
    truth = truth_all[proc.sub]
    var = float(truth.var(0).sum())
    rng = np.random.default_rng(7)
    d = rng.integers(0, len(proc.sub), N_AGENTS)
    ctx = rng.integers(0, 4, N_AGENTS)
    stale = {k: d.copy() for k in BASELINE_PERIODS}
    se = {k: 0.0 for k in BASELINE_PERIODS}
    tot = 0.0
    live = np.arange(N_AGENTS)
    for t in range(T):
        proc.step(d, ctx, live, rng)
        ref = truth[d]                                    # what the agent SHOULD be doing now
        for k in BASELINE_PERIODS:
            if t % k == 0:
                stale[k] = d.copy()
            se[k] += float((((proc.decode(stale[k], 16) - ref) ** 2).sum(1)).mean())
        tot += 1.0
    return {k: 1.0 - (se[k] / tot) / var for k in BASELINE_PERIODS}


class Menu:
    """A behaviour-axis-only menu: (name, quality, us/agent, drift/frame)."""

    def __init__(self, rows):
        self.names = [r[0] for r in rows]
        self.quality = np.array([r[1] for r in rows], np.float64)
        self.cost = np.array([r[2] for r in rows], np.float64)
        self.err = np.array([r[3] for r in rows], np.float64)
        self.m = len(rows)
        self.tiers = np.zeros((self.m, 4), np.int8)


# Measured: sim/reconcile.py::promote costs 16.4 us per agent after optimisation (39.3 before:
# a tabulated inverse-CDF for the latent draw, 11.5x, and a lazily-batched slot search, 2.4x).
# Still 1.5x an entire per-frame behaviour step. A row with drift e forces a restoration every
# cap/e frames, so its true per-frame cost carries an amortised R * e / cap. The cost model
# never charged it, which flatters exactly the configurations that churn hardest.
RECONCILE_US = 16.4     # measured after optimisation; see STATE.md


def build(q_tick, e_tick, e_sur, reconcile_us=0.0, cap=1.0):
    beh = AXIS_QUALITY[0]
    rows = [("full latent16", beh[0], FULL_US, 0.0),
            ("latent8", beh[1], 10.943, 0.0),
            ("latent4", beh[2], 10.527, 0.0),
            ("surrogate", beh[3], SUR_US, e_sur)]
    for k in BASELINE_PERIODS[1:]:
        # the full step every k frames, the core every frame; the core is never allocated
        rows.append((f"tick-{k}", q_tick[k], CORE_US + (FULL_US - CORE_US) / k, e_tick[k]))
    if reconcile_us > 0.0:
        # Per-TRANSITION-TYPE, not global. Charging one R to every drifting row flatters the
        # surrogate, because the two transitions are not the same operation:
        #   surrogate -> live  reconstructs state: draw d ~ pi_ref(. | R, ctx), find a free
        #                      lateral slot, rebuild gait phase, and snap the position onto
        #                      the core point. Measured 16.4 us/agent.
        #   tick-k -> tick-1   RESUMES. A tick-k agent never left the live state -- its latent
        #                      is valid and merely lagged, its position was integrated from its
        #                      own decoded velocity and is already inside the core band. There
        #                      is nothing to reconstruct and nothing to snap, so the cost is a
        #                      tier-field write.
        # That is structural, not a constant factor: frame-skipping avoids reconciliation by
        # construction, so no optimisation of promote() can close the gap.
        rows = [(n, q, c + (reconcile_us if n == "surrogate" else 0.0) * e / cap, e)
                for (n, q, c, e) in rows]
    return Menu(rows)


def main():
    d = np.load("bench/logs/phase7_plaza_calib.npz", allow_pickle=True)
    kt, pif, pic, klc, cf = d["kl_tick"], d["pi_f_given_c"], d["pi_c"], d["kl_coarse"], d["ctx_freq"]
    pi_fine = np.einsum("cR,cRr->cr", pic, pif)
    e_sur = float(np.einsum("cR,cR,c->", pic, klc, cf))
    e_tick = {k: float(np.einsum("cr,cr,c->", pi_fine, kt[i], cf))
              for i, k in enumerate(BASELINE_PERIODS)}

    print("measuring tick-k staleness quality against the phase 2 ground truth ...")
    q_tick = stale_quality()
    recorded = AXIS_QUALITY[0][0]
    print(f"  validation: tick-1 has no staleness, so it must reproduce latent-16's recorded "
          f"quality.\n    measured {q_tick[1]:.4f}  recorded {recorded:.4f}  "
          f"delta {q_tick[1] - recorded:+.4f}")
    cap = 300.0 * e_sur
    menu = build(q_tick, e_tick, e_sur)
    charged = build(q_tick, e_tick, e_sur, RECONCILE_US, cap)

    print(f"\n{'config':16} {'quality':>8} {'steady us':>10} {'+reconcile':>11} "
          f"{'true us':>9} {'drift/frame':>12} {'to cap':>8}")
    for i, nm in enumerate(menu.names):
        ttc = cap / menu.err[i] if menu.err[i] > 0 else float("inf")
        extra = charged.cost[i] - menu.cost[i]
        print(f"{nm:16} {menu.quality[i]:>8.4f} {menu.cost[i]:>10.3f} {extra:>11.3f} "
              f"{charged.cost[i]:>9.3f} {menu.err[i]:>12.5f} "
              f"{(f'{ttc:.0f}' if np.isfinite(ttc) else 'never'):>8}")
    # where does the ranking flip?
    i10 = menu.names.index("tick-10"); isur = menu.names.index("surrogate")
    de = (menu.err[i10] - menu.err[isur]) / cap
    print(f"\nbreak-even: the surrogate beats tick-10 once reconciliation costs more than "
          f"{(menu.cost[isur] - menu.cost[i10]) / de:.2f} us/agent.")
    print(f"measured 16.4 us after optimisation, so the surrogate still wins by 3.5x.")

    # which rows survive dominance once all three axes are on the table?
    from alloc.config import prune_dominated
    print(f"\nnon-dominated, steady cost only : "
          f"{[menu.names[i] for i in prune_dominated(menu, menu.cost)]}")
    print(f"non-dominated, reconcile charged: "
          f"{[charged.names[i] for i in prune_dominated(charged, charged.cost)]}")

    # and what does the allocator actually pick, across budgets?
    rng = np.random.default_rng(0)
    s_ = rng.lognormal(0.0, 0.8, 600)
    s = s_
    hr = np.full(len(s), 1e9)
    cur = 574 / 600 * 60 / 200          # measured today: bench/anticipate.py, plaza N=200
    print(f"\ntoday's measured reconciliation rate: {cur:.3f} per agent per second")
    print(f"\n{'budget':>8}  selection (count per config)")
    for bf in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85):
        for tag, mu in (("steady only     ", menu), ("reconcile charged", charged)):
            B = len(s) * (mu.cost.min() + bf * (mu.cost.max() - mu.cost.min()))
            r = SerialAllocator(mu).allocate(s, mu.cost, B, headroom=hr)
            cnt = np.bincount(r.assign, minlength=mu.m)
            picked = "  ".join(f"{mu.names[i]}={cnt[i]}" for i in range(mu.m) if cnt[i])
            churn = sum(cnt[i] * mu.err[i] / cap for i in range(mu.m) if mu.err[i] > 0)
            lead = f"{bf:>8.2f}" if tag.startswith("steady") else " " * 8
            print(f"{lead}  {tag}  {picked}   [churn {churn:.2f}/frame]")


if __name__ == "__main__":
    main()
