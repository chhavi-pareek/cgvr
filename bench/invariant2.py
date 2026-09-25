"""Why the invariant-2 ablation produces no signal, and what would give it one.

    python -m bench.invariant2

`parity_authored` swaps the learned per-axis quality (behaviour and animation as 1 - held-out
NMSE from the phase 2 nested truncation, so the two share a unit through the latent) for
authored ramps and hand weights. The sweep records it as byte-identical to `parity`. That is
not evidence against invariant 2; it is evidence that phase 7 cannot test it.

Two facts do it:

  1. Only the behaviour axis executes in phase 7, so it is the only axis with cost.
  2. Its measured costs are 10.930 / 10.943 / 10.527 / 4.854 us per agent -- tier 1 costs MORE
     than tier 0, and all three live tiers sit within 4% of one another against a surrogate at
     less than half. Tiers 1 and 2 are therefore dominated: same cost as tier 0, less quality.

So the allocator's real choice is binary, tier 0 or surrogate. For a binary choice on one axis
the quality gain (q0 - q3) is a constant across agents, so it cancels out of the ranking and
agents sort by salience alone. ANY monotone quality model then yields the identical
allocation, whatever its values. The ablation is structurally incapable of separating.

It separates as soon as more than one axis carries cost, because then the *relative* weighting
between axes -- which is exactly what invariant 2 claims to derive rather than author -- starts
to matter. The Unity build calibrates all four axes and is where this ablation belongs.
"""
import numpy as np

from alloc.costmodel import row_costs_from_theta
from alloc.serial import SerialAllocator
from bench.sweep import _authored_quality
from sim.tiered import phase7_table

TABLE = phase7_table(0.06934)

# measured, sim/tiered.py::measure_costs on plaza
PHASE7_BEH = np.array([10.930, 10.943, 10.527, 4.854])
# measured in-engine, CrowdWorld.CalibrateAxes (us/agent/frame), all four axes
UNITY = np.array([[0.472, 0.440, 0.425, 0.0], [0.001, 0.001, 0.004, 0.0],
                  [0.428, 0.324, 0.155, 0.0], [0.013, 0.017, 0.016, 0.0]])


class _View:
    """The table with a swapped quality column; nothing else differs."""

    def __init__(self, q):
        self.tiers, self.quality, self.err, self.m = TABLE.tiers, q, TABLE.err, TABLE.m


def compare(theta, core, label, n=200, seed=0):
    cost = row_costs_from_theta(TABLE, core, theta)
    rng = np.random.default_rng(seed)
    s = rng.lognormal(0.0, 0.8, n)
    hr = np.full(n, 1e9)
    learned, authored = TABLE.quality.copy(), _authored_quality(TABLE)
    diff = 0
    used = set()
    for bf in np.linspace(0.1, 0.9, 9):
        B = n * (float(cost.min()) + bf * float(cost.max() - cost.min()))
        ra = SerialAllocator(_View(learned)).allocate(s, cost, B, headroom=hr)
        rb = SerialAllocator(_View(authored)).allocate(s, cost, B, headroom=hr)
        diff += int((ra.assign != rb.assign).sum())
        used |= set(np.unique(TABLE.tiers[ra.assign, 0]).tolist())
    print(f"  {label:50} {diff:>6} {str(sorted(used)):>14}")
    return diff


def main():
    print("Does swapping learned quality for authored ramps change the allocation?")
    print(f"  {'cost structure':50} {'differ':>6} {'beh tiers':>14}")
    th7 = np.zeros((4, 4))
    th7[0, :3] = (PHASE7_BEH[:3] - PHASE7_BEH[3]) * 1e-3
    a = compare(th7, PHASE7_BEH[3] * 1e-3, "phase 7 as measured: 1 costed axis, cost-flat tiers")

    b = compare(UNITY * 1e-3, 0.191e-3, "Unity calibration: 4 costed axes")

    thd = np.zeros((4, 4))
    thd[0, :3] = np.array([16.0, 8.0, 4.0]) * 1e-3     # decode cost genuinely scales with k
    thd[2, :3] = np.array([12.0, 8.0, 3.0]) * 1e-3
    c = compare(thd, 2.0e-3, "decode-dominated: behaviour and animation both separate")

    print(f"\nphase 7 gives {a} differing assignments, so the recorded null is a property of the")
    print(f"experiment, not of invariant 2. With four costed axes it separates ({b}), and when the")
    print(f"decode genuinely dominates per-agent cost the intermediate latent tiers are used at")
    print(f"all ({c} differing). The ablation belongs in the Unity build.")


if __name__ == "__main__":
    main()
