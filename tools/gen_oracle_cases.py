"""Export allocator cases for the C# port, then diff the port's answers against alloc/serial.py.

The Unity runtime is fp32 and the Python oracle is fp64, so bit identity is neither expected
nor claimed (unity/PORT_SPEC.md section 1 says as much). What must hold is: the same feasible
set, the same infeasibility verdict, no budget overrun, no error-mask violation, and a utility
gap at the level of float32 rounding rather than a different allocation policy.

    .venv/bin/python -m tools.gen_oracle_cases            # write cases, then compare
"""
import json
import os
import subprocess
import sys

import numpy as np

from alloc.config import build_table
from alloc.costmodel import row_costs_from_theta
from alloc.factored import FactoredAllocator, split_quality
from alloc.serial import SerialAllocator
from sim.tiered import phase7_table

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.join(HERE, "unity", "OracleCheck")
CASES = os.path.join(PROJ, "cases.json")
RESULTS = os.path.join(PROJ, "results.json")
FILL_MAX = 100000  # match Python's unbounded fill so the two run the same algorithm

# The C# port carries the phase-7 error column (only the surrogate diverges), not
# alloc/config.py's raw truncation NMSE, so the oracle has to be the same table or the
# feasibility masks differ and every comparison is meaningless.
E_MAX = 0.05716  # ParityTable.PlazaEMax; bench/logs/phase7_plaza_calib.npz
TABLE = phase7_table(E_MAX)


def instance(seed, n, mask):
    rng = np.random.default_rng(seed)
    s = rng.lognormal(0.0, 0.8, n)
    th = np.zeros((4, 4))
    for ax in range(4):
        th[ax, :3] = np.sort(rng.uniform(2, 40, 3))[::-1]
    cost = row_costs_from_theta(TABLE, 0.0, th) * 1e-3  # us -> ms
    hr = rng.uniform(0.15, 1.5, n) if mask else None
    return s, cost, hr


def build_cases():
    cases, meta = [], []
    for seed in range(4):
        for n in (64, 200, 700):
            for mask in (False, True):
                s, cost, hr = instance(seed, n, mask)
                t0 = SerialAllocator(TABLE).allocate(s, cost, np.inf, headroom=hr).cost
                for frac in (0.12, 0.3, 0.55, 0.8, 1.05):
                    b = float(frac * t0)
                    cases.append(dict(n=n, budget=b, salience=[float(v) for v in s],
                                      rowCost=[float(v) for v in cost],
                                      headroom=None if hr is None else [float(v) for v in hr],
                                      fillMax=FILL_MAX, warm=False))
                    meta.append((seed, n, mask, frac, s, cost, hr, b))
    return cases, meta


def main():
    cases, meta = build_cases()
    os.makedirs(PROJ, exist_ok=True)
    with open(CASES, "w") as f:
        json.dump(cases, f)
    print(f"wrote {len(cases)} cases")

    env = dict(os.environ, PATH="/opt/homebrew/bin:" + os.environ.get("PATH", ""))
    r = subprocess.run(["dotnet", "run", "--project", PROJ, "-c", "Release", "--", CASES, RESULTS],
                       capture_output=True, text=True, env=env, cwd=HERE)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); sys.exit(1)
    print(r.stdout.strip().splitlines()[-2:])

    got = json.load(open(RESULTS))
    assert len(got) == len(meta)

    n_assign_diff = n_cases = 0
    worst_gap = 0.0
    worst_desc = ""
    over_budget = mask_viol = verdict_diff = 0
    for (seed, n, mask, frac, s, cost, hr, b), g in zip(meta, got):
        py = SerialAllocator(TABLE).allocate(s, cost, b, headroom=hr)
        cs = np.asarray(g["assign"], np.int64)
        n_cases += 1
        if bool(py.infeasible) != bool(g["infeasible"]):
            verdict_diff += 1
        if not g["infeasible"] and float(cost[cs].sum()) > b * (1 + 1e-5):
            over_budget += 1
        if hr is not None and np.any(TABLE.err[cs] > np.asarray(hr) + 1e-6):
            mask_viol += 1
        d = int((cs != py.assign).sum())
        n_assign_diff += d
        u_cs = float((s * TABLE.quality[cs]).sum())
        gap = (py.utility - u_cs) / abs(py.utility) if py.utility else 0.0
        if gap > worst_gap:
            worst_gap, worst_desc = gap, f"seed{seed} n={n} mask={int(mask)} frac={frac}"

    tot = sum(m[1] for m in meta)
    print(f"\ncases {n_cases}   agent-decisions {tot}")
    print(f"  infeasibility verdict disagreements : {verdict_diff}")
    print(f"  budget overruns (C#)                : {over_budget}")
    print(f"  error-mask violations (C#)          : {mask_viol}")
    print(f"  differing agent assignments         : {n_assign_diff}  ({100*n_assign_diff/tot:.4f}%)")
    print(f"  worst utility gap vs fp64 oracle    : {worst_gap*100:.5f}%  ({worst_desc})")
    ok = verdict_diff == 0 and over_budget == 0 and mask_viol == 0 and worst_gap < 1e-3
    print("  ->", "cold path agrees with alloc/serial.py" if ok else "COLD PATH DISAGREES")

    ok &= warm_path(env)
    ok &= capped_fill(env)
    ok &= factored(env)
    print("\nC# port agrees with alloc/serial.py" if ok else "\nC# PORT DISAGREES -- investigate")
    sys.exit(0 if ok else 1)


def _run(cases, env, tag):
    with open(CASES, "w") as f:
        json.dump(cases, f)
    r = subprocess.run(["dotnet", "run", "--project", PROJ, "-c", "Release", "--", CASES, RESULTS],
                       capture_output=True, text=True, env=env, cwd=HERE)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)
    return json.load(open(RESULTS))


def warm_path(env):
    """Frame-to-frame reuse of the multiplier: the path that actually runs every frame. The
    budget wobbles a few percent as a measured frame time would, so the bracket has to expand
    and contract rather than start cold."""
    s, cost, hr = instance(9, 400, True)
    t0 = SerialAllocator(TABLE).allocate(s, cost, np.inf, headroom=hr).cost
    rng = np.random.default_rng(3)
    budgets = [float(0.45 * t0 * (1 + v)) for v in rng.uniform(-0.03, 0.03, 40)]
    cases = [dict(n=400, budget=b, salience=[float(v) for v in s],
                  rowCost=[float(v) for v in cost], headroom=[float(v) for v in hr],
                  fillMax=FILL_MAX, warm=True) for b in budgets]
    got = _run(cases, env, "warm")

    py = SerialAllocator(TABLE)   # also reused, so its lambda warms the same way
    diff = over = 0
    evals = []
    for b, g in zip(budgets, got):
        r = py.allocate(s, cost, b, headroom=hr)
        cs = np.asarray(g["assign"], np.int64)
        diff += int((cs != r.assign).sum())
        if float(cost[cs].sum()) > b * (1 + 1e-5):
            over += 1
        evals.append(g["evals"])
    print(f"\nwarm-start sequence: 40 frames, N=400, budget wobbling +-3%")
    print(f"  differing agent assignments         : {diff}  of {40*400}")
    print(f"  budget overruns                     : {over}")
    print(f"  evals per frame  first {evals[0]}, then median {int(np.median(evals[1:]))}, max {max(evals[1:])}")
    ok = diff == 0 and over == 0
    print("  ->", "warm path agrees" if ok else "WARM PATH DISAGREES")
    return ok


def factored(env):
    """The two-salience allocator (alloc/factored.py) on the full table: state salience for
    behaviour and navigation, projected-size salience for animation and geometry, 30% of the
    crowd out of view. Same acceptance as the cold path."""
    q_s, q_v = split_quality(TABLE)
    cases, meta = [], []
    for seed in range(4):
        for n in (64, 200, 700, 3000):
            for mask in (False, True):
                s, cost, hr = instance(seed, n, mask)
                rng = np.random.default_rng(100 + seed)
                v = np.where(rng.random(n) < 0.3, 0.0, np.minimum(1.0, 10.0 / rng.uniform(2, 120, n)))
                top = FactoredAllocator(TABLE).allocate(s, cost, np.inf, headroom=hr, view_salience=v).cost
                for frac in (0.12, 0.3, 0.55, 0.8, 1.05):
                    b = float(frac * top)
                    cases.append(dict(n=n, budget=b, salience=[float(x) for x in s], view=[float(x) for x in v],
                                      rowCost=[float(x) for x in cost],
                                      headroom=None if hr is None else [float(x) for x in hr],
                                      fillMax=0, warm=False))
                    meta.append((n, s, v, cost, hr, b))
    got = _run(cases, env, "factored")
    verdict = over = mask_v = diff = 0
    tot = 0
    worst = 0.0
    ms = []
    for (n, s, v, cost, hr, b), g in zip(meta, got):
        py = FactoredAllocator(TABLE).allocate(s, cost, b, headroom=hr, view_salience=v)
        cs = np.asarray(g["assign"], np.int64)
        tot += n
        verdict += int(bool(py.infeasible) != bool(g["infeasible"]))
        if not g["infeasible"] and float(cost[cs].sum()) > b * (1 + 1e-5):
            over += 1
        if hr is not None and np.any(TABLE.err[cs] > np.asarray(hr) + 1e-6):
            mask_v += 1
        diff += int((cs != py.assign).sum())
        u = float((s * q_s[cs] + v * q_v[cs]).sum())
        if py.utility:
            worst = max(worst, (py.utility - u) / abs(py.utility))
        if n == 3000:
            ms.append(g["ms"])
    print(f"\nfactored (two saliences, full table): {len(meta)} cases, {tot} agent-decisions")
    print(f"  infeasibility verdict disagreements : {verdict}")
    print(f"  budget overruns (C#)                : {over}")
    print(f"  error-mask violations (C#)          : {mask_v}")
    print(f"  differing agent assignments         : {diff}  ({100*diff/tot:.4f}%)")
    print(f"  worst utility gap vs fp64 oracle    : {worst*100:.5f}%")
    print(f"  C# solve at N=3000                  : median {np.median(ms):.3f} ms")
    ok = verdict == 0 and over == 0 and mask_v == 0 and worst < 1e-3
    print("  ->", "factored port agrees with alloc/factored.py" if ok else "FACTORED PORT DISAGREES")
    return ok


def capped_fill(env):
    """FillMax = 8 is unity/PORT_SPEC.md's engine decision. It is a deliberate departure from
    the unbounded Python fill, so the question is not identity but how much utility it costs
    and whether the result is still feasible."""
    cases, meta = build_cases()
    for c in cases:
        c["fillMax"] = 8
    got = _run(cases, env, "fill8")
    over = mask = 0
    gaps = []
    for (seed, n, m, frac, s, cost, hr, b), g in zip(meta, got):
        py = SerialAllocator(TABLE).allocate(s, cost, b, headroom=hr)
        cs = np.asarray(g["assign"], np.int64)
        if not g["infeasible"] and float(cost[cs].sum()) > b * (1 + 1e-5):
            over += 1
        if hr is not None and np.any(TABLE.err[cs] > np.asarray(hr) + 1e-6):
            mask += 1
        u = float((s * TABLE.quality[cs]).sum())
        if py.utility:
            gaps.append((py.utility - u) / abs(py.utility))
    gaps = np.array(gaps)
    print(f"\nFILL_MAX = 8 (the engine setting) vs the unbounded oracle, {len(gaps)} cases")
    print(f"  budget overruns                     : {over}")
    print(f"  error-mask violations               : {mask}")
    print(f"  utility loss   mean {gaps.mean()*100:.4f}%   max {gaps.max()*100:.4f}%")
    ok = over == 0 and mask == 0 and gaps.max() < 0.02
    print("  ->", "capped fill is feasible and near-oracle" if ok else "CAPPED FILL PROBLEM")
    return ok


if __name__ == "__main__":
    main()
