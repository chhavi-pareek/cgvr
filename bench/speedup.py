"""Speedup, efficiency and allocator share of the frame budget (Amdahl framing).

python -m bench.speedup --sizes 1000 10000 --frames 8 --threads 1 2 4 8
python -m bench.speedup --sizes 1000 10000 100000     # T4 only; >10k is not for development

Per frame the budget wobbles by a few percent so the warm-start path is the one
timed, as in a running crowd. Writes bench/logs/speedup.csv and prints a table.
"""
import argparse
import csv
import os
import time

import numpy as np

from alloc.config import build_table
from alloc.costmodel import row_costs_from_theta
from alloc.serial import SerialAllocator
from alloc.threaded import ThreadedAllocator

FRAME_MS = 1000.0 / 60.0
LOG = os.path.join(os.path.dirname(__file__), "logs", "speedup.csv")


def instance(seed, n, table):
    rng = np.random.default_rng(seed)
    s = rng.lognormal(0.0, 0.8, n)
    th = np.zeros((4, 4))
    for ax in range(4):
        th[ax, :3] = np.sort(rng.uniform(2, 40, 3))[::-1]
    cost = row_costs_from_theta(table, 0.0, th) * 1e-3  # us -> ms per agent
    return s, cost


def time_alloc(alloc, s, cost, budgets, sync=None):
    alloc.allocate(s, cost, budgets[0])  # warm-up: JIT, warm lambda
    ts = []
    ref = None
    for b in budgets:
        t = time.perf_counter()
        r = alloc.allocate(s, cost, b)
        if sync is not None:
            sync()
        ts.append(time.perf_counter() - t)
        ref = r
    return float(np.median(ts)) * 1e3, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--frac", type=float, default=0.5, help="budget as a fraction of T(0)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = build_table()
    impls = []
    try:
        from alloc import cuda as accuda
        if not accuda.SIMULATED:
            impls.append(("cuda", lambda: accuda.CudaAllocator(table), lambda: accuda.cuda.synchronize()))
        else:
            print("cuda: simulator only on this machine, skipped")
    except Exception as ex:  # noqa: BLE001
        print("cuda unavailable:", ex)
    try:
        from alloc import openacc
        if openacc.available():
            impls.append((f"openacc[{openacc.backend()}]", lambda: openacc.OpenACCAllocator(table), None))
    except Exception as ex:  # noqa: BLE001
        print("openacc unavailable:", ex)

    rows = []
    rng = np.random.default_rng(args.seed + 1)
    for n in args.sizes:
        s, cost = instance(args.seed, n, table)
        t0 = SerialAllocator(table).allocate(s, cost, np.inf).cost
        budgets = args.frac * t0 * (1 + rng.uniform(-0.03, 0.03, args.frames))
        t_serial, ref = time_alloc(SerialAllocator(table), s, cost, budgets)
        rows.append(dict(n=n, impl="serial", threads=1, ms=t_serial, evals=ref.evals, fill=ref.fill_steps))
        print(f"N={n:>7} serial          {t_serial:9.3f} ms  evals={ref.evals} fill={ref.fill_steps}")
        t_one = None
        for k in args.threads:
            t, r = time_alloc(ThreadedAllocator(table, threads=k), s, cost, budgets)
            ok = np.array_equal(r.assign, ref.assign)
            t_one = t if t_one is None else t_one  # efficiency vs the same JIT code on one thread
            rows.append(dict(n=n, impl="threaded", threads=k, ms=t, evals=r.evals, fill=r.fill_steps, match=ok))
            print(f"N={n:>7} threaded x{k:<2}    {t:9.3f} ms  speedup={t_serial / t:6.2f} "
                  f"eff(vs x{args.threads[0]})={t_one / t / (k / args.threads[0]):5.2f} match={ok}")
        for name, make, sync in impls:
            t, r = time_alloc(make(), s, cost, budgets, sync)
            ok = np.array_equal(r.assign, ref.assign)
            rows.append(dict(n=n, impl=name, threads=0, ms=t, evals=r.evals, fill=r.fill_steps, match=ok))
            print(f"N={n:>7} {name:<16}{t:9.3f} ms  speedup={t_serial / t:6.2f} match={ok}")
        # Amdahl: allocator share p of a 60 Hz frame whose other work fills the rest
        best = min(r["ms"] for r in rows if r["n"] == n)
        p = min(t_serial / FRAME_MS, 1.0)
        S = t_serial / best
        print(f"N={n:>7} allocator share of {FRAME_MS:.2f} ms frame: serial {100 * p:5.1f}%  "
              f"best {100 * best / FRAME_MS:5.1f}%  frame speedup bound 1/((1-p)+p/S) = "
              f"{1 / ((1 - p) + p / S):5.2f}")

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "impl", "threads", "ms", "evals", "fill", "match"])
        w.writeheader()
        for r in rows:
            w.writerow({**{"match": ""}, **r})
    print("wrote", LOG)


if __name__ == "__main__":
    main()
