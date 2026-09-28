"""What the scheduler itself costs: wall time per simulated second, by crowd size.

    python -m bench.llm_overhead

Each policy runs the same station for the same seconds; surrogate_only asks the model nothing and
schedules nothing, so its time is the crowd's own (dynamics, surrogate decisions). The difference is
the scheduler: predicting contexts, charging drift, ranking requests, the ledger. Median of the
per-second times after a 60 s warm-up, single thread, pure Python/numpy (llm/schedule.py).
"""
import argparse
import time

import numpy as np

from llm.policy import build_table
from llm.schedule import Run

CONDS = (("surrogate_only", {}), ("parity", {}), ("parity/reh", dict(rehearse=0.2)), ("cascade", {}), ("view_lod", {}))


def per_second(n, pol, kw, P, lat, seconds, seed=0):
    r = Run(n, pol, P, lat, seed=seed, **kw)
    ts = []
    for t in range(seconds):
        t0 = time.perf_counter()
        r.step(t)
        ts.append(time.perf_counter() - t0)
    return np.array(ts[60:]), r.stats["llm"] + r.stats["sur"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", type=int, default=[1000, 2000, 5000, 10000])
    ap.add_argument("--seconds", type=int, default=600)
    a = ap.parse_args()
    P, lat = build_table(log=None)
    print(f"ms per simulated second (median, p95), {a.seconds} s runs; decisions per second in brackets")
    for n in a.agents:
        base = None
        row = []
        for name, kw in CONDS:
            ts, dec = per_second(n, name.split("/")[0], kw, P, lat, a.seconds)
            med = 1e3 * np.median(ts)
            base = med if base is None else base
            extra = "" if name == "surrogate_only" else f", +{med - base:.2f} over it"
            row.append(f"{name} {med:.2f} / {1e3 * np.percentile(ts, 95):.2f}{extra}")
        print(f"N {n:6d} [{dec / a.seconds:.1f}/s]: " + "; ".join(row), flush=True)


if __name__ == "__main__":
    main()
