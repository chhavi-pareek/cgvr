"""One setting at a time, around the defaults: does the result depend on a tuned constant?

    python -m bench.llm_sensitivity --out bench/logs/llm_sensitivity.csv

Station (N 1000, 30 min): the cap, how often the surrogate is refitted, the salience of agents the
viewer cannot see, and the call budget (the share of the model's capacity the crowd may use), for
PARITY, PARITY without its ledger and the cascade. Museum (N 1000, 15 min): the rehearsal share and
the wait bound, for PARITY. Five seeds each, order-free reference unless --fixed-order. One CSV row per
run; a short summary per setting is printed.
"""
import argparse
import csv

import numpy as np

from llm import museum, station
from llm.policy import build_rotations, build_table
from llm.schedule import Run
from bench.llm_agents import evac_w1

STATION = [("cap", "cap", [3.0, 5.0, 7.5, 10.0, 20.0, 40.0]),
           ("refit_every", "refit_every", [10, 20, 40, 80, 160]),
           ("far_salience", "far_salience", [0.1, 0.25, 0.5, 1.0]),
           ("util", "util", [0.3, 0.6, 0.9, 1.2])]
MUSEUM = [("rehearse", "rehearse", [0.0, 0.1, 0.2, 0.3, 0.5]),
          ("max_wait", "max_wait", [0, 2, 5, 15, 60, None])]
KEYS = ("dmax", "kl_per_agent_h", "kl_steady_per_agent_h", "kl_view_per_agent_h", "overflow_frac",
        "stall_per_agent_h", "forced_per_agent_h", "calls_per_s", "llm_frac")


def tables(scn, order_free):
    P = build_table(scn=None if scn is station else scn, log=None)[0]
    lat = build_table(log=None)[1]
    if not order_free:
        return P, lat, {}
    R = build_rotations(scn=None if scn is station else scn, log=None)[0]
    return R.mean(0), lat, dict(rotations=R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/logs/llm_sensitivity.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--fixed-order", action="store_true")
    a = ap.parse_args()
    rows = []

    def record(scen, param, value, cond, rs, cap):
        for sd, r in zip(a.seeds, rs):
            rows.append(dict(scenario=scen, param=param, value="none" if value is None else value, condition=cond, seed=sd, n=a.n,
                             worst_over_cap=r["dmax"] / cap, **{k: r[k] for k in KEYS if k in r},
                             **({"w1": r["w1"]} if "w1" in r else {})))
        m = lambda k: np.nanmean([r[k] for r in rs])  # noqa: E731
        w1 = f", evac W1 {m('w1'):.0f} s" if "w1" in rs[0] else ""
        print(f"  {scen:7s} {param:12s} {str(value):6s} {cond:13s} worst {max(r['dmax'] for r in rs) / cap:5.2f}x  "
              f"drift {m('kl_per_agent_h'):6.2f}  steady {m('kl_steady_per_agent_h'):6.2f}  overflow {100 * m('overflow_frac'):5.2f}%  "
              f"waits {m('stall_per_agent_h'):7.1f}  forced {m('forced_per_agent_h'):5.1f}{w1}", flush=True)

    P, lat, extra = tables(station, not a.fixed_order)
    for param, kw, values in STATION:
        for v in values:
            for cond in ("parity", "parity_nocap", "cascade"):
                cap = v if param == "cap" else 10.0
                opts = {kw: v} if param != "cap" else {}
                rs = [Run(a.n, cond, P, lat, seed=s, cap=cap, **opts, **extra).run(1800).summary() for s in a.seeds]
                record("station", param, v, cond, rs, cap)

    P, lat, extra = tables(museum, not a.fixed_order)
    ref = [Run(a.n, "reference", P, lat, seed=s, scenario=museum).run(900).summary() for s in a.seeds]
    base = dict(rehearse=0.2, max_wait=5)
    for param, kw, values in MUSEUM:
        for v in values:
            opts = dict(base, **{kw: v})
            rs = [Run(a.n, "parity", P, lat, seed=s, scenario=museum, **opts, **extra).run(900).summary() for s in a.seeds]
            for r, r0 in zip(rs, ref):
                r["w1"] = evac_w1(r, r0)
            record("museum", param, v, "parity", rs, 10.0)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs written to {a.out}")


if __name__ == "__main__":
    main()
