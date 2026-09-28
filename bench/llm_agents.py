"""LLM-driven agents: which decisions get the model, and how far does everyone else drift?

    python -m bench.llm_agents                     # 100 / 300 / 1000 / 2000 agents, 30 min, 2 seeds

The model is qwen2.5:7b through Ollama (llm/policy.py), one call ~0.69 s, so it can serve about
1.3 decisions a second. Everything below spends that same budget; only the choice of WHICH
decisions get the model differs. The divergence of every surrogate decision is measured against
the model's exact answer for that situation (tabulated once, never shown to the scheduler).

  reference       every decision by the model, instantly -- the behaviour being approximated
  parity          ledger + worst-case charge for unseen situations + salience x drift
  view_lod        the LOD way: agents the viewer can see first, then the rest; no ledger
  round_robin     everyone in turn; no ledger
  parity/wait W   the ledger holds an agent at most W seconds, then it decides by the surrogate and
                  the overflow is counted (forced/agent-h)
  cascade         the LLM-cascade rule: the least confident surrogate decisions first; no ledger
  parity_nocap    PARITY's priority with the ledger switched off (the ablation)
  surrogate_only  never ask the model after warm-up

    python -m bench.llm_agents --scenario museum   # an evacuation: every context changes at the alarm
    python -m bench.llm_agents --rehearse 0.2      # adds PARITY buying replies ahead (llm/schedule.py)
    python -m bench.llm_agents --model qwen2.5:14b # another reference model
    python -m bench.llm_agents --order-free        # the reference asked with the options shuffled

With more than two seeds each column is the mean with a 95% t-interval over seeds, and every
policy is also compared with PARITY seed by seed (same station, same arrivals: paired).

    python -m bench.llm_agents --novel             # contexts that never repeat
    python -m bench.llm_agents --menu              # a smaller model (1.5B) as a middle option

With --novel no paid reply is ever reused as an exact bound (as when memory or dialogue makes every
prompt unique), and PARITY's charge for a surrogate decision is the worst case, a plug-in estimate
with no margin, the split-conformal bound, or conformal risk control (llm/schedule.py DriftBound).
Coverage is the share of surrogate decisions whose true drift was at most what was charged;
overflow is the share of stretches (from one model decision, or a new person, to the next) whose
TRUE drift passed the cap -- the quantity risk control bounds by delta.
"""
import argparse
import csv

import numpy as np
from scipy import stats

from llm import museum, station
from llm.policy import build_rotations, build_table
from llm.schedule import Run

# crowd outcomes printed per scenario (Station.outcomes / Museum.outcomes)
OUTCOMES = {"station": ("boarded", "missed"), "museum": ("t50", "inside", "w1")}


def evac_w1(run, ref):
    """W1 (seconds) between a run's evacuation-time distribution and the reference's with the same
    seed: the area between the two evacuation curves on the 30 s grid, over 600 s after the alarm."""
    return 30.0 * float(np.abs(run["curve"] - ref["curve"]).sum())

POLICIES = ("parity", "parity_nocap", "cascade", "view_lod", "round_robin", "surrogate_only")


def ci(x):
    """Mean and half-width of the 95% t-interval over seeds."""
    x = np.asarray(x, float)
    if x.size < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def paired(base, other, key):
    """PARITY minus another policy, seed by seed: mean difference, its 95% interval, the Wilcoxon
    signed-rank p-value, and how many seeds PARITY was lower on."""
    d = np.array([b[key] - o[key] for b, o in zip(base, other)])
    m, h = ci(d)
    p = stats.wilcoxon(d).pvalue if d.size >= 5 and np.any(d != 0) else float("nan")
    return m, h, p, int((d < 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", type=int, default=[100, 300, 1000, 2000])
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--novel", action="store_true")
    ap.add_argument("--menu", action="store_true", help="add the small model (qwen2.5:1.5b) as a middle option")
    ap.add_argument("--scenario", choices=("station", "museum"), default="station")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--rehearse", type=float, default=0.0, help="also run PARITY with this rehearsal share")
    ap.add_argument("--max-wait", nargs="+", type=float, default=[],
                    help="also run PARITY with each of these bounded waits (seconds; with --rehearse, rehearsing too)")
    ap.add_argument("--policies", nargs="+", default=None, help="only these conditions (names as printed)")
    ap.add_argument("--surrogate", choices=("distilled", "marginal"), default="distilled")
    ap.add_argument("--csv", default=None, help="also write one row per run (policy, N, seed, every summary field)")
    ap.add_argument("--order-free", action="store_true",
                    help="reference = the model with the options in a random order (mean over six rotations)")
    a = ap.parse_args()
    scn = museum if a.scenario == "museum" else station
    conds = [(p, {}) for p in POLICIES]
    if a.rehearse > 0:
        conds.insert(1, (f"parity/reh {a.rehearse:g}", dict(rehearse=a.rehearse)))
        conds.insert(4, (f"cascade/reh {a.rehearse:g}", dict(rehearse=a.rehearse)))
    for w in a.max_wait:
        kw = dict(max_wait=w, rehearse=a.rehearse) if a.rehearse > 0 else dict(max_wait=w)
        conds.insert(len(conds) - len(POLICIES) + 1, (f"parity/wait {w:g}" + (" reh" if a.rehearse > 0 else ""), kw))
    if a.policies:
        conds = [c for c in conds if c[0] in a.policies]
    if a.novel:
        conds = [("parity/worst", dict(novel=True, bound="worst")),
                 ("parity/plugin", dict(novel=True, bound="plugin")),
                 ("parity/cp 0.10", dict(novel=True, bound="conformal", alpha=0.10)),
                 ("parity/cp 0.05", dict(novel=True, bound="conformal", alpha=0.05)),
                 ("parity/crc 0.05", dict(novel=True, bound="crc", delta=0.05, eta=0.25)),
                 ("parity/crc 0.20", dict(novel=True, bound="crc", delta=0.20, eta=0.50)),
                 ("parity/crc mean", dict(novel=True, bound="crc", delta=1.0, eta=0.25)),
                 ("view_lod", {}), ("round_robin", {})]
    # the call latencies are the station table's for the same model (same length of prompt), so
    # every scenario is served at one measured speed
    P = build_table(a.model, scn=None if scn is station else scn)[0]
    lat = build_table(a.model)[1]
    extra = {}
    if a.order_free:
        R = build_rotations(a.model, scn=None if scn is station else scn)[0]
        P, extra = R.mean(0), dict(rotations=R)
    for i, (name, kw) in enumerate(conds):
        conds[i] = (name, dict(kw, scenario=scn, surrogate=a.surrogate, **extra))
    outk = OUTCOMES[a.scenario]
    rows_out, series = [], {}
    if a.menu:
        assert scn is station, "the small model is tabulated for the station only"
        small = build_table("qwen2.5:1.5b")
        conds = [("parity", {}), ("parity/7B+1.5B", dict(small=small)), ("view_lod", {}),
                 ("model_lod", dict(small=small)), ("round_robin", {})]
    print(f"model latency median {1e3 * np.median(lat):.0f} ms -> budget {0.9 / lat.mean():.2f} calls/s; "
          f"cap {a.cap} nats; {a.seconds // 60} simulated minutes; seeds {a.seeds}")
    for n in a.agents:
        ref = [Run(n, "reference", P, lat, seed=s, scenario=scn).run(a.seconds) for s in a.seeds]
        dec_rate = np.mean([r.stats["llm"] for r in ref]) / a.seconds
        rsum = [r.summary() for r in ref]
        if a.csv:
            for sd, r in zip(a.seeds, ref):
                if hasattr(r.st, "out_t"):
                    series[f"reference|{n}|{sd}|out_t"] = np.array(r.st.out_t, float)
                    series[f"reference|{n}|{sd}|at_alarm"] = np.array([r.st.at_alarm or 0])
        if "curve" in rsum[0]:
            for r in rsum:
                r["w1"] = 0.0
        ro = {k: np.nanmean([r[k] for r in rsum]) for k in outk}
        print(f"\nN = {n}: {dec_rate:.1f} decisions/s needed for every decision to be the model's "
              f"({100 * min(1.0, 0.9 / lat.mean() / dec_rate):.0f}% servable);  reference "
              + ", ".join(f"{k} {v:.0f}" for k, v in ro.items()))
        many = len(a.seeds) > 2
        print("  policy           worst/cap  over-cap   drift/agent-h  steady  in view   model share  calls/s  waits/agent-h  forced  "
              + "  ".join(f"{k:>7s}" for k in outk) + "  coverage  overflow")
        runs = {}
        for name, kw in conds:
            pol = "view_lod" if name == "model_lod" else name.split("/")[0]
            rr = [Run(n, pol, P, lat, seed=s, cap=a.cap, **kw).run(a.seconds) for s in a.seeds]
            rs = [r.summary() for r in rr]
            if a.csv:
                for sd, r in zip(a.seeds, rr):
                    series[f"{name}|{n}|{sd}|kl"] = np.array(r.kl_t)
                    series[f"{name}|{n}|{sd}|agent_s"] = np.array(r.agent_s_t)
                    if hasattr(r.st, "out_t"):
                        series[f"{name}|{n}|{sd}|out_t"] = np.array(r.st.out_t, float)
                        series[f"{name}|{n}|{sd}|at_alarm"] = np.array([r.st.at_alarm or 0])
            if "curve" in rs[0]:
                for r, r0 in zip(rs, rsum):
                    r["w1"] = evac_w1(r, r0)
            if a.csv:
                rows_out += [dict(scenario=a.scenario, model=a.model, order_free=a.order_free, cap=a.cap, condition=name,
                                  n=n, seed=sd, **{k: v for k, v in r.items() if k not in ("curve", "policy", "n")})
                             for sd, r in zip(a.seeds, rs)]
            runs[name] = rs
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            print(f"  {name:15s} {max(r['dmax'] for r in rs) / a.cap:8.2f}x {100 * m('over_cap_frac'):8.2f}% "
                  f"{m('kl_per_agent_h'):13.2f} {m('kl_steady_per_agent_h'):7.2f} {m('kl_view_per_agent_h'):9.2f} {m('llm_frac'):12.2f} {m('calls_per_s'):8.2f} "
                  f"{m('stall_per_agent_h'):13.1f}  {m('forced_per_agent_h'):6.1f}  " + "  ".join(f"{np.nanmean([r[k] for r in rs]):7.0f}" for k in outk)
                  + f"  {100 * m('coverage'):6.1f}%  {100 * m('overflow_frac'):6.2f}%")
        if not many:
            continue
        print(f"  over {len(a.seeds)} seeds, mean +- 95% CI:  worst/cap | drift/agent-h | steady | in view | overflow % | "
              + " | ".join(outk))
        for name, rs in runs.items():
            cols = [ci([r["dmax"] / a.cap for r in rs]), ci([r["kl_per_agent_h"] for r in rs]),
                    ci([r["kl_steady_per_agent_h"] for r in rs]), ci([r["kl_view_per_agent_h"] for r in rs]),
                    ci([100 * r["overflow_frac"] for r in rs])] + [ci([r[k] for r in rs]) for k in outk]
            print(f"  {name:15s} " + " | ".join(f"{mu:6.2f} +- {h:4.2f}" for mu, h in cols))
        base = runs[conds[0][0]]
        print(f"  {conds[0][0]} minus each, paired (mean +- 95% CI, Wilcoxon p, seeds {conds[0][0]} lower):")
        for name, rs in list(runs.items())[1:]:
            parts = []
            keys = (("kl_per_agent_h", "drift"), ("kl_steady_per_agent_h", "steady"), ("kl_view_per_agent_h", "in view"), ("dmax", "worst"))
            if "w1" in rs[0]:
                keys += (("w1", "evac W1"),)
            for key, lab in keys:
                mu, h, p, w = paired(base, rs, key)
                parts.append(f"{lab} {mu:+.2f} +- {h:.2f} (p {p:.3f}, {w}/{len(rs)})")
            print(f"    vs {name:13s} " + "; ".join(parts))
    if a.csv and rows_out:
        keys = sorted({k for r in rows_out for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows_out)
        np.savez_compressed(a.csv.rsplit(".", 1)[0] + "_series.npz", **series)
        print(f"{len(rows_out)} runs written to {a.csv} (time series beside it)")

if __name__ == "__main__":
    main()
