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
  surrogate_only  never ask the model after warm-up

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

import numpy as np

from llm.policy import build_table
from llm.schedule import Run

POLICIES = ("parity", "view_lod", "round_robin", "surrogate_only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", type=int, default=[100, 300, 1000, 2000])
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--novel", action="store_true")
    ap.add_argument("--menu", action="store_true", help="add the small model (qwen2.5:1.5b) as a middle option")
    a = ap.parse_args()
    conds = [(p, {}) for p in POLICIES]
    if a.novel:
        conds = [("parity/worst", dict(novel=True, bound="worst")),
                 ("parity/plugin", dict(novel=True, bound="plugin")),
                 ("parity/cp 0.10", dict(novel=True, bound="conformal", alpha=0.10)),
                 ("parity/cp 0.05", dict(novel=True, bound="conformal", alpha=0.05)),
                 ("parity/crc 0.05", dict(novel=True, bound="crc", delta=0.05, eta=0.25)),
                 ("parity/crc 0.20", dict(novel=True, bound="crc", delta=0.20, eta=0.50)),
                 ("parity/crc mean", dict(novel=True, bound="crc", delta=1.0, eta=0.25)),
                 ("view_lod", {}), ("round_robin", {})]
    P, lat = build_table()
    if a.menu:
        small = build_table("qwen2.5:1.5b")
        conds = [("parity", {}), ("parity/7B+1.5B", dict(small=small)), ("view_lod", {}),
                 ("model_lod", dict(small=small)), ("round_robin", {})]
    print(f"model latency median {1e3 * np.median(lat):.0f} ms -> budget {0.9 / lat.mean():.2f} calls/s; "
          f"cap {a.cap} nats; {a.seconds // 60} simulated minutes; seeds {a.seeds}")
    for n in a.agents:
        ref = [Run(n, "reference", P, lat, seed=s).run(a.seconds) for s in a.seeds]
        dec_rate = np.mean([r.stats["llm"] for r in ref]) / a.seconds
        rb = np.mean([r.st.boarded for r in ref]); rm = np.mean([r.st.missed for r in ref])
        print(f"\nN = {n}: {dec_rate:.1f} decisions/s needed for every decision to be the model's "
              f"({100 * min(1.0, 0.9 / lat.mean() / dec_rate):.0f}% servable);  reference boarded {rb:.0f}, missed {rm:.0f}")
        print("  policy           worst/cap  over-cap   drift/agent-h  steady  in view   model share  calls/s  waits/agent-h  boarded  missed  coverage  overflow")
        for name, kw in conds:
            pol = "view_lod" if name == "model_lod" else name.split("/")[0]
            rs = [Run(n, pol, P, lat, seed=s, cap=a.cap, **kw).run(a.seconds).summary() for s in a.seeds]
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            print(f"  {name:15s} {max(r['dmax'] for r in rs) / a.cap:8.2f}x {100 * m('over_cap_frac'):8.2f}% "
                  f"{m('kl_per_agent_h'):13.2f} {m('kl_steady_per_agent_h'):7.2f} {m('kl_view_per_agent_h'):9.2f} {m('llm_frac'):12.2f} {m('calls_per_s'):8.2f} "
                  f"{m('stall_per_agent_h'):13.1f} {m('boarded'):8.0f} {m('missed'):7.0f}  {100 * m('coverage'):6.1f}%  {100 * m('overflow_frac'):6.2f}%")


if __name__ == "__main__":
    main()
