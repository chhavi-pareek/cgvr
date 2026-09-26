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
    a = ap.parse_args()
    P, lat = build_table()
    print(f"model latency median {1e3 * np.median(lat):.0f} ms -> budget {0.9 / lat.mean():.2f} calls/s; "
          f"cap {a.cap} nats; {a.seconds // 60} simulated minutes; seeds {a.seeds}")
    for n in a.agents:
        ref = [Run(n, "reference", P, lat, seed=s).run(a.seconds) for s in a.seeds]
        dec_rate = np.mean([r.stats["llm"] for r in ref]) / a.seconds
        rb = np.mean([r.st.boarded for r in ref]); rm = np.mean([r.st.missed for r in ref])
        print(f"\nN = {n}: {dec_rate:.1f} decisions/s needed for every decision to be the model's "
              f"({100 * min(1.0, 0.9 / lat.mean() / dec_rate):.0f}% servable);  reference boarded {rb:.0f}, missed {rm:.0f}")
        print("  policy           worst/cap  over-cap   drift/agent-h  in view   model share  calls/s  waits/agent-h  boarded  missed")
        for pol in POLICIES:
            rs = [Run(n, pol, P, lat, seed=s, cap=a.cap).run(a.seconds).summary() for s in a.seeds]
            m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
            print(f"  {pol:15s} {max(r['dmax'] for r in rs) / a.cap:8.2f}x {100 * m('over_cap_frac'):8.2f}% "
                  f"{m('kl_per_agent_h'):13.2f} {m('kl_view_per_agent_h'):9.2f} {m('llm_frac'):12.2f} {m('calls_per_s'):8.2f} "
                  f"{m('stall_per_agent_h'):13.1f} {m('boarded'):8.0f} {m('missed'):7.0f}")


if __name__ == "__main__":
    main()
