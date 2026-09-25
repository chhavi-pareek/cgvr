"""Figure 5: the in-engine result, drawn from unity/bench/engine_bench.csv.

    python -m figures.engine_plot

Written by Unity's BenchmarkRunner with rendering off, both policies from one seed on one
camera path, priced by the same calibrated cost model. This is the engine column the Python
sweep cannot produce: the allocator running inside a real frame loop rather than a harness.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(os.path.dirname(HERE), "unity", "bench", "engine_bench.csv")
OUT = os.path.join(HERE, "out")
COL = {"Parity": "#0072B2", "Baseline": "#D55E00"}
NAME = {"Parity": "PARITY", "Baseline": "MassLOD baseline"}
CAP = 3.97  # 300 * e_sur, bench/logs/phase7_plaza_calib.npz
TIER_COL = ["#0072B2", "#009E73", "#E69F00", "#BBBBBB"]


def load():
    rows = list(csv.DictReader(open(CSV)))
    ns = sorted({int(r["n"]) for r in rows})
    d = {(int(r["n"]), r["policy"]): r for r in rows}
    return ns, d


def main():
    ns, d = load()
    x = np.array(ns, float)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    fig.subplots_adjust(hspace=0.34, wspace=0.26, left=0.08, right=0.98, top=0.93, bottom=0.09)

    # (a) the result: divergence against crowd size
    ax = axes[0, 0]
    for pol in ("Baseline", "Parity"):
        y = [float(d[(n, pol)]["kl_max_end"]) for n in ns]
        ax.plot(x, y, "o-", color=COL[pol], lw=1.6, ms=4, label=NAME[pol])
    ax.axhline(CAP, ls="--", lw=0.9, color="#009E73")
    ax.text(x[0], CAP * 1.06, "cap", color="#009E73", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("agents"); ax.set_ylabel("worst-agent divergence (nats)")
    ax.set_title("(a) the baseline's divergence grows with the crowd; PARITY's does not",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # (b) what it costs
    ax = axes[0, 1]
    base = np.array([float(d[(n, "Baseline")]["step_ms_mean"]) for n in ns])
    step = np.array([float(d[(n, "Parity")]["step_ms_mean"]) for n in ns])
    alloc = np.array([float(d[(n, "Parity")]["alloc_ms_mean"]) for n in ns])
    ax.plot(x, base, "o-", color=COL["Baseline"], lw=1.6, ms=4, label="baseline crowd")
    ax.plot(x, step - alloc, "o-", color=COL["Parity"], lw=1.6, ms=4, label="PARITY crowd")
    ax.plot(x, alloc, "s--", color=COL["Parity"], lw=1.3, ms=3.5, alpha=0.75, label="PARITY allocator")
    ax.axhline(16.7, ls=":", lw=0.9, color="#444444")
    ax.text(x[0], 17.6, "60 Hz", fontsize=7.5, color="#444444")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("agents"); ax.set_ylabel("ms per frame")
    ax.set_title("(b) the allocator is the cost, and its share falls with crowd size",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # (c) the efficiency claim: evaluations do not grow with the crowd
    ax = axes[1, 0]
    ev = np.array([float(d[(n, "Parity")]["evals_mean"]) for n in ns])
    fl = np.array([float(d[(n, "Parity")]["fill_mean"]) for n in ns])
    ax.plot(x, ev, "o-", color=COL["Parity"], lw=1.6, ms=4, label="Lagrangian evaluations")
    ax.plot(x, fl, "s-", color="#009E73", lw=1.4, ms=3.5, label="greedy fill steps")
    ax.axhline(8, ls="--", lw=0.9, color="#CC79A7")
    ax.text(x[0], 8.3, "FILL_MAX", color="#CC79A7", fontsize=7.5)
    ax.set_xscale("log"); ax.set_ylim(0, 20)
    ax.set_xlabel("agents"); ax.set_ylabel("per frame")
    ax.set_title("(c) evaluations are flat in N; the fill cap is what binds first",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left")

    # (d) where the fidelity went
    ax = axes[1, 1]
    w = 0.38
    idx = np.arange(len(ns))
    for k, pol in enumerate(("Baseline", "Parity")):
        bot = np.zeros(len(ns))
        for t in range(4):
            v = np.array([int(d[(n, pol)][f"beh{t}"]) for n in ns], float)
            v = v / np.array([float(n) for n in ns]) * 100
            ax.bar(idx + (k - 0.5) * w, v, w * 0.92, bottom=bot, color=TIER_COL[t],
                   edgecolor="white", linewidth=0.4,
                   label=f"tier {t}" + (" (surrogate)" if t == 3 else "") if k == 0 else None)
            bot += v
    ax.set_xticks(idx)
    ax.set_xticklabels([str(n) for n in ns], fontsize=8)
    ax.set_xlabel("agents   (left bar: baseline,  right bar: PARITY)", fontsize=8)
    ax.set_ylabel("% of agents, behaviour axis")
    ax.set_title("(d) PARITY is bang-bang: full fidelity or surrogate, no middle",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    # acceptance checks: assert the claims the figure makes
    kl_b = np.array([float(d[(n, "Baseline")]["kl_max_end"]) for n in ns])
    kl_p = np.array([float(d[(n, "Parity")]["kl_max_end"]) for n in ns])
    assert np.all(kl_p <= CAP + 1e-3), f"PARITY exceeded the cap: {kl_p}"
    assert kl_b[-1] > 3 * kl_b[0], "the baseline did not actually grow with N"
    assert np.all([float(d[(n, "Parity")]["budget_met_frac"]) == 1.0 for n in ns]), "budget missed"
    assert ev.max() / ev.min() < 1.5, "evaluation count is not flat in N"

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT, f"fig5_engine.{ext}")
        fig.savefig(p, dpi=170, bbox_inches="tight")
    print("wrote", os.path.join(OUT, "fig5_engine.png"))
    print(f"  baseline divergence {kl_b[0]:.2f} -> {kl_b[-1]:.2f} nats over {ns[0]}..{ns[-1]} agents")
    print(f"  PARITY   divergence {kl_p[0]:.2f} -> {kl_p[-1]:.2f} nats, cap {CAP}")
    print(f"  evaluations {ev.min():.1f}-{ev.max():.1f}, flat across a {ns[-1]//ns[0]}x range of N")
    print(f"  allocator share {100*alloc[0]/step[0]:.0f}% at N={ns[0]} -> {100*alloc[-1]/step[-1]:.0f}% at N={ns[-1]}")


if __name__ == "__main__":
    main()
