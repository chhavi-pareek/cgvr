"""Figures and LaTeX tables for the LLM-agent results, from the per-run CSVs in paper/data/.

    python -m figures.llm_plots

Reads paper/data/{station_7b,museum_7b,station_14b,station_marginal,sensitivity}.csv (and
museum_7b_series.npz) as written by bench.llm_agents --csv and bench.llm_sensitivity; skips any that
are absent. Writes figures/out/llm_*.{pdf,png} and paper/generated/tab_*.tex. Every number drawn is a
mean over seeds with a 95% t-interval.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from figures.plot import panel_letter, save, style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "paper", "data")
GEN = os.path.join(ROOT, "paper", "generated")

COL = {"parity": "#0072B2", "parity/reh 0.2": "#56B4E9", "parity/wait 5 reh": "#009E73",
       "parity_nocap": "#E69F00", "cascade": "#CC79A7", "cascade/reh 0.2": "#CC79A7",
       "view_lod": "#D55E00", "round_robin": "#999999", "reference": "#000000", "surrogate_only": "#000000"}
LS = {"cascade/reh 0.2": "--", "parity/reh 0.2": "-", "parity/wait 5 reh": "-"}
NAME = {"parity": "PARITY", "parity/reh 0.2": "PARITY + rehearsal", "parity/wait 5 reh": "PARITY + rehearsal + wait 5 s",
        "parity_nocap": "PARITY, no ledger", "cascade": "Cascade", "cascade/reh 0.2": "Cascade + rehearsal",
        "view_lod": "View-LOD", "round_robin": "Round-robin", "surrogate_only": "Surrogate only", "reference": "All-LLM reference"}


def read(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"  SKIP: {name} not found")
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def ci(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if x.size < 2:
        return (float(x.mean()) if x.size else np.nan), np.nan
    return float(x.mean()), float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def station_figure(rows, stem, title):
    ns = sorted({r["n"] for r in rows})
    conds = [c for c in ("parity", "parity_nocap", "cascade", "view_lod", "round_robin") if sel(rows, condition=c)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, key, lab in zip(axes, ("kl_per_agent_h", "kl_steady_per_agent_h", "dmax"),
                            ("drift, nats per agent-hour", "steady drift (after 5 min)", "worst agent / cap")):
        for c in conds:
            m = [ci([r[key] / (r["cap"] if key == "dmax" else 1) for r in sel(rows, condition=c, n=n)]) for n in ns]
            ax.errorbar(ns, [x[0] for x in m], yerr=[x[1] for x in m], color=COL[c], capsize=2, label=NAME[c],
                        ls=LS.get(c, "-"), marker="s" if c == "parity_nocap" else "o",
                        mfc="none" if c == "parity_nocap" else COL[c])
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_xlabel("agents")
        ax.set_ylabel(lab)
        if key == "dmax":
            ax.axhline(1.0, color="k", lw=0.6, ls=":")
    axes[0].legend(frameon=False, loc="upper left")
    for ax, l in zip(axes, "abc"):
        panel_letter(ax, l)
    fig.subplots_adjust(wspace=0.42)
    print(f"  {stem}: {title}")
    save(fig, stem)


def cap_figure(rows):
    caps = sorted({r["value"] for r in sel(rows, scenario="station", param="cap")})
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2))
    for c in ("parity", "parity_nocap", "cascade"):
        m = [ci([100 * r["overflow_frac"] for r in sel(rows, scenario="station", param="cap", value=v, condition=c)]) for v in caps]
        axes[0].errorbar(caps, [x[0] for x in m], yerr=[x[1] for x in m], color=COL[c], marker="o", capsize=2, label=NAME[c])
    m = [ci([r["stall_per_agent_h"] for r in sel(rows, scenario="station", param="cap", value=v, condition="parity")]) for v in caps]
    axes[1].errorbar(caps, [x[0] for x in m], yerr=[x[1] for x in m], color=COL["parity"], marker="o", capsize=2)
    for ax in axes:
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(caps)
        ax.set_xticklabels([f"{v:g}" for v in caps])
        ax.set_xlabel("cap, nats")
        ax.axvline(9.21, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("stretches over the cap, %")
    axes[1].set_ylabel("PARITY waits per agent-hour")
    axes[0].legend(frameon=False)
    for ax, l in zip(axes, "ab"):
        panel_letter(ax, l)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "llm_cap")


def museum_timeline(series, n=1000):
    conds = ["parity", "parity/reh 0.2", "parity/wait 5 reh", "cascade", "cascade/reh 0.2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for c in conds:
        keys = [k for k in series.files if k.startswith(f"{c}|{n}|") and k.endswith("|kl")]
        if not keys:
            continue
        cum = np.mean([series[k] / n for k in keys], 0)          # total drift so far, per agent
        axes[0].plot(np.arange(1, len(cum) + 1), cum, color=COL[c], ls=LS.get(c, "-"), label=NAME[c])
    for x, lab in ((300, "alarm"), (480, "smoke")):
        axes[0].axvline(x, color="k", lw=0.6, ls=":")
        axes[0].text(x + 8, axes[0].get_ylim()[1] * 0.95, lab, fontsize=6, va="top")
    axes[0].set_xlabel("seconds")
    axes[0].set_ylabel("cumulative drift per agent, nats")
    grid = np.arange(0, 601, 10)
    for c in ["reference"] + conds:
        keys = [k for k in series.files if k.startswith(f"{c}|{n}|") and k.endswith("|out_t")]
        if not keys:
            continue
        curves = []
        for k in keys:
            m = float(series[k.replace("|out_t", "|at_alarm")][0])
            curves.append(np.searchsorted(np.sort(series[k]), grid, side="right") / max(m, 1))
        axes[1].plot(grid, np.mean(curves, 0), color=COL[c], ls=LS.get(c, "-"), lw=1.6 if c == "reference" else 1.0,
                     label=NAME[c])
    axes[1].set_xlabel("seconds after the alarm")
    axes[1].set_ylabel("share of people out")
    axes[1].legend(frameon=False, loc="lower right")
    for ax, l in zip(axes, "ab"):
        panel_letter(ax, l)
    save(fig, "llm_museum_timeline")


def museum_sensitivity(rows):
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2))
    reh = sel(rows, scenario="museum", param="rehearse")
    vs = sorted({r["value"] for r in reh})
    for key, lab, ls in (("kl_steady_per_agent_h", "post-alarm drift", "-"), ("w1", "evacuation W1, s", "--")):
        m = [ci([r[key] for r in sel(reh, value=v)]) for v in vs]
        axes[0].errorbar(vs, [x[0] for x in m], yerr=[x[1] for x in m], color=COL["parity"], ls=ls, marker="o", capsize=2, label=lab)
    axes[0].set_xlabel("rehearsal share of the call budget")
    axes[0].legend(frameon=False)
    wt = sel(rows, scenario="museum", param="max_wait")
    order = [v for v in (0.0, 2.0, 5.0, 15.0, 60.0) if sel(wt, value=v)] + (["none"] if sel(wt, value="none") else [])
    xs = np.arange(len(order))
    m1 = [ci([r["stall_per_agent_h"] for r in sel(wt, value=v)]) for v in order]
    m2 = [ci([100 * r["overflow_frac"] for r in sel(wt, value=v)]) for v in order]
    axes[1].errorbar(xs, [x[0] for x in m1], yerr=[x[1] for x in m1], color=COL["parity"], marker="o", capsize=2, label="waits per agent-hour")
    ax2 = axes[1].twinx()
    ax2.errorbar(xs, [x[0] for x in m2], yerr=[x[1] for x in m2], color=COL["parity_nocap"], marker="s", capsize=2, label="stretches over, %")
    ax2.set_ylabel("stretches over the cap, %")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{v:g}" if v != "none" else "none" for v in order])
    axes[1].set_xlabel("wait bound, s")
    axes[1].set_ylabel("waits per agent-hour")
    for ax, l in zip(axes, "ab"):
        panel_letter(ax, l)
    fig.subplots_adjust(wspace=0.6)
    save(fig, "llm_museum_sensitivity")


def fmt(m, h, d=2):
    return f"{m:.{d}f}" if not np.isfinite(h) else f"{m:.{d}f} $\\pm$ {h:.{d}f}"


def table(rows, conds, cols, path, caption, label):
    ns = sorted({r["n"] for r in rows})
    head = " & ".join(["$N$", "Policy"] + [c[1] for c in cols])
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule", head + " \\\\", "\\midrule"]
    for n in ns:
        for i, c in enumerate(conds):
            rs = sel(rows, n=n, condition=c)
            if not rs:
                continue
            cells = []
            for key, _, scale, d in cols:
                m, h = ci([r[key] * scale / (r["cap"] if key == "dmax" else 1) for r in rs])
                cells.append(fmt(m, h, d))
            lines.append(" & ".join([f"{int(n)}" if i == 0 else "", NAME[c]] + cells) + " \\\\")
        lines.append("\\midrule" if n != ns[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table*}"]
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, path), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote paper/generated/{path}")


def main():
    style()
    cols = [("dmax", "Worst / cap", 1, 2), ("overflow_frac", "Over, \\%", 100, 2), ("kl_per_agent_h", "Drift", 1, 2),
            ("kl_steady_per_agent_h", "Steady", 1, 2), ("stall_per_agent_h", "Waits", 1, 0)]
    st = read("station_7b.csv")
    if st:
        station_figure(st, "llm_station", "Station, qwen2.5:7b, order-free reference")
        table(st, ["parity", "parity_nocap", "cascade", "view_lod", "round_robin", "surrogate_only"], cols,
              "tab_station.tex", "Station: drift in nats per agent-hour and the per-agent cap ($\\kappa=10$), 10 seeds, mean $\\pm$ 95\\% CI.",
              "tab:station")
    for name, stem, title in (("station_14b.csv", "llm_station_14b", "Station, qwen2.5:14b reference"),
                              ("station_marginal.csv", "llm_station_marginal", "Station, per-persona average surrogate")):
        r = read(name)
        if r:
            station_figure(r, stem, title)
    mu = read("museum_7b.csv")
    if mu:
        mcols = cols[:3] + [("kl_steady_per_agent_h", "Post-alarm", 1, 1), ("stall_per_agent_h", "Waits", 1, 0), ("w1", "Evac.\\ W1, s", 1, 0)]
        table(mu, ["parity", "parity/reh 0.2", "parity/wait 5 reh", "parity_nocap", "cascade", "cascade/reh 0.2",
                   "view_lod", "round_robin", "surrogate_only"], mcols, "tab_museum.tex",
              "Museum evacuation (alarm at 5 min): drift, the cap, waiting and evacuation-time W1 to the same-seed all-LLM run, 10 seeds.",
              "tab:museum")
    path = os.path.join(DATA, "museum_7b_series.npz")
    if os.path.exists(path):
        museum_timeline(np.load(path))
    se = read("sensitivity.csv")
    if se:
        cap_figure(se)
        museum_sensitivity(se)


if __name__ == "__main__":
    main()
