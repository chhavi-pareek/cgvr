"""Draw the four primary PARITY figures from the sweep CSVs, per figures/SPEC.md.

    python -m figures.plot                # all figures that have inputs
    python -m figures.plot --only 1 2     # a subset
    python -m figures.plot --target 100   # force figure 3's target frame time (ms)

Every figure asserts its acceptance checks before drawing and prints the numbers it
annotated, so the captions can quote them. A figure whose input CSV is missing is
skipped with a named reason; figure 4 draws hollow bars where the T4 rows are absent.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DT = 1.0 / 30.0
LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "logs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

COLOUR = {"parity": "#0072B2", "baseline": "#D55E00", "reference": "#000000",
          "parity_nocap": "#E69F00", "parity_authored": "#009E73", "grey": "#999999"}
NAME = {"parity": "PARITY", "baseline": "MassLOD baseline", "reference": "Reference (full fidelity)",
        "parity_nocap": "PARITY, no error cap", "parity_authored": "PARITY, authored per-axis weights"}
CAM_MARKER = {"orbit": "o", "flythrough": "s", "static_wide": "^", "static_choke": "v", "sweep": "D"}
SCENES = ("plaza", "hub", "corridor")
COND_ORDER = ("reference", "baseline", "parity", "parity_nocap", "parity_authored")
BAND_S = 2.0  # BAND (2 m) at 1 m/s: the guarantee's ceiling on visible lag


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


TAG = ""   # --tag: read sweep_*{TAG}.csv and write fig*{TAG}.*, so a re-run sits beside the record


def load(name, required):
    """Read a CSV from bench/logs; refuse (return None) if absent or missing a column."""
    if name.startswith("sweep_"):
        name = name.replace(".csv", f"{TAG}.csv")
    path = os.path.join(LOGS, name)
    if not os.path.exists(path):
        print(f"  SKIP: {name} not found ({path})")
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"  SKIP: {name} is empty")
        return None
    missing = [c for c in required if c not in rows[0]]
    if missing:
        print(f"  SKIP: {name} missing column(s): {', '.join(missing)}")
        return None
    return rows


def panel_letter(ax, letter):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom")


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}{TAG}.{ext}"))
    plt.close(fig)
    print(f"  wrote {OUT}/{stem}{TAG}.pdf and .png")


# ---------------------------------------------------------------- figure 1

def fig1():
    print("figure 1: behavioural divergence vs elapsed time")
    rows = load("sweep_kl.csv", ["scene", "cond", "cam", "n", "seed", "frame", "kl_p95", "kl_max", "cap"])
    if rows is None:
        return
    cells = load("sweep_cells.csv", ["scene", "cond", "n", "cam", "kl_slope_per_1k"]) or []
    rows = [r for r in rows if int(r["n"]) == 200 and r["cam"] == "orbit"]
    conds = [c for c in COND_ORDER if c != "reference" and any(r["cond"] == c for r in rows)]

    series, caps = {}, {}
    for s in SCENES:
        for c in conds:
            sel = [r for r in rows if r["scene"] == s and r["cond"] == c]
            if not sel:
                continue
            by_seed = defaultdict(dict)
            for r in sel:
                by_seed[r["seed"]][int(r["frame"])] = (float(r["kl_p95"]), float(r["kl_max"]))
            frames = sorted(set.intersection(*(set(d) for d in by_seed.values())))
            p95 = np.array([[by_seed[sd][f][0] for f in frames] for sd in by_seed])
            mx = np.array([[by_seed[sd][f][1] for f in frames] for sd in by_seed])
            series[(s, c)] = (np.array(frames) * DT, p95.mean(0), mx.mean(0), p95.min(0), p95.max(0))
            if c == "parity":
                caps[s] = float(sel[0]["cap"])

    # checks
    ratios = {}
    for s in SCENES:
        for c in conds:
            assert (s, c) in series, f"figure 1: no rows for {s}/{c} at the final frame"
        if "parity" in conds and s in caps:
            r = series[(s, "parity")][2].max() / caps[s]
            ratios[s] = r
            assert r <= 2.0, f"figure 1: parity kl_max is {r:.2f}x cap in {s} (limit 2.0)"
        if "baseline" in conds and "parity" in conds:
            assert series[(s, "baseline")][2][-1] > series[(s, "parity")][2][-1], \
                f"figure 1: baseline does not exceed parity at the end in {s}"
    print("  parity kl_max / cap:", ", ".join(f"{s} {v:.2f}x" for s, v in ratios.items()))
    if "parity" in conds and "parity_authored" in conds:
        same = all(np.array_equal(series[(s, "parity")][1], series[(s, "parity_authored")][1])
                   for s in SCENES)
        if same:
            print("  NOTE: parity_authored is identical to parity in every scene - the invariant-2 "
                  "ablation does not separate. Drawn dashed so both remain visible.")

    top = max(v[2].max() for v in series.values())
    ymax = 10 ** np.ceil(np.log10(top))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, s, letter in zip(axes, SCENES, "abc"):
        for c in conds:
            x, p95, mx, lo, hi = series[(s, c)]
            # parity_authored is byte-identical to parity here (the invariant-2 ablation does
            # not separate), so draw it dashed or it hides the parity line completely.
            ls = "--" if c == "parity_authored" else "-"
            ax.plot(x, np.maximum(p95, 1e-3), color=COLOUR[c], lw=1.2, ls=ls, label=NAME[c])
            ax.plot(x, np.maximum(mx, 1e-3), color=COLOUR[c], lw=0.7, ls=ls, alpha=0.5)
            if not np.allclose(lo, hi):
                ax.fill_between(x, np.maximum(lo, 1e-3), np.maximum(hi, 1e-3), color=COLOUR[c], alpha=0.2, lw=0)
        if s in caps:
            ax.axhline(caps[s], ls="--", lw=0.7, color=COLOUR["grey"])
            ax.text(ax.get_xlim()[1], caps[s], " cap", fontsize=7, color=COLOUR["grey"], va="center")
        ax.set_yscale("log")
        ax.set_ylim(0.1, ymax)
        ax.set_xlabel("elapsed time (s)")
        ax.set_title(s, fontsize=8, pad=3)
        panel_letter(ax, letter)
    axes[0].set_ylabel("accumulated divergence (nats)")
    axes[0].legend(loc="lower right", frameon=False)

    notes = []
    for c in ("baseline", "parity_nocap"):
        v = [float(r["kl_slope_per_1k"]) for r in cells
             if r["cond"] == c and r["cam"] == "orbit" and int(r["n"]) == 200]
        if v:
            notes.append(f"{NAME[c]}: {np.mean(v):+.0f} nats / 1000 frames")
    if notes:
        axes[-1].text(0.97, 0.05, "\n".join(notes), transform=axes[-1].transAxes, fontsize=7,
                      color=COLOUR["grey"], ha="right", va="bottom")
        print("  slopes:", "; ".join(notes))
    save(fig, "fig1_divergence")


# ---------------------------------------------------------------- figure 2

def fig2():
    print("figure 2: outcome variance across camera paths")
    rows = load("sweep_egress.csv", ["scene", "cond", "n", "cam", "seed", "agent", "trip", "frame"])
    if rows is None:
        return
    rows = [r for r in rows if int(r["n"]) == 200 and int(r["seed"]) == 0
            and r["cond"] in ("parity", "baseline")]
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        wasserstein_distance = None
        print("  note: scipy unavailable, W1 not computed")

    trips, stats = defaultdict(dict), {}
    for r in rows:
        trips[(r["scene"], r["cond"], r["cam"])][(r["agent"], r["trip"])] = int(r["frame"])

    deltas, ntrips = defaultdict(list), {}
    for s in SCENES:
        for c in ("parity", "baseline"):
            ref = trips.get((s, c, "orbit"))
            if not ref:
                continue
            cams = sorted({k[2] for k in trips if k[0] == s and k[1] == c})
            for cam in cams:
                cur = trips[(s, c, cam)]
                ntrips[(s, c, cam)] = len(cur)
                if cam == "orbit":
                    continue
                shared = set(ref) & set(cur)
                d = [abs(cur[k] - ref[k]) * DT for k in shared]
                deltas[(s, c)] += d
                w1 = (wasserstein_distance([v * DT for v in ref.values()],
                                           [v * DT for v in cur.values()])
                      if wasserstein_distance else float("nan"))
                stats[(s, c, cam)] = (len(cur), len(shared), max(d) if d else 0.0,
                                      len(set(ref) ^ set(cur)), w1)

    # checks: acceptance (b) of phase 7
    for s in SCENES:
        pc = [ntrips[k] for k in ntrips if k[0] == s and k[1] == "parity"]
        if pc:
            assert len(set(pc)) == 1, f"figure 2: parity trip counts differ across cams in {s}: {pc}"
            assert max(deltas[(s, 'parity')], default=0.0) == 0.0, \
                f"figure 2: parity max|delta| != 0 in {s}"
    for k in sorted(stats):
        n, m, mx, un, w1 = stats[k]
        print(f"  {k[0]:9s} {k[1]:9s} {k[2]:13s} trips={n:5d} matched={m:5d} "
              f"max|d|={mx:7.2f}s unmatched={un:3d} W1={w1:6.2f}s")

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    for i, (s, letter) in enumerate(zip(SCENES, "abc")):
        sub = gs[i].subgridspec(1, 2, width_ratios=[0.7, 0.3], wspace=0.08)
        ax, inset = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        xmax = max((max(deltas[(s, c)], default=0.0) for c in ("parity", "baseline")), default=1.0)
        for c in ("parity", "baseline"):
            d = np.sort(np.asarray(deltas[(s, c)], float))
            if d.size == 0:
                continue
            if d.max() == 0:
                ax.plot([0, max(xmax, BAND_S)], [1, 1], color=COLOUR[c], lw=1.2, label=NAME[c])
                ax.plot([0], [1], marker="o", color=COLOUR[c], ms=3.5)
                ax.text(0.04, 0.88, "max |Δ| = 0.00 s", transform=ax.transAxes,
                        fontsize=7, color=COLOUR[c])
            else:
                ax.step(np.r_[0, d], np.r_[0, np.arange(1, d.size + 1) / d.size],
                        where="post", color=COLOUR[c], lw=1.2, label=NAME[c])
        ax.set_xscale("symlog", linthresh=DT)
        ax.set_xlim(0, max(xmax, BAND_S) * 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 1e-1, 1e0, 1e1])
        ax.axvline(BAND_S, ls="--", lw=0.7, color=COLOUR["grey"])
        ax.text(BAND_S, 0.5, " band", fontsize=7, color=COLOUR["grey"], rotation=90, va="center")
        ax.set_xlabel("|Δ| egress time (s)")
        ax.set_title(s, fontsize=8, pad=3)
        panel_letter(ax, letter)
        if i == 0:
            ax.set_ylabel("fraction of trips")
            ax.legend(loc="lower right", frameon=False)

        for c in ("parity", "baseline"):
            cams = [cm for cm in CAM_MARKER if (s, c, cm) in ntrips]
            for j, cm in enumerate(cams):
                inset.plot([j], [ntrips[(s, c, cm)]], marker=CAM_MARKER[cm], color=COLOUR[c], ms=3.5)
            if cams and len({ntrips[(s, c, cm)] for cm in cams}) == 1:
                inset.axhline(ntrips[(s, c, cams[0])], color=COLOUR[c], lw=0.7)
        inset.set_xticks(range(len(CAM_MARKER)))
        inset.set_xticklabels([])
        inset.set_xlim(-0.6, len(CAM_MARKER) - 0.4)
        inset.yaxis.tick_right()                      # keep ticks off the main panel
        inset.tick_params(labelsize=6, pad=1)
        inset.yaxis.set_major_locator(plt.MaxNLocator(3))
        inset.set_title("trips", fontsize=7, pad=2)

        bl = [stats[k] for k in stats if k[0] == s and k[1] == "baseline"]
        if bl:
            ax.text(0.03, 0.30, f"base: max |Δ| {max(b[2] for b in bl):.1f} s\n"
                                f"W1 ≤ {max(b[4] for b in bl):.1f} s\nPARITY: 0 / 0",
                    transform=ax.transAxes, fontsize=6.5, color=COLOUR["grey"],
                    ha="left", va="top", linespacing=1.3)
    save(fig, "fig2_camera_variance")


# ---------------------------------------------------------------- figure 3

def fig3(target=None):
    print("figure 3: sustained agent count at a fixed target frame time")
    rows = load("sweep_cells.csv", ["scene", "cond", "n", "cam", "seed",
                                    "ft_mean_ms", "ft_low1_ms", "alloc_share"])
    if rows is None:
        return
    rows = [r for r in rows if r["cam"] == "orbit"]
    curves = {}
    for s in SCENES:
        for c in COND_ORDER:
            sel = [r for r in rows if r["scene"] == s and r["cond"] == c]
            if not sel:
                continue
            by_n = defaultdict(list)
            for r in sel:
                by_n[int(r["n"])].append((float(r["ft_mean_ms"]), float(r["ft_low1_ms"])))
            ns = sorted(by_n)
            curves[(s, c)] = (np.array(ns, float),
                              np.array([np.mean([v[0] for v in by_n[n]]) for n in ns]),
                              np.array([np.mean([v[1] for v in by_n[n]]) for n in ns]))

    # checks
    counts = {len(v[0]) for v in curves.values()}
    assert counts == {6}, f"figure 3: not every (scene, cond) has six agent counts: {counts}"
    bad = [r for r in rows if float(r["ft_low1_ms"]) < float(r["ft_mean_ms"])]
    assert not bad, f"figure 3: {len(bad)} rows have ft_low1_ms < ft_mean_ms"

    def sustained(ns, ft, t):
        ok = ns[ft <= t]
        if ok.size == 0:
            return 0.0, "<min"
        if ok.size == ns.size:
            return float(ns[-1]), ">=max"
        i = int(np.searchsorted(ns, ok[-1]))
        x0, x1, y0, y1 = ns[i], ns[i + 1], ft[i], ft[i + 1]
        if y1 == y0:
            return float(x0), "ok"
        f = (np.log10(t) - np.log10(y0)) / (np.log10(y1) - np.log10(y0))
        return float(10 ** (np.log10(x0) + f * (np.log10(x1) - np.log10(x0)))), "ok"

    if target is None:
        for t in (33.3, 100.0, 333.0):
            crossed = sum(1 for k, (ns, _, lo) in curves.items()
                          if lo[0] <= t < lo[-1])
            if crossed >= 3:
                target = t
                break
        target = target or 333.0
    print(f"  target frame time T* = {target:.1f} ms (prototype; Python times, ratios only)")

    sus = {k: (sustained(v[0], v[2], target), sustained(v[0], v[1], target)) for k, v in curves.items()}
    for k in sorted(sus):
        print(f"  {k[0]:9s} {k[1]:17s} sustained(1% low)={sus[k][0][0]:8.0f} {sus[k][0][1]:6s} "
              f"sustained(mean)={sus[k][1][0]:8.0f}")
    for s in SCENES:
        if (s, "reference") in sus and (s, "parity") in sus and \
                sus[(s, "reference")][0][0] > sus[(s, "parity")][0][0]:
            print(f"  WARNING: reference sustains more than parity in {s} - allocator overhead "
                  f"exceeded its saving; the caption must say so")
    for s in SCENES:
        for c in COND_ORDER:
            sel = [r for r in rows if r["scene"] == s and r["cond"] == c]
            if sel:
                big = max(sel, key=lambda r: int(r["n"]))
                print(f"  alloc_share {s:9s} {c:17s} at N={big['n']:>5s}: {float(big['alloc_share']):.3f}")

    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    for c in COND_ORDER:
        if ("hub", c) not in curves:
            continue
        ns, mean, lo = curves[("hub", c)]
        ax.plot(ns, mean, color=COLOUR[c], lw=1.2, label=NAME[c])
        ax.plot(ns, lo, color=COLOUR[c], lw=1.2, ls="--")
        ax.fill_between(ns, mean, lo, color=COLOUR[c], alpha=0.15, lw=0)
    ax.axhline(target, ls="--", lw=0.7, color=COLOUR["grey"])
    ax.text(ax.get_xlim()[1], target, f" T*={target:.0f} ms", fontsize=7, color=COLOUR["grey"], va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ns_all = curves[("hub", "parity")][0]
    ax.set_xticks(ns_all)
    ax.set_xticklabels([f"{int(n)}" for n in ns_all])
    ax.minorticks_off()
    ax.set_xlabel("agents")
    ax.set_ylabel("frame time (ms)")
    ax.set_title("hub", fontsize=8, pad=3)
    ax.legend(loc="upper left", frameon=False)
    panel_letter(ax, "a")

    ax2 = fig.add_subplot(gs[1])
    w = 0.8 / len(COND_ORDER)
    for j, c in enumerate(COND_ORDER):
        xs = [i + (j - (len(COND_ORDER) - 1) / 2) * w for i, s in enumerate(SCENES) if (s, c) in sus]
        lo = [sus[(s, c)][0][0] for s in SCENES if (s, c) in sus]
        mn = [sus[(s, c)][1][0] for s in SCENES if (s, c) in sus]
        ax2.bar(xs, mn, width=w, color=COLOUR[c], alpha=0.35, lw=0)
        ax2.bar(xs, lo, width=w, color=COLOUR[c], label=NAME[c])
        for x, v in zip(xs, lo):
            ax2.text(x, v, f"{v:.0f}", fontsize=6, ha="center", va="bottom", rotation=90)
    ax2.set_xticks(range(len(SCENES)))
    ax2.set_xticklabels(SCENES)
    ax2.set_ylabel(f"agents sustained at T*={target:.0f} ms (1% low)")
    panel_letter(ax2, "b")
    save(fig, "fig3_sustained_agents")


# ---------------------------------------------------------------- figure 4

def fig4():
    print("figure 4: ordering Pareto front and four-way speedup")
    order = load("order_sweep.csv", ["density", "n", "seed", "w", "bytes_per_agent", "wee"])
    speed = load("speedup.csv", ["n", "impl", "ms"])
    if order is None and speed is None:
        print("  SKIP: neither order_sweep.csv nor speedup.csv present - run t4_runs.ipynb")
        return

    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 2, wspace=0.55)  # room for (a)'s colourbar label
    ax = fig.add_subplot(gs[0])
    if order is not None:
        sel = [r for r in order if int(r["n"]) == 50000 and int(r["seed"]) == 0]
        if "wee_meas" in order[0]:
            mism = [r for r in sel if r["wee_meas"] not in ("", "nan")
                    and abs(float(r["wee_meas"]) - float(r["wee"])) > 1e-9]
            assert not mism, f"figure 4: wee_meas != wee on {len(mism)} rows (counter check failed)"
        shape = {"sparse": "o", "mixed": "s", "dense": "^"}
        for d, mk in shape.items():
            pts = defaultdict(list)
            for r in sel:
                if r["density"] == d:
                    pts[float(r["w"])].append((float(r["bytes_per_agent"]), float(r["wee"])))
            if not pts:
                continue
            for w, vs in sorted(pts.items()):
                b, e = np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])
                if w < 0:
                    ax.plot(b, e, marker=mk, mfc="none", color=COLOUR["grey"], ms=4)
                else:
                    ax.scatter([b], [e], marker=mk, c=[w], cmap="viridis", vmin=0, vmax=1, s=14)
        ax.set_xscale("log")
        ax.set_xlabel("bytes moved per agent per frame (B)")
        ax.set_ylabel("warp execution efficiency")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("spatial weight w", fontsize=7)
    else:
        ax.text(0.5, 0.5, "T4 pending\n(order_sweep.csv)", ha="center", va="center",
                fontsize=8, color=COLOUR["grey"], transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    panel_letter(ax, "a")

    ax2 = fig.add_subplot(gs[1])
    IMPL = {"serial": "#999999", "threaded": "#56B4E9", "openacc": "#CC79A7", "cuda": "#0072B2"}
    sizes = [1000, 10000, 100000]
    if speed is not None:
        if "match" in speed[0]:
            bad = [r for r in speed if r["impl"] != "serial" and r["match"].lower() not in ("true", "1")]
            assert not bad, f"figure 4: {len(bad)} non-serial rows have match != True"
        best = defaultdict(dict)
        for r in speed:
            impl = r["impl"].split("[")[0]
            best[int(r["n"])].setdefault(impl, []).append(float(r["ms"]))
        missing = []
        for j, impl in enumerate(IMPL):
            xs, ys = [], []
            for i, n in enumerate(sizes):
                got = best.get(n, {})
                if "serial" not in got or impl not in got:
                    missing.append(f"{impl}@{n}")
                    continue
                xs.append(i + (j - 1.5) * 0.2)
                ys.append(min(got["serial"]) / min(got[impl]))
            ax2.bar(xs, ys, width=0.2, color=IMPL[impl], label=impl)
        if missing:
            print("  T4 pending, hollow:", ", ".join(missing))
            ax2.text(0.5, 0.92, "T4 pending: " + ", ".join(sorted({m.split('@')[0] for m in missing})),
                     transform=ax2.transAxes, ha="center", fontsize=7, color=COLOUR["grey"])
        ax2.set_yscale("log")
        ax2.legend(loc="upper left", frameon=False, ncol=2)
    else:
        ax2.text(0.5, 0.5, "T4 pending\n(speedup.csv)", ha="center", va="center",
                 fontsize=8, color=COLOUR["grey"], transform=ax2.transAxes)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f"{n // 1000}k" for n in sizes])
    ax2.set_xlabel("agents")
    ax2.set_ylabel("speedup over serial")
    panel_letter(ax2, "b")
    save(fig, "fig4_ordering_and_speedup")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", type=int, choices=[1, 2, 3, 4])
    ap.add_argument("--target", type=float, default=None, help="figure 3 target frame time (ms)")
    ap.add_argument("--tag", default="", help="sweep tag, e.g. _v2")
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    style()
    want = a.only or [1, 2, 3, 4]
    for i in want:
        (fig1 if i == 1 else fig2 if i == 2 else (lambda: fig3(a.target)) if i == 3 else fig4)()


if __name__ == "__main__":
    main()
