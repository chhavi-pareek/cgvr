"""Palace egress figure: peak crowding by Fruin level, one panel per run.

    python -m figures.palace s48000_nearest_w0_x1 s48000_balanced_w0_x1 s48000_measured_w0_x1

Reads bench/logs/palace/<tag>_density.npz and <tag>_exits.csv; writes figures/out/palace_egress.png.
Crowding is the peak walking density per 2 m cell over the run, binned at Fruin's walkway levels;
outlines are the seat blocks.
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from sim.site import RES, Site  # noqa: E402

LOGS = pathlib.Path("bench/logs/palace")
OUT = pathlib.Path("figures/out/palace_egress.png")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
GROUND, BUILDING = "#f0efec", "#b9b7b0"
LEVELS = [(1 / 1.39, "D  (0.72-1.08 /m²)", "#86b6ef"), (1 / 0.93, "E  (1.08-2.17 /m²)", "#2a78d6"),
          (1 / 0.46, "F  (over 2.17 /m²)", "#104281")]
PLAN = {"nearest": "Everyone to the nearest gate", "balanced": "Seats shared by gate width",
        "measured": "Seats shared by measured gate rates"}
NAME = {"north": "North (Jayarama, Balarama)", "south": "Varaha (south)", "east": "Jayamarthanda (east)",
        "east_n": "East side, N", "east_s": "East side, S", "west_n": "Brahmapuri? (west)",
        "west_s": "Karikal Thotti? (west)"}


def entry_panel(ax, st, tag):
    """Arrival through screening: peak crowd outside and inside, and per gate who passed and
    how long it took from arriving at the approach zone to clearing screening."""
    d = np.load(LOGS / f"{tag}_density.npz")
    peak, cell = d["peak"], float(d["cell"])
    scr = np.loadtxt(LOGS / f"{tag}_screened.csv", delimiter=",", skiprows=1)
    txt = (LOGS / f"{tag}.txt").read_text().splitlines()
    ext = (0, st.shape[0] * RES, 0, st.shape[1] * RES)
    base = np.where(st.building, 2, np.where(st.walk, 1, 0))
    ax.imshow(base.T, origin="lower", extent=ext, cmap=ListedColormap([SURFACE, GROUND, BUILDING]),
              vmin=0, vmax=2, interpolation="nearest")
    lvl = np.zeros(peak.shape)
    for k, (lo, _, _) in enumerate(LEVELS):
        lvl[peak > lo] = k + 1
    lvl = np.ma.masked_where(lvl == 0, lvl)
    ax.imshow(lvl.T, origin="lower", extent=(0, peak.shape[0] * cell, 0, peak.shape[1] * cell),
              cmap=ListedColormap([c for _, _, c in LEVELS]), vmin=0.5, vmax=3.5, interpolation="nearest")
    for line in txt:
        line = line.strip()
        for k in st.open:
            g = st.gates[k]
            if not line.startswith(g["key"] + " "):
                continue
            m = scr[:, 1] == k
            east = g["at"][0] > 0.6 * st.size[0]
            p95 = line.split("p95")[1].split()[0] if "p95" in line else "-"
            lanes = line.split("m, ")[1].split(" lanes")[0] if " lanes" in line else "?"
            ax.plot(*g["at"], marker="^", ms=8, color=INK, mec=SURFACE, mew=1.5)
            ax.annotate(f"{NAME.get(g['key'], g['key'])}, {g['width']:g} m, {lanes} lanes\n{int(m.sum()):,} in, 95% within {p95} min",
                        g["at"], xytext=(-8 if east else 8, 6), textcoords="offset points", fontsize=8, color=INK,
                        ha="right" if east else "left", path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])
    seated = next((l for l in txt if l.startswith("seated when")), "")
    pct = seated.split("(")[1].split(")")[0] if "(" in seated else "?"
    ax.set_title(f"Arrival through screening: {pct} seated when the procession starts", fontsize=11, color=INK, loc="left")
    ax.set_xlim(10, st.size[0] - 40)
    ax.set_ylim(10, st.size[1] - 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def panel(ax, st, tag):
    if (LOGS / f"{tag}_screened.csv").exists():
        return entry_panel(ax, st, tag)
    d = np.load(LOGS / f"{tag}_density.npz")
    peak, cell = d["peak"], float(d["cell"])
    ex = np.loadtxt(LOGS / f"{tag}_exits.csv", delimiter=",", skiprows=1)
    ext = (0, st.shape[0] * RES, 0, st.shape[1] * RES)
    base = np.where(st.building, 2, np.where(st.walk, 1, 0))
    ax.imshow(base.T, origin="lower", extent=ext, cmap=ListedColormap([SURFACE, GROUND, BUILDING]),
              vmin=0, vmax=2, interpolation="nearest")
    lvl = np.zeros(peak.shape)
    for k, (lo, _, _) in enumerate(LEVELS):
        lvl[peak > lo] = k + 1
    lvl = np.ma.masked_where(lvl == 0, lvl)
    ax.imshow(lvl.T, origin="lower", extent=(0, peak.shape[0] * cell, 0, peak.shape[1] * cell),
              cmap=ListedColormap([c for _, _, c in LEVELS]), vmin=0.5, vmax=3.5, interpolation="nearest")
    from matplotlib.patches import Rectangle

    from sim.scenes import Palace

    bw, bh = Palace.ROWS * Palace.ROW, Palace.COLS * Palace.SEAT
    for x, y in Palace(site=st).blocks(-(-len(ex) // (Palace.ROWS * Palace.COLS))):
        ax.add_patch(Rectangle((x, y), bw, bh, fill=False, lw=0.6, ec=MUTED))
    rel = 0  # release frame: the first block always goes at frame 0
    for k in st.open:
        g = st.gates[k]
        m = ex[:, 1] == k
        ax.plot(*g["at"], marker="^", ms=8, color=INK, mec=SURFACE, mew=1.5)
        txt = f"{NAME.get(g['key'], g['key'])}, {g['width']:g} m\n" + (f"{int(m.sum()):,} out, last {((ex[m, 2].max() - rel) / 30 / 60):.0f} min"
                                                    if m.any() else "unused")
        east = g["at"][0] > 0.6 * st.size[0]  # labels on the east edge go left, inside the panel
        ax.annotate(txt, g["at"], xytext=(-8 if east else 8, 6), textcoords="offset points", fontsize=8,
                    color=INK, ha="right" if east else "left",
                    path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])
    t100 = (ex[:, 2].max() - rel) / 30 / 60 if len(ex) else float("nan")
    plan = next((v for k, v in PLAN.items() if f"_{k}_" in tag), tag)
    ax.set_title(f"{plan}: all out in {t100:.0f} min", fontsize=11, color=INK, loc="left")
    ax.set_xlim(10, st.size[0] - 40)
    ax.set_ylim(10, st.size[1] - 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main(tags):
    st = Site()
    fig, axes = plt.subplots(1, len(tags), figsize=(7.5 * len(tags), 6.2), facecolor=SURFACE)
    seats = set()
    for ax, tag in zip(np.atleast_1d(axes), tags):
        panel(ax, st, tag)
        f = LOGS / f"{tag}_exits.csv"
        seats.add(len(np.loadtxt(f if f.exists() else LOGS / f"{tag}_screened.csv", delimiter=",", skiprows=1)))
    fig.subplots_adjust(bottom=0.12, wspace=0.02)
    fig.legend(handles=[Patch(color=c, label=lab) for _, lab, c in LEVELS], loc="lower center", ncol=3,
               frameon=False, fontsize=9, title="Peak walking crowd, Fruin level of service", title_fontsize=9,
               labelcolor=MUTED, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, -0.01, f"Mysore Palace grounds, {'/'.join(f'{n:,}' for n in sorted(seats))} people. Gate widths are estimated "
             "from photographs or assumed (docs/site/mysore_palace.md, 5b); compare the plans, not the absolute minutes. "
             "Map (c) OpenStreetMap contributors.", fontsize=7.5, color=MUTED, ha="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main(sys.argv[1:] or ["s48000_nearest_w0_x1", "s48000_balanced_w0_x1", "s48000_measured_w0_x1"])
