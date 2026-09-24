"""Side-by-side visual demo: MassLOD baseline vs PARITY, same seed, same camera.

    python -m figures.demo                          # corridor, 900 frames -> mp4
    python -m figures.demo --scene hub --frames 600
    python -m figures.demo --gif                    # gif instead of mp4

Each agent is drawn at its simulated position and coloured by its accumulated
behavioural divergence. Under the baseline agents redden monotonically and never
recover; under PARITY the ledger forces reconciliation, so they redden and snap back.
The trace underneath is the worst agent's divergence against the cap.

This is the one result a static figure cannot carry: the divergence is a *process*,
and the recovery is what the error ledger buys.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from bench.camerapaths import camera  # noqa: E402
from sim.tiered import DT, Run, calibrate  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
COL = {"parity": "#0072B2", "baseline": "#D55E00"}
NAME = {"parity": "PARITY", "baseline": "MassLOD baseline"}
CONDS = ("baseline", "parity")


def capture(scene, n, cond, frames, seed, cam, cal, stride):
    """Run one condition, snapshotting positions and per-agent divergence."""
    r = Run(scene, n, cond, seed=seed, cam=camera(scene, cam, 1), calib=cal)
    pos, div, kmax, egress = [], [], [], []
    for f in range(frames):
        r.step(f)
        kmax.append(float(r.D_meas.max()))
        egress.append(len(r.world.egress))
        if f % stride == 0:
            pos.append(r.a.pos.copy())
            div.append(r.D_meas.copy())
    cap = float(getattr(r, "cap", np.nan))
    return dict(pos=np.array(pos), div=np.array(div), kmax=np.array(kmax),
                egress=np.array(egress), cap=cap, wall=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="plaza", choices=["plaza", "hub", "corridor"])
    ap.add_argument("--agents", type=int, default=200)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--stride", type=int, default=2, help="simulate every frame, draw every stride-th")
    ap.add_argument("--cam", default="orbit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()

    print(f"calibrating {a.scene} (cached after the first run) ...")
    cal = calibrate(a.scene, n=200, seed=0)

    data = {}
    for c in CONDS:
        print(f"simulating {c:9s} {a.frames} frames, N={a.agents} ...")
        data[c] = capture(a.scene, a.agents, c, a.frames, a.seed, a.cam, cal, a.stride)
    cap = data["parity"]["cap"]
    print(f"cap = {cap:.2f} nats")
    for c in CONDS:
        d = data[c]
        print(f"  {NAME[c]:18s} end max divergence {d['kmax'][-1]:8.2f} nats   trips {d['egress'][-1]}")

    # shared colour scale so the two panels are directly comparable
    vmax = max(d["div"].max() for d in data.values())
    norm = LogNorm(vmin=max(cap * 0.05, 1e-2), vmax=max(vmax, cap * 2))

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": 0.6})
    P = data["parity"]["pos"]
    lo = np.nanmin(np.concatenate([d["pos"].reshape(-1, 2) for d in data.values()]), 0)
    hi = np.nanmax(np.concatenate([d["pos"].reshape(-1, 2) for d in data.values()]), 0)
    pad = 0.05 * (hi - lo + 1e-9)

    # size the canvas to the scene's aspect so a wide scene does not leave the panels tiny
    span = (hi - lo) + 2 * pad
    panel_w = 4.6
    panel_h = np.clip(panel_w * span[1] / max(span[0], 1e-9), 1.4, 5.2)
    fig = plt.figure(figsize=(2 * panel_w + 1.2, panel_h + 2.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[panel_h, 1.7], hspace=0.28, wspace=0.10)

    scat, axes = {}, {}
    for i, c in enumerate(CONDS):
        ax = fig.add_subplot(gs[0, i])
        axes[c] = ax
        scat[c] = ax.scatter(data[c]["pos"][0][:, 0], data[c]["pos"][0][:, 1],
                             c=np.maximum(data[c]["div"][0], norm.vmin),
                             cmap="inferno", norm=norm, s=14, linewidths=0)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(NAME[c], fontsize=11, color=COL[c], pad=6)
    cb = fig.colorbar(scat["parity"], ax=[axes[c] for c in CONDS], fraction=0.025, pad=0.01)
    cb.set_label("accumulated divergence (nats, log)", fontsize=8)
    cb.ax.axhline(cap, color="#00FF88", lw=1.4)
    cb.ax.text(1.8, cap, " cap", color="#00A060", fontsize=7, va="center")

    axk = fig.add_subplot(gs[1, :])
    t = np.arange(a.frames) * DT
    for c in CONDS:
        axk.plot(t, np.maximum(data[c]["kmax"], 1e-2), color=COL[c], lw=1.4, label=NAME[c])
    axk.axhline(cap, ls="--", lw=0.8, color="#00A060")
    axk.text(0.995, cap, "cap ", color="#00A060", fontsize=8, va="bottom", ha="right",
             transform=axk.get_yaxis_transform())
    axk.set_yscale("log")
    axk.set_xlim(0, t[-1])
    axk.set_xlabel("elapsed time (s)")
    axk.set_ylabel("worst-agent\ndivergence (nats)")
    axk.legend(loc="center right", frameon=False, fontsize=8)
    marker = axk.axvline(0, color="#333333", lw=1.0)
    clock = axk.text(0.01, 0.88, "", transform=axk.transAxes, fontsize=8, family="monospace")

    nframes = len(P)

    def update(k):
        for c in CONDS:
            d = data[c]
            scat[c].set_offsets(d["pos"][k])
            scat[c].set_array(np.maximum(d["div"][k], norm.vmin))
        f = k * a.stride
        marker.set_xdata([f * DT, f * DT])
        clock.set_text(f"t = {f * DT:6.1f} s   baseline max {data['baseline']['kmax'][f]:8.1f}"
                       f"   PARITY max {data['parity']['kmax'][f]:6.2f} nats")
        return ()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"demo_{a.scene}.{'gif' if a.gif else 'mp4'}")
    writer = PillowWriter(fps=a.fps) if a.gif else FFMpegWriter(fps=a.fps, bitrate=2400)
    print(f"rendering {nframes} frames -> {path}")
    with writer.saving(fig, path, dpi=110):
        for k in range(nframes):
            update(k)
            writer.grab_frame()
            if k % 100 == 0:
                print(f"  {k}/{nframes}")
    plt.close(fig)
    print("wrote", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
