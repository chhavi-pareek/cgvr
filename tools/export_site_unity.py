"""Palace site -> Unity (StreamingAssets/palace): the grids, gates, seat-block candidates and
building outlines the engine scene reads, plus an oracle the C# port must reproduce.

    python tools/export_site_unity.py

grid.bytes   magic b"PSG1", int32 nx, ny, float32 res, shift x, shift y, int32 gates, then
             packbits (big-endian bit order, row-major over [x cell, y cell]) of walk, clear,
             building, and one sink mask per gate (all zero for a closed gate)
site.txt     one record per line, scene frame (sim/site.py: compound corner at (MARGIN, MARGIN)):
               size <w> <h>
               gate <index> <key> <open> <x> <y> <width> <out x> <out y>
               block <x> <y>                 seat-block corners, nearest the palace front first
               building <height> <n> x y ...  footprint rings of whole buildings (height 0: unmapped)
               part <height> <n> x y ...      building:part rings (the palace's towers and wings)
               compound <n> x y ...
               wall <n> x y ...
               garden <n> x y ...
oracle.txt   field <gate> <i> <j> <metres>   sampled distance-field values
             route <gate> <x0> <y0> <length> <n> x y ...   string-pulled routes from block sides

The grids are Python's (sim/site.py is the reference); the C# side recomputes the fields and
routes from them, and ParitySmoke checks it against oracle.txt.
"""
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sim.scenes import Palace  # noqa: E402
from sim.site import RES, Site  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DST = ROOT / "unity/Parity3D/Assets/StreamingAssets/palace"
N_BLOCKS = 120


def _bits(mask):
    return np.packbits(mask.astype(np.uint8).ravel()).tobytes()


def _ring(tag, pts, shift, extra=""):
    p = np.asarray(pts, np.float64) + shift
    return f"{tag}{extra} {len(p)} " + " ".join(f"{x:.2f} {y:.2f}" for x, y in p)


def main():
    st = Site()
    DST.mkdir(parents=True, exist_ok=True)
    nx, ny = st.shape
    with open(DST / "grid.bytes", "wb") as f:
        f.write(b"PSG1" + struct.pack("<iifffi", nx, ny, RES, st.shift[0], st.shift[1], len(st.gates)))
        for m in (st.walk, st.clear, st.building):
            f.write(_bits(m))
        for k in range(len(st.gates)):
            f.write(_bits(st.sink[k] if st.sink[k] is not None else np.zeros(st.shape, bool)))

    raw, s = st.raw, st.shift
    lines = [f"size {st.size[0]:.2f} {st.size[1]:.2f}"]
    for k, g in enumerate(st.gates):
        out = g.get("out", np.zeros(2))
        lines.append(f"gate {k} {g['key']} {int(g['open'])} {g['at'][0]:.2f} {g['at'][1]:.2f} {g['width']:.2f} "
                     f"{out[0]:.5f} {out[1]:.5f}")
    sc = Palace(site=st)
    for x, y in sc.blocks(N_BLOCKS):
        lines.append(f"block {x:.3f} {y:.3f}")
    for tag, group in (("building", raw["buildings"]), ("part", raw["parts"])):
        for b in group:
            for ring in b["rings"]:
                lines.append(_ring(tag, ring, s, f" {b['height']:.1f}"))
    lines.append(_ring("compound", raw["compound"], s))
    lines += [_ring("wall", w, s) for w in raw["walls"]]
    lines += [_ring("garden", g, s) for g in raw["gardens"]]
    (DST / "site.txt").write_text("\n".join(lines) + "\n")

    rng = np.random.default_rng(0)
    orc = []
    for k in st.open:
        f = st.field(k)
        cells = np.argwhere(np.isfinite(f))
        for i, j in cells[rng.choice(len(cells), 40, replace=False)]:
            orc.append(f"field {k} {i} {j} {f[i, j]:.6f}")
    corners = sc.blocks(12)
    bw, bh = Palace.ROWS * Palace.ROW, Palace.COLS * Palace.SEAT
    starts = np.concatenate([corners + (bw / 2, -Palace.AISLE / 2), corners + (bw / 2, bh + Palace.AISLE / 2)])
    for p in starts:
        for k in st.open:
            r = st.route(p, k)
            L = float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())
            orc.append(f"route {k} {p[0]:.4f} {p[1]:.4f} {L:.4f} {len(r)} " + " ".join(f"{x:.4f} {y:.4f}" for x, y in r))
    (DST / "oracle.txt").write_text("\n".join(orc) + "\n")
    print(f"{DST.relative_to(ROOT)}: grid {nx} x {ny} ({(DST / 'grid.bytes').stat().st_size // 1024} KB), "
          f"{len(lines)} site records, {len(orc)} oracle records")


if __name__ == "__main__":
    main()
