"""OpenStreetMap extract -> site file for the palace scene (sim/site.py).

    python tools/osm_site.py

Reads docs/site/mysore_palace_osm.json (API 0.6 map call, fetched 2026-10-05) and writes
sim/data/mysore_palace_site.json in a local frame: origin 12.3050 N 76.6560 E, x east, y north,
metres. The data is (c) OpenStreetMap contributors, ODbL, and the site file inherits that licence.

Gates are where mapped paths cross the compound outline (clustered within 12 m). Names, roles and
clear widths come from the gate audit in docs/site/mysore_palace.md (section 5b): widths are
estimated from photographs or assumed, never published, so each carries a range to sweep.
"""
import json
import math
import pathlib
import re

import numpy as np
from matplotlib.path import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/site/mysore_palace_osm.json"
DST = ROOT / "sim/data/mysore_palace_site.json"
LAT0, LON0 = 12.3050, 76.6560
COMPOUND = 656323590
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 110574.0

# (near x, near y): key, name, role on Jamboo Savari day, clear width, (low, high), how the width
# was obtained. Roles: "entry+exit" (police-staffed spectator gates in the news reports),
# "procession" (the Jamboo Savari leaves by it; closed to spectators unless a scenario opens it),
# "unknown" (in no report; closed by default). docs/site/mysore_palace.md, section 5b, has the
# evidence for every row.
GATES = {
    (-34, 257): ("north", "North: Jayarama and Balarama", "procession", 17.0, (13.0, 21.0),
                 "DER: two arches, each 4.7 m central + 2 x 1.9 m side, scooters for scale (Commons photo)"),
    (199, -4): ("east", "East: Jayamarthanda (main gate)", "entry+exit", 10.0, (8.0, 12.0),
                "DER: 0.32 of the 34 m mapped structure (frontal photo); side ground arches not counted"),
    (217, -51): ("east_s", "East forecourt, south side opening", "unknown", 3.0, (2.0, 4.0), "ASSUMED"),
    (226, 39): ("east_n", "East forecourt, north side opening", "unknown", 3.0, (2.0, 4.0), "ASSUMED"),
    (-75, -169): ("south", "South: Varaha", "entry+exit", 4.2, (3.5, 6.4),
                  "DER: central arch, people for scale (Commons photo); high end adds the two side doors"),
    (-326, 146): ("west_n", "West: Brahmapuri (identity not confirmed)", "entry+exit", 4.0, (3.0, 6.0), "ASSUMED"),
    (-318, -114): ("west_s", "West: Karikal Thotti (identity not confirmed)", "entry+exit", 3.0, (2.0, 4.0), "ASSUMED"),
}
OUTSIDE = 70.0  # metres from a gate within which outside buildings and walls are kept


def _xy(n):
    return [round((n["lon"] - LON0) * KX, 2), round((n["lat"] - LAT0) * KY, 2)]


def _rings(ways, nodes, refs):
    """Stitch member ways into closed rings (multipolygon outers are often split)."""
    segs = [list(ways[r]["nodes"]) for r in refs if r in ways]
    rings = []
    while segs:
        ring = segs.pop(0)
        grown = True
        while ring[0] != ring[-1] and grown:
            grown = False
            for k, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[::-1][1:]
                else:
                    continue
                segs.pop(k)
                grown = True
                break
        if ring[0] == ring[-1] and all(i in nodes for i in ring):
            rings.append([_xy(nodes[i]) for i in ring])
    return rings


def _height(t):
    m = re.match(r"\s*([0-9.]+)", t.get("height", ""))
    return float(m.group(1)) if m else 0.0


def _cross(p, q, a, b):
    d = (q[0] - p[0]) * (b[1] - a[1]) - (q[1] - p[1]) * (b[0] - a[0])
    if abs(d) < 1e-12:
        return None
    t = ((a[0] - p[0]) * (b[1] - a[1]) - (a[1] - p[1]) * (b[0] - a[0])) / d
    u = ((a[0] - p[0]) * (q[1] - p[1]) - (a[1] - p[1]) * (q[0] - p[0])) / d
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])) if 0 <= t <= 1 and 0 <= u <= 1 else None


def main():
    els = json.loads(SRC.read_text())["elements"]
    nodes = {e["id"]: e for e in els if e["type"] == "node"}
    ways = {e["id"]: e for e in els if e["type"] == "way"}
    rels = [e for e in els if e["type"] == "relation"]
    comp = [_xy(nodes[i]) for i in ways[COMPOUND]["nodes"]]
    inside = Path(comp)

    def within(pts):
        return bool(inside.contains_points(np.asarray(pts)).any())

    gate_at = np.array([k for k in GATES], np.float64)

    def near_gate(pts):
        p = np.asarray(pts, np.float64)
        return bool((np.linalg.norm(p[:, None] - gate_at[None], axis=2) < OUTSIDE).any())

    buildings, parts, walls, gardens, paths = [], [], [], [], []
    out_buildings, out_walls = [], []
    for w in ways.values():
        t = w.get("tags", {})
        if w["id"] == COMPOUND or not all(i in nodes for i in w["nodes"]):
            continue
        pts = [_xy(nodes[i]) for i in w["nodes"]]
        closed = w["nodes"][0] == w["nodes"][-1]
        if not within(pts):
            # outside the walls: only what shapes the approach to a gate
            if near_gate(pts) and closed and ("building" in t or "building:part" in t):
                out_buildings.append({"height": _height(t), "rings": [pts]})
            elif near_gate(pts) and t.get("barrier") in ("wall", "fence"):
                out_walls.append(pts)
            continue
        h = _height(t)
        if closed and "building" in t:
            buildings.append({"name": t.get("name", ""), "height": h, "rings": [pts]})
        elif closed and "building:part" in t:
            parts.append({"height": h, "rings": [pts]})
        elif t.get("barrier") in ("wall", "fence"):
            walls.append(pts)
        elif closed and (t.get("leisure") in ("garden", "park") or t.get("landuse") == "grass"):
            gardens.append(pts)
        elif "highway" in t:
            paths.append({"kind": t["highway"], "pts": pts})
    for r in rels:
        t = r.get("tags", {})
        outer = _rings(ways, nodes, [m["ref"] for m in r["members"] if m["type"] == "way" and m["role"] == "outer"])
        if not outer or not within(outer[0]):
            continue
        h = _height(t)
        if "building" in t:
            buildings.append({"name": t.get("name", ""), "height": h, "rings": outer})
        elif "building:part" in t:
            parts.append({"height": h, "rings": outer})

    hits = []
    for w in ways.values():
        if "highway" not in w.get("tags", {}) or not all(i in nodes for i in w["nodes"]):
            continue
        p = [_xy(nodes[i]) for i in w["nodes"]]
        for k in range(len(p) - 1):
            for j in range(len(comp) - 1):
                x = _cross(p[k], p[k + 1], comp[j], comp[j + 1])
                if x:
                    hits.append(x)
    gates = []
    for (gx, gy), (key, name, role, width, (lo, hi), how) in GATES.items():
        near = [h for h in hits if math.dist(h, (gx, gy)) < 12.0]
        if not near:
            raise SystemExit(f"no mapped crossing near gate {key} {gx, gy}")
        at = [round(float(np.mean([h[0] for h in near])), 2), round(float(np.mean([h[1] for h in near])), 2)]
        gates.append({"key": key, "name": name, "at": at, "role": role, "width": width,
                      "width_range": [lo, hi], "width_is": how})
    xs, ys = [p[0] for p in comp], [p[1] for p in comp]
    site = {
        "name": "Mysore Palace compound",
        "source": "(c) OpenStreetMap contributors, ODbL; docs/site/mysore_palace_osm.json, fetched 2026-10-05",
        "frame": {"origin_lat": LAT0, "origin_lon": LON0, "axes": "x east, y north, metres"},
        "bounds": [min(xs), min(ys), max(xs), max(ys)],
        "compound": comp,
        "gates": gates,
        "buildings": buildings,
        "parts": parts,
        "walls": walls,
        "gardens": gardens,
        "paths": paths,
        "outside_buildings": out_buildings,
        "outside_walls": out_walls,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(site, separators=(",", ":")))
    print(f"{DST.relative_to(ROOT)}: {len(gates)} gates, {len(buildings)} buildings, {len(parts)} parts, "
          f"{len(out_buildings)} outside buildings, {len(out_walls)} outside walls, "
          f"{len(walls)} walls, {len(gardens)} gardens, {len(paths)} paths, {DST.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
