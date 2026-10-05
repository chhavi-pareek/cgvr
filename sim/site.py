"""Real-site geometry: walkable grid, one distance field per gate, routes as polylines.

The site file (sim/data/<name>_site.json, from tools/osm_site.py) is in a local frame, x east and
y north in metres. Site shifts it so the compound's lower-left corner sits at (MARGIN, MARGIN):
every scene keeps positions in [0, size].

Grid cells are RES metres. `walk` is where a body centre may be (inside the compound, outside
buildings, walls and, unless the event uses them, gardens; plus each open gate and the apron
beyond it). `clear` is `walk` eroded by one cell, and routes are planned on it so they keep off
walls. Outside each open gate is its approach zone: the open ground within APPROACH metres of
the gate (outside buildings and walls excluded). A gate's sink is the zone's rim, and its field is
the shortest 8-connected path length to that rim; agents that reach a sink have left.
"""
import functools
import json
import pathlib

import numpy as np
from matplotlib.path import Path
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

DATA = pathlib.Path(__file__).resolve().parent / "data"
RES = 0.5
MARGIN = 20.0
WALL_T = 0.6  # metres of wall thickness when a mapped wall is a line
APRON = 8.0  # metres the carved gate passage runs outside the outline
APPROACH = 40.0  # metres of outside ground in front of each open gate; its rim is the sink
MAX_GATEHOUSE = 30.0  # metres a gate passage may run inward through buildings
SQ2 = np.sqrt(2.0)
NB = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))


def _fill(mask, ring, shift, value=True):
    """Set the cells whose centres fall inside a polygon (world coordinates)."""
    p = np.asarray(ring, np.float64) + shift
    i0, j0 = np.floor(p.min(0) / RES).astype(int)
    i1, j1 = np.ceil(p.max(0) / RES).astype(int) + 1
    i0, j0 = max(i0, 0), max(j0, 0)
    i1, j1 = min(i1, mask.shape[0]), min(j1, mask.shape[1])
    if i1 <= i0 or j1 <= j0:
        return
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    c = np.stack([(ii.ravel() + 0.5) * RES, (jj.ravel() + 0.5) * RES], 1)
    inside = Path(p).contains_points(c).reshape(ii.shape)
    mask[i0:i1, j0:j1][inside] = value


def _stroke(mask, pts, shift, half, value):
    """Set the cells within `half` metres of a polyline."""
    p = np.asarray(pts, np.float64) + shift
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / (RES * 0.5))), 1)
        for t in np.linspace(0.0, 1.0, n + 1):
            c = a + t * (b - a)
            i0, j0 = np.floor((c - half) / RES).astype(int)
            i1, j1 = np.floor((c + half) / RES).astype(int) + 1
            ii, jj = np.meshgrid(np.arange(max(i0, 0), min(i1, mask.shape[0])),
                                 np.arange(max(j0, 0), min(j1, mask.shape[1])), indexing="ij")
            near = np.hypot((ii + 0.5) * RES - c[0], (jj + 0.5) * RES - c[1]) <= half
            mask[ii[near], jj[near]] = value


class Site:
    def __init__(self, name="mysore_palace", gardens_walkable=True, widths=None, open_gates=None):
        """widths: {gate key: metres} overriding the site file's assumed widths.
        open_gates: keys of the open gates (default: the spectator gates, role "entry+exit")."""
        raw = json.loads((DATA / f"{name}_site.json").read_text())
        self.raw = raw
        x0, y0, x1, y1 = raw["bounds"]
        self.shift = np.array([MARGIN - x0, MARGIN - y0])
        self.size = (x1 - x0 + 2 * MARGIN, y1 - y0 + 2 * MARGIN)
        self.shape = (int(np.ceil(self.size[0] / RES)), int(np.ceil(self.size[1] / RES)))
        gates = [dict(g) for g in raw["gates"]]
        for g in gates:
            g["at"] = np.asarray(g["at"], np.float64) + self.shift
            if widths and g["key"] in widths:
                g["width"] = float(widths[g["key"]])
            g["open"] = g["role"] == "entry+exit" if open_gates is None else g["key"] in open_gates
        self.gates = gates
        self.open = [k for k, g in enumerate(gates) if g["open"]]
        self.compound = np.asarray(raw["compound"], np.float64) + self.shift
        self._rasterise(gardens_walkable)

    def _outward(self, at):
        """Unit normal pointing out of the compound at a gate."""
        poly = Path(self.compound)
        best, nrm = np.inf, None
        for a, b in zip(self.compound[:-1], self.compound[1:]):
            ab = b - a
            t = np.clip(np.dot(at - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            d = np.linalg.norm(at - (a + t * ab))
            if d < best:
                best, nrm = d, np.array([ab[1], -ab[0]]) / max(np.linalg.norm(ab), 1e-9)
        if poly.contains_point(at + 2.0 * nrm):
            nrm = -nrm
        return nrm

    def _rasterise(self, gardens_walkable):
        raw, s = self.raw, self.shift
        inside = np.zeros(self.shape, bool)
        _fill(inside, self.compound, 0.0)
        blocked = np.zeros(self.shape, bool)
        for b in raw["buildings"] + raw["parts"]:
            for ring in b["rings"]:
                _fill(blocked, ring, s)
        self.building = blocked.copy()
        for w in raw["walls"]:
            _stroke(blocked, w, s, WALL_T / 2, True)
        if not gardens_walkable:
            for g in raw["gardens"]:
                _fill(blocked, g, s)
        walk = inside & ~blocked
        # the compound outline is a wall everywhere except the gates
        _stroke(walk, self.compound, 0.0, WALL_T / 2, False)
        outside_block = np.zeros(self.shape, bool)
        _stroke(outside_block, self.compound, 0.0, WALL_T / 2, True)
        for b in raw.get("outside_buildings", []):
            for ring in b["rings"]:
                _fill(self.building, ring, s)
                _fill(outside_block, ring, s)
        for w in raw.get("outside_walls", []):
            _stroke(outside_block, w, s, WALL_T / 2, True)
        centres = (np.indices(self.shape).transpose(1, 2, 0) + 0.5) * RES
        self.zone = np.zeros(self.shape, bool)
        self.sink = []
        self.passage = np.zeros(self.shape, bool)
        self.gate_passage = {}
        for g in self.gates:
            if not g["open"]:
                self.sink.append(None)
                continue
            out = self._outward(g["at"])
            g["out"] = out
            along = np.array([-out[1], out[0]])
            half = g["width"] / 2
            # the outline is lined with mapped buildings (the fort ranges), so the passage runs
            # in through its gatehouse until it meets open ground
            depth = 2.0
            for t in np.arange(0.0, MAX_GATEHOUSE, RES / 2):
                if self.walkable(g["at"] - out * t, walk) and self.walkable(g["at"] - out * (t + 1.0), walk):
                    depth = t + 1.0
                    break
            g["depth"] = depth
            ring = [g["at"] - out * depth - along * half, g["at"] - out * depth + along * half,
                    g["at"] + out * APRON + along * half, g["at"] + out * APRON - along * half]
            gate = np.zeros(self.shape, bool)
            _fill(gate, ring, 0.0)
            walk |= gate
            self.passage |= gate
            self.gate_passage[self.gates.index(g)] = gate
            # the approach zone: outside ground within APPROACH of the gate, connected to its passage
            d = np.hypot(centres[..., 0] - g["at"][0], centres[..., 1] - g["at"][1])
            zone = ((d <= APPROACH) & ~inside & ~outside_block) | (gate & ~inside)
            lab, _ = ndimage.label(zone)
            end = self.cell(g["at"] + out * (APRON - 1.0))
            zone = lab == lab[tuple(end)]
            walk |= zone
            self.zone |= zone
            self.sink.append(zone & (d >= APPROACH - 3.0))
        self.walk = walk
        self.sink_any = np.any([self.sink[k] for k in self.open], 0)
        self.clear = ndimage.binary_erosion(walk, structure=np.ones((3, 3), bool))
        for k in self.open:
            self.clear |= self.sink[k]

    # --- coordinates -------------------------------------------------------------------------
    def cell(self, pos):
        ij = np.floor(np.asarray(pos, np.float64) / RES).astype(np.int64)
        ij[..., 0] = np.clip(ij[..., 0], 0, self.shape[0] - 1)
        ij[..., 1] = np.clip(ij[..., 1], 0, self.shape[1] - 1)
        return ij

    def centre(self, ij):
        return (np.asarray(ij, np.float64) + 0.5) * RES

    def walkable(self, pos, grid=None):
        g = self.walk if grid is None else grid
        p = np.asarray(pos, np.float64)
        ok = (p[..., 0] >= 0) & (p[..., 1] >= 0) & (p[..., 0] < self.shape[0] * RES) & (p[..., 1] < self.shape[1] * RES)
        ij = self.cell(p)
        return ok & g[ij[..., 0], ij[..., 1]]

    # --- distance fields ---------------------------------------------------------------------
    @functools.cached_property
    def _graph(self):
        idx = np.full(self.shape, -1, np.int64)
        cells = np.argwhere(self.clear)
        idx[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
        rows, cols, w = [], [], []
        for di, dj, c in NB:
            a = cells
            b = a + (di, dj)
            ok = (b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 0] < self.shape[0]) & (b[:, 1] < self.shape[1])
            a, b = a[ok], b[ok]
            ok = self.clear[b[:, 0], b[:, 1]]
            if di and dj:  # no corner cutting
                ok &= self.clear[a[:, 0] + di, a[:, 1]] & self.clear[a[:, 0], a[:, 1] + dj]
            a, b = a[ok], b[ok]
            rows.append(idx[a[:, 0], a[:, 1]])
            cols.append(idx[b[:, 0], b[:, 1]])
            w.append(np.full(len(a), c * RES))
        n = len(cells)
        g = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        return idx, cells, g

    @functools.lru_cache(maxsize=None)
    def field(self, k):
        """Metres from each cell to gate k's sink along `clear` (inf where unreachable)."""
        idx, cells, g = self._graph
        src = idx[self.sink[k] & self.clear]
        src = src[src >= 0]
        d = dijkstra(g, directed=False, indices=src, min_only=True)
        out = np.full(self.shape, np.inf)
        out[cells[:, 0], cells[:, 1]] = d
        return out

    @functools.lru_cache(maxsize=None)
    def flow(self, k):
        """Unit step toward gate k per cell: down the field on `clear`, and from a walkable cell
        off `clear` (beside a wall) toward its nearest `clear` cell."""
        f = self.field(k)
        best = f.copy()
        step = np.zeros(self.shape + (2,))
        pad = np.pad(f, 1, constant_values=np.inf)
        for di, dj, c in NB:
            nb = pad[1 + di: 1 + di + self.shape[0], 1 + dj: 1 + dj + self.shape[1]]
            better = nb < best
            best = np.where(better, nb, best)
            step[better] = (di / c, dj / c)
        _, (ii, jj) = ndimage.distance_transform_edt(~self.clear, return_indices=True)
        off = ~self.clear
        here = np.indices(self.shape).transpose(1, 2, 0)
        to = np.stack([ii, jj], -1) - here
        n = np.linalg.norm(to, axis=-1, keepdims=True)
        step[off] = (to / np.maximum(n, 1e-9))[off]
        return step

    def steer(self, pos, k):
        """Per-agent unit direction from gate k's flow (k may be an array, one gate per agent)."""
        ij = self.cell(pos)
        k = np.broadcast_to(np.asarray(k), ij.shape[:-1])
        out = np.zeros(ij.shape)
        for g in np.unique(k):
            m = k == g
            out[m] = self.flow(int(g))[ij[m, 0], ij[m, 1]]
        return out

    def distance(self, pos, k):
        ij = self.cell(pos)
        return self.field(k)[ij[..., 0], ij[..., 1]]

    # --- routes ------------------------------------------------------------------------------
    def _descend(self, ij, k):
        """Cell path from ij down gate k's field to its sink."""
        f = self.field(k)
        path = [tuple(ij)]
        cur = tuple(ij)
        while not self.sink[k][cur]:
            best, nxt = f[cur], None
            for di, dj, _ in NB:
                c = (cur[0] + di, cur[1] + dj)
                if 0 <= c[0] < self.shape[0] and 0 <= c[1] < self.shape[1] and f[c] < best:
                    best, nxt = f[c], c
            if nxt is None:
                raise ValueError(f"no descent from {ij} toward gate {self.gates[k]['key']}")
            path.append(nxt)
            cur = nxt
        return path

    def visible(self, a, b, grid=None):
        """True when the straight segment a-b stays on walkable cells (`clear` by default)."""
        g = self.clear if grid is None else grid
        n = max(int(np.ceil(np.linalg.norm(np.asarray(b) - np.asarray(a)) / (RES * 0.5))), 1)
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        p = np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
        return bool(self.walkable(p, g).all())

    def route(self, start, k):
        """Polyline from start to gate k's sink: the field descent, string-pulled on `clear`."""
        ij = self.cell(start)
        if not np.isfinite(self.field(k)[tuple(ij)]):
            raise ValueError(f"start {start} cannot reach gate {self.gates[k]['key']}")
        pts = [self.centre(c) for c in self._descend(ij, k)]
        out = [np.asarray(start, np.float64)]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.visible(out[-1], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.asarray(out)

    def nearest_gate(self, pos):
        """Open gate with the shortest field distance from each position (the core's exit choice)."""
        d = np.stack([self.distance(pos, k) for k in self.open], -1)
        return np.asarray(self.open)[np.argmin(d, -1)], d.min(-1)

    # --- wall constraint ---------------------------------------------------------------------
    def constrain(self, pos, prev):
        """Moves that end off `walk` slide along one axis, or are undone."""
        bad = ~self.walkable(pos)
        if not bad.any():
            return pos
        p, q = pos[bad].copy(), prev[bad]
        sx = np.stack([p[:, 0], q[:, 1]], 1)
        sy = np.stack([q[:, 0], p[:, 1]], 1)
        okx, oky = self.walkable(sx), self.walkable(sy)
        p = np.where(okx[:, None], sx, np.where(oky[:, None], sy, q))
        pos[bad] = p
        return pos
