"""Cell-list separation for large crowds: the same force as sim.run._separation (its oracle,
tests/test_site.py), found through a uniform grid of r-sized cells instead of all pairs."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _bins(pos, active, r):
    n = pos.shape[0]
    lo0, lo1 = np.inf, np.inf
    hi0, hi1 = -np.inf, -np.inf
    for i in range(n):
        if active[i]:
            lo0, lo1 = min(lo0, pos[i, 0]), min(lo1, pos[i, 1])
            hi0, hi1 = max(hi0, pos[i, 0]), max(hi1, pos[i, 1])
    nx = int((hi0 - lo0) / r) + 1
    ny = int((hi1 - lo1) / r) + 1
    cell = np.full(n, -1, np.int64)
    count = np.zeros(nx * ny + 1, np.int64)
    for i in range(n):
        if active[i]:
            c = int((pos[i, 0] - lo0) / r) * ny + int((pos[i, 1] - lo1) / r)
            cell[i] = c
            count[c + 1] += 1
    start = np.cumsum(count)
    fill = start[:-1].copy()
    order = np.empty(start[-1], np.int64)
    for i in range(n):
        if cell[i] >= 0:
            order[fill[cell[i]]] = i
            fill[cell[i]] += 1
    return lo0, lo1, nx, ny, cell, start, order


@njit(parallel=True, cache=True)
def _sep(pos, active, r, lo0, lo1, nx, ny, cell, start, order):
    n = pos.shape[0]
    out = np.zeros((n, 2))
    for i in prange(n):
        if not active[i]:
            continue
        cx = int((pos[i, 0] - lo0) / r)
        cy = int((pos[i, 1] - lo1) / r)
        sx, sy = 0.0, 0.0
        for gx in range(max(cx - 1, 0), min(cx + 2, nx)):
            for gy in range(max(cy - 1, 0), min(cy + 2, ny)):
                c = gx * ny + gy
                for q in range(start[c], start[c + 1]):
                    j = order[q]
                    if j == i:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 < r * r:
                        d = np.sqrt(d2)
                        w = (1.0 - d / r) / max(d, 1e-6)
                        sx += w * dx
                        sy += w * dy
        out[i, 0] = sx
        out[i, 1] = sy
    return out


def separation(pos, r, active=None):
    """Sum over neighbours within r of (1 - d/r) * unit(p_i - p_j); inactive agents neither
    push nor feel a push."""
    pos = np.ascontiguousarray(pos, np.float64)
    active = np.ones(len(pos), np.bool_) if active is None else np.ascontiguousarray(active, np.bool_)
    if not active.any():
        return np.zeros_like(pos)
    return _sep(pos, active, float(r), *_bins(pos, active, float(r)))


@njit(parallel=True, cache=True)
def _push(pos, active, fixed, r, lo0, lo1, nx, ny, cell, start, order):
    n = pos.shape[0]
    out = np.zeros((n, 2))
    reach = 2.0 * r
    for i in prange(n):
        if not active[i] or fixed[i]:
            continue
        cx = int((pos[i, 0] - lo0) / r)
        cy = int((pos[i, 1] - lo1) / r)
        sx, sy = 0.0, 0.0
        for gx in range(max(cx - 2, 0), min(cx + 3, nx)):
            for gy in range(max(cy - 2, 0), min(cy + 3, ny)):
                c = gx * ny + gy
                for q in range(start[c], start[c + 1]):
                    j = order[q]
                    if j == i:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 < reach * reach:
                        d = np.sqrt(d2)
                        if d < 1e-9:  # coincident: split along a fixed axis by index
                            dx, dy, d = (1.0 if i < j else -1.0), 0.0, 1.0
                            gap = reach
                        else:
                            gap = reach - d
                        share = 1.0 if fixed[j] else 0.5
                        sx += share * gap * dx / d
                        sy += share * gap * dy / d
        out[i, 0] = sx
        out[i, 1] = sy
    return out


def resolve(pos, radius, active=None, fixed=None, iters=4):
    """Bodies of `radius` may not overlap: Jacobi position projection, each overlapping pair
    pushed apart along its centre line (half each, or all of it against a fixed agent).
    Returns the corrected positions; inactive agents are ignored, fixed ones never move."""
    pos = np.array(pos, np.float64)
    n = len(pos)
    active = np.ones(n, np.bool_) if active is None else np.ascontiguousarray(active, np.bool_)
    fixed = np.zeros(n, np.bool_) if fixed is None else np.ascontiguousarray(fixed, np.bool_)
    if not active.any():
        return pos
    for _ in range(iters):
        bins = _bins(pos, active, float(radius))
        d = _push(pos, active, fixed, float(radius), *bins)
        if not np.abs(d).max() > 1e-9:
            break
        pos += 0.8 * d
    return pos


@njit(parallel=True, cache=True)
def _count(pos, active, r, lo0, lo1, nx, ny, cell, start, order):
    n = pos.shape[0]
    out = np.zeros(n)
    for i in prange(n):
        if not active[i]:
            continue
        cx = int((pos[i, 0] - lo0) / r)
        cy = int((pos[i, 1] - lo1) / r)
        c_n = 0
        for gx in range(max(cx - 1, 0), min(cx + 2, nx)):
            for gy in range(max(cy - 1, 0), min(cy + 2, ny)):
                c = gx * ny + gy
                for q in range(start[c], start[c + 1]):
                    j = order[q]
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    if dx * dx + dy * dy < r * r:
                        c_n += 1
        out[i] = c_n
    return out


def local_density(pos, r, active=None):
    """Persons per m2 in the disc of radius r around each active agent, the agent included."""
    pos = np.ascontiguousarray(pos, np.float64)
    active = np.ones(len(pos), np.bool_) if active is None else np.ascontiguousarray(active, np.bool_)
    if not active.any():
        return np.zeros(len(pos))
    return _count(pos, active, float(r), *_bins(pos, active, float(r))) / (np.pi * r * r)


def weidmann(rho, jam=5.4, gamma=1.913):
    """Weidmann (1993) speed-density relation as a fraction of free speed: 1 when sparse,
    0 at the jam density (persons per m2)."""
    rho = np.maximum(np.asarray(rho, np.float64), 1e-6)
    return np.clip(1.0 - np.exp(-gamma * (1.0 / rho - 1.0 / jam)), 0.0, 1.0)
