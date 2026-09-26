"""ORCA local collision avoidance (van den Berg, Guy, Lin and Manocha 2011), agents only.

The serial reference for the engine's navigation tiers (unity/.../Runtime/Orca.cs) and its
correctness oracle. Each neighbour contributes the half-plane of velocities that avoid it for
`tau` seconds if both agents take half the responsibility; the new velocity is the one closest to
the preferred velocity inside the speed circle and all half-planes (a 2D incremental linear
program), or, when those are infeasible, the one that violates them least (the RVO2
formulation: linear_program_3 over the projected constraints).

Neighbours are the k nearest within `reach`, nearest first, ties by index, so a port that finds
them with a grid picks exactly the same set. Everything is float64.
"""
from dataclasses import dataclass

import numpy as np

EPS = 1e-5
RADIUS = 0.3        # m, body radius
REACH = 2.5         # m, neighbour search radius (one engine grid cell)
TAU = 1.5           # s, time horizon


@dataclass
class Line:
    point: np.ndarray
    direction: np.ndarray


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def neighbours(pos, i, k, reach=REACH):
    """Indices of the k nearest agents to i within reach, nearest first, ties by index."""
    d = pos - pos[i]
    d2 = np.einsum("ij,ij->i", d, d)
    cand = np.flatnonzero((d2 < reach * reach) & (np.arange(len(pos)) != i))
    order = np.lexsort((cand, d2[cand]))
    return cand[order[:k]]


def half_planes(p, v, pn, vn, radius=RADIUS, tau=TAU, dt=1.0 / 30.0):
    """ORCA half-planes of an agent at p with velocity v against neighbours pn, vn."""
    lines = []
    inv_tau = 1.0 / tau
    r = 2.0 * radius
    r2 = r * r
    for pb, vb in zip(pn, vn):
        rel_pos = pb - p
        rel_vel = v - vb
        dist2 = rel_pos @ rel_pos
        if dist2 > r2:
            w = rel_vel - inv_tau * rel_pos
            w2 = w @ w
            dot1 = w @ rel_pos
            if dot1 < 0.0 and dot1 * dot1 > r2 * w2:
                # the velocity obstacle's cut-off circle
                wl = np.sqrt(w2)
                unit = w / wl
                direction = np.array([unit[1], -unit[0]])
                u = (r * inv_tau - wl) * unit
            else:
                # one of its legs
                leg = np.sqrt(dist2 - r2)
                if _det(rel_pos, w) > 0.0:
                    direction = np.array([rel_pos[0] * leg - rel_pos[1] * r, rel_pos[0] * r + rel_pos[1] * leg]) / dist2
                else:
                    direction = -np.array([rel_pos[0] * leg + rel_pos[1] * r, -rel_pos[0] * r + rel_pos[1] * leg]) / dist2
                u = (rel_vel @ direction) * direction - rel_vel
        else:
            # already overlapping: resolve within one step
            inv_dt = 1.0 / dt
            w = rel_vel - inv_dt * rel_pos
            wl = np.sqrt(w @ w)
            unit = w / wl
            direction = np.array([unit[1], -unit[0]])
            u = (r * inv_dt - wl) * unit
        lines.append(Line(v + 0.5 * u, direction))
    return lines


def _lp1(lines, n, radius, opt, direction_opt):
    ln = lines[n]
    dot = ln.point @ ln.direction
    disc = dot * dot + radius * radius - ln.point @ ln.point
    if disc < 0.0:
        return None
    s = np.sqrt(disc)
    t_left, t_right = -dot - s, -dot + s
    for i in range(n):
        den = _det(ln.direction, lines[i].direction)
        num = _det(lines[i].direction, ln.point - lines[i].point)
        if abs(den) <= EPS:
            if num < 0.0:
                return None
            continue
        t = num / den
        if den >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return None
    if direction_opt:
        return ln.point + (t_right if opt @ ln.direction > 0.0 else t_left) * ln.direction
    t = ln.direction @ (opt - ln.point)
    return ln.point + min(max(t, t_left), t_right) * ln.direction


def _lp2(lines, radius, opt, direction_opt):
    if direction_opt:
        result = opt * radius
    elif opt @ opt > radius * radius:
        result = opt / np.sqrt(opt @ opt) * radius
    else:
        result = opt.copy()
    for i in range(len(lines)):
        if _det(lines[i].direction, lines[i].point - result) > 0.0:
            r = _lp1(lines, i, radius, opt, direction_opt)
            if r is None:
                return i, result
            result = r
    return len(lines), result


def _lp3(lines, begin, radius, result):
    distance = 0.0
    for i in range(begin, len(lines)):
        if _det(lines[i].direction, lines[i].point - result) > distance:
            proj = []
            for j in range(i):
                det = _det(lines[i].direction, lines[j].direction)
                if abs(det) <= EPS:
                    if lines[i].direction @ lines[j].direction > 0.0:
                        continue
                    point = 0.5 * (lines[i].point + lines[j].point)
                else:
                    point = lines[i].point + (_det(lines[j].direction, lines[i].point - lines[j].point) / det) * lines[i].direction
                d = lines[j].direction - lines[i].direction
                proj.append(Line(point, d / np.sqrt(d @ d)))
            opt = np.array([-lines[i].direction[1], lines[i].direction[0]])
            fail, r = _lp2(proj, radius, opt, True)
            if fail >= len(proj):
                result = r
            distance = _det(lines[i].direction, lines[i].point - result)
    return result


def new_velocity(p, v, pref, max_speed, pn, vn, radius=RADIUS, tau=TAU, dt=1.0 / 30.0):
    lines = half_planes(p, v, pn, vn, radius, tau, dt)
    fail, result = _lp2(lines, max_speed, pref, False)
    if fail < len(lines):
        result = _lp3(lines, fail, max_speed, result)
    return result


def step(pos, vel, pref, max_speed, k, dt=1.0 / 30.0, reach=REACH):
    """One synchronous (Jacobi) ORCA step for every agent: new velocities from the snapshot."""
    out = np.empty_like(vel)
    for i in range(len(pos)):
        nb = neighbours(pos, i, k, reach)
        out[i] = new_velocity(pos[i], vel[i], pref[i], max_speed[i], pos[nb], vel[nb], dt=dt)
    return out
