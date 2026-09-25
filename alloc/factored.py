"""Two saliences, one budget: the allocator for a view-dependent objective.

Every other allocator here maximises  sum_i s_i * quality(row_i)  with ONE salience per agent.
That weights an agent's geometry exactly as much as its behaviour, which is wrong in both
directions: behaviour is simulation state and diverges whether or not anyone is looking, while
geometry and animation are only ever *seen*, and how much they matter is the agent's projected
size -- the screen-space error LOD has always been driven by (Funkhouser & Sequin 1993). Priced
honestly (render time measured in the engine), a view-independent objective spreads mesh detail
evenly over the crowd, including agents behind the camera, and the result is the wrong LOD.

So the objective splits along the table's own seams:

    sum_i  a_i * q_state(b_i, n_i)  +  b_i * q_view(a_i, g_i)

with a_i the state salience (behaviour, navigation) and b_i the view salience (animation,
geometry). With a_i = b_i it is the old objective exactly.

The split costs nothing, because the table already factorises. `alloc.config.allowed` couples
behaviour only with navigation (field navigation cannot carry gestures) and animation only with
geometry (no IK under an impostor), row cost is additive per axis, and the error rate lives on
the state axes. So the 180 rows are exactly 12 state pairs x 15 view pairs, and for a given
multiplier lam each agent's best row is its best state pair plus its best view pair, chosen
independently. Each half is a shared menu at shared prices -- the rank-one case alloc/hull.py
solves by sorting -- so the whole problem is two hull problems summed under one multiplier.

Two consequences worth stating:

* Dominance pruning on the *combined* quality (alloc.config.prune_dominated) is invalid here. A
  row with lower total quality but better view quality is the right choice for an agent in full
  view. Each hull prunes its own half instead, which is exactly the pruning that stays valid.
* The error mask acts on the state half only. Agents are grouped by feasible state set, as in
  alloc/hull.py, so the number of hulls is bounded by the number of error levels, not by n.
"""
import numpy as np

from alloc.config import AXIS_QUALITY, N_AXES, N_TIERS
from alloc.hull import upper_hull, hull_thetas, HullAllocator
from alloc.serial import Result

STATE_AXES = (0, 1)  # behaviour, navigation
VIEW_AXES = (2, 3)   # animation, geometry


def split_quality(table, axis_quality=None):
    """(q_state, q_view) per row, each the axis qualities / N_AXES, so q_state + q_view equals
    the table's mean quality. `axis_quality` replaces AXIS_QUALITY (the engine passes its
    measured geometry column); with the default, raises if the table is not the axis mean."""
    aq = AXIS_QUALITY if axis_quality is None else np.asarray(axis_quality, np.float64)
    t = table.tiers.astype(np.int64)
    q_s = sum(aq[ax, t[:, ax]] for ax in STATE_AXES) / N_AXES
    q_v = sum(aq[ax, t[:, ax]] for ax in VIEW_AXES) / N_AXES
    if axis_quality is None and not np.allclose(q_s + q_v, table.quality, atol=1e-12):
        raise ValueError("table quality is not the per-axis mean; cannot split it")
    return q_s, q_v


class Factorisation:
    """The table as (state pair) x (view pair), with costs split additively.

    `row[s, v]` is the table row of state pair s with view pair v; `state_of[r]`, `view_of[r]`
    invert it. Raises if the table is not the full product (a pruned table is not), if the cost
    vector is not additive over the two halves, or if the error rate depends on the view half."""

    def __init__(self, table, cost, atol=1e-9, axis_quality=None):
        t = table.tiers.astype(np.int64)
        skey = t[:, STATE_AXES[0]] * N_TIERS + t[:, STATE_AXES[1]]
        vkey = t[:, VIEW_AXES[0]] * N_TIERS + t[:, VIEW_AXES[1]]
        self.skeys, si = np.unique(skey, return_inverse=True)
        self.vkeys, vi = np.unique(vkey, return_inverse=True)
        S, V = len(self.skeys), len(self.vkeys)
        if S * V != table.m:
            raise ValueError(f"table has {table.m} rows, not the full {S} x {V} product; "
                             "the factored allocator needs the unpruned table")
        self.row = np.full((S, V), -1, np.int64)
        self.row[si, vi] = np.arange(table.m)
        if (self.row < 0).any():
            raise ValueError("table is not a product of state and view pairs")
        self.state_of, self.view_of = si.astype(np.int64), vi.astype(np.int64)

        q_s, q_v = split_quality(table, axis_quality)
        self.q_state = q_s[self.row[:, 0]]
        self.q_view = q_v[self.row[0, :]]
        cost = np.asarray(cost, np.float64)
        # Split against the CHEAPEST view pair, so a view cost is a non-negative increment over
        # the tier-3 floor and the state half carries the core. With one viewer any reference
        # would do (the halves add back to the row cost); with several, each viewer pays its own
        # increment, and a reference above the floor would make the second viewer's cost negative.
        ref = int(np.argmin(cost[self.row[0, :]]))
        self.c_state = cost[self.row[:, ref]]
        self.c_view = cost[self.row[0, :]] - cost[self.row[0, ref]]
        recon = self.c_state[:, None] + self.c_view[None, :]
        tol = atol + 1e-6 * np.abs(cost).max()   # fp32-rounded engine costs add up to ~1e-7
        if np.abs(recon - cost[self.row]).max() > tol:
            raise ValueError("row cost is not additive over the state and view halves")
        e = np.asarray(table.err, np.float64)
        self.e_state = e[self.row[:, 0]]
        if np.abs(e[self.row] - self.e_state[:, None]).max() > 1e-12:
            raise ValueError("error rate depends on the view half")


class _Half:
    """One shared-menu problem: agents sorted by salience, one hull per feasible group, plus a
    constant for agents whose choice is fixed (a view pair held by the pop ledger).

    With `prev` and `bonus` it is a switching-cost problem: each agent has one extra option, its
    previous choice, worth `bonus` more than the menu says. That option is agent-specific, so it
    cannot join the shared hull; instead every evaluation compares each agent's hull choice with
    its own previous choice -- O(n) per multiplier instead of O(K log n), still exact, and total
    cost is still non-increasing in the multiplier because each agent's choice is an argmax of
    value - lam * cost over a fixed option set."""

    def __init__(self, sal, q, c, groups, fixed_cost=0.0, prev=None, bonus=None):
        self.parts = []
        for idx, mask in groups:
            if len(idx) == 0:
                continue
            menu = np.arange(len(q)) if mask is None else np.flatnonzero(mask)
            h = menu[upper_hull(c[menu], q[menu])]
            order = idx[np.argsort(-sal[idx], kind="stable")]
            self.parts.append((order, sal[order], h, hull_thetas(c, q, h)))
        self.c, self.q, self.sal = c, q, sal
        self.fixed = float(fixed_cost)
        self.prev = prev if (prev is not None and bonus is not None and (bonus > 0).any()) else None
        self.bonus = bonus

    def _hull_pick(self, lam, pick):
        for order, ss, h, th in self.parts:
            bounds, verts = HullAllocator._blocks(lam, ss, th)
            for k in range(len(verts)):
                lo, hi = int(bounds[k]), int(bounds[k + 1])
                if hi > lo:
                    pick[order[lo:hi]] = h[verts[k]]
        return pick

    def _sticky(self, lam, pick):
        """Replace each agent's hull choice by its previous one where keeping it is worth more."""
        ids = np.concatenate([p[0] for p in self.parts]) if self.parts else np.zeros(0, np.int64)
        pv = self.prev[ids]
        ok = pv >= 0
        ids, pv = ids[ok], pv[ok]
        hv = pick[ids]
        s = self.sal[ids]
        keep = (s * self.q[pv] + self.bonus[ids] - lam * self.c[pv]) >= (s * self.q[hv] - lam * self.c[hv])
        pick[ids[keep]] = pv[keep]
        return pick

    def total(self, lam):
        if self.prev is None:
            t = self.fixed
            for order, ss, h, th in self.parts:
                bounds, verts = HullAllocator._blocks(lam, ss, th)
                t += float((np.diff(bounds) * self.c[h[verts]]).sum())
            return t
        pick = np.full(len(self.sal), -1, np.int64)
        self._sticky(lam, self._hull_pick(lam, pick))
        return self.fixed + float(self.c[pick[pick >= 0]].sum())

    def choose(self, lam, pick):
        self._hull_pick(lam, pick)
        if self.prev is not None:
            self._sticky(lam, pick)
        return pick


_FLOOR = 1e300   # a multiplier large enough that every agent takes its cheapest vertex


class FactoredAllocator:
    """allocate(state_salience, cost, budget, headroom, view_salience, view_lock) over the FULL
    table.

    * view_salience None: view = state salience, the single-salience objective of every other
      allocator here, solved as alloc/hull.py solves it.
    * view_salience (n,): one viewer. (V, n): V viewers sharing ONE simulation -- one state pair
      per agent (behaviour is simulated once) and one view pair per agent PER VIEWER (each
      viewer draws its own mesh). The objective and the cost sum over viewers, so the problem
      is 1 + V shared-menu halves under one multiplier; `assign` is then (V, n), viewer k's
      row for each agent.
    * view_prev + switch_cost: each agent's previous view pair per viewer (or -1), and a price
      on changing it, in the same units as quality, scaled by the agent's view salience -- so a
      change nobody sees is free and a change in full view must earn switch_cost more quality
      per unit salience than it costs. A preference in the objective, priced against the budget
      like everything else; the pop ledger remains the hard bound.
    * view_lock, same shape as view_salience: a view-pair index to hold, or -1. Held agents
      drop out of that viewer's half with a fixed cost. If the held costs alone make the budget
      unreachable, holds are released, most expensive first, and `self.released` counts them:
      the frame budget outranks the pop budget, as the error cap outranks the frame budget.
    """

    def __init__(self, table, rtol=1e-4, btol=0.0, max_iter=64, fill=True, axis_quality=None, warm=False):
        self.table = table
        self.warm = warm
        self.rtol, self.btol, self.max_iter = rtol, btol, max_iter
        self.axis_quality = axis_quality
        self.lam = None
        self.released = 0
        self._fac = None

    def factor(self, cost):
        cost = np.asarray(cost, np.float64)
        f = self._fac
        if f is None or not np.array_equal(f[0], cost):
            f = (cost.copy(), Factorisation(self.table, cost, axis_quality=self.axis_quality))
            self._fac = f
        return f[1]

    def allocate(self, salience, cost, budget, headroom=None, view_salience=None, view_lock=None,
                 view_prev=None, switch_cost=0.0):
        a = np.asarray(salience, np.float64)
        B = a if view_salience is None else np.asarray(view_salience, np.float64)
        multi = B.ndim == 2
        B = B if multi else B[None, :]
        V, n = B.shape
        if (a < 0).any() or (B < 0).any():
            raise ValueError("salience must be non-negative")
        L = np.full((V, n), -1, np.int64) if view_lock is None else np.asarray(view_lock, np.int64).reshape(V, n)
        cost = np.asarray(cost, np.float64)
        F = self.factor(cost)

        if headroom is None:
            sgroups = [(np.arange(n), None)]
        else:
            hr = np.asarray(headroom, np.float64)
            levels = np.unique(F.e_state)
            g = np.searchsorted(levels, hr, side="right")
            if (g == 0).any():
                raise ValueError("agent with no feasible configuration")
            sgroups = [(np.flatnonzero(g == gv), F.e_state <= levels[gv - 1] + 1e-12) for gv in np.unique(g)]
        state = _Half(a, F.q_state, F.c_state, sgroups)

        # the pop budget yields to the frame budget: release the costliest holds until the
        # cheapest reachable total fits
        L = L.copy()
        self.released = 0
        held = np.argwhere(L >= 0)
        if held.size:
            floor = state.total(_FLOOR) + sum(
                F.c_view[L[k][L[k] >= 0]].sum() + (L[k] < 0).sum() * F.c_view.min() for k in range(V))
            if floor > budget:
                extra = F.c_view[L[held[:, 0], held[:, 1]]] - F.c_view.min()
                for j in np.argsort(-extra, kind="stable"):
                    if floor <= budget:
                        break
                    k, i = held[j]
                    floor -= extra[j]
                    L[k, i] = -1
                    self.released += 1

        P = None if view_prev is None else np.asarray(view_prev, np.int64).reshape(V, n)
        views = [_Half(B[k], F.q_view, F.c_view, [(np.flatnonzero(L[k] < 0), None)],
                       fixed_cost=F.c_view[L[k][L[k] >= 0]].sum(),
                       prev=None if P is None else P[k], bonus=switch_cost * B[k])
                 for k in range(V)]

        def total(lam):
            return state.total(lam) + sum(h.total(lam) for h in views)

        # Warm start: the multiplier moves little between frames, so bracket around last frame's
        # instead of growing from 1. The C# port brackets identically, so the two stay bit-equal.
        # Stop once the spend at the feasible end is within btol of the budget (the serial
        # allocator's rule): the remaining multiplier interval can only move a sliver of budget.
        warm = self.lam if (self.warm and self.lam) else 0.0
        if total(0.0) <= budget:
            lam = 0.0
        else:
            if warm > 0.0:
                hi = warm
                t_hi = total(hi)
                while t_hi > budget and hi < 1e18:
                    hi *= 2.0
                    t_hi = total(hi)
                lo = hi * 0.5
                while lo > 1e-12:
                    t_lo = total(lo)
                    if t_lo > budget:
                        break
                    hi, t_hi, lo = lo, t_lo, lo * 0.5
            else:
                hi = 1.0
                t_hi = total(hi)
                while t_hi > budget and hi < 1e18:
                    hi *= 4.0
                    t_hi = total(hi)
                lo = 0.0
            for _ in range(self.max_iter):
                if hi - lo <= self.rtol * max(hi, 1e-12) or budget - t_hi <= self.btol * abs(budget):
                    break
                mid = 0.5 * (lo + hi)
                t_mid = total(mid)
                if t_mid <= budget:
                    hi, t_hi = mid, t_mid
                else:
                    lo = mid
            lam = hi
        self.lam = lam

        sp = state.choose(lam, np.empty(n, np.int64))
        vp = np.empty((V, n), np.int64)
        for k in range(V):
            vp[k] = np.where(L[k] >= 0, L[k], 0)
            views[k].choose(lam, vp[k])
        self.state_pair, self.view_pair = sp, vp
        rows = F.row[sp[None, :], vp]
        tot = float(F.c_state[sp].sum() + F.c_view[vp].sum())
        util = float((a * F.q_state[sp]).sum() + (B * F.q_view[vp]).sum())
        return Result(rows if multi else rows[0], tot, util, lam, self.max_iter,
                      tot > budget * (1 + 1e-9), 0)


def view_salience(dist, in_view, d0=10.0):
    """Projected-area salience for the visual axes: 1 inside d0 metres, (d0 / d)^2 beyond, and 0
    for an agent the camera cannot see. Area rather than height because the geometry quality it
    multiplies is a count of wrong pixels measured at d0 (the engine's pixel judge), and a
    figure's pixel count falls as 1 / d^2."""
    d = np.maximum(np.asarray(dist, np.float64), 1e-6)
    return np.where(np.asarray(in_view, bool), np.minimum(1.0, (d0 / d) ** 2), 0.0)
