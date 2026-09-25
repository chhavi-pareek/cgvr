"""View-invariant state core (invariant 4).

Core (every agent, every frame, fixed cost, tier-independent): committed route (up to three
waypoints), exit choice, queue membership and slot, context, coarse progress s along the route
advanced at nominal speed x context-mean speed scale, and chokepoint passage serialised by a
capacity counter calibrated from a full-fidelity run. Egress is declared from s.

View-dependent (allocated): jump state / surrogate region and decoded modifier, fine position and
velocity, gait phase, gesture weights. Fine position is bound to the core by |s_fine - s| <= BAND.
"""
import numpy as np

from .behaviour import CHOKEPOINT, NEAR_GOAL, QUEUED, WALKING

BAND = 2.0  # metres of allowed fine-vs-core progress disagreement
BIND = True  # invariant 4; bench/invariant4.py ablates it
BIND_ITERS = 4  # v2: re-project after correcting, so a corner cannot leave the band; 1 = v1
NEAR = 5.0  # metres of route remaining that count as near_goal
CHOKE_AHEAD = 3.0  # metres before a chokepoint that count as chokepoint context
MAX_WP = 3


class Core:
    def __init__(self, n, mbar, kappa=np.inf):
        self.n = n
        self.mbar = np.asarray(mbar, np.float64)  # context-mean speed scale, from calibration
        self.kappa = float(kappa)  # chokepoint capacity, agents per frame
        self.tokens = 1.0
        self.wp = np.zeros((n, MAX_WP, 2), np.float64)
        self.seg_len = np.zeros((n, MAX_WP - 1), np.float64)
        self.L = np.zeros(n, np.float64)
        self.s = np.zeros(n, np.float64)
        self.choke_s = np.full(n, np.inf)
        self.queued = np.zeros(n, bool)
        self.ctx = np.full(n, WALKING, np.int32)
        self.waits = 0  # frames agents spent held at the chokepoint (diagnostic)

    def set_route(self, idx, pts, choke_x=None):
        """pts: (len(idx), MAX_WP, 2); unused trailing waypoints repeat the last point."""
        idx = np.atleast_1d(idx)
        pts = np.asarray(pts, np.float64).reshape(len(idx), MAX_WP, 2)
        self.wp[idx] = pts
        seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
        self.seg_len[idx] = seg
        self.L[idx] = seg.sum(1)
        self.s[idx] = 0.0
        self.choke_s[idx] = np.inf
        if choke_x is not None:
            cum = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(seg, 1)], 1)
            for k in range(MAX_WP - 1):
                x0, x1 = pts[:, k, 0], pts[:, k + 1, 0]
                hit = (x0 < choke_x) & (x1 >= choke_x) & (seg[:, k] > 0)
                f = np.where(hit, (choke_x - x0) / np.where(x1 != x0, x1 - x0, 1.0), 0.0)
                cs = cum[:, k] + f * seg[:, k]
                self.choke_s[idx] = np.where(hit & ~np.isfinite(self.choke_s[idx]), cs, self.choke_s[idx])

    def point(self, s, idx=None):
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        s = np.asarray(s, np.float64)
        wp, seg = self.wp[idx], self.seg_len[idx]
        out = wp[:, -1].copy()
        done = np.zeros(len(idx), bool)
        rem = s.copy()
        for k in range(MAX_WP - 1):
            here = ~done & (rem <= seg[:, k] + 1e-9) & (seg[:, k] > 0)
            f = np.where(seg[:, k] > 0, rem / np.maximum(seg[:, k], 1e-9), 0.0)[:, None]
            out = np.where(here[:, None], wp[:, k] + f * (wp[:, k + 1] - wp[:, k]), out)
            done |= here
            rem = rem - seg[:, k]
        return out

    def tangent(self, idx=None):
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        wp, seg = self.wp[idx], self.seg_len[idx]
        s = self.s[idx]
        k = np.where(s <= seg[:, 0] + 1e-9, 0, 1)
        k = np.where(seg[np.arange(len(idx)), k] > 0, k, 0)
        d = wp[np.arange(len(idx)), k + 1] - wp[np.arange(len(idx)), k]
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        return np.where(nrm > 1e-9, d / np.maximum(nrm, 1e-9), np.array([[1.0, 0.0]]))

    def project(self, pos, idx=None):
        """Arc-length of the nearest point on the route to pos."""
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        wp, seg = self.wp[idx], self.seg_len[idx]
        best_d = np.full(len(idx), np.inf)
        best_s = np.zeros(len(idx))
        cum = 0.0
        for k in range(MAX_WP - 1):
            a, b = wp[:, k], wp[:, k + 1]
            ab = b - a
            t = np.clip(np.einsum("ij,ij->i", pos - a, ab) / np.maximum(seg[:, k] ** 2, 1e-9), 0, 1)
            t = np.where(seg[:, k] > 0, t, 0.0)
            d = np.linalg.norm(pos - (a + t[:, None] * ab), axis=1)
            take = d < best_d
            best_s = np.where(take, cum + t * seg[:, k], best_s)
            best_d = np.where(take, d, best_d)
            cum = cum + seg[:, k]
        return best_s

    def update_ctx(self):
        rem = self.L - self.s
        to_choke = self.choke_s - self.s
        ctx = np.full(self.n, WALKING, np.int32)
        ctx[rem < NEAR] = NEAR_GOAL
        ctx[(to_choke >= 0) & (to_choke < CHOKE_AHEAD)] = CHOKEPOINT
        ctx[self.queued] = QUEUED
        self.ctx = ctx
        return ctx

    def step(self, speed, dt):
        """Advance coarse progress; returns the mask of agents that completed their route."""
        self.update_ctx()
        adv = np.where(self.queued, 0.0, speed * self.mbar[self.ctx] * dt)
        s_new = np.minimum(self.s + adv, self.L)
        crossing = (self.s < self.choke_s) & (s_new >= self.choke_s)
        if crossing.any():
            self.tokens = min(self.tokens + self.kappa, 1.0 + self.kappa)
            order = np.flatnonzero(crossing)[np.argsort(-self.s[crossing], kind="stable")]
            for i in order:
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                else:
                    s_new[i] = self.choke_s[i] - 1e-3
                    self.waits += 1
        else:
            self.tokens = min(self.tokens + self.kappa, 1.0 + self.kappa)
        self.s = s_new
        return (self.s >= self.L - 1e-9) & ~self.queued

    def bind(self, pos, idx=None):
        """Clamp fine positions into the progress band around the core (returns corrected pos).

        This is invariant 4 acting on agents at EVERY tier, not just the surrogate: the core is
        simulated at fixed cost and the fine simulation is never allowed to disagree with it by
        more than BAND. Set BIND = False to ablate it -- see bench/invariant4.py for what that
        costs, which is the reconciliation snap losing its bound."""
        if not BIND:
            return pos
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        pos = pos.copy()
        # The corrected position is the target point plus the agent's PERPENDICULAR offset from
        # the route. v1 translated by (point at target - point at projection), which keeps the
        # whole offset, including any component ALONG the route. For an interior projection that
        # component is zero, but past a route end (projection clamped to L) or across a corner it
        # is not, and the translated point stays outside the band -- the corridor's 5.9%
        # overshoot, and an agent 1 cm past its end-point that v1 could never pull back.
        # Dropping the along-route part and re-projecting converges within a couple of steps.
        for _ in range(BIND_ITERS):
            sf = self.project(pos, idx)
            err = sf - self.s[idx]
            over = np.abs(err) > BAND * (1.0 + 1e-9)
            if not over.any():
                break
            target = self.s[idx] + np.clip(err, -BAND, BAND)
            off = pos - self.point(sf, idx)
            if BIND_ITERS > 1:
                t = self._tangent_at(target, idx)
                off = off - (off * t).sum(1)[:, None] * t
            pos[over] = (self.point(target, idx) + off)[over]
        return pos

    def _tangent_at(self, s, idx):
        """Unit tangent of the route segment containing arc length s."""
        wp, seg = self.wp[idx], self.seg_len[idx]
        k = np.zeros(len(idx), np.int64)
        cum = seg[:, 0].copy()
        for j in range(1, MAX_WP - 1):         # past the end of segment j-1, onto a real segment j
            k = np.where((s > cum) & (seg[:, j] > 0), j, k)
            cum = cum + seg[:, j]
        rows = np.arange(len(idx))
        d = wp[rows, k + 1] - wp[rows, k]
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        return np.where(nrm > 0, d / np.maximum(nrm, 1e-12), np.array([1.0, 0.0]))


def partition():
    """Names of the core and view-dependent fields, for the record and for tests."""
    core = ("route", "exit_choice", "queued", "slot", "ctx", "s")
    view = ("d", "region", "dec", "pos", "vel", "gait_phase", "gesture")
    return core, view
