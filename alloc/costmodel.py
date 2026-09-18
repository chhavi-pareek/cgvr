"""Recursive-least-squares fit of per-tier cost from measured frame time.

Model: frame_ms = theta_core * N + sum_{axis, tier<3} theta[axis, tier] * n[axis, tier].
Tier 3 (lowest) of every axis is the reference level with zero incremental
cost, so the intercept-per-agent theta_core is identifiable and equals the
fixed cost of the view-invariant core plus whatever the lowest tiers cost.
"""
import numpy as np

from alloc.config import N_AXES, N_TIERS

N_FEAT = 1 + N_AXES * (N_TIERS - 1)


def features(counts):
    counts = np.asarray(counts, np.float64)
    return np.concatenate([[counts[0].sum()], counts[:, : N_TIERS - 1].ravel()])


class RLSCostModel:
    def __init__(self, forget=0.98, p0=1.0, p_max=1e4):
        self.forget = forget
        self.p_max = p_max
        self.theta = np.zeros(N_FEAT)
        self.P = np.eye(N_FEAT) * p0
        self.n = 0
        self.resid_var = 0.0

    def predict(self, counts):
        return float(features(counts) @ self.theta)

    def update(self, counts, frame_ms):
        x = features(counts)
        e = float(frame_ms) - x @ self.theta
        Px = self.P @ x
        k = Px / (self.forget + x @ Px)
        self.theta = self.theta + k * e  # unclamped: clamping here destabilises RLS
        self.P = (self.P - np.outer(k, Px)) / self.forget
        tr = np.trace(self.P)
        if tr > self.p_max * N_FEAT:
            self.P *= self.p_max * N_FEAT / tr
        self.n += 1
        a = 0.05 if self.n > 20 else 1.0 / self.n
        self.resid_var += a * (e * e - self.resid_var)
        return e

    @property
    def theta_axis(self):
        """(N_AXES, N_TIERS) incremental cost per agent, tier 3 = 0, clamped >= 0."""
        t = np.zeros((N_AXES, N_TIERS))
        t[:, : N_TIERS - 1] = np.maximum(self.theta[1:], 0.0).reshape(N_AXES, N_TIERS - 1)
        return t

    @property
    def theta_core(self):
        return max(float(self.theta[0]), 0.0)

    def row_costs(self, table):
        """Per-row predicted ms per agent."""
        ax = np.arange(N_AXES)
        return self.theta_core + self.theta_axis[ax, table.tiers].sum(1)


def row_costs_from_theta(table, theta_core, theta_axis):
    ax = np.arange(N_AXES)
    return theta_core + np.asarray(theta_axis)[ax, table.tiers].sum(1)
