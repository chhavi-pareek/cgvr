"""Pruned per-agent configuration table.

Four axes, four tiers each (0 = highest fidelity, 3 = lowest). Behaviour and
animation quality are 1 - held-out normalised MSE from the phase 2 nested
truncation study (mean of 10 seeds), so those two axes share a unit through the
latent. Navigation and geometry fidelities are placeholders until measured
(see STATE.md OPEN QUESTIONS).
"""
import itertools

import numpy as np

AXES = ("behaviour", "navigation", "animation", "geometry")
TIERS = (
    ("latent16", "latent8", "latent4", "core"),
    ("orca", "orca_sparse", "field", "core"),
    ("skeletal_ik", "skeletal", "vat", "none"),
    ("high", "mid", "low", "impostor"),
)
N_AXES = 4
N_TIERS = 4

# phase 2 nested truncation, held-out normalised MSE, k = 16 / 8 / 4
_BEH_NMSE = (0.132, 0.213, 0.449, 1.0)
_ANIM_NMSE = (0.196, 0.435, 0.746, 1.0)

AXIS_QUALITY = np.array(
    [
        [1.0 - e for e in _BEH_NMSE],
        [1.0, 0.9, 0.6, 0.3],  # placeholder: trajectory deviation vs full ORCA
        [1.0 - e for e in _ANIM_NMSE],
        [1.0, 0.75, 0.5, 0.25],  # placeholder: screen-space geometric error
    ],
    dtype=np.float64,
)

# per-frame state-divergence rate; visual axes do not diverge simulation state
AXIS_ERR = np.array(
    [
        list(_BEH_NMSE),
        [0.0, 0.05, 0.2, 0.4],  # placeholder: measured nav deviation
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

GESTURE_BEARING = (0, 1)  # behaviour tiers whose decode includes gesture weights
FIELD_NAVIGATED = (2, 3)
IMPOSTOR = 3
FULL_IK = 0


def allowed(t):
    b, n, a, g = t
    if g == IMPOSTOR and a == FULL_IK:
        return False
    if n in FIELD_NAVIGATED and b in GESTURE_BEARING:
        return False
    return True


class ConfigTable:
    def __init__(self, tiers, quality, err):
        self.tiers = np.asarray(tiers, np.int8)
        self.quality = np.asarray(quality, np.float64)
        self.err = np.asarray(err, np.float64)
        self.m = len(self.tiers)

    def name(self, c):
        return "/".join(TIERS[ax][self.tiers[c, ax]] for ax in range(N_AXES))

    def counts(self, assign):
        """(N_AXES, N_TIERS) histogram of agents per axis-tier for an assignment."""
        h = np.zeros((N_AXES, N_TIERS), np.int64)
        t = self.tiers[assign]
        for ax in range(N_AXES):
            h[ax] = np.bincount(t[:, ax], minlength=N_TIERS)
        return h

    def subset(self, idx):
        return ConfigTable(self.tiers[idx], self.quality[idx], self.err[idx])


def build_table():
    rows = [t for t in itertools.product(range(N_TIERS), repeat=N_AXES) if allowed(t)]
    tiers = np.array(rows, np.int8)
    ax = np.arange(N_AXES)
    quality = AXIS_QUALITY[ax, tiers].mean(1)
    err = AXIS_ERR[ax, tiers].sum(1)
    return ConfigTable(tiers, quality, err)


def prune_dominated(table, cost):
    """Drop rows beaten on quality, cost and error by another row (one strict)."""
    q, e, c = table.quality, table.err, np.asarray(cost)
    keep = np.ones(table.m, bool)
    for i in range(table.m):
        dom = (q >= q[i]) & (c <= c[i]) & (e <= e[i]) & ((q > q[i]) | (c < c[i]) | (e < e[i]))
        if dom.any():
            keep[i] = False
    return np.flatnonzero(keep)
