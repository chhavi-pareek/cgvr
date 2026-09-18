"""Full-fidelity behaviour reference: a context-conditioned jump process over corpus latents.

State per agent is a corpus index d; z = z(d) exactly (nested AE seed 0). Each frame the
agent stays with probability 1 - h(ctx) or jumps to d' with probability proportional to
exp(-|z_d - z_d'|^2 / 2 tau^2) * compat(d', ctx). Tiers 0-2 run this process and decode z
truncated to 16 / 8 / 4 dims; tier 3 replaces it by the surrogate (sim/surrogate.py).
"""
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(__file__))
_CACHE = os.path.join(_ROOT, "latent", "cache", "behaviour_s0.npz")
_ZCACHE = os.path.join(_ROOT, "latent", "cache", "z_nested_s0.npy")
_CORPUS = os.path.join(_ROOT, "latent", "data", "corpus.jsonl")

M = 1000  # corpus subset used as the jump-process state space
K_TIERS = (16, 8, 4)
CTX_NAMES = ("queued", "walking", "near_goal", "chokepoint")
QUEUED, WALKING, NEAR_GOAL, CHOKEPOINT = range(4)
N_CTX = 4
# jump hazard per frame by context (dwell 4 s / 3 s / 1.5 s / 1.5 s)
HAZARD = np.array([1 / 120, 1 / 90, 1 / 45, 1 / 45])
# intent ids (latent/corpus.py INTENT order) favoured by each context
_CTX_INTENTS = {
    QUEUED: (4, 5, 6),  # waiting in line, lingering, browsing
    WALKING: (0, 1, 2, 3, 12, 13, 14),
    NEAR_GOAL: (7, 10, 9, 14),  # approaching, stopping, turning back, scanning
    CHOKEPOINT: (8, 15, 2, 3),  # veering, dashing through a gap, hurrying, jogging
}
COMPAT_BOOST = 4.0
DEC_DIM = 19  # vel 3 + gaze 3 + gesture 8 + gait 5


def _build():
    import torch

    from latent.train import get_data, load_model

    m, order, fill = load_model(0, True)
    x = get_data()[0]
    with torch.no_grad():
        z_all = m.enc(x)
        decs = {}
        for k in K_TIERS:
            zk = fill.expand_as(z_all).clone()
            zk[:, order[:k]] = z_all[:, order[:k]]
            out = m.decode(zk)
            decs[f"dec{k}"] = torch.cat([out[g] for g in ("vel", "gaze", "gesture", "gait")], 1).numpy()
    z = z_all[:, order].numpy()
    with open(_CORPUS) as f:
        ids = np.array([json.loads(line)["ids"] for line in f], np.int32)
    np.savez(_CACHE, z=z, ids=ids, **decs)


def load_tables():
    if not os.path.exists(_CACHE):
        _build()
    t = np.load(_CACHE)
    return {k: t[k] for k in t.files}


class BehaviourProcess:
    """Reference jump process on an M-point corpus subset; cumulative kernel rows per context."""

    def __init__(self, seed=0):
        t = load_tables()
        rng = np.random.default_rng(1000 + seed)
        self.sub = np.sort(rng.choice(len(t["z"]), M, replace=False))
        self.z = t["z"][self.sub].astype(np.float64)
        self.intent = t["ids"][self.sub, 0]
        self.dec = {k: t[f"dec{k}"][self.sub].astype(np.float64) for k in K_TIERS}
        d2 = ((self.z[:, None, :] - self.z[None, :, :]) ** 2).sum(-1)
        nn = np.sqrt(np.sort(d2, 1)[:, 1])
        self.tau = float(np.median(nn))  # local jumps: kernel width = median nearest-neighbour gap
        kern = np.exp(-d2 / (2 * self.tau**2))
        np.fill_diagonal(kern, 0.0)
        self.cum = np.empty((N_CTX, M, M))
        self.kernel = np.empty((N_CTX, M, M))
        for c in range(N_CTX):
            w = np.where(np.isin(self.intent, _CTX_INTENTS[c]), 1.0 + COMPAT_BOOST, 1.0)
            p = kern * w[None, :]
            p /= p.sum(1, keepdims=True)
            self.kernel[c] = p
            self.cum[c] = np.cumsum(p, 1)

    def init(self, n, ctx, rng):
        """Draw initial states from the context-compatible marginal."""
        d = np.empty(n, np.int32)
        for c in range(N_CTX):
            idx = np.flatnonzero(ctx == c)
            if idx.size:
                w = np.where(np.isin(self.intent, _CTX_INTENTS[c]), 1.0 + COMPAT_BOOST, 1.0)
                d[idx] = rng.choice(M, idx.size, p=w / w.sum())
        return d

    def step(self, d, ctx, active, rng):
        """One frame for agents in `active` (index array). Returns the updated d (in place)."""
        if active.size == 0:
            return d
        u = rng.random(active.size)
        jump = active[u < HAZARD[ctx[active]]]
        if jump.size:
            v = rng.random(jump.size)
            for k, i in enumerate(jump):
                d[i] = min(int(np.searchsorted(self.cum[ctx[i], d[i]], v[k])), M - 1)
        return d

    def decode(self, d, k):
        return self.dec[k][d]


def speed_scale(dec):
    """Preferred-velocity head: tanh-squashed vel[0] -> multiplicative speed scale in [0.6, 1.4]."""
    return 1.0 + 0.4 * np.clip(dec[..., 0], -1, 1)


def lateral_bias(dec):
    return 0.5 * np.clip(dec[..., 1], -1, 1)


def stride(dec):
    """Gait head: stride length in metres."""
    return 0.7 + 0.2 * np.clip(dec[..., 14], -1, 1)
