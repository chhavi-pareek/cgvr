"""Reconciliation on promotion out of the surrogate tier.

Given the surrogate's region trajectory (current region R, context, distance the core has
carried the agent since demotion), reconstruct the fine behaviour state so that it is a sample
from the reference law conditioned on the coarse history:
  intent   d ~ pi_ref(d | R, ctx)             (corpus point inside the surrogate's region)
  gait     phase = phase_at_demotion + core distance / stride(decoded gait)   (mod 1)
  social   position = route point at core progress + nearest neighbour-free lateral slot,
           velocity = route tangent x nominal speed x decoded speed scale
On demotion into the surrogate the region is set from the live latent, so the chain continues
from the true state.
"""
import numpy as np

from .behaviour import speed_scale, stride

LATERAL = (0.0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8)
CLEAR = 0.7  # metres of clearance for a lateral slot to count as free


def demote(idx, d, region, surrogate, core, phase, dist_at_demote):
    region[idx] = surrogate.region_of(d[idx])
    dist_at_demote[idx] = core.s[idx]
    return region


def promote(idx, d, region, surrogate, core, a, phase, dist_at_demote, rng):
    idx = np.atleast_1d(idx)
    if idx.size == 0:
        return
    # intent continuity: corpus point consistent with the surrogate's region and context
    for i in idx:
        w = surrogate.pi_d[core.ctx[i]] * (surrogate.coarse == region[i])
        d[i] = rng.choice(len(w), p=w / w.sum())
    dec = surrogate.proc.decode(d[idx], 16)
    # gait phase from distance travelled under the core, not from elapsed time
    travelled = np.maximum(core.s[idx] - dist_at_demote[idx], 0.0)
    phase[idx] = (phase[idx] + travelled / stride(dec)) % 1.0
    # social context: nearest free lateral slot around the core point, tangent velocity
    p0 = core.point(core.s[idx], idx)
    t = core.tangent(idx)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    others = np.ones(a.n, bool)
    for k, i in enumerate(idx):
        others[i] = False
        pos = p0[k]
        for off in LATERAL:
            cand = p0[k] + off * nrm[k]
            dd = np.linalg.norm(a.pos[others] - cand, axis=1)
            if dd.size == 0 or dd.min() >= CLEAR:
                pos = cand
                break
        others[i] = True
        a.pos[i] = pos
    a.vel[idx] = t * (a.speed[idx] * speed_scale(dec))[:, None]
