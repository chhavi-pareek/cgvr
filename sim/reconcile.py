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

# Coupled reconciliation. Both draws below are free to pick any outcome consistent with the
# reference law; the default implementation picks one *independently* of where the agent
# already is, which makes the restoration a visible teleport. Measured on plaza: position
# jump p95 0.566 m and max 1.82 m -- the latter being exactly the widest lateral slot --
# against an ordinary per-frame movement of 0.071 m, and the latent redrawn on 98.1% of
# promotions. Since 83.6% of reconciliations happen on camera (bench/anticipate.py), that
# discontinuity is delivered where it is seen.
#
# Coupling changes only the tie-break, never the law:
#   lateral  the slot search starts from the agent's CURRENT offset instead of the centreline.
#            Every slot in the list is equally valid; the list is a collision-avoidance device,
#            not a distributional one, so preferring the one the agent already occupies is free.
#   latent   keep d when its coarse region already equals the agent's current region. On
#            demotion the region is set FROM the live latent, so conditional on region_of(d)
#            == R the retained d is distributed as pi_ref(. | R, ctx) -- exactly what the
#            redraw would have sampled. The mixture is therefore pi_ref either way, and the
#            marginal is preserved exactly rather than approximately.
COUPLE = False


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
        if COUPLE and surrogate.coarse[d[i]] == region[i]:
            continue          # already a valid pi_ref(. | R, ctx) draw; keeping it is exact
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
        slots = LATERAL
        if COUPLE:
            # the agent's own current offset first, then the standard list as fallback
            cur = float(np.dot(a.pos[i] - p0[k], nrm[k]))
            cur = float(np.clip(cur, -max(LATERAL), max(LATERAL)))
            slots = (cur,) + LATERAL
        for off in slots:
            cand = p0[k] + off * nrm[k]
            dd = np.linalg.norm(a.pos[others] - cand, axis=1)
            if dd.size == 0 or dd.min() >= CLEAR:
                pos = cand
                break
        others[i] = True
        a.pos[i] = pos
    a.vel[idx] = t * (a.speed[idx] * speed_scale(dec))[:, None]
