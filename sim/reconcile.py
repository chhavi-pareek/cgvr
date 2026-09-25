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
    # intent continuity: corpus point consistent with the surrogate's region and context.
    # Tabulated inverse-CDF rather than a per-agent rng.choice over the whole corpus; same
    # law, O(log |R|) instead of O(M). See Surrogate.lift_tables.
    cand, cdf = surrogate.lift_tables()
    for i in idx:
        if COUPLE and surrogate.coarse[d[i]] == region[i]:
            continue          # already a valid pi_ref(. | R, ctx) draw; keeping it is exact
        R = int(region[i])
        c = cdf[int(core.ctx[i])][R]
        d[i] = cand[R][min(int(np.searchsorted(c, rng.random())), c.size - 1)]
    dec = surrogate.proc.decode(d[idx], 16)
    # gait phase from distance travelled under the core, not from elapsed time
    travelled = np.maximum(core.s[idx] - dist_at_demote[idx], 0.0)
    phase[idx] = (phase[idx] + travelled / stride(dec)) % 1.0
    # social context: nearest free lateral slot around the core point, tangent velocity
    p0 = core.point(core.s[idx], idx)
    t = core.tangent(idx)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    # Slot search. Two phases, because the cost here was never arithmetic -- it was numpy
    # dispatch: the original did one O(n) np.linalg.norm per candidate per agent, 140 separate
    # calls for a batch of 20, at 1-2 us of call overhead each (16.5 us/agent measured).
    #
    # Phase 1 tests clearance against the agents that are NOT reconciling. That set is fixed
    # for the whole call and order-independent, so it is one vectorised test per slot -- at
    # most 7 numpy calls total, regardless of batch size.
    # Phase 2 tests only against reconciling agents already placed in this call. That part IS
    # order-dependent (agent j must see where agent i < j was just put), so it stays
    # sequential, but it runs on a handful of points in plain Python and never touches numpy.
    static = np.ones(a.n, bool)
    static[idx] = False
    sp = a.pos[static]
    clear2 = CLEAR * CLEAR

    if COUPLE:
        cur = ((a.pos[idx] - p0) * nrm).sum(1)
        lim = max(LATERAL)
        offs = np.concatenate([np.clip(cur, -lim, lim)[:, None],
                               np.tile(LATERAL, (len(idx), 1))], 1)
    else:
        offs = np.tile(LATERAL, (len(idx), 1))

    # ok_static[:, si] is computed lazily, vectorised across ALL agents at once, the first
    # time any agent asks for slot si. The original recomputed `a.pos[others]` -- a boolean
    # mask reallocating a fresh array -- once per candidate per agent; that allocation, not
    # the arithmetic, was the 16.5 us. Evaluating every slot up front instead is worse: the
    # loop nearly always succeeds on the first slot, so eager evaluation does 7x the work.
    ok_static = np.zeros(offs.shape, bool)
    done_col = np.zeros(offs.shape[1], bool)

    def static_col(si):
        if not done_col[si]:
            if sp.size:
                c = p0 + offs[:, si, None] * nrm
                dx = c[:, None, :] - sp[None, :, :]
                ok_static[:, si] = (dx * dx).sum(-1).min(1) >= clear2
            else:
                ok_static[:, si] = True
            done_col[si] = True
        return ok_static[:, si]

    placed = np.empty((len(idx), 2), np.float64)
    for k, i in enumerate(idx):
        pos = p0[k]
        for si in range(offs.shape[1]):
            if not static_col(si)[k]:
                continue
            cx = p0[k, 0] + offs[k, si] * nrm[k, 0]
            cy = p0[k, 1] + offs[k, si] * nrm[k, 1]
            hit = False
            for j in range(k):                               # tiny; plain Python beats numpy
                ddx = placed[j, 0] - cx
                ddy = placed[j, 1] - cy
                if ddx * ddx + ddy * ddy < clear2:
                    hit = True
                    break
            if hit:
                continue
            pos = np.array((cx, cy))
            break
        placed[k] = pos
        a.pos[i] = pos
    a.vel[idx] = t * (a.speed[idx] * speed_scale(dec))[:, None]
