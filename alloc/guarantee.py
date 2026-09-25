"""Invariant 3 as a stated theorem, with machine-checked verification.

Everywhere else in this repo the divergence bound is an emergent property of code: the mask
is built from the headroom, the ledger is charged afterwards, and the assertions happen to
hold. That is not the same as a guarantee. This module states the bound, proves it, names the
assumption it rests on, and verifies it against adversarial schedules that the simulator would
never produce on its own.

-- Setup ---------------------------------------------------------------------------------

Agent i carries an accumulator D_i(t) >= 0 against a cap C > 0. At frame t the allocator is
handed a headroom h_i(t) = max(C - D_i(t), 0) and may select only from

    A_i(t) = { c : err(c) <= h_i(t) }                                   (the mask)

After the frame the accumulator is charged the *realised* rate of the configuration in the
context the agent was actually in,

    D_i(t+1) = D_i(t) + r(c_i(t), k_i(t)),   or 0 if agent i was reconciled.

-- The two assumptions -------------------------------------------------------------------

(A1) ADMISSION DOMINANCE.  r(c, k) <= err(c)  for every configuration c and every context k.

     In PARITY: err(c) = e_admit for behaviour-tier-3 rows and 0 otherwise, while
     r(c, k) = e_rate[k] for those rows and 0 otherwise, with e_admit = max_k e_rate[k].
     The table admits against the worst context; the ledger spends the actual one.

(A2) NON-EMPTY MASK.  There exists c0 with err(c0) = 0.

     In PARITY: every row whose behaviour tier is below 3 runs the live process and diverges
     by nothing, so rate-0 rows always exist and A_i(t) is never empty for any h_i(t) >= 0.

-- Theorem -------------------------------------------------------------------------------

    Under (A1) and (A2), if D_i(0) <= C then D_i(t) <= C for every agent i and every frame t.

Proof by induction on t. The base case is the hypothesis. Assume D_i(t) <= C. Then
h_i(t) = C - D_i(t) >= 0, so by (A2) the mask is non-empty and some c_i(t) is selected from
it. The mask gives err(c_i(t)) <= h_i(t), and (A1) gives r(c_i(t), k) <= err(c_i(t)). Hence

    D_i(t+1) = D_i(t) + r(c_i(t), k_i(t)) <= D_i(t) + h_i(t) = C.

Reconciliation only sets D_i(t+1) = 0 <= C. []

The bound is TIGHT: an agent admitted at exactly h_i(t) whose realised rate equals the
admission rate lands on C exactly, so no smaller constant works.

-- Why the asymmetry is the inventive step -----------------------------------------------

The obvious design charges the mask what it charges the ledger. That requires knowing
k_i(t) -- the context the agent will be in during the frame -- at the moment of the decision,
which is not available: the allocator runs before the step. Admitting at the worst case makes
the bound hold using only information already in hand, and costs only the gap between
e_admit and the realised rate, which is spent capacity rather than a violated guarantee.

-- Corollary (deferred application) -------------------------------------------------------

If the configuration chosen against h_i(t) is not applied until frame t + L (an engine with
asynchronous readback: unity/PORT_SPEC.md section 4), the agent keeps accruing at its OLD
configuration for L frames. Reserving that in advance,

    h_i(t) = max(C - D_i(t) - L_max * e_admit, 0)

restores the theorem for any actual latency L <= L_max, by the same induction with the
inductive hypothesis D_i(t) <= C - L_max * e_admit + (frames already charged).
"""
import numpy as np

__all__ = ["admissible", "headroom", "step_ledger", "verify_schedule", "check_assumptions"]


def headroom(D, cap, latency=0, e_admit=0.0):
    """h_i = max(cap - D_i - L*e_admit, 0). The latency term is the corollary's margin."""
    return np.maximum(np.asarray(cap, np.float64) - np.asarray(D, np.float64)
                      - latency * e_admit, 0.0)


def admissible(err, h):
    """Boolean [n, m] mask: configuration j is admissible for agent i iff err[j] <= h[i]."""
    return np.asarray(err, np.float64)[None, :] <= np.asarray(h, np.float64)[:, None]


def step_ledger(D, chosen_rate, reconciled=None):
    """One charge. `reconciled` indices are zeroed instead, which is the only reset."""
    D = np.asarray(D, np.float64) + np.asarray(chosen_rate, np.float64)
    if reconciled is not None and len(reconciled):
        D[np.asarray(reconciled, np.int64)] = 0.0
    return D


def check_assumptions(err, rate):
    """(A1) and (A2) for a table. `rate` is [m, n_ctx] realised rates per configuration.

    Returns (ok, message). A1 must hold for the theorem; A2 must hold for the mask to be
    non-empty at zero headroom, which is what stops an agent from having no legal move."""
    err = np.asarray(err, np.float64)
    rate = np.asarray(rate, np.float64)
    if rate.ndim == 1:
        rate = rate[:, None]
    worst = rate.max(axis=1)
    bad = np.flatnonzero(worst > err + 1e-12)
    if bad.size:
        j = int(bad[0])
        return False, (f"(A1) violated: configuration {j} is admitted at err={err[j]:.6g} "
                       f"but can be charged {worst[j]:.6g}")
    if not np.any(err <= 0.0):
        return False, "(A2) violated: no rate-0 configuration, so the mask can empty"
    return True, "(A1) admission dominance and (A2) non-empty mask both hold"


def verify_schedule(err, rate, cap, ctx, pick, D0, latency=0, reconcile=None):
    """Replay a schedule and return (ok, max_D, first_violation_frame).

    `pick` chooses an admissible configuration per agent per frame -- including adversarially,
    which is the point: the theorem must hold for EVERY selection rule the mask allows, not
    just the one the allocator happens to use.
    """
    err = np.asarray(err, np.float64)
    rate = np.asarray(rate, np.float64)
    if rate.ndim == 1:
        rate = rate[:, None]
    e_admit = float(err.max())
    D = np.array(D0, np.float64)
    n, T = len(D), len(ctx)
    worst = 0.0
    for t in range(T):
        h = headroom(D, cap, latency, e_admit)
        mask = admissible(err, h)
        if not mask.any(axis=1).all():
            return False, float(D.max()), t          # (A2) failed: an agent has no legal move
        c = pick(t, mask, D, h)
        assert mask[np.arange(n), c].all(), "pick() returned an inadmissible configuration"
        D = step_ledger(D, rate[c, ctx[t]], None if reconcile is None else reconcile(t, D, cap))
        worst = max(worst, float(D.max()))
        if D.max() > cap + 1e-9:
            return False, worst, t
    return True, worst, -1


def safe_admission_rate(calib, occupancy_floor=1e-6):
    """The smallest admission rate that makes (A1) hold for the *measured* divergence.

    sim/surrogate.py builds e_rate as E_R[kl_coarse | ctx] -- an average over latent regions --
    but sim/tiered.py charges D_meas the agent's own kl_coarse[ctx, region]. Any region above
    its context mean therefore accrues faster than the mask reserved, so the cap bounds the
    ledger accumulator but NOT the quantity the figures plot. Measured shortfall: 1.8-2.3x
    across plaza, hub and corridor.

    Taking the maximum over reachable (context, region) pairs restores the theorem for the
    measured quantity. A one-step-reachable bound through P_sur would be tighter in principle,
    but the Metropolis-Hastings correction leaves the coarse kernel dense -- one step reaches
    every region -- so it is identical to the global maximum here and is not worth the
    machinery.
    """
    klc = np.asarray(calib["kl_coarse"], np.float64)
    pic = np.asarray(calib["pi_c"], np.float64)
    occ = pic > occupancy_floor
    if not occ.any():
        return float(klc.max())
    return float(klc[occ].max())


def admission_shortfall(calib, occupancy_floor=1e-6):
    """(current_rate, safe_rate, ratio) for the table's admission constant."""
    klc = np.asarray(calib["kl_coarse"], np.float64)
    pic = np.asarray(calib["pi_c"], np.float64)
    cf = np.asarray(calib["ctx_freq"], np.float64)
    e_rate = np.einsum("cR,cR->c", pic, klc)
    live = cf > 0.01
    current = float(e_rate[live].max()) if live.any() else float(e_rate.max())
    safe = safe_admission_rate(calib, occupancy_floor)
    return current, safe, safe / max(current, 1e-12)


# -- the companion bound, from invariant 4 rather than invariant 3 -------------------------
#
# Everything above bounds BEHAVIOURAL divergence. It says nothing about position, and position
# is where the visible artefact lives: a demoted agent's fine position is replaced by its core
# point, and the size of that replacement is exactly how far the fine simulation had been
# allowed to drift from the core.
#
# Invariant 4 bounds it. Core.bind clamps the fine position of agents at EVERY tier to within
# BAND of core progress, so:
#
#   THEOREM (positional discontinuity). Under Core.bind with band B and lateral clip L, on a
#   route whose progress values lie on a single segment, the position change at demotion into
#   the surrogate is at most B + L.
#
#   Proof. Demotion sets pos <- core.point(s) + n * clip(lateral, -L, L). The along-route
#   component of the change is |project(pos) - s|, which bind holds at <= B. The lateral
#   component is |lateral - clip(lateral, -L, L)|, which is 0 when |lateral| <= L and at most
#   |lateral| - L otherwise; the fine simulation's lateral excursion is itself bounded by the
#   scene's clearance. []
#
#   The single-segment condition is not cosmetic, and writing the proof is what surfaced it.
#   bind corrects by the rigid displacement point(target) - point(sf), which relocates the
#   agent to the right progress only when both values sit on the same segment; across a corner
#   the two route points are not colinear with the agent and one application leaves a residual.
#   Measured over 600 frames: plaza and hub hold the band at exactly 2.0000 m (their routes are
#   single-segment), while corridor -- the one scene with an intermediate waypoint -- reaches
#   2.1182 m, an overshoot of 5.9%. So the bound in general is B(1 + eps) + L with eps the
#   per-corner projection residual. Iterating the clamp to a fixed point, or clamping per
#   segment, would remove it; neither is applied here because it changes the core simulation
#   and would invalidate the recorded sweep.
#
# Measured (bench/invariant4.py), corridor, the scene with a real chokepoint:
#
#   bind on   fine-vs-core drift max  2.12 m    demote snap max   2.39 m
#   bind off  fine-vs-core drift max 20.30 m    demote snap max  20.69 m
#
# and the ledger's bound is untouched either way (5.402 vs 5.454 nats against a 5.484 cap),
# which is the point: these are two independent guarantees over two different quantities, and
# neither implies the other. The error ledger bounds what an agent DOES; the core bounds where
# it IS. Ablating invariant 4 leaves invariant 3 exactly intact and produces a 20 m teleport.
