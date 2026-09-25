"""The ledger's drift rate is an estimate; its upper bound must cover the true rate."""
import numpy as np
import pytest

from sim.surrogate import Surrogate, K_FINE, N_CTX, _ctx_rows, kl_rows
from sim.tiered import Run


@pytest.fixture(scope="module")
def truth():
    """A surrogate fitted on one run, and a 'true' reference kernel from a long pooled run.
    Datasets are then drawn from the true kernel, so the true rate is known exactly."""
    r = Run("plaza", 200, "calib", seed=0).run(2400, log=False)
    sur = Surrogate(r.proc).fit(np.stack(r.ctx_log)[200:], np.stack(r.d_log)[200:])
    P_true = _ctx_rows(sur.Cf)
    part = sur.part
    lift = np.empty((N_CTX, K_FINE, K_FINE))
    for c in range(N_CTX):
        stay = sur.P_sur[c, part, part]
        lift[c] = sur.P_lift[c, part, :] - stay[:, None] * sur.pi_f_given_c[c, part, :] + stay[:, None] * np.eye(K_FINE)
    kc = np.einsum("cRr,cr->cR", sur.pi_f_given_c, kl_rows(lift, P_true))
    true_rate = np.einsum("cR,cR->c", sur.pi_c, kc)
    return sur, P_true, sur.Cf.sum(-1).astype(np.int64), true_rate


def draw(P, n_row, rng):
    C = np.zeros(P.shape)
    for c in range(N_CTX):
        for r in range(K_FINE):
            if n_row[c, r]:
                C[c, r] = rng.multinomial(n_row[c, r], P[c, r])
    return C


def test_plug_in_rate_is_inflated_and_the_bound_still_covers(truth):
    sur, P_true, n_row, true_rate = truth
    rng = np.random.default_rng(1)
    busy = sur.ctx_freq > 0.01
    covered, plug_hi, upper_lt_plug = 0, 0, 0
    reps = 30
    for k in range(reps):
        b = sur.rate_bounds(draw(P_true, n_row, rng), delta=0.05, draws=60, rng=rng)
        covered += int((b["upper"][busy] >= true_rate[busy] * (1 - 1e-9)).all())
        plug_hi += int((b["plug"][busy] > true_rate[busy]).all())
        upper_lt_plug += int((b["upper"][busy] < b["plug"][busy]).all())
    # Jensen: the plug-in overestimates on (nearly) every draw
    assert plug_hi >= reps - 1
    # the bound is valid at the nominal level, allowing for Monte Carlo slack
    assert covered >= int(0.85 * reps)
    # and it is tighter than the plug-in it replaces
    assert upper_lt_plug >= int(0.8 * reps)
