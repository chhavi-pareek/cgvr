"""PARITY for LLM-driven agents, on a synthetic policy table: no model needed."""
import numpy as np
import pytest

from llm.policy import Distilled, Marginal, kl
from llm.schedule import E_MAX, DriftBound, Run
from llm.station import N_ACT, N_CTX, ctx_index, ctx_parts


def synthetic_policy(seed=0, peak=6.0):
    """A context-dependent policy with interactions the additive surrogate cannot fully capture."""
    rng = np.random.default_rng(seed)
    p, t, k, q, z = ctx_parts(np.arange(N_CTX))
    logit = (rng.normal(0, 1, (6, N_ACT))[p] + rng.normal(0, 1, (4, N_ACT))[t] + rng.normal(0, 1, (2, N_ACT))[k]
             + rng.normal(0, 1, (3, N_ACT))[q] + rng.normal(0, 1, (4, N_ACT))[z]
             + 1.5 * rng.normal(0, 1, (N_CTX, N_ACT)))
    P = np.exp(peak / 3.0 * logit)
    P = np.maximum(P / P.sum(1, keepdims=True), 1e-4)
    return P / P.sum(1, keepdims=True)


P = synthetic_policy()
LAT = np.full(50, 0.7)          # a 7B-sized call


def test_context_index_round_trips():
    c = np.arange(N_CTX)
    assert (ctx_index(*ctx_parts(c)) == c).all()


def test_distilled_surrogate_generalises_better_than_the_marginal():
    rng = np.random.default_rng(0)
    train = rng.choice(N_CTX, 150, replace=False)
    test = np.setdiff1d(np.arange(N_CTX), train)
    m, d = Marginal(), Distilled()
    m.observe(train, P[train]); d.observe(train, P[train]); d.fit(iters=600)
    assert kl(d(test), P[test]).mean() < 0.7 * kl(m(test), P[test]).mean()


@pytest.mark.parametrize("seed", [0, 1])
def test_parity_keeps_every_agent_under_the_cap_and_lod_does_not(seed):
    runs = {pol: Run(300, pol, P, LAT, seed=seed, cap=10.0).run(900).summary()
            for pol in ("parity", "view_lod", "round_robin")}
    par = runs["parity"]
    # the ledger charges a bound learned from replies, not the truth; it must still hold the truth
    assert par["over_cap_frac"] == 0.0 and par["dmax"] <= 10.0 + 1e-9
    assert runs["view_lod"]["dmax"] > 10.0 and runs["round_robin"]["dmax"] > 10.0
    # and it spends the model where it matters: less total drift per agent-hour at the same budget
    assert par["kl_per_agent_h"] < runs["view_lod"]["kl_per_agent_h"]


def test_the_call_budget_is_respected_except_for_mandatory_requests():
    r = Run(300, "parity", P, LAT, seed=3, cap=10.0).run(600)
    s = r.summary()
    budget = 0.9 / 0.7
    # calls beyond the budget are only ever the ones the cap forced
    assert s["calls_per_s"] <= budget + s["overrun"] / 600 + 0.05


def test_reference_is_the_llm_everywhere():
    s = Run(100, "reference", P, LAT, seed=0).run(300).summary()
    assert s["llm_frac"] == 1.0 and s["kl_per_agent_h"] == 0.0


def test_conformal_charge_covers_fresh_contexts_at_its_level():
    # split conformal: replies at random contexts calibrate the margin, and the charge must cover
    # the true drift of at least ~1 - alpha of fresh random contexts the predictor never saw
    rng = np.random.default_rng(3)
    sur = Distilled()
    for c in rng.integers(0, N_CTX, 300):
        sur.observe(c, P[c])
    sur.fit()
    for alpha in (0.2, 0.1):
        d = DriftBound(novel=True, bound="conformal", alpha=alpha)
        for c in rng.integers(0, N_CTX, 300):
            d.observe(c, P[c])
        for c in rng.integers(0, N_CTX, 200):
            d.observe(c, P[c], calib=True)
        d.refresh(sur)
        assert np.isfinite(d.q_hat)
        fresh = rng.integers(0, N_CTX, 4000)
        cover = np.mean(kl(sur(fresh), P[fresh]) <= d(fresh) + 1e-12)
        assert cover >= 1 - alpha - 0.03, (alpha, cover)


def test_conformal_charge_is_the_worst_case_until_calibrated():
    d = DriftBound(novel=True, bound="conformal", alpha=0.1)
    for c in range(20):
        d.observe(c, P[c])
    for c in range(5):
        d.observe(100 + c, P[100 + c], calib=True)      # 5 < 1 / 0.1 - 1: no quantile yet
    sur = Distilled()
    sur.observe(np.arange(20), P[:20]); sur.fit()
    d.refresh(sur)
    assert np.all(d(np.arange(N_CTX)) == E_MAX)
