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
    # (round-robin, which also serves everyone in turn, can stay under the cap in a short run once
    # a new person in a slot starts a clean ledger; visibility-first LOD does not)
    assert runs["view_lod"]["dmax"] > 10.0
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


def test_risk_control_bounds_the_under_charge_and_the_stretch_overflow():
    # conformal risk control: for decisions exchangeable with the calibration ones, the expected
    # under-charge is at most eps = delta * eta times the expected charge, and a ledger run to
    # cap / (1 + eta) sees its TRUE drift pass the cap in at most ~delta of stretches
    rng = np.random.default_rng(5)
    sur = Distilled()
    for c in rng.integers(0, N_CTX, 300):
        sur.observe(c, P[c])
    sur.fit()
    delta, eta, cap = 0.05, 0.25, 10.0
    d = DriftBound(novel=True, bound="crc", delta=delta, eta=eta)
    for c in rng.integers(0, N_CTX, 300):
        d.observe(c, P[c])
    for c in rng.integers(0, N_CTX, 300):
        d.observe(c, P[c], calib=True)
    d.refresh(sur)
    assert np.isfinite(d.q_hat)
    fresh = rng.integers(0, N_CTX, 20000)
    true, charged = kl(sur(fresh), P[fresh]), d(fresh)
    assert np.maximum(true - charged, 0).mean() <= delta * eta * charged.mean() + 0.01
    # stretches of i.i.d. decisions; one the ledger cannot absorb goes to the model and ends it
    over = stretches = 0
    L = D = 0.0
    n = 0
    for c, t in zip(charged, true):
        if L + c > cap / (1 + eta):
            if n:
                stretches += 1; over += D > cap
            L = D = 0.0; n = 0
            continue
        L += c; D += t; n += 1
    assert stretches > 500
    assert over / stretches <= delta + 0.02, over / stretches


def test_risk_control_is_the_worst_case_until_calibrated():
    d = DriftBound(novel=True, bound="crc", delta=0.05, eta=0.25)
    for c in range(40):
        d.observe(c, P[c])
    for c in range(50):
        d.observe(200 + c, P[200 + c], calib=True)    # 50 < 1 / eps = 80: no margin can exist
    sur = Distilled()
    sur.observe(np.arange(40), P[:40]); sur.fit()
    d.refresh(sur)
    assert np.all(d(np.arange(N_CTX)) == E_MAX)
