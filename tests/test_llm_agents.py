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
    m.observe(train, P[train]); m.fit(); d.observe(train, P[train]); d.fit(iters=600)
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


LAT15 = np.full(50, 0.175)      # a 1.5B-sized call, a quarter of the 7B one


def test_menu_uses_a_small_model_where_it_is_closer_than_the_surrogate():
    # a small model close to the reference everywhere: worth a quarter of a call, so PARITY
    # spends on it and, once past the opening burst, drifts less than with the reference and the
    # surrogate alone (over the whole run it does not yet: an LLM reply also trains the surrogate
    # and resets the ledger, which the per-decision choice does not value -- see STATE.md)
    P15 = 0.9 * P + 0.1 / N_ACT
    two = Run(300, "parity", P, LAT, seed=0, cap=10.0).run(900).summary()
    three = Run(300, "parity", P, LAT, seed=0, cap=10.0, small=(P15, LAT15)).run(900).summary()
    assert three["small_frac"] > 0.05
    assert three["kl_steady_per_agent_h"] < two["kl_steady_per_agent_h"]
    assert three["over_cap_frac"] == 0.0 and three["dmax"] <= 10.0 + 1e-9


def test_menu_stops_probing_a_small_model_that_never_helps():
    # a degenerate small model (always the first action): after the first probes show it is
    # rarely closer than the surrogate, probing stops, and it serves only the few situations
    # where the reference itself all but always picks that action
    P15 = np.full_like(P, 1e-4)
    P15[:, 0] = 1.0
    P15 /= P15.sum(1, keepdims=True)
    s = Run(300, "parity", P, LAT, seed=0, cap=10.0, small=(P15, LAT15)).run(900).summary()
    assert s["small_frac"] < 0.01 and s["probes"] <= 35
    assert s["over_cap_frac"] == 0.0 and s["dmax"] <= 10.0 + 1e-9


def museum_policy(seed=0):
    """Synthetic museum policy: calm people browse, alarmed people mostly head for an exit."""
    from llm import museum as M
    rng = np.random.default_rng(seed)
    p, a, x, g, z = M.ctx_parts(np.arange(M.N_CTX))
    logit = (rng.normal(0, 1, (6, N_ACT))[p] + rng.normal(0, 1, (3, N_ACT))[x] + rng.normal(0, 1, (3, N_ACT))[g]
             + rng.normal(0, 1, (4, N_ACT))[z] + 0.7 * rng.normal(0, 1, (M.N_CTX, N_ACT)))
    logit[:, M.A_BROWSE] += 3.0 * (a == 0)
    logit[:, M.A_MAIN] += 2.5 * (a > 0)
    logit[:, M.A_EMERG] += 2.0 * (a > 0)
    P = np.maximum(np.exp(logit) / np.exp(logit).sum(1, keepdims=True), 1e-4)
    return M, P / P.sum(1, keepdims=True)


def test_museum_contexts_round_trip_and_the_crowd_drains_after_the_alarm():
    M, PM = museum_policy()
    c = np.arange(M.N_CTX)
    assert (M.ctx_index(*M.ctx_parts(c)) == c).all()
    assert "smell smoke" in M.prompt(M.ctx_index(0, 2, 0, 0, 0)) and "no alarm" in M.prompt(0)
    r = Run(300, "reference", PM, LAT, seed=0, scenario=M).run(900)
    s = r.summary()
    # nobody new comes in after the alarm, everyone inside then is either out or still inside
    assert s["inside"] + len(r.st.out_t) == r.st.at_alarm
    assert np.isfinite(s["t50"]) and not s["t50"] > s["t90"]         # t90 is nan if 90% never got out
    # a departed slot never decides again and never counts as present
    gone = np.flatnonzero(~r.st.inside)
    assert gone.size > 0 and (r.st.busy_until[gone] == M.NEVER).all() and r.st.present() == s["inside"]


def test_rehearsal_buys_unseen_neighbours_and_serves_no_decision():
    M, PM = museum_policy()
    r = Run(1000, "parity", PM, LAT, seed=0, scenario=M, rehearse=0.25)
    asked = []
    submit = r.server.submit

    def spy(agent, ctx, step, model=0):
        if agent < 0:
            asked.append((ctx, ctx in r.drift.p))
        submit(agent, ctx, step, model)
    r.server.submit = spy
    r.run(600)
    s = r.summary()
    assert r.stats["rehearsals"] == len(asked) > 0
    assert not any(seen for _, seen in asked)             # only contexts nobody had asked about
    assert len({c for c, _ in asked}) == len(asked)       # each at most once
    # every rehearsal reply is learned, and the ledger still holds
    assert all(c in r.drift.p for c, _ in asked[:-5])
    assert s["dmax"] <= 10.0 + 1e-9
    # the share it takes from the budget is the share it was given
    assert len(asked) <= 0.25 * r.rate * 600 + 2


def test_the_cap_holds_with_a_surrogate_that_learns_between_refits():
    # the charges are exact only for the surrogate they were computed for; a surrogate that moved
    # with every reply (the marginal did) made them stale and true drift passed the cap
    s = Run(1000, "parity", P, LAT, seed=0, cap=10.0, surrogate="marginal").run(600).summary()
    assert s["dmax"] <= 10.0 + 1e-9 and s["overflow_frac"] == 0.0


def rotated_policy(seed=0):
    """Six answers per context, as if the options had been shown in six orders: the order-free
    reference is their mean."""
    rng = np.random.default_rng(seed)
    R = np.stack([P * np.exp(0.8 * rng.normal(0, 1, P.shape)) for _ in range(N_ACT)])
    R = np.maximum(R / R.sum(-1, keepdims=True), 1e-4)
    R = R / R.sum(-1, keepdims=True)
    return R, R.mean(0)


def test_order_free_charge_bounds_the_mixture_and_is_exact_once_all_orders_are_seen():
    R, M = rotated_policy()
    d = DriftBound(rotations=True)
    rng = np.random.default_rng(1)
    q = Distilled()
    q.observe(np.arange(0, N_CTX, 3), M[::3]); q.fit(iters=300)
    for c in rng.choice(N_CTX, 40, replace=False):
        qc = q(c)
        prev = E_MAX
        for r in rng.permutation(N_ACT):
            d.observe(c, R[r, c], rot=int(r))
            b = d.certify(c, qc)
            assert b >= float(kl(qc, M[c])[0]) - 1e-9     # never below the true drift
            assert b <= prev + 1e-9                       # every order seen tightens it
            prev = b
        assert b == pytest.approx(float(kl(qc, M[c])[0]), abs=1e-9)   # all six: exact


def test_parity_holds_the_cap_against_the_order_free_reference():
    R, M = rotated_policy()
    s = Run(1000, "parity", M, LAT, seed=0, cap=10.0, rotations=R).run(600).summary()
    assert s["dmax"] <= 10.0 + 1e-9 and s["overflow_frac"] == 0.0


def test_risk_charge_is_exact_where_seen_and_the_worst_case_until_calibrated():
    b = DriftBound(bound="risk")
    sur = Distilled()
    for c in range(0, N_CTX, 7):
        sur.observe(c, P[c]); b.observe(c, P[c])
    sur.fit(); b.refresh(sur)
    seen, unseen = 0, 1
    assert b.charge_at(seen, 0.1)[0] == pytest.approx(float(kl(sur(seen), P[seen])[0]))
    assert b.charge_at(unseen, 0.1)[0] == E_MAX                       # no residuals yet
    for c in range(1, N_CTX, 7)[:40]:
        b.residual(c, sur(c)[0], P[c])
    lo, hi = b.charge_at(unseen, 0.2)[0], b.charge_at(unseen, 0.05)[0]
    assert lo < E_MAX and lo <= hi                                    # a smaller risk costs more
    assert b.charge_at(unseen, 1 / 1000)[0] == E_MAX                  # 40 residuals cannot certify 0.1%


def test_risk_mode_calibrates_only_unseen_contexts_and_keeps_stretch_overflow_under_the_budget():
    M, PM = museum_policy()
    r = Run(1000, "parity", PM, LAT, seed=0, scenario=M, bound="risk", alpha=0.2, explore=0.3)
    s = r.run(900).summary()
    assert len(r.drift.res) > 20 and s["est"] > 0
    # per stretch: P(true drift > cap) <= alpha (union bound over alpha/2 + alpha/4 + ...)
    assert s["overflow_frac"] <= 0.2
    w = Run(1000, "parity", PM, LAT, seed=0, scenario=M).run(900).summary()
    assert s["stall_per_agent_h"] <= w["stall_per_agent_h"]           # never more waits than the worst case


def test_risk_mode_leaves_the_default_ledger_untouched():
    a = Run(300, "parity", P, LAT, seed=3).run(600).summary()
    b = Run(300, "parity", P, LAT, seed=3, bound="worst").run(600).summary()
    assert a["kl_per_agent_h"] == b["kl_per_agent_h"] and a["est"] == 0


def test_the_model_queue_serves_held_agents_first():
    from llm.schedule import Server
    s = Server(P, LAT, np.random.default_rng(0))
    for a in (1, 2, 3):
        s.submit(a, a, 0)
    s.prio = lambda agent, model: 1 if agent == 3 else 3
    assert [r[0] for r in s.poll(0)][0] == 3


def test_pruning_skips_replies_nobody_can_use_and_refunds_them():
    M, PM = museum_policy()
    kw = dict(scenario=M, bound="risk", alpha=0.2, explore=0.3)
    a = Run(1000, "parity", PM, LAT, seed=0, **kw).run(900)
    b = Run(1000, "parity", PM, LAT, seed=0, prune=True, **kw).run(900)
    assert b.server.dropped > 0 and b.late <= a.late
    assert b.summary()["overflow_frac"] <= 0.2


def test_the_calibration_share_is_zero_when_nothing_is_unseen():
    r = Run(300, "parity", P, LAT, seed=0, bound="risk", alpha=0.2, explore=0.5, cal_adapt=True)
    r.drift.p = {c: P[c] for c in range(N_CTX)}
    cand = np.arange(50)
    r._size_calibration(cand, cand % N_CTX, np.zeros(50, bool))
    assert r.explore_eff == 0.0


def test_order_free_residual_uses_a_bound_that_never_undercounts():
    R, PM = rotated_policy()
    b = DriftBound(bound="risk", rotations=True)
    q = np.full(N_ACT, 1 / N_ACT)
    for c in range(0, 60):
        for r in range(N_ACT):
            b.pred[c] = 0.0
            b.residual(c, q, R[r, c])
            assert b.res[-1] >= float(kl(q, PM[c])[0]) - 1e-9        # one order's bound >= the true drift


def test_a_rehearsal_is_asked_under_the_option_order_it_names():
    from llm.schedule import Server
    R, PM = rotated_policy()
    s = Server(PM, LAT, np.random.default_rng(0), rotations=R, rot_rng=np.random.default_rng(1))
    s.rot_req[7] = 4
    s.submit(-1, 7, 0)
    (agent, c, p, model, rot), = s.poll(0)
    assert rot == 4 and np.allclose(p, R[4, 7])


@pytest.mark.parametrize("name, base", [("station_mem", 576), ("museum_mem", 648)])
def test_memory_contexts_round_trip_and_the_prompt_names_the_memory(name, base):
    import importlib
    M = importlib.import_module(f"llm.{name}")
    c = np.arange(M.N_CTX)
    assert M.N_CTX == base * len(M.MEMORY) and (M.ctx_index(*M.ctx_parts(c)) == c).all()
    assert all(text in M.prompt(m) for m, text in enumerate(M.MEMORY))     # memory is the last factor


def test_station_memory_is_settled_by_its_errand_and_set_by_a_missed_train():
    from llm import station_mem as S
    from llm.station import A_FOOD, A_INFO
    w = S.World(8, np.random.default_rng(0))
    w.busy_until[:] = 10 ** 9
    w.mem[:5] = [S.M_HUNGRY, S.M_FRIEND, S.M_PLATFORM, S.M_SUITCASE, S.M_HUNGRY]
    w.apply(np.arange(5), [A_FOOD, A_INFO, A_INFO, A_FOOD, A_INFO], 0)
    assert w.mem[:5].tolist() == [S.M_NONE, S.M_NONE, S.M_NONE, S.M_SUITCASE, S.M_HUNGRY]
    w.mem[5], w.activity[5] = S.M_HUNGRY, A_FOOD                  # mid-errand: settled when it next decides
    assert w.next_state([5])[2][0] == S.M_NONE and w.context(0, [5], mem=[S.M_NONE])[0] % S.N_M == S.M_NONE
    w.mem[6], w.train_at[6], gen = S.M_NONE, 0, w.gen[6]
    w.tick(100)
    assert w.mem[6] == S.M_MISSED and w.gen[6] == gen


def test_museum_memory_bag_is_collected_with_the_coat():
    from llm import museum_mem as M
    w = M.World(4, np.random.default_rng(0))
    w.mem[:2] = M.M_BAG
    w.apply([0, 1], [M.A_COAT, M.A_BROWSE], 0)
    assert w.mem[0] == M.M_NONE and w.mem[1] == M.M_BAG


@pytest.mark.parametrize("policy", ["parity", "round_robin", "cascade"])
def test_the_reply_cache_serves_answered_contexts_and_never_asks_for_them_again(policy):
    kw = dict(bound="risk", alpha=0.2, explore=0.5, prune=True, cal_adapt=True) if policy == "parity" else {}
    r = Run(1000, policy, P, LAT, seed=0, cache=True, **kw)
    again = []
    submit = r.server.submit

    def spy(agent, ctx, step, *a, **kw):
        if agent >= 0 and ctx in r.drift.p:
            again.append(ctx)
        submit(agent, ctx, step, *a, **kw)
    r.server.submit = spy
    r.run(900)
    s = r.summary()
    assert s["cached_frac"] > 0.3 and not again


def test_the_cache_is_off_where_one_reply_is_not_the_reference():
    R, PM = rotated_policy()
    assert Run(100, "parity", P, LAT, seed=0, cache=True).cache
    assert not Run(100, "parity", PM, LAT, seed=0, cache=True, rotations=R).cache


def _residuals():
    q, p = np.full(N_ACT, 1 / N_ACT), np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    return q, p, float(kl(q, p)[0])     # residual(c, q, p, pred=d0 - r) appends exactly r


def test_aci_holds_the_long_run_miss_rate_at_its_level_when_the_residuals_drift():
    """Gibbs and Candes (2021): whatever the residuals do, the adapted level's average miss rate is
    within (1 - a_1 + gamma) / (gamma T) of the target; a fixed window falls behind a drift."""
    q, p, d0 = _residuals()
    rng = np.random.default_rng(0)
    g = 0.05
    fixed, adapt = DriftBound(bound="risk", alpha=0.2), DriftBound(bound="risk", alpha=0.2, aci=g)
    miss = []
    for t in range(4000):
        r = t / 100 + rng.normal()
        qf = float(fixed._quantile(0.1)[0])
        if np.isfinite(qf):
            miss.append(r > qf)
        fixed.residual(0, q, p, pred=d0 - r)
        adapt.residual(0, q, p, pred=d0 - r)
    T = len(adapt.aci_err)
    assert np.mean(miss) > 0.15
    assert T > 3900 and abs(np.mean(adapt.aci_err) - 0.1) <= (0.9 + g) / (g * T)


def test_the_aci_level_keeps_adapting_where_its_quantile_is_infinite():
    q, p, d0 = _residuals()
    b = DriftBound(bound="risk", alpha=0.2, aci=0.05, window=50)
    for _ in range(50):
        b.residual(0, q, p, pred=d0)
    b.a_t = 0.01                       # below 1 / (m + 1): an infinite quantile, which covers
    n = len(b.aci_err)
    b.residual(0, q, p, pred=d0)
    assert len(b.aci_err) == n + 1 and b.aci_err[-1] == 0.0 and b.a_t > 0.01


def test_fill_spends_the_idle_budget_on_unseen_contexts_and_stays_within_it():
    r = Run(1000, "round_robin", P, LAT, seed=0, cache=True, fill=True)
    asked = []
    submit = r.server.submit

    def spy(agent, ctx, step, *a, **kw):
        if agent < 0:
            asked.append((ctx, ctx in r.drift.p))
        submit(agent, ctx, step, *a, **kw)
    r.server.submit = spy
    r.run(600)
    s = r.summary()
    idle = Run(1000, "round_robin", P, LAT, seed=0, cache=True).run(600).summary()
    assert r.stats["fills"] == len(asked) > 0 and not any(seen for _, seen in asked)
    assert len({c for c, _ in asked}) == len(asked)
    assert idle["model_util"] < s["model_util"] <= 0.9 + 0.05            # the default budget: 90% busy



def test_spill_passes_the_calibration_budget_it_cannot_hold_to_fill_and_is_off_by_default():
    kw = dict(bound="risk", alpha=0.2, explore=0.5, prune=True, cache=True, fill=True, rehearse=0.2, max_wait=30)
    frozen = Run(1000, "parity", P, LAT, seed=0, **kw).run(600).summary()
    off = Run(1000, "parity", P, LAT, seed=0, spill=False, **kw).run(600).summary()
    r = Run(1000, "parity", P, LAT, seed=0, spill=True, **kw).run(600)
    on = r.summary()
    assert frozen == off                                                  # the frozen method, unchanged
    assert r.ctokens <= 3 * r.explore * r.rate + 1
    assert frozen["model_util"] < on["model_util"] <= 0.9 + 0.05 and r.stats["fills"] > 0


def test_aci_on_spent_charges_brings_their_miss_rate_to_the_level_under_selection():
    """Spent charges see residuals shifted from the calibration ones (admission selects): scored on
    the charges themselves once their contexts are answered, the level makes them miss at its rate."""
    q, p, d0 = _residuals()
    rng = np.random.default_rng(1)
    b = DriftBound(bound="risk", alpha=0.2, aci=0.05, aci_mode="spent")
    for _ in range(200):
        b.residual(0, q, p, pred=d0 - rng.normal())             # calibration residuals ~ N(0, 1)
    assert b.a_t == 0.1                                         # "spent" ignores calibration scores
    errs = []
    for _ in range(4000):
        err = rng.normal(1.0, 1.0) > float(b._quantile(b.a_t)[0])   # spent ones ~ N(1, 1)
        b.spent(err, 0.1)
        errs.append(err)
    assert abs(np.mean(errs) - 0.1) < 0.02                      # unadapted, they would miss ~39%


def test_debit_pays_a_held_agents_request_from_the_budget_and_is_off_by_default():
    M, PM = museum_policy()
    kw = dict(scenario=M, bound="risk", alpha=0.2, explore=0.3, prune=True, max_wait=30, fill=True, cap=1.0)  # many holds
    frozen = Run(1000, "parity", PM, LAT, seed=0, **kw).run(900)
    off = Run(1000, "parity", PM, LAT, seed=0, debit=False, **kw).run(900)
    r = Run(1000, "parity", PM, LAT, seed=0, debit=True, **kw).run(900)
    scalar = lambda d: {k: repr(v) for k, v in d.items() if np.isscalar(v)}       # nan == nan
    assert scalar(frozen.summary()) == scalar(off.summary()) and r.stats["overrun"] > 0   # agents were held
    assert r.stats["fills"] < frozen.stats["fills"]           # held requests now cost what fill would have bought
    assert r.summary()["model_util"] < frozen.summary()["model_util"]


def test_weighted_margin_reduces_to_the_split_conformal_quantile_and_follows_its_weights():
    rng = np.random.default_rng(0)
    b = DriftBound(bound="risk", alpha=0.2)
    for x in rng.normal(size=150):
        b.res.append(float(x)); b.res_c.append(int(rng.integers(N_CTX))); b.res_i.append(b.res_n); b.res_n += 1
    lv = np.array([0.1, 0.05, 0.01, 0.003, 0.0])
    c = np.arange(lv.size)
    assert np.array_equal(b._wquantile(c, lv), b._quantile(lv))            # equal weights: split conformal
    near, far = 0, N_CTX - 1                                                 # contexts differing in every factor
    loc = DriftBound(bound="risk", alpha=0.2, wtau=0.5)
    old = DriftBound(bound="risk", alpha=0.2, wdecay=0.98)
    for k in range(200):
        ctx, r = (near, 0.0) if k % 2 else (far, 5.0)
        for d, rr in ((loc, r), (old, 5.0 if k < 100 else 0.0)):
            d.res.append(rr); d.res_c.append(ctx); d.res_i.append(d.res_n); d.res_n += 1
    assert loc._wquantile([near], [0.1])[0] < loc._quantile([0.1])[0]      # nearby residuals dominate
    assert old._wquantile([near], [0.2])[0] < old._quantile([0.2])[0]      # recent residuals dominate


def test_revision_draw_is_the_models_distribution_and_changes_at_the_tv_rate():
    from llm.schedule import revise_draw
    rng = np.random.default_rng(0)
    for _ in range(3):
        q = rng.dirichlet(np.ones(N_ACT)); p = rng.dirichlet(np.ones(N_ACT) * 0.5)
        a = rng.choice(N_ACT, size=60000, p=q)
        b = np.array([revise_draw(rng, q, p, int(x)) for x in a])
        assert np.abs(np.bincount(b, minlength=N_ACT) / b.size - p).max() < 0.01
        assert abs((b != a).mean() - 0.5 * np.abs(q - p).sum()) < 0.01


def test_revision_removes_drift_it_revises_and_is_off_by_default():
    kw = dict(cache=True, fill=True, bound="risk", alpha=0.2, explore=0.5, prune=True, rehearse=0.2, max_wait=30, strict=True,
              spill=True, debit=True)
    off = Run(300, "parity", P, LAT, seed=3, **kw).run(300).summary()
    assert off["revised_per_agent_h"] == 0 and off["kl_revised_per_agent_h"] == 0
    for pol, extra in (("parity", {}), ("cascade", {}), ("parity", dict(rev_aware=True)), ("cascade", dict(rev_aware=True))):
        r = Run(300, pol, P, LAT, seed=3, revise=60, **(kw if pol == "parity" else dict(cache=True, fill=True)), **extra).run(300)
        s = r.summary()
        assert s["revised_per_agent_h"] > 0 and r.stats["kl"] >= -1e-9
        assert s["switched_per_agent_h"] <= s["revised_per_agent_h"]


def test_unapply_restores_what_the_action_set_and_keeps_what_the_world_changed_since():
    from llm import museum_mem
    st = museum_mem.World(50, np.random.default_rng(0))
    i = 3
    pre = st.snap(i)
    st.apply(np.array([i]), np.array([museum_mem.A_BROWSE]), 10)
    post = st.snap(i)
    st.persona[i] = (st.persona[i] + 1) % 6          # the world changes a field the action did not set
    st.unapply(i, pre, post, museum_mem.A_BROWSE, 10)
    for k in ("zone", "busy_until", "activity", "comp", "mem"):
        assert getattr(st, k)[i] == pre[k]
    assert st.persona[i] != pre["persona"]
