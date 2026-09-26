"""PARITY for LLM-driven agents: ration the model, bound how far any agent drifts from it.

Every decision is served by the LLM or by a cheap surrogate. The LLM is slow (hundreds of ms a
call on one local model) and serial, so it can serve a small fraction of a crowd's decisions.
The question is which ones, and what the others are allowed to do meanwhile.

PARITY's answer is the one it gives for simulation fidelity, moved to decisions:
  * a per-agent ledger L_i accumulates the drift charged for every surrogate decision since the
    agent's last LLM decision (KL(surrogate || LLM) at that decision; by the chain rule their sum
    is the KL between the two trajectory laws over that stretch), and an LLM decision resets it;
  * a surrogate decision is admissible only if it keeps L_i under the cap -- otherwise the agent
    waits for the model (the cap outranks the call budget, as it outranks the frame budget);
  * replies take time, so requests are made AHEAD, for the decision each agent will make next,
    from its predicted context; an agent whose ledger could not absorb its next surrogate decision
    is requested first (mandatory), and the remaining budget goes to the agents the viewer can see
    whose surrogate would be most wrong (salience x expected drift -- the Lagrangian choice for a
    two-option menu at unit cost);
  * the drift charged per (persona, zone) class is learned from the replies themselves: every paid
    call shows what the surrogate would have said, so the scheduler never needs ground truth.

The benchmark measures the TRUE drift of every surrogate decision against the tabulated LLM
answer, so it can check whether the charged bound held.

Server: requests queue in order; the model serves one at a time, each taking a latency drawn from
the measured calls. A reply is the model's exact distribution for the prompt, which is
deterministic at temperature 0 -- so reading the table is exactly what a live call returns.
"""
from collections import deque

import numpy as np

from .policy import FLOOR, Distilled, Marginal, kl
from .station import N_ACT, N_CTX, N_Z, Station, Z_GATES, ctx_parts

VIEW_PERIOD = 300          # the viewer moves to another zone every five minutes
FAR_SALIENCE = 0.25


class Server:
    """One GPU serving requests in order. Model 0 is the reference (7B); model 1, when given, a
    smaller one (1.5B) whose calls take its own measured latencies."""

    def __init__(self, P, latency, rng, small=None):
        self.P = [P] + ([small[0]] if small else [])
        self.lat = [np.asarray(latency)] + ([np.asarray(small[1])] if small else [])
        self.rng = rng
        self.q = deque()
        self.busy_until = 0.0
        self.busy_s = 0.0                   # model-seconds used
        self.done = []
        self.calls = 0                      # reference-model calls
        self.small_calls = 0

    def submit(self, agent, ctx, step, model=0):
        self.q.append((int(agent), int(ctx), float(step), int(model)))
        if model == 0:
            self.calls += 1
        else:
            self.small_calls += 1

    def backlog(self, step):
        """Seconds until a request made now would be answered, on average."""
        return max(self.busy_until - step, 0.0) + sum(float(self.lat[m].mean()) for *_, m in self.q)

    def poll(self, step):
        """Replies complete by the end of this step, as (agent, ctx, p, model)."""
        out = []
        while self.q and max(self.busy_until, self.q[0][2]) <= step + 1:
            agent, ctx, t_req, model = self.q.popleft()
            start = max(self.busy_until, t_req)
            dt = float(self.rng.choice(self.lat[model]))
            self.busy_until = start + dt
            self.busy_s += dt
            self.done.append((self.busy_until, agent, ctx, model))
        keep = []
        for item in self.done:
            t, agent, ctx, model = item
            if t <= step + 1:
                out.append((agent, ctx, self.P[model][ctx], model))
            else:
                keep.append(item)
        self.done = keep
        return out


# The reference policy carries a probability floor (llm/policy.FLOOR), so no action is ever below
# FLOOR / (1 + N_ACT * FLOOR), and ONE surrogate decision can never drift more than this from it.
E_MAX = float(np.log((1 + N_ACT * FLOOR) / FLOOR))


class DriftBound:
    """The drift the ledger charges for a surrogate decision at context c -- never an estimate.

    Seen contexts: EXACTLY KL(surrogate || LLM) there. A paid reply for c reveals the LLM's answer
    at c, so the surrogate's drift at c is known (and recomputed whenever the surrogate is refitted).
    Unseen contexts: E_MAX, the largest drift one decision can have at all. An estimate (the largest
    drift seen in the class, even with a 50% margin) under-charged unseen contexts by ~0.8 nats a
    decision; the worst case cannot, so true drift <= charged drift <= cap holds deterministically.
    Charging the worst case also makes an unfamiliar situation the first thing worth paying for."""

    # novel=True emulates contexts that never repeat (memory or dialogue in the prompt): a paid
    # reply says nothing EXACT about any later decision, so the exact branch is unavailable and
    # every decision is charged one of
    #   worst      E_MAX: the deterministic guarantee, at one surrogate decision per LLM decision;
    #   plugin     the predicted drift, no margin: no guarantee at all;
    #   conformal  the predicted drift plus a split-conformal margin: P(true <= charged) >= 1 - alpha
    #              for a decision exchangeable with the calibration decisions. The prediction is a
    #              kernel average of drift over paid replies at OTHER contexts (Hamming distance on
    #              the five factors); the margin is the ceil((m+1)(1-alpha))-th smallest residual on
    #              m calibration replies bought for uniformly random decisions and kept out of the
    #              predictor. Too few of them to form the quantile and the charge is E_MAX, so the
    #              bound degrades to the worst case rather than to a guess.
    #   crc        the same prediction plus a margin from conformal risk control (Angelopoulos et
    #              al. 2022) on the SIZE of the under-charge, not its probability -- a ledger adds
    #              decisions up, and per-decision coverage says nothing about how far the misses
    #              go. The margin is the smallest lam with (m/(m+1)) R(lam) + E_MAX/(m+1) <= 0 for
    #              the loss (true - charged)+ - eps * charged (non-increasing in lam, at most
    #              E_MAX), so for an exchangeable decision E[(true - charged)+] <= eps E[charged].
    #              Over a stretch (a stopping time of i.i.d. decisions, by Wald) the expected
    #              under-charge is at most eps times what was charged, at most eps * cap'; with the
    #              ledger run to cap' = cap / (1 + eta), Markov bounds the chance that a stretch's
    #              TRUE drift passes the cap by eps / eta = delta. delta = 1 is the expectation
    #              version: a stretch's expected true drift is within the cap. The margin needs
    #              about E_MAX / (eps E[charge]) calibration replies to exist at all, so a small
    #              delta is only affordable when the model serves a large share of decisions.
    def __init__(self, novel=False, bound="worst", alpha=0.1, tau=0.5, delta=0.05, eta=0.25):
        self.p = {}                    # context -> LLM distribution, from paid replies
        self.exact = {}
        self.p15, self.exact15 = {}, {}  # the small model's answers, and its exact drift from the LLM
        self.novel, self.bound, self.alpha, self.tau = novel, bound, alpha, tau
        self.delta, self.eta = delta, eta
        self.train, self.cal = [], []  # (ctx, p) of paid replies; cal = the random ones
        self.charge = np.full(N_CTX, E_MAX)
        self.q_hat = np.inf
        if novel:
            parts = np.stack(ctx_parts(np.arange(N_CTX)), 1)
            h = (parts[:, None, :] != parts[None, :, :]).sum(2).astype(np.float64)
            self.K = np.exp(-h / tau)
            np.fill_diagonal(self.K, 0.0)            # a context never predicts itself

    def observe(self, c, p, calib=False):
        self.p[int(c)] = p
        (self.cal if calib else self.train).append((int(c), p))
        if int(c) in self.p15:
            self.exact15[int(c)] = float(kl(self.p15[int(c)], p)[0])

    def observe_small(self, c, q):
        self.p15[int(c)] = q
        if int(c) in self.p:
            self.exact15[int(c)] = float(kl(q, self.p[int(c)])[0])

    def small_charge(self, c):
        """A small-model decision at c: its drift from the LLM, exact once both have answered at c,
        else the worst case."""
        return np.array([self.exact15.get(int(x), E_MAX) for x in np.atleast_1d(c)])

    def refresh(self, sur):
        if not self.p:
            return
        c = np.fromiter(self.p.keys(), np.int64)
        k = kl(sur(c), np.vstack([self.p[int(x)] for x in c]))
        self.exact = dict(zip(c.tolist(), k.tolist()))
        if self.novel and self.bound != "worst":
            self._conformal(sur)

    def _conformal(self, sur):
        if not self.train:
            return
        tc = np.array([x for x, _ in self.train])
        td = kl(sur(tc), np.vstack([p for _, p in self.train]))
        tot, cnt = np.zeros(N_CTX), np.zeros(N_CTX)
        np.add.at(tot, tc, td); np.add.at(cnt, tc, 1.0)
        w = self.K @ cnt
        pred = np.where(w > 0, (self.K @ tot) / np.maximum(w, 1e-300), E_MAX)
        self.pred = pred
        if self.bound == "plugin":
            self.charge = np.clip(pred, 0.0, E_MAX)
            return
        m = len(self.cal)
        if self.bound == "crc":
            self._risk_control(sur, pred)
            return
        k = int(np.ceil((m + 1) * (1 - self.alpha)))
        if k > m:
            self.q_hat = np.inf
            self.charge = np.full(N_CTX, E_MAX)
            return
        cc = np.array([x for x, _ in self.cal])
        cd = kl(sur(cc), np.vstack([p for _, p in self.cal]))
        self.q_hat = float(np.sort(cd - pred[cc])[k - 1])
        self.charge = np.clip(pred + self.q_hat, 0.0, E_MAX)

    def _risk_control(self, sur, pred):
        m = len(self.cal)
        if m == 0:
            return
        eps = self.delta * self.eta
        cc = np.array([x for x, _ in self.cal])
        cd = kl(sur(cc), np.vstack([p for _, p in self.cal]))
        lam = np.linspace(0.0, E_MAX, 922)
        charged = np.minimum(pred[cc][None, :] + lam[:, None], E_MAX)
        risk = (np.maximum(cd[None, :] - charged, 0.0) - eps * charged).mean(1)
        ok = (m / (m + 1)) * risk + E_MAX / (m + 1) <= 0.0
        if not ok.any():                      # fewer than ~1/eps calibration replies yet
            self.q_hat = np.inf
            self.charge = np.full(N_CTX, E_MAX)
            return
        self.q_hat = float(lam[np.argmax(ok)])
        self.charge = np.clip(pred + self.q_hat, 0.0, E_MAX)

    def __call__(self, c):
        if self.novel:
            return self.charge[np.atleast_1d(c)]
        return np.array([self.exact.get(int(x), E_MAX) for x in np.atleast_1d(c)])

    def is_exact(self, c):
        return np.array([int(x) in self.exact for x in np.atleast_1d(c)])


class Run:
    """policy: parity | view_lod | round_robin | surrogate_only | reference"""

    def __init__(self, n, policy, P, latency, seed=0, cap=10.0, surrogate="distilled", util=0.9,
                 calib_calls=60, refit_every=40, novel=False, bound="worst", alpha=0.1, explore=0.1,
                 delta=0.05, eta=0.25, small=None):
        self.n, self.policy, self.P, self.cap = n, policy, P, cap
        # small = (P, latency) of a second, cheaper model: a three-option menu per decision
        self.small = small is not None
        self.P15 = small[0] if small else None
        self.cost15 = float(np.mean(small[1]) / np.mean(latency)) if small else 1.0
        self.probed = set()
        self.probe_q = deque()
        self.rng = np.random.default_rng(seed)
        self.st = Station(n, np.random.default_rng(seed + 1))
        self.server = Server(P, latency, np.random.default_rng(seed + 2), small)
        self.rate = util / float(np.mean(latency))            # calls per second the model sustains
        self.tokens = 0.0
        self.sur = Distilled() if surrogate == "distilled" else Marginal()
        self.drift = DriftBound(novel, bound, alpha, delta=delta, eta=eta)
        # risk control runs a stretch that holds any estimated charge to cap / (1 + eta), leaving
        # room for the under-charge; a stretch charged only E_MAX (exact upper bounds, which
        # cannot under-charge) still runs to the cap -- else before calibration nothing fits
        self.cap_eff = cap / (1 + eta) if novel and bound == "crc" else cap
        # novel contexts: this share of the call budget goes to uniformly random decisions, whose
        # replies calibrate the conformal margin (and still serve those decisions)
        self.explore = explore if novel and bound in ("conformal", "crc") else 0.0
        self.calib_req = set()
        self.refit_every, self.seen = refit_every, 0
        self.L = np.zeros(n)                                  # charged ledger
        self.D = np.zeros(n)                                  # true drift since last LLM decision
        self.nsur = np.zeros(n, np.int64)                     # surrogate decisions in this stretch
        self.Ls = np.zeros(n)                                 # the estimated (non-E_MAX) part of L
        self.over = np.zeros(n, bool)                         # true drift passed the cap in it
        self.gen_seen = self.st.gen.copy()
        self.inflight = np.zeros(n, bool)
        self.pending = {}                                     # agent -> (ctx, p, generation)
        self.stalled = set()                                  # agents waiting for the model
        self.req_gen = {}                                     # generation each request was made for
        self.ndec = np.zeros(n, np.int64)                     # decisions made per slot
        self.req_dec = {}                                     # which decision each request is for
        self.late = 0
        self.rr = 0
        self.stats = dict(llm=0, sur=0, stale=0, stall=0, overrun=0, kl=0.0, kl_view=0.0, dmax=0.0,
                          over_cap=0, agent_s=0, view_s=0, calib=0, covered=0, stretches=0, overflowed=0,
                          small=0, probes=0)
        self.dmax_t = []
        # nearly all drift comes in the first minute (every agent deciding at once, the surrogate
        # trained on a 60-reply warm-up), so drift is also reported from here on, in steady state
        self.steady_from, self.kl_at, self.agent_s_at = 300, None, None
        if policy not in ("reference",):
            # a warm-up batch so the surrogate and the drift bound start from something
            c = self.st.context(0)[self.rng.choice(n, min(calib_calls, n), replace=False)]
            for x in c:
                self._learn(x, P[x])
            self.stats["calib"] = len(c)
            self._refit()

    # -- learning from replies --------------------------------------------------------
    def _learn(self, c, p, calib=False):
        self.sur.observe(c, p)
        self.drift.observe(c, p, calib)
        self.seen += 1

    def _refit(self):
        if hasattr(self.sur, "fit"):
            self.sur.fit()
        self.drift.refresh(self.sur)

    def _close(self, idx):
        """End the stretches of these slots: an LLM decision, or a new person in the slot (who
        starts with a clean ledger -- the last person's drift is not theirs)."""
        idx = np.atleast_1d(idx)
        had = idx[self.nsur[idx] > 0]
        self.stats["stretches"] += int(had.size)
        self.stats["overflowed"] += int(self.over[had].sum())
        self.L[idx] = 0.0; self.Ls[idx] = 0.0; self.D[idx] = 0.0; self.nsur[idx] = 0; self.over[idx] = False

    def _respawned(self):
        new = np.flatnonzero(self.st.gen != self.gen_seen)
        if new.size:
            self._close(new)
            self.gen_seen[new] = self.st.gen[new]

    def _limit(self, i, e):
        """The ledger limit a surrogate decision charged e must fit under."""
        est = (np.asarray(e) < E_MAX) | (self.Ls[i] > 0)
        return np.where(est, self.cap_eff, self.cap)

    # -- one second ----------------------------------------------------------------------
    def salience(self, step):
        view = (step // VIEW_PERIOD) % N_Z
        return np.where(self.st.zone == view, 1.0, FAR_SALIENCE)

    def step(self, t):
        st = self.st
        st.tick(t)
        self._respawned()
        for agent, c, p, model in self.server.poll(t):
            if model == 1:
                # the small model's answer: learned for its drift, and a decision if one asked
                self.drift.observe_small(c, p)
                if agent >= 0:
                    self.inflight[agent] = False
                    if self.req_dec.get(agent, -1) == self.ndec[agent]:
                        self.pending[agent] = (c, p, self.req_gen.get(agent, -1), 1)
                    else:
                        self.late += 1
                continue
            self.inflight[agent] = False
            if self.req_dec.get(agent, -1) == self.ndec[agent]:
                self.pending[agent] = (c, p, self.req_gen.get(agent, -1), 0)
            else:
                self.late += 1                # the decision it was for has already been made
            if self.small and c not in self.drift.p15 and c not in self.probed:
                # a probe: the small model at a context the LLM has now answered makes the small
                # model's drift there exact, so the scheduler knows where it can stand in. Probes
                # wait for budget no decision wanted (_probe): paid into debt, even 30 of them
                # cost ~13% more drift, all of it in the first minute, when every call is worth
                # most (the surrogate is untrained and every agent is deciding)
                self.probed.add(c)
                self.probe_q.append(c)
            fresh = c not in self.drift.p
            calib = agent in self.calib_req
            self.calib_req.discard(agent)
            self._learn(c, p, calib)
            if self.seen % self.refit_every == 0:
                self._refit()
            elif fresh and not self.drift.novel:
                self.drift.exact[c] = float(kl(self.sur(c), p)[0])

        dec = np.flatnonzero(st.busy_until <= t)
        if dec.size:
            x = st.context(t, dec)
            acts, go = [], []
            for i, c in zip(dec, x):
                a = self._decide(int(i), int(c), t)
                if a is not None:
                    acts.append(a); go.append(i)
            if go:
                st.apply(np.array(go), np.array(acts), t)
                self._respawned()
        self._schedule(t)
        if t == self.steady_from:
            self.kl_at, self.agent_s_at = self.stats["kl"], self.stats["agent_s"]
        self.stats["agent_s"] += self.n
        self.stats["view_s"] += int((self.salience(t) == 1.0).sum())
        self.stats["dmax"] = max(self.stats["dmax"], float(self.D.max()))
        self.stats["over_cap"] += int((self.D > self.cap + 1e-9).sum())
        self.dmax_t.append(float(self.D.max()))

    def _decide(self, i, c, t):
        P, rng = self.P, self.rng
        if self.policy == "reference":
            self.stats["llm"] += 1
            return int(rng.choice(N_ACT, p=P[c]))
        rep = self.pending.pop(i, None)
        if rep is not None and rep[2] != self.st.gen[i]:
            rep = None                         # asked for the person who used to be in this slot
            self.stats["stale"] += 1
        if rep is not None and rep[0] == c and rep[3] == 0:
            self.stats["llm"] += 1
            self._close(i)
            self.stalled.discard(i)
            self.ndec[i] += 1
            return int(rng.choice(N_ACT, p=rep[1]))
        if rep is not None and rep[0] == c and rep[3] == 1:
            # a small-model decision: not a reset, charged its drift from the LLM like any other
            e = float(self.drift.small_charge(c)[0])
            if self.policy != "parity" or self.L[i] + e <= self.cap:
                d = float(kl(rep[1], P[c])[0])
                self.L[i] += e
                self.D[i] += d
                self.nsur[i] += 1
                self.over[i] |= self.D[i] > self.cap + 1e-9
                self.stats["small"] += 1
                self.stats["kl"] += d
                if self.salience(t)[i] == 1.0:
                    self.stats["kl_view"] += d
                self.stalled.discard(i)
                self.ndec[i] += 1
                return int(rng.choice(N_ACT, p=rep[1]))
            rep = None
        if rep is not None:
            # the reply answers a situation the agent is no longer in (the queue changed while it
            # was on its way); acting on it would be drift nobody charged, so it is discarded
            self.stats["stale"] += 1
        e = float(self.drift(c)[0])
        if self.policy == "parity" and self.L[i] + e > self._limit(i, e):
            # the ledger cannot absorb another surrogate decision: wait for the model
            self.st.busy_until[i] = t + 1
            self.stats["stall"] += 1
            self.stalled.add(i)
            if not self.inflight[i]:
                self.server.submit(i, c, t)
                self.req_gen[i] = int(self.st.gen[i])
                self.req_dec[i] = int(self.ndec[i])
                self.inflight[i] = True
                self.stats["overrun"] += 1
            return None
        q = self.sur(c)[0]
        d = float(kl(q, P[c])[0])
        self.L[i] += e
        if e < E_MAX:
            self.Ls[i] += e
        self.D[i] += d
        self.nsur[i] += 1
        self.over[i] |= self.D[i] > self.cap + 1e-9
        self.stats["covered"] += int(d <= e + 1e-12)
        self.stats["sur"] += 1
        self.stats["kl"] += d
        if self.salience(t)[i] == 1.0:
            self.stats["kl_view"] += d
        self.ndec[i] += 1
        return int(rng.choice(N_ACT, p=q))

    def _schedule(self, t):
        if self.policy in ("reference", "surrogate_only"):
            return
        st = self.st
        self.tokens = min(self.tokens + self.rate, 3 * self.rate + 1)
        # people in the ticket queue included: their decision time is estimated from their place
        cand = np.flatnonzero(~self.inflight & (st.busy_until > t))
        cand = np.array([i for i in cand if i not in self.pending], np.int64)
        # aim each reply to land just before its decision: sooner than `earliest` it cannot
        # arrive in time (a wasted call), much later and the prediction goes stale
        earliest = t + self.server.backlog(t) + float(self.server.lat[0].max())
        if cand.size:
            when = st.next_decision_step(t, cand)
            keep = when <= earliest + 3
            cand, when = cand[keep], when[keep]
        if cand.size == 0:
            return
        in_time = when >= earliest
        zone, ticket = st.next_state(cand)
        xhat = st.context(t, cand, zone=zone, ticket=ticket, at_step=when)
        if self.small:
            in15 = when >= t + self.server.backlog(t) + float(self.server.lat[1].max())
            if self.policy == "parity":
                self._schedule3(t, cand, xhat, in_time, in15)
                self._probe(t)
                return
            if self.policy == "view_lod":
                # model LOD, the MassLOD idea with models as tiers: the big model for the agents
                # the viewer can see, the small one for the rest, soonest decision first; no ledger
                sal = self.salience(t)[cand]
                near = np.flatnonzero((sal == 1.0) & in_time)
                far = np.flatnonzero((sal < 1.0) & in15)
                for j, m in [(j, 0) for j in near[np.argsort(when[near])]] + [(j, 1) for j in far[np.argsort(when[far])]]:
                    cost = 1.0 if m == 0 else self.cost15
                    if self.tokens < cost:
                        continue
                    self._request(cand[j], xhat[j], t, m)
                    self.tokens -= cost
                return
        if self.policy == "parity":
            # calibration draws on the call budget like any request, except while there is no
            # margin at all: then every surrogate decision costs E_MAX, every agent's next
            # request is mandatory, the budget never recovers, and the margin would never form.
            # A request for a random decision is what breaks that, so it overrides the budget
            # (as a mandatory request does). The calibration set is those decisions only -- not
            # the warm-up at t = 0, whose contexts (everyone just arrived) are unrepresentative.
            uncal = not np.isfinite(self.drift.q_hat)
            if (self.explore > 0 and (self.tokens >= 1 or uncal) and in_time.any()
                    and (uncal or self.rng.random() < self.explore)):
                j = int(self.rng.choice(np.flatnonzero(in_time)))
                self.server.submit(cand[j], xhat[j], t)
                self.req_gen[int(cand[j])] = int(self.st.gen[cand[j]])
                self.req_dec[int(cand[j])] = int(self.ndec[cand[j]])
                self.inflight[cand[j]] = True
                self.calib_req.add(int(cand[j]))
                self.tokens -= 1
                keep = np.arange(len(cand)) != j
                cand, when, in_time, xhat = cand[keep], when[keep], in_time[keep], xhat[keep]
                if cand.size == 0:
                    return
            e = self.drift(xhat)
            # an agent whose ledger cannot take its next surrogate decision is asked for now,
            # even too late to avoid a wait; anyone else only if the reply can arrive in time
            must = self.L[cand] + e > self._limit(cand, e)
            val = np.where(in_time, self.salience(t)[cand] * e, -1.0)
            order = list(np.flatnonzero(must)) + [j for j in np.argsort(-val) if not must[j] and val[j] >= 0]
            for j in order:
                if not must[j] and self.tokens < 1:
                    break
                if must[j] and self.tokens < 1:
                    self.stats["overrun"] += 1
                self.server.submit(cand[j], xhat[j], t)
                self.req_gen[int(cand[j])] = int(self.st.gen[cand[j]])
                self.req_dec[int(cand[j])] = int(self.ndec[cand[j]])
                self.inflight[cand[j]] = True
                self.tokens -= 1
        elif self.policy == "view_lod":
            # LOD by visibility, the MassLOD idea: everyone the viewer can see first, soonest
            # decision first, then the leftover budget to everyone else -- no ledger
            sal = self.salience(t)[cand]
            near = np.flatnonzero((sal == 1.0) & in_time)
            far = np.flatnonzero((sal < 1.0) & in_time)
            for j in np.concatenate([near[np.argsort(when[near])], far[np.argsort(when[far])]]):
                if self.tokens < 1:
                    break
                self.server.submit(cand[j], xhat[j], t)
                self.req_gen[int(cand[j])] = int(self.st.gen[cand[j]])
                self.req_dec[int(cand[j])] = int(self.ndec[cand[j]])
                self.inflight[cand[j]] = True
                self.tokens -= 1
        elif self.policy == "round_robin":
            order = np.argsort((cand - self.rr) % self.n)
            for j in order[in_time[order]]:
                if self.tokens < 1:
                    break
                self.server.submit(cand[j], xhat[j], t)
                self.req_gen[int(cand[j])] = int(self.st.gen[cand[j]])
                self.req_dec[int(cand[j])] = int(self.ndec[cand[j]])
                self.inflight[cand[j]] = True
                self.tokens -= 1
                self.rr = (cand[j] + 1) % self.n

    def _probe(self, t):
        """Probes from the budget left after this step's decisions, while they might pay."""
        while self.probe_q and self.tokens >= self.cost15 and self._worth_probing():
            self.server.submit(-1, self.probe_q.popleft(), t, model=1)
            self.tokens -= self.cost15
            self.stats["probes"] += 1

    def _worth_probing(self, first=30, share=0.10):
        """Probe while the small model might pay: always for the first probes, then only while it
        beat the surrogate at a useful share of the contexts it was probed at."""
        known = self.drift.exact15
        if len(known) < first:
            return True
        better = sum(1 for c, e in known.items() if e < self.drift.exact.get(c, E_MAX))
        return better >= share * len(known)

    def _request(self, i, x, t, model):
        self.server.submit(i, x, t, model)
        self.req_gen[int(i)] = int(self.st.gen[i])
        self.req_dec[int(i)] = int(self.ndec[i])
        self.inflight[i] = True

    def _schedule3(self, t, cand, xhat, in_time, in15):
        """PARITY over a three-option menu per decision -- the LLM (cost 1, charge 0, resets the
        ledger), the small model (cost15, its exact drift where learned) and the surrogate (free,
        its charge) -- as a multiple-choice knapsack: every candidate's options on their upper
        hull, the upgrades taken in order of salience-weighted drift saved per unit of model time.
        A candidate whose ledger cannot take its surrogate decision gets the cheapest option that
        keeps it under the cap first, over budget if need be."""
        e_s = self.drift(xhat)
        e_m = self.drift.small_charge(xhat)
        sal = self.salience(t)[cand]
        L = self.L[cand]
        ok_s = L + e_s <= self._limit(cand, e_s)
        ok_m = (L + e_m <= self.cap) & in15
        steps = []
        for j in range(len(cand)):
            if not ok_s[j]:
                m = 1 if ok_m[j] and e_m[j] < e_s[j] and not in_time[j] else 0
                m = 1 if ok_m[j] and self.tokens < 1 else m
                cost = self.cost15 if m == 1 else 1.0
                if self.tokens < cost:
                    self.stats["overrun"] += 1
                self._request(cand[j], xhat[j], t, m)
                self.tokens -= cost
                continue
            opts = [(0.0, e_s[j], -1)]
            if ok_m[j] and e_m[j] < e_s[j]:
                opts.append((self.cost15, e_m[j], 1))
            if in_time[j]:
                opts.append((1.0, 0.0, 0))
            hull = [opts[0]]
            for o in opts[1:]:
                while len(hull) >= 2 and (hull[-1][1] - o[1]) * (hull[-1][0] - hull[-2][0]) >= (hull[-2][1] - hull[-1][1]) * (o[0] - hull[-1][0]):
                    hull.pop()
                hull.append(o)
            for k, (a, b) in enumerate(zip(hull, hull[1:])):
                steps.append((sal[j] * (a[1] - b[1]) / (b[0] - a[0]), j, k, b[0] - a[0], b[2]))
        steps.sort(key=lambda x: -x[0])
        level, choice = {}, {}
        for eff, j, k, dc, m in steps:
            if level.get(j, 0) != k or self.tokens < dc:
                continue
            self.tokens -= dc
            level[j] = k + 1
            choice[j] = m
        for j, m in choice.items():
            self._request(cand[j], xhat[j], t, m)

    def run(self, seconds):
        for t in range(seconds):
            self.step(t)
        return self

    def summary(self):
        s, st = self.stats, self.st
        hours = s["agent_s"] / 3600.0
        open_ = self.nsur > 0
        stretches = s["stretches"] + int(open_.sum())
        overflowed = s["overflowed"] + int(self.over[open_].sum())
        dec = s["llm"] + s["sur"]
        dec += s["small"]
        steady = ((s["kl"] - self.kl_at) / max((s["agent_s"] - self.agent_s_at) / 3600.0, 1e-9)
                  if self.kl_at is not None else float("nan"))
        return dict(policy=self.policy, n=self.n, llm_frac=s["llm"] / max(dec, 1), small_frac=s["small"] / max(dec, 1),
                    model_util=self.server.busy_s / max(len(self.dmax_t), 1), probes=s["probes"],
                    calls_per_s=self.server.calls / max(len(self.dmax_t), 1), kl_per_agent_h=s["kl"] / max(hours, 1e-9),
                    kl_steady_per_agent_h=steady,
                    kl_view_per_agent_h=s["kl_view"] / max(s["view_s"] / 3600.0, 1e-9),
                    dmax=s["dmax"], over_cap_frac=s["over_cap"] / max(s["agent_s"], 1),
                    stall_per_agent_h=s["stall"] / max(hours, 1e-9), stale=s["stale"], late=self.late,
                    overrun=s["overrun"], coverage=s["covered"] / max(s["sur"], 1),
                    stretches=stretches, overflow_frac=overflowed / max(stretches, 1),
                    boarded=st.boarded, missed=st.missed, left=st.left, turned=st.turned_away)
