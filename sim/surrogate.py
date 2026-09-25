"""Compact Markov surrogate over latent regions (behaviour tier 3, "core" source).

State (R, ctx): R one of K_COARSE k-means regions of the 16-dim latent, ctx from the core.
One K x K kernel per context, fitted by transition counts from a full-fidelity calibration
run and Metropolis-Hastings corrected so its stationary distribution equals the reference's
empirical occupancy pi_ref(R | ctx) exactly. Divergence is measured on K_FINE nested regions,
where the surrogate lifted through pi_ref(r | R, ctx) is what reconciliation samples from.
"""
import numpy as np

from .behaviour import M, N_CTX, speed_scale

K_COARSE = 16
K_FINE_PER = 4
K_FINE = K_COARSE * K_FINE_PER
SMOOTH = 0.05  # Dirichlet pseudo-count on transition rows
BASELINE_PERIODS = (1, 3, 10)  # MassLOD HIGH / MED / LOW tick periods (OFF = frozen)


def kmeans(x, k, rng, iters=50):
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
    return lab, c


def regions(z, seed=0):
    """Nested partition of the M corpus points: coarse (K_COARSE) and fine (K_FINE)."""
    rng = np.random.default_rng(seed)
    coarse, _ = kmeans(z, K_COARSE, rng)
    fine = np.empty(len(z), np.int32)
    for R in range(K_COARSE):
        m = np.flatnonzero(coarse == R)
        sub, _ = kmeans(z[m], min(K_FINE_PER, len(m)), rng) if len(m) > K_FINE_PER else (np.arange(len(m)), None)
        fine[m] = R * K_FINE_PER + sub
    return coarse.astype(np.int32), fine


def _rows(counts, backoff=None, beta=2.0):
    """Row-normalise with a Dirichlet prior: uniform (SMOOTH) or `beta` pseudo-counts of `backoff`."""
    p = counts + (SMOOTH if backoff is None else beta * backoff)
    return p / p.sum(-1, keepdims=True)


def _ctx_rows(C):
    """Per-context kernels backed off to the context-pooled kernel for sparse rows."""
    pooled = _rows(C.sum(0))
    return np.stack([_rows(C[c], pooled) for c in range(C.shape[0])])


def mh_correct(Q, pi):
    """Metropolis-Hastings kernel with proposal Q and exact stationary distribution pi."""
    K = len(pi)
    ratio = (pi[None, :] * Q.T) / np.maximum(pi[:, None] * Q, 1e-300)
    P = Q * np.minimum(1.0, ratio)
    np.fill_diagonal(P, 0.0)
    P[np.arange(K), np.arange(K)] = 1.0 - P.sum(1)
    return P


def kl_rows(P, Q):
    """Row-wise KL(P || Q) for stochastic matrices (0 log 0 = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log(P / Q), 0.0)
    return t.sum(-1)


def _tables(Cf, occ_f, occ_d, part, n_coarse=K_COARSE):
    """Every fitted table from fine-level counts and a fine -> coarse partition `part`.

    The coarse counts are the fine counts aggregated through the partition, which is exactly what
    counting coarse labels directly gives, so the default (nested k-means) partition reproduces
    the original fit bit for bit. Pure function of its inputs, so a partition search can call it."""
    A = np.zeros((K_FINE, n_coarse))
    A[np.arange(K_FINE), part] = 1.0
    Cc = np.einsum("cij,ia,jb->cab", Cf, A, A)
    occ_c = occ_f @ A
    t = {}
    t["P_ref"] = _ctx_rows(Cf)
    t["pi_f"] = _ctx_rows(occ_f[:, None, :])[:, 0]
    t["pi_c"] = _ctx_rows(occ_c[:, None, :])[:, 0]
    t["pi_d"] = occ_d + 0.1
    t["ctx_freq"] = occ_c.sum(1) / occ_c.sum()
    Qc = _ctx_rows(Cc)
    t["P_sur"] = np.stack([mh_correct(Qc[c], t["pi_c"][c]) for c in range(N_CTX)])
    # pi_ref(r | R, ctx) and the lift of the surrogate to the fine level
    pfc = (t["pi_f"][:, None, :] * A.T[None, :, :])
    t["pi_f_given_c"] = pfc / pfc.sum(-1, keepdims=True)
    # Sticky lift: the surrogate holds the fine state while its region is unchanged and draws
    # r' ~ pi_ref(r' | R', ctx) when the region changes. Its fine-level kernel from fine state r is
    #   P_sur(R|R) delta_r  +  sum_{R' != R} P_sur(R'|R) pi(.|R')
    # and kl_lift[ctx, r] = KL(that || P_ref(.|r)); kl_coarse averages r over pi(r | R, ctx).
    t["P_lift"] = np.einsum("cRS,cSr->cRr", t["P_sur"], t["pi_f_given_c"])  # memoryless part
    eye = np.eye(K_FINE)
    lift_f = np.empty((N_CTX, K_FINE, K_FINE))
    for c in range(N_CTX):
        stay = t["P_sur"][c, part, part]  # P_sur(R|R) for each fine r
        move = t["P_lift"][c, part, :] - stay[:, None] * t["pi_f_given_c"][c, part, :]
        lift_f[c] = move + stay[:, None] * eye
    t["lift_f"] = lift_f
    t["kl_lift"] = kl_rows(lift_f, t["P_ref"])  # per (ctx, r): KL(surrogate || reference)
    t["kl_coarse"] = np.einsum("cRr,cr->cR", t["pi_f_given_c"], t["kl_lift"])  # E_r|R
    t["e_rate"] = np.einsum("cR,cR->c", t["pi_c"], t["kl_coarse"])  # nats/frame per ctx
    return t


def objective(t, which="e_sur"):
    """e_sur: occupancy-weighted mean rate (sets the cap); e_max: worst frequent context (admission)."""
    if which == "e_max":
        return float(t["e_rate"][t["ctx_freq"] > 0.01].max())
    return float(t["e_rate"] @ t["ctx_freq"])


def optimise_partition(Cf, occ_f, occ_d, part, which="e_sur", sweeps=6):
    """Local search over fine -> coarse partitions for the surrogate's own KL rate.

    The k-means regions group latent points that are CLOSE; what the ledger pays for is how well
    the lumped chain reproduces the reference's TRANSITIONS, which is a different criterion
    (optimal Kullback-Leibler aggregation of a Markov chain; Deng, Mehta & Meyn 2011). The rate
    of any partition is computable exactly from the calibration counts, so moving one fine region
    at a time to whichever coarse region lowers it most, until no move helps, is a direct descent
    on the quantity itself. It starts from the k-means partition and only accepts improvements,
    so it can never do worse on the data it is fitted to; held-out counts are the real test."""
    part = np.asarray(part, np.int64).copy()
    best = objective(_tables(Cf, occ_f, occ_d, part), which)
    history = [best]
    for _ in range(sweeps):
        improved = False
        for r in range(K_FINE):
            src = part[r]
            if (part == src).sum() <= 1:
                continue                                  # every coarse region keeps a member
            cand_best, cand_R = best, src
            for R in range(K_COARSE):
                if R == src:
                    continue
                part[r] = R
                v = objective(_tables(Cf, occ_f, occ_d, part), which)
                if v < cand_best - 1e-15:
                    cand_best, cand_R = v, R
            part[r] = cand_R
            if cand_R != src:
                best, improved = cand_best, True
        history.append(best)
        if not improved:
            break
    return part, history


class Surrogate:
    def __init__(self, proc, seed=0):
        self.proc = proc
        self.coarse, self.fine = regions(proc.z, seed)
        self.part = np.arange(K_FINE) // K_FINE_PER     # fine region -> coarse region
        self.fitted = False

    def set_partition(self, part):
        """Re-group the fine regions (e.g. optimise_partition's output) and refit from the counts."""
        self.part = np.asarray(part, np.int64)
        self.coarse = self.part[self.fine].astype(np.int32)
        self._lift = None
        if getattr(self, "Cf", None) is not None:
            self._apply(_tables(self.Cf, self.occ_f_n, self.occ_d_n, self.part))
        return self

    # -- fitting ---------------------------------------------------------------------
    def fit(self, ctx_log, d_log):
        """ctx_log, d_log: (frames, n) int arrays from a full-fidelity run (fine-authoritative)."""
        r = self.fine[d_log]
        c0, c1 = ctx_log[:-1].ravel(), ctx_log[1:].ravel()
        same = c0 == c1  # transitions counted within a context only
        Cf = np.zeros((N_CTX, K_FINE, K_FINE))
        occ_f = np.zeros((N_CTX, K_FINE))
        occ_d = np.zeros((N_CTX, M))
        r0, r1 = r[:-1].ravel()[same], r[1:].ravel()[same]
        np.add.at(Cf, (c0[same], r0, r1), 1.0)
        np.add.at(occ_f, (ctx_log.ravel(), r.ravel()), 1.0)
        np.add.at(occ_d, (ctx_log.ravel(), d_log.ravel()), 1.0)
        self.Cf, self.occ_f_n, self.occ_d_n = Cf, occ_f, occ_d
        self._apply(_tables(Cf, occ_f, occ_d, self.part))
        return self

    def _apply(self, t):
        for k in ("P_ref", "pi_f", "pi_c", "pi_d", "ctx_freq", "P_sur", "pi_f_given_c", "P_lift",
                  "kl_lift", "kl_coarse", "e_rate"):
            setattr(self, k, t[k])
        self.kl_frozen = -np.log(self.P_ref[:, np.arange(K_FINE), np.arange(K_FINE)])  # KL(delta || ref)
        self.kl_tick = {}
        for k in BASELINE_PERIODS:
            Pk = np.stack([np.linalg.matrix_power(self.P_ref[c], k) for c in range(N_CTX)])
            self.kl_tick[k] = kl_rows(Pk, self.P_ref)
        # region outputs and context-mean speed scale
        dec16 = self.proc.dec[16]
        self.dec_region = np.zeros((K_COARSE, dec16.shape[1]))
        wd = self.pi_d.sum(0)
        for R_ in range(K_COARSE):
            m = self.coarse == R_
            self.dec_region[R_] = (dec16[m] * wd[m, None]).sum(0) / wd[m].sum()
        pd = self.pi_d / self.pi_d.sum(1, keepdims=True)
        self.mbar = pd @ speed_scale(dec16)
        self.fitted = True

    def rates_against(self, Cf_other):
        """This surrogate's KL rate against a reference kernel estimated from OTHER counts: the
        held-out test of a partition fitted to this surrogate's own calibration."""
        P_other = _ctx_rows(Cf_other)
        part = self.part
        lift_f = np.empty((N_CTX, K_FINE, K_FINE))
        eye = np.eye(K_FINE)
        for c in range(N_CTX):
            stay = self.P_sur[c, part, part]
            move = self.P_lift[c, part, :] - stay[:, None] * self.pi_f_given_c[c, part, :]
            lift_f[c] = move + stay[:, None] * eye
        kl = kl_rows(lift_f, P_other)
        kc = np.einsum("cRr,cr->cR", self.pi_f_given_c, kl)
        return np.einsum("cR,cR->c", self.pi_c, kc), kc

    def rate_bounds(self, Cf_eval, delta=0.05, draws=200, rng=None):
        """Plug-in, bias-corrected and upper-bound KL rates per context, from evaluation counts.

        The rate the ledger charges is KL(surrogate || reference) with the reference kernel
        estimated from finite counts. KL is convex in that kernel, so by Jensen the plug-in is
        biased UP by the estimation noise -- measured: the same surrogate reads 0.0132 nats/frame
        against one 6000-frame run and 0.0082 against two pooled. A parametric bootstrap (rows
        resampled Multinomial(n_row, P_hat)) measures that inflation. The basic-bootstrap upper
        bound U = 2 * plug-in - q_delta(bootstrap) is the bias-corrected rate plus a sampling
        margin: valid at level 1 - delta (Bonferroni over contexts) and, when the noise inflation
        dominates the margin, BELOW the plug-in. Evaluate on counts the surrogate was not fitted
        to, or the surrogate's own overfit enters the estimate the other way."""
        rng = np.random.default_rng(0) if rng is None else rng
        Cf_eval = np.asarray(Cf_eval, np.float64)
        part = self.part
        lift_f = np.empty((N_CTX, K_FINE, K_FINE))
        eye = np.eye(K_FINE)
        for c in range(N_CTX):
            stay = self.P_sur[c, part, part]
            move = self.P_lift[c, part, :] - stay[:, None] * self.pi_f_given_c[c, part, :]
            lift_f[c] = move + stay[:, None] * eye

        def rate(C):
            kl = kl_rows(lift_f, _ctx_rows(C))
            kc = np.einsum("cRr,cr->cR", self.pi_f_given_c, kl)
            return np.einsum("cR,cR->c", self.pi_c, kc), kc

        plug, kc = rate(Cf_eval)
        P_hat = _ctx_rows(Cf_eval)
        n_row = Cf_eval.sum(-1).astype(np.int64)
        boot = np.empty((draws, N_CTX))
        for b in range(draws):
            Cs = np.zeros_like(Cf_eval)
            for c in range(N_CTX):
                for r in range(K_FINE):
                    if n_row[c, r]:
                        Cs[c, r] = rng.multinomial(n_row[c, r], P_hat[c, r])
            boot[b] = rate(Cs)[0]
        q = np.quantile(boot, delta / N_CTX, axis=0)
        upper = np.maximum(2.0 * plug - q, 0.0)
        corrected = np.maximum(2.0 * plug - boot.mean(0), 0.0)
        return dict(plug=plug, corrected=corrected, upper=upper, kl_coarse=kc,
                    inflation=boot.mean(0) - plug)

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in ("P_ref", "pi_f", "pi_c", "pi_d", "ctx_freq", "P_sur",
                                                          "pi_f_given_c", "P_lift", "kl_lift", "kl_coarse", "kl_frozen",
                                                          "dec_region", "mbar", "e_rate", "part")},
                 kl_tick=np.stack([self.kl_tick[k] for k in BASELINE_PERIODS]))

    def load(self, path):
        t = np.load(path)
        for k in t.files:
            if k != "kl_tick":
                setattr(self, k, t[k])
        if "part" in t.files:                             # older calibrations are the default nesting
            self.coarse = self.part[self.fine].astype(np.int32)
        self.kl_tick = {k: t["kl_tick"][i] for i, k in enumerate(BASELINE_PERIODS)}
        self.fitted = True
        return self

    # -- runtime ---------------------------------------------------------------------
    def region_of(self, d):
        return self.coarse[d]

    def lift_tables(self):
        """Precomputed inverse-CDF tables for pi_ref(d | R, ctx), built once and cached.

        sim/reconcile.py::promote used to rebuild `pi_d[ctx] * (coarse == R)` and call
        rng.choice(p=...) once PER AGENT, which is O(M) with a fresh normalisation each time --
        measured at 21 us per reconciled agent, twice the cost of an entire per-frame behaviour
        step. The candidate set depends only on R and the weights only on ctx, so there are
        4 x 16 = 64 distributions in total; tabulating them turns the draw into one
        searchsorted. Same law, O(log |R|) instead of O(M)."""
        if getattr(self, "_lift", None) is None:
            nctx, nR = self.pi_d.shape[0], int(self.coarse.max()) + 1
            cand = [np.flatnonzero(self.coarse == R) for R in range(nR)]
            cdf = [[None] * nR for _ in range(nctx)]
            for c in range(nctx):
                for R in range(nR):
                    if cand[R].size:
                        w = self.pi_d[c][cand[R]]
                        cdf[c][R] = np.cumsum(w) / w.sum()
            self._lift = (cand, cdf)
        return self._lift

    def step(self, R, ctx, active, rng):
        if active.size == 0:
            return R
        cum = np.cumsum(self.P_sur[ctx[active], R[active]], 1)
        u = rng.random(active.size)
        R[active] = np.minimum((u[:, None] > cum).sum(1), K_COARSE - 1)
        return R

    def decode(self, R):
        return self.dec_region[R]
