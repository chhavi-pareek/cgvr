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


class Surrogate:
    def __init__(self, proc, seed=0):
        self.proc = proc
        self.coarse, self.fine = regions(proc.z, seed)
        self.fitted = False

    # -- fitting ---------------------------------------------------------------------
    def fit(self, ctx_log, d_log):
        """ctx_log, d_log: (frames, n) int arrays from a full-fidelity run (fine-authoritative)."""
        T, n = d_log.shape
        r, R = self.fine[d_log], self.coarse[d_log]
        c0, c1 = ctx_log[:-1].ravel(), ctx_log[1:].ravel()
        same = c0 == c1  # transitions counted within a context only
        Cf = np.zeros((N_CTX, K_FINE, K_FINE))
        Cc = np.zeros((N_CTX, K_COARSE, K_COARSE))
        occ_f = np.zeros((N_CTX, K_FINE))
        occ_c = np.zeros((N_CTX, K_COARSE))
        occ_d = np.zeros((N_CTX, M))
        r0, r1, R0, R1 = r[:-1].ravel()[same], r[1:].ravel()[same], R[:-1].ravel()[same], R[1:].ravel()[same]
        cc = c0[same]
        np.add.at(Cf, (cc, r0, r1), 1.0)
        np.add.at(Cc, (cc, R0, R1), 1.0)
        np.add.at(occ_f, (ctx_log.ravel(), r.ravel()), 1.0)
        np.add.at(occ_c, (ctx_log.ravel(), R.ravel()), 1.0)
        np.add.at(occ_d, (ctx_log.ravel(), d_log.ravel()), 1.0)
        self.P_ref = _ctx_rows(Cf)  # (ctx, K_FINE, K_FINE) reference one-step kernel on fine regions
        self.pi_f = _ctx_rows(occ_f[:, None, :])[:, 0]
        self.pi_c = _ctx_rows(occ_c[:, None, :])[:, 0]
        self.pi_d = occ_d + 0.1
        self.ctx_freq = occ_c.sum(1) / occ_c.sum()
        Qc = _ctx_rows(Cc)
        self.P_sur = np.stack([mh_correct(Qc[c], self.pi_c[c]) for c in range(N_CTX)])
        # pi_ref(r | R, ctx) and the lift of the surrogate to the fine level
        fine_of = np.arange(K_FINE) // K_FINE_PER
        self.pi_f_given_c = np.zeros((N_CTX, K_COARSE, K_FINE))
        for c in range(N_CTX):
            for R_ in range(K_COARSE):
                m = fine_of == R_
                w = self.pi_f[c] * m
                self.pi_f_given_c[c, R_] = w / w.sum()
        # Sticky lift: the surrogate holds the fine state while its region is unchanged and draws
        # r' ~ pi_ref(r' | R', ctx) when the region changes. Its fine-level kernel from fine state r is
        #   P_sur(R|R) delta_r  +  sum_{R' != R} P_sur(R'|R) pi(.|R')
        # and kl_lift[ctx, r] = KL(that || P_ref(.|r)); kl_coarse averages r over pi(r | R, ctx).
        self.P_lift = np.einsum("cRS,cSr->cRr", self.P_sur, self.pi_f_given_c)  # memoryless part
        eye = np.eye(K_FINE)
        lift_f = np.empty((N_CTX, K_FINE, K_FINE))
        for c in range(N_CTX):
            stay = self.P_sur[c, fine_of, fine_of]  # P_sur(R|R) for each fine r
            move = self.P_lift[c, fine_of, :] - stay[:, None] * self.pi_f_given_c[c, fine_of, :]
            lift_f[c] = move + stay[:, None] * eye
        self.kl_lift = kl_rows(lift_f, self.P_ref)  # per (ctx, r): KL(surrogate || reference)
        self.kl_coarse = np.einsum("cRr,cr->cR", self.pi_f_given_c, self.kl_lift)  # E_r|R
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
        self.e_rate = np.einsum("cR,cR->c", self.pi_c, self.kl_coarse)  # nats/frame per ctx
        self.fitted = True
        return self

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in ("P_ref", "pi_f", "pi_c", "pi_d", "ctx_freq", "P_sur",
                                                          "pi_f_given_c", "P_lift", "kl_lift", "kl_coarse", "kl_frozen",
                                                          "dec_region", "mbar", "e_rate")},
                 kl_tick=np.stack([self.kl_tick[k] for k in BASELINE_PERIODS]))

    def load(self, path):
        t = np.load(path)
        for k in t.files:
            if k != "kl_tick":
                setattr(self, k, t[k])
        self.kl_tick = {k: t["kl_tick"][i] for i, k in enumerate(BASELINE_PERIODS)}
        self.fitted = True
        return self

    # -- runtime ---------------------------------------------------------------------
    def region_of(self, d):
        return self.coarse[d]

    def step(self, R, ctx, active, rng):
        if active.size == 0:
            return R
        cum = np.cumsum(self.P_sur[ctx[active], R[active]], 1)
        u = rng.random(active.size)
        R[active] = np.minimum((u[:, None] > cum).sum(1), K_COARSE - 1)
        return R

    def decode(self, R):
        return self.dec_region[R]
