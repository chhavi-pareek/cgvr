"""INT8 tensor-core scoring of the allocator's candidate table with a derived error interval.

Scoring
-------
Agent i has a nested latent z_i of width d. Its feature row is the retained-
energy profile phi_i = z_i^2 / |z_i|^2, K = d. Candidate c has a basis column
b_c = 1/4 * (1[j < k_beh] + 1[j < k_anim]) where k is the tier's kept width
(d, d/2, d/4, 0), and an agent-independent term kappa_c = 1/4 * (q_nav + q_geo)
from the phase 3 table. u_ic = phi_i . b_c + kappa_c is the mean of four
per-axis qualities, behaviour and animation quality being the fraction of the
agent's own latent energy the tier keeps. Score sigma_ic = s_i u_ic - lam cost_c.
kappa_c is added in the epilogue rather than carried as a constant feature: a
constant 1 would set every row scale to 1/127 while the energy shares are far
below 1, wasting most of the INT8 range.

Quantisation (symmetric, per row of Phi and per column of B)
-----------------------------------------------------------
Delta_i = max_j |phi_ij| / 127, phihat_ij = round(phi_ij / Delta_i), |phi_ij - Delta_i phihat_ij| <= ebar_i <= Delta_i / 2
delta_c = max_j |b_jc| / 126,   bhat_jc  = round(b_jc / delta_c),    |b_jc - delta_c bhat_jc|   <= fbar_c <= delta_c / 2
ebar_i and fbar_c are the measured per-row / per-column maximum residuals
(exact, one pass). Basis entries are in {0, 1/4, 1/2}, so with scale max/126
the columns quantise exactly and fbar_c = 0.
A_ic = sum_j phihat_ij bhat_jc is exact in INT32 (|A| <= K 127^2 < 2^31), utilde_ic = Delta_i delta_c A_ic.

Interval
--------
With phi_ij = Delta_i phihat_ij + e_ij and b_jc = delta_c bhat_jc + f_jc,
  u_ic = utilde_ic + Delta_i sum_j phihat_ij f_jc + delta_c sum_j bhat_jc e_ij + sum_j e_ij f_jc
  |u_ic - utilde_ic| <= eps_ic = Delta_i |phihat_i|_1 fbar_c + delta_c |bhat_c|_1 ebar_i + K ebar_i fbar_c
(with the caps Delta_i/2 and delta_c/2 this is Delta_i delta_c (|phihat_i|_1/2 + |bhat_c|_1/2 + K/4)).
The L1 norms are of the integer vectors, hence exact. The FP32 epilogue
sigmatilde_ic = fl(fl(fl(fl(fl(s_i) fl(Delta_i)) fl(delta_c)) A_ic) + fl(fl(s_i) fl(kappa_c))) - fl(fl(lam) fl(cost_c))
has at most 9 roundings on any term at u = 2^-24, and the FP64 reference has 4
at 2^-53, so with M_ic = |s_i Delta_i delta_c A_ic| + s_i kappa_c + lam cost_c
  eta_ic = s_i eps_ic + 10u M_ic + 5 2^-53 M_ic
and both the exact score and the reference's computed score lie in
[sigmatilde_ic - eta_ic, sigmatilde_ic + eta_ic].

Decision
--------
c* = INT8 argmax. C_i = {c feasible : sigmatilde_ic + eta_ic >= sigmatilde_ic* - eta_ic*}.
Any c outside C_i is strictly below c* in exact and in reference arithmetic, so
the reference argmax is in C_i. |C_i| = 1 is a provably identical decision with
no FP32 work; otherwise only C_i is re-ranked in the reference's FP64 arithmetic
with the reference comparator, which reproduces its choice exactly. Without
re-ranking the regret sigma_i,ref - sigma_i,c* <= 2 max_{c in C_i} eta_ic.

Pairwise interval (default, pairwise=True)
------------------------------------------
The same expansion applied to the difference u_ic - u_ic*: the error terms are
linear in b, so
  |(u_ic - utilde_ic) - (u_ic* - utilde_ic*)|
     <= Delta_i |phihat_i|_1 (fbar_c + fbar_c*) + ebar_i |delta_c bhat_c - delta_c* bhat_c*|_1 + K ebar_i (fbar_c + fbar_c*)
     =: eps_i(c, c*)
Candidates sharing c*'s behaviour and animation tiers have identical columns and
eps = 0; only the fp32/fp64 rounding terms of both scores remain. Then
  sigma_ic - sigma_ic* <= sigmatilde_ic - sigmatilde_ic* + s_i eps_i(c, c*) + rnd_ic + rnd_ic*  =: slack_ic
C_i = {c feasible : slack_ic >= 0}, and the no-re-rank regret is max_{c in C_i} slack_ic.
The per-candidate interval (pairwise=False) is the special case that bounds
both errors separately.

GEMM
----
Numba-CUDA has no WMMA/mma intrinsic, so the INT8 GEMM is torch._int_mm, which
on sm_75 dispatches to cuBLASLt IMMA tensor-core kernels. K is zero-padded to
32 and m to a multiple of 8 (exact). Without a CUDA device the accumulator
comes from a NumPy INT32 matmul, which is also the oracle on the T4.
"""
import numpy as np
from numba import njit

from alloc.config import AXIS_QUALITY

U32 = 2.0 ** -24
U64 = 2.0 ** -53
K_PAD = 32
M_PAD = 8

try:
    import torch

    _CUDA = torch.cuda.is_available()
except Exception:  # torch missing or broken: numpy path only
    torch = None
    _CUDA = False

GEMM_BACKEND = "torch._int_mm (cuBLASLt IMMA)" if _CUDA else "numpy int32"


def latent_features(z):
    z = np.asarray(z, np.float64)
    e = z * z
    e /= np.maximum(e.sum(1, keepdims=True), 1e-30)
    return e


def kept_width(d, tier):
    return 0 if tier >= 3 else d >> tier


def basis(table, d):
    """(d, m) basis columns and (m,) agent-independent term for the table at latent width d."""
    m = len(table.tiers)
    B = np.zeros((d, m))
    kappa = np.zeros(m)
    j = np.arange(d)
    for c, t in enumerate(table.tiers):
        B[:, c] = 0.25 * ((j < kept_width(d, t[0])) + (j < kept_width(d, t[2])))
        kappa[c] = 0.25 * (AXIS_QUALITY[1, t[1]] + AXIS_QUALITY[3, t[3]])
    return B, kappa


def quantize_rows(X, levels=127):
    """Symmetric per-row INT8. Returns integers, scales and the measured max residual per row."""
    X = np.asarray(X, np.float64)
    scale = np.abs(X).max(1) / levels
    scale = np.where(scale > 0, scale, 1.0)
    q = np.rint(X / scale[:, None])
    assert np.abs(q).max() <= 127
    resid = np.abs(X - scale[:, None] * q).max(1)
    return q.astype(np.int8), scale, resid


def quantize_cols(B, levels=126):
    q, scale, resid = quantize_rows(np.asarray(B).T, levels)
    return np.ascontiguousarray(q.T), scale, resid


def gemm_ref(Xq, Bq):
    return Xq.astype(np.int32) @ Bq.astype(np.int32)


def gemm_int8(Xq, Bq):
    """INT32 accumulator of Xq (N x K) @ Bq (K x m); tensor cores when a CUDA device exists."""
    if not _CUDA:
        return gemm_ref(Xq, Bq)
    n, k = Xq.shape
    m = Bq.shape[1]
    kp = -(-k // K_PAD) * K_PAD
    mp = -(-m // M_PAD) * M_PAD
    a = torch.zeros((max(n, 17), kp), dtype=torch.int8, device="cuda")
    b = torch.zeros((kp, mp), dtype=torch.int8, device="cuda")
    a[:n, :k] = torch.from_numpy(Xq).cuda()
    b[:k, :m] = torch.from_numpy(Bq).cuda()
    return torch._int_mm(a, b)[:n, :m].cpu().numpy()


@njit(cache=True)
def _dots(phi, B, kappa, rows, cols, out):
    """Sequential-order fp64 dot products plus kappa for the listed (row, col) pairs.
    Used by both the reference and the re-rank so their arithmetic is identical."""
    for p in range(len(rows)):
        i = rows[p]
        c = cols[p]
        acc = 0.0
        for j in range(phi.shape[1]):
            acc += phi[i, j] * B[j, c]
        out[p] = acc + kappa[c]


def utility_fp64(phi, B, kappa, rows, cols):
    out = np.empty(len(rows))
    _dots(np.ascontiguousarray(phi, np.float64), np.ascontiguousarray(B, np.float64), np.ascontiguousarray(kappa, np.float64),
          np.ascontiguousarray(rows, np.int64), np.ascontiguousarray(cols, np.int64), out)
    return out


def _argmax_tiebreak(score, cost, util):
    """Per row: highest score, then lowest cost, then highest utility, then lowest index."""
    n, m = score.shape
    idx = np.broadcast_to(np.arange(m), (n, m))
    order = np.lexsort((idx, -util, np.broadcast_to(cost, (n, m)), -score), axis=1)
    return order[:, 0]


def reference(s, phi, B, kappa, lam, cost, headroom=None, err=None):
    """FP64 argmax over all candidates: the oracle the INT8 path is measured against."""
    n, m = len(s), B.shape[1]
    rows = np.repeat(np.arange(n), m)
    cols = np.tile(np.arange(m), n)
    U = utility_fp64(phi, B, kappa, rows, cols).reshape(n, m)
    score = s[:, None] * U - lam * cost[None, :]
    if headroom is not None:
        score = np.where(err[None, :] <= headroom[:, None], score, -np.inf)
    return _argmax_tiebreak(score, cost, U), score


class Int8Scorer:
    def __init__(self, table, d, cost):
        self.table, self.d = table, d
        self.B, self.kappa = basis(table, d)
        self.Bq, self.delta, self.fbar = quantize_cols(self.B)
        self.b_l1 = np.abs(self.Bq.astype(np.int64)).sum(0)
        self.cost = np.ascontiguousarray(cost, np.float64)
        self.err = table.err
        self.K = d
        Bdq = self.Bq.astype(np.float64) * self.delta[None, :]
        self.col_dist = np.abs(Bdq[:, :, None] - Bdq[:, None, :]).sum(0) * (1 + 2.0 ** -40)

    def score(self, s, phi, lam, headroom=None, rerank=True, pairwise=True):
        s = np.asarray(s, np.float64)
        Xq, Delta, ebar = quantize_rows(phi)
        x_l1 = np.abs(Xq.astype(np.int64)).sum(1)
        A = gemm_int8(Xq, self.Bq)
        # fp32 epilogue, roundings as in the module docstring
        s32 = s.astype(np.float32)
        g = (s32 * Delta.astype(np.float32))[:, None] * self.delta.astype(np.float32)[None, :]
        lc = np.float32(lam) * self.cost.astype(np.float32)
        sk = s32[:, None] * self.kappa.astype(np.float32)[None, :]
        util = g * A.astype(np.float32) + sk
        st = util - lc[None, :]
        # interval
        eps = (Delta[:, None] * self.fbar[None, :]) * x_l1[:, None] + (self.delta[None, :] * ebar[:, None]) * self.b_l1[None, :] \
            + self.K * ebar[:, None] * self.fbar[None, :]
        mag = np.abs((g * A.astype(np.float32)).astype(np.float64)) + sk.astype(np.float64) + np.abs(lc.astype(np.float64))[None, :]
        eta = s[:, None] * eps + (10 * U32 + 5 * U64) * mag
        st64 = st.astype(np.float64)
        feasible = np.ones_like(st, bool) if headroom is None else (self.err[None, :] <= np.asarray(headroom)[:, None])
        st_masked = np.where(feasible, st64, -np.inf)
        a8 = _argmax_tiebreak(st_masked, self.cost, util.astype(np.float64))
        n = len(s)
        ar = np.arange(n)
        rnd = (10 * U32 + 5 * U64) * mag
        if pairwise:
            # error of the score *difference* against c*: the expansion is linear in b,
            # so the quantisation error of u_c - u_c* is bounded against b_c - b_c*.
            dq = self.col_dist[a8]                                   # (n, m) |delta_c bhat_c - delta_c* bhat_c*|_1
            fb = self.fbar[None, :] + self.fbar[a8][:, None]
            eps_pair = Delta[:, None] * x_l1[:, None] * fb + ebar[:, None] * dq + self.K * ebar[:, None] * fb
            eta_pair = s[:, None] * eps_pair + rnd + rnd[ar, a8][:, None]
            slack = st64 - st64[ar, a8][:, None] + eta_pair        # upper bound on sigma_c - sigma_c*
            cand = feasible & (slack >= 0)
            regret_bound = np.where(cand, slack, 0.0).max(1)
        else:
            lo_star = st64[ar, a8] - eta[ar, a8]
            cand = feasible & (st64 + eta >= lo_star[:, None])
            regret_bound = 2.0 * np.where(cand, eta, 0.0).max(1)
        ncand = cand.sum(1)
        provable = ncand == 1
        regret_bound[provable] = 0.0
        assign = a8.copy()
        if rerank:
            rows, cols = np.nonzero(cand & ~provable[:, None])
            if len(rows):
                u = utility_fp64(phi, self.B, self.kappa, rows, cols)
                sc = s[rows] * u - lam * self.cost[cols]
                full = np.full((n, self.B.shape[1]), -np.inf)
                full[rows, cols] = sc
                uf = np.zeros_like(full)
                uf[rows, cols] = u
                rr = _argmax_tiebreak(full, self.cost, uf)
                assign[~provable] = rr[~provable]
        return dict(assign=assign, assign_int8=a8, provable=provable, ncand=ncand, eta=eta, eps=eps,
                    score_int8=st, regret_bound=regret_bound, A=A, A_ref=gemm_ref(Xq, self.Bq))
