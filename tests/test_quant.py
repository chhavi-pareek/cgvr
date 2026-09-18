"""INT8 scoring: interval validity, provable decisions, re-rank identity with the FP64 reference."""
import numpy as np
import pytest

from alloc.config import build_table
from alloc.cuda import score_int8 as q8
from tests.test_parallel import _instance

TABLE = build_table()


def _latents(n, d, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n, d)) * (0.5 ** np.arange(d))[None]  # nested-like energy decay
    return z


def _setup(n, d, seed):
    s, cost = _instance(seed, n, ticks=False)
    phi = q8.latent_features(_latents(n, d, seed))
    return s, cost, phi, q8.Int8Scorer(TABLE, d, cost)


def test_quantisation_caps_and_exact_columns():
    X = np.random.default_rng(0).uniform(-1, 1, (50, 16))
    q, sc, resid = q8.quantize_rows(X)
    assert (resid <= sc / 2 + 1e-15).all()
    assert np.array_equal(q.astype(np.float64), np.rint(X / sc[:, None]))
    B, kappa = q8.basis(TABLE, 16)
    Bq, delta, fbar = q8.quantize_cols(B)
    assert (fbar == 0).all()
    np.testing.assert_array_equal(Bq.astype(np.float64) * delta[None, :], B)


@pytest.mark.parametrize("d", [16, 8, 4])
def test_utility_interval_contains_truth(d):
    s, cost, phi, sc = _setup(400, d, 1)
    r = sc.score(s, phi, 1.0, pairwise=False)
    n, m = len(s), len(cost)
    rows, cols = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
    U = q8.utility_fp64(phi, sc.B, sc.kappa, rows, cols).reshape(n, m)
    Xq, Delta, _ = q8.quantize_rows(phi)
    ut = Delta[:, None] * sc.delta[None, :] * r["A"].astype(np.float64) + sc.kappa[None, :]
    assert (np.abs(U - ut) <= r["eps"] + 1e-15).all()
    score_ref = s[:, None] * U - 1.0 * cost[None, :]
    assert (np.abs(score_ref - r["score_int8"].astype(np.float64)) <= r["eta"]).all()


@pytest.mark.parametrize("d", [16, 8, 4])
@pytest.mark.parametrize("pairwise", [False, True])
def test_provable_and_rerank_match_reference(d, pairwise):
    s, cost, phi, sc = _setup(500, d, 2)
    for lam in (0.0, 0.5, 3.0, 20.0):
        r = sc.score(s, phi, lam, pairwise=pairwise)
        ref, score_ref = q8.reference(s, phi, sc.B, sc.kappa, lam, cost)
        assert np.array_equal(r["assign"], ref)
        assert np.array_equal(r["assign_int8"][r["provable"]], ref[r["provable"]])
        ar = np.arange(len(s))
        regret = score_ref[ar, ref] - score_ref[ar, r["assign_int8"]]
        assert (regret <= r["regret_bound"] + 1e-12).all()
        assert np.array_equal(r["A"], r["A_ref"])


def test_pairwise_never_looser_than_candidate():
    s, cost, phi, sc = _setup(300, 16, 3)
    a = sc.score(s, phi, 2.0, pairwise=False)
    b = sc.score(s, phi, 2.0, pairwise=True)
    assert (b["ncand"] <= a["ncand"]).all()
    assert b["provable"].sum() >= a["provable"].sum()


def test_headroom_mask_respected():
    s, cost, phi, sc = _setup(300, 8, 4)
    h = np.random.default_rng(4).uniform(TABLE.err.min(), TABLE.err.max(), len(s))
    r = sc.score(s, phi, 1.0, headroom=h)
    ref, _ = q8.reference(s, phi, sc.B, sc.kappa, 1.0, cost, headroom=h, err=TABLE.err)
    assert np.array_equal(r["assign"], ref)
    assert (TABLE.err[r["assign"]] <= h).all()
