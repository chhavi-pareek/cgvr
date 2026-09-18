"""INT8 scoring versus the FP64 reference: provable-identical fraction, re-rank cost, regret bound.

python -m bench.quant --agents 2000 --dims 16 8 4

Latents: the phase 2 corpus encoded by the nested AE (seed 0), sampled per agent
and truncated to the first d dims. Salience and costs as in bench/speedup.py;
lambda values from a serial allocation at three budgets plus lambda = 0.
Writes bench/logs/quant.csv.
"""
import argparse
import csv
import itertools
import os
import time

import numpy as np

from alloc.config import build_table
from alloc.cuda import score_int8 as q8
from alloc.serial import SerialAllocator
from bench.speedup import instance

LOG = os.path.join(os.path.dirname(__file__), "logs", "quant.csv")
ZCACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latent", "cache", "z_nested_s0.npy")
FIELDS = ["d", "n", "lam", "interval", "provable", "int8_match", "rerank_match", "mean_cand", "pair_frac",
          "bound_max", "bound_mean", "regret_max", "gemm_bitexact", "gemm_us", "fp32_us", "backend"]


def corpus_latents():
    if os.path.exists(ZCACHE):
        return np.load(ZCACHE)
    import torch

    from latent.train import get_data, load_model

    m, order, _ = load_model(0, True)
    with torch.no_grad():
        z = m.enc(get_data()[0])[:, order].numpy()
    np.save(ZCACHE, z)
    return z


def agent_latents(n, d, seed):
    z = corpus_latents()
    rng = np.random.default_rng(seed)
    return z[rng.integers(0, len(z), n), :d]


def lambdas(table, s, cost, fracs=(0.3, 0.5, 0.7)):
    alloc = SerialAllocator(table, fill=False)
    t0 = alloc.allocate(s, cost, np.inf).cost
    return [0.0] + [alloc.allocate(s, cost, f * t0).lam for f in fracs]


def time_gemm(Xq, Bq, reps=20):
    if not q8._CUDA:
        return float("nan"), float("nan")
    import torch

    n, k = Xq.shape
    kp = -(-k // q8.K_PAD) * q8.K_PAD
    mp = -(-Bq.shape[1] // q8.M_PAD) * q8.M_PAD
    a = torch.zeros((max(n, 17), kp), dtype=torch.int8, device="cuda")
    b = torch.zeros((kp, mp), dtype=torch.int8, device="cuda")
    a[:n, :k] = torch.from_numpy(Xq).cuda()
    b[:k, :Bq.shape[1]] = torch.from_numpy(Bq).cuda()
    af, bf = a.float(), b.float()
    out = []
    for fn in (lambda: torch._int_mm(a, b), lambda: af @ bf):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        out.append(float(np.median(ts)))
    return out[0], out[1]


def run(d, n, seed, table, s, cost, lams):
    phi = q8.latent_features(agent_latents(n, d, seed))
    scorer = q8.Int8Scorer(table, d, cost)
    rows = []
    for lam, pw in itertools.product(lams, (False, True)):
        r = scorer.score(s, phi, lam, pairwise=pw)
        ref, score_ref = q8.reference(s, phi, scorer.B, scorer.kappa, lam, cost)
        ar = np.arange(n)
        regret = score_ref[ar, ref] - score_ref[ar, r["assign_int8"]]
        assert (regret <= r["regret_bound"] + 1e-12).all()
        assert (r["assign_int8"][r["provable"]] == ref[r["provable"]]).all()
        g_us, f_us = time_gemm(q8.quantize_rows(phi)[0], scorer.Bq)
        np_ = ~r["provable"]
        rows.append(dict(
            d=d, n=n, lam=lam, interval="pairwise" if pw else "candidate", provable=r["provable"].mean(), int8_match=(r["assign_int8"] == ref).mean(),
            rerank_match=(r["assign"] == ref).mean(), mean_cand=r["ncand"].mean(),
            pair_frac=r["ncand"][np_].sum() / (n * len(table.tiers)),
            bound_max=r["regret_bound"].max(), bound_mean=r["regret_bound"][np_].mean() if np_.any() else 0.0,
            regret_max=regret.max(), gemm_bitexact=bool(np.array_equal(r["A"], r["A_ref"])),
            gemm_us=g_us, fp32_us=f_us, backend=q8.GEMM_BACKEND))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=2000)
    ap.add_argument("--dims", nargs="+", type=int, default=[16, 8, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=LOG)
    args = ap.parse_args()
    table = build_table()
    s, cost = instance(args.seed, args.agents, table)
    lams = lambdas(table, s, cost)
    rows = []
    for d in args.dims:
        t = time.perf_counter()
        rows += run(d, args.agents, args.seed, table, s, cost, lams)
        print(f"d={d}: {time.perf_counter() - t:.1f} s")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"backend: {rows[0]['backend']}; N={args.agents}, m={len(table.tiers)}, lambdas={[f'{l:.3g}' for l in lams]}")
    print("d   lam     interval  provable int8=ref rerank=ref cand/agent pairs  bound_max  regret_max gemm_us")
    for r in rows:
        print(f"{r['d']:2d}  {r['lam']:8.3g} {r['interval']:9s} {r['provable']:8.4f} {r['int8_match']:8.4f} {r['rerank_match']:10.4f} "
              f"{r['mean_cand']:10.2f} {r['pair_frac']:6.3f} {r['bound_max']:10.3g} {r['regret_max']:10.3g} {r['gemm_us']:7.1f}")


if __name__ == "__main__":
    main()
