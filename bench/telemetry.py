"""Per-tier timing harness.

The simulator records tiers but does not yet execute tiered work, so this
module times stand-in numpy workloads whose size scales with the number of
agents in each axis-tier. Per-group times are ground truth for validating the
cost model; the cost model itself is fitted from total frame time only.
"""
import argparse
import csv
import pathlib
import time

import numpy as np

from alloc.config import AXES, N_AXES, N_TIERS, TIERS
from alloc.costmodel import RLSCostModel

_K = (16, 8, 4, 0)
_NEIGH = (32, 8, 1, 1)
_VERTS = (4000, 1000, 250, 4)
_BONES = 64


class Workloads:
    def __init__(self, n, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.W = rng.standard_normal((16, 19)).astype(np.float32)
        self.emb = rng.standard_normal((n, 16)).astype(np.float32)
        self.pos = rng.uniform(0, 50, (n, 2)).astype(np.float32)
        self.vel = rng.standard_normal((n, 2)).astype(np.float32)
        self.field = rng.standard_normal((64, 64, 2)).astype(np.float32)
        self.bones = rng.standard_normal((_BONES, 4, 4)).astype(np.float32)
        self.local = rng.standard_normal((n, _BONES, 4, 4)).astype(np.float32)
        self.chain = rng.standard_normal((n, 4, 3)).astype(np.float32)
        self.verts = rng.standard_normal((_VERTS[0], 3)).astype(np.float32)
        self.M = rng.standard_normal((n, 3, 3)).astype(np.float32)

    def behaviour(self, c, tier):
        k = _K[tier]
        if c == 0 or k == 0:
            return
        np.tanh(self.emb[:c, :k] @ self.W[:k])

    def navigation(self, c, tier):
        if c == 0:
            return
        p = self.pos[:c]
        if tier < 2:
            acc = np.zeros((c, 2), np.float32)
            for j in range(_NEIGH[tier]):
                d = p - np.roll(p, j + 1, axis=0)
                dist = np.sqrt((d * d).sum(1, keepdims=True)) + 1e-3
                acc += d / dist * np.maximum(0, 2.0 - dist)
            np.clip(acc, -1, 1)
        elif tier == 2:
            ij = np.clip((p * (64 / 50)).astype(np.int32), 0, 63)
            self.field[ij[:, 0], ij[:, 1]] * 0.1 + self.vel[:c]
        else:
            p + self.vel[:c] * (1 / 30)

    def animation(self, c, tier):
        if c == 0 or tier == 3:
            return
        if tier < 2:
            np.einsum("bij,abjk->abik", self.bones, self.local[:c])
            if tier == 0:
                ch = self.chain[:c].copy()
                for _ in range(8):
                    d = ch[:, 1:] - ch[:, :-1]
                    ch[:, 1:] -= 0.1 * d / (np.linalg.norm(d, axis=2, keepdims=True) + 1e-3)
        else:
            self.verts[(np.arange(c) * 7) % _VERTS[0]]

    def geometry(self, c, tier):
        if c == 0:
            return
        np.einsum("vj,aij->avi", self.verts[: _VERTS[tier]], self.M[:c])

    def frame(self, counts):
        """Run one frame. Returns (total_ms, per_group_ms[N_AXES, N_TIERS])."""
        fns = (self.behaviour, self.navigation, self.animation, self.geometry)
        per = np.zeros((N_AXES, N_TIERS))
        t_frame = time.perf_counter()
        # invariant core: fixed cost, every agent
        self.pos + self.vel * (1 / 30)
        for ax in range(N_AXES):
            for tier in range(N_TIERS):
                t0 = time.perf_counter()
                fns[ax](int(counts[ax, tier]), tier)
                per[ax, tier] = (time.perf_counter() - t0) * 1e3
        return (time.perf_counter() - t_frame) * 1e3, per


def random_counts(rng, n):
    h = np.zeros((N_AXES, N_TIERS), np.int64)
    for ax in range(N_AXES):
        p = rng.dirichlet(np.ones(N_TIERS))
        h[ax] = rng.multinomial(n, p)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=200)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench/logs/telemetry.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    w = Workloads(a.agents, a.seed)
    model = RLSCostModel()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    per_sum = np.zeros((N_AXES, N_TIERS))
    cnt_sum = np.zeros((N_AXES, N_TIERS))
    tail = []
    for _ in range(3):
        w.frame(random_counts(rng, a.agents))  # warm caches
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "total_ms", "pred_ms"] + [f"{AXES[x]}_{TIERS[x][t]}" for x in range(N_AXES) for t in range(N_TIERS)])
        for fr in range(a.frames):
            h = random_counts(rng, a.agents)
            pred = model.predict(h)
            total, per = w.frame(h)
            model.update(h, total)
            per_sum += per
            cnt_sum += h
            if fr >= a.frames - 50:
                tail.append((total, pred))
            wr.writerow([fr, f"{total:.4f}", f"{pred:.4f}"] + [f"{v:.4f}" for v in per.ravel()])
    tail = np.array(tail)
    rel = np.sqrt(np.mean(((tail[:, 0] - tail[:, 1]) / tail[:, 0]) ** 2))
    meas = per_sum / np.maximum(cnt_sum, 1) * 1e3
    fit = model.theta_axis * 1e3
    print(f"N={a.agents} frames={a.frames}  last-50 rel RMS error of predicted total: {rel:.3f}")
    print(f"core (us/agent, fitted): {model.theta_core*1e3:.2f}")
    print("us/agent  measured(per-group) | fitted(incremental over tier 3)")
    for ax in range(N_AXES):
        m = " ".join(f"{v:7.2f}" for v in meas[ax])
        f_ = " ".join(f"{v:7.2f}" for v in fit[ax])
        print(f"{AXES[ax]:11s} {m} | {f_}")


if __name__ == "__main__":
    main()
