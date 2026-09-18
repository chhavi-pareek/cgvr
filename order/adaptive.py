"""Per-frame choice of the key weight w from density statistics.

Features, both O(N) from data the kernel's grid build already has:
  rho  : mean neighbour-list length (agents visited per agent), log1p-scaled
  geff : exp(entropy) of the (tier, class) histogram, the effective number of
         warp-coherence groups
The policy is calibrated from a sweep CSV: every calibration frame contributes
its time-versus-w curve (normalised to its own minimum), and a frame at
feature f takes the argmin of the Gaussian-weighted mean curve of the
calibration frames nearest to f. Time comes from whichever source the sweep had
(CUDA events on the T4, the model under the simulator).

python -m order.adaptive --calib bench/logs/order_sweep.csv --test bench/logs/order_sweep_test.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from .sweep import BLIND


def features(rho, geff):
    return np.array([np.log1p(rho), geff], np.float64)


def _groups(rows):
    g = defaultdict(dict)
    meta = {}
    for r in rows:
        w = float(r["w"])
        if w == BLIND:
            continue
        k = (r["density"], int(r["n"]), int(r["seed"]), int(r["frame"]))
        g[k][w] = float(r["t_us"])
        meta[k] = features(float(r["rho"]), float(r["geff"]))
    return g, meta


class Policy:
    def __init__(self, bandwidth=0.5):
        self.bandwidth = bandwidth

    def fit(self, rows):
        g, meta = _groups(rows)
        keys = sorted(g)
        self.ws = np.array(sorted(g[keys[0]]))
        self.curves = np.array([[g[k][w] for w in self.ws] for k in keys])
        self.curves /= self.curves.min(1, keepdims=True)
        f = np.array([meta[k] for k in keys])
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-9
        self.f = (f - self.mu) / self.sd
        return self

    def choose(self, rho, geff):
        z = (features(rho, geff) - self.mu) / self.sd
        d2 = ((self.f - z) ** 2).sum(1)
        wt = np.exp(-0.5 * d2 / self.bandwidth**2)
        if wt.sum() < 1e-12:
            wt = np.exp(-0.5 * (d2 - d2.min()) / self.bandwidth**2)
        curve = (wt[:, None] * self.curves).sum(0) / wt.sum()
        return float(self.ws[int(np.argmin(curve))])


def evaluate(policy, rows, tol=1e-3):
    """Per (density, n): mean time of the best fixed w (chosen with hindsight on
    these rows) versus the policy's per-frame choice."""
    g, meta = _groups(rows)
    out, ok = [], True
    for d, n in sorted({(k[0], k[1]) for k in g}, key=lambda x: (x[1], x[0])):
        keys = [k for k in g if k[0] == d and k[1] == n]
        ws = sorted(g[keys[0]])
        fixed = {w: np.mean([g[k][w] for k in keys]) for w in ws}
        wbest = min(fixed, key=fixed.get)
        choices = [policy.choose(np.expm1(meta[k][0]), meta[k][1]) for k in keys]
        t_ad = float(np.mean([g[k][c] for k, c in zip(keys, choices)]))
        passed = t_ad <= fixed[wbest] * (1 + tol)
        ok &= passed
        out.append(f"{d:6s} n={n:6d} best fixed w={wbest:.1f} {fixed[wbest]:7.1f} us | "
                   f"adaptive {t_ad:7.1f} us, chose {sorted(set(choices))} | "
                   f"w=0 {fixed[0.0]:7.1f} w=1 {fixed[1.0]:7.1f} | {'ok' if passed else 'WORSE'}")
    out.append("adaptive no worse than the better fixed w at every density: " + ("PASS" if ok else "FAIL"))
    return "\n".join(out), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="bench/logs/order_sweep.csv")
    ap.add_argument("--test", default="bench/logs/order_sweep_test.csv")
    ap.add_argument("--bandwidth", type=float, default=0.5)
    args = ap.parse_args()
    pol = Policy(args.bandwidth).fit(list(csv.DictReader(open(args.calib))))
    print(f"calibrated on {len(pol.curves)} frames, w grid {pol.ws.tolist()}")
    for name, path in (("calibration", args.calib), ("held-out", args.test)):
        try:
            rows = list(csv.DictReader(open(path)))
        except FileNotFoundError:
            print(f"{name}: {path} missing")
            continue
        text, _ = evaluate(pol, rows)
        print(f"[{name}]\n{text}")


if __name__ == "__main__":
    main()
