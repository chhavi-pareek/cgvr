import json
import sys

import numpy as np
import torch

from latent.corpus import ANIMATION, BEHAVIOUR, DATA
from latent.train import D, get_data, head_err, head_var, load_model

KS = list(range(D, 3, -1))  # 16 .. 4
SEEDS = range(3)
TOL = 1e-3  # allowed non-monotone wobble, in normalised-MSE units
CLIFF = 0.35  # no single k-step may carry more than this share of the total 16->4 degradation
TABLE_KS = [16, 12, 10, 8, 6, 5, 4]


def curves(nested, seed):
    x, y, tr, te = get_data()
    m, order, fill = load_model(seed, nested)
    var = {"behaviour": head_var(y, tr, BEHAVIOUR), "animation": head_var(y, tr, ANIMATION)}
    out = {"behaviour": [], "animation": []}
    with torch.no_grad():
        z = m.enc(x[te])
        for k in KS:
            zk = fill.expand_as(z).clone()
            zk[:, order[:k]] = z[:, order[:k]]
            pred = m.decode(zk)
            out["behaviour"].append(head_err(pred, y, te, BEHAVIOUR, var["behaviour"]).item())
            out["animation"].append(head_err(pred, y, te, ANIMATION, var["animation"]).item())
    return {h: np.array(v) for h, v in out.items()}


def check(err):
    d = np.diff(err)  # k=16 -> 4, so each step is one more dim dropped
    total = err[-1] - err[0]
    return {
        "monotone": bool((d >= -TOL).all()),
        "worst_wobble": float(max(0.0, -d.min())),
        "total_increase": float(total),
        "max_step_share": float(d.max() / total) if total > 0 else float("nan"),
        "smooth": bool(total > 0 and d.max() <= CLIFF * total),
    }


def main():
    res = {}
    for nested in (True, False):
        name = "nested" if nested else "plain"
        per_seed = [curves(nested, s) for s in SEEDS]
        res[name] = {
            h: {
                "mean": np.mean([c[h] for c in per_seed], 0).tolist(),
                "std": np.std([c[h] for c in per_seed], 0).tolist(),
                "mean_check": check(np.mean([c[h] for c in per_seed], 0)),
                "seed_checks": [check(c[h]) for c in per_seed],
            }
            for h in ("behaviour", "animation")
        }
    (DATA / "truncation.json").write_text(json.dumps({"ks": KS, "results": res}, indent=1))

    print("| k | nested beh | nested anim | plain beh | plain anim |\n|---|---|---|---|---|")
    for k in TABLE_KS:
        i = KS.index(k)
        print(f"| {k} | " + " | ".join(f"{res[n][h]['mean'][i]:.4f}" for n in ("nested", "plain") for h in ("behaviour", "animation")) + " |")
    ok = True
    for n in ("nested", "plain"):
        for h in ("behaviour", "animation"):
            c = res[n][h]
            seeds_ok = all(s["monotone"] and s["smooth"] for s in c["seed_checks"])
            mc = c["mean_check"]
            print(f"{n}/{h}: mean monotone={mc['monotone']} smooth={mc['smooth']} wobble={mc['worst_wobble']:.4f} "
                  f"total={mc['total_increase']:.4f} max_step_share={mc['max_step_share']:.2f} all_seeds_ok={seeds_ok}")
            if n == "nested":
                ok &= mc["monotone"] and mc["smooth"] and seeds_ok
    print("GATE (nested):", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
