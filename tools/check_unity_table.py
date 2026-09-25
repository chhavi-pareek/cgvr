"""Cross-check the C# config table against alloc/config.py.

The Unity port reconstructs the table rather than embedding it, so a drift in either file is
a real divergence. This reads the constants straight back out of ParityTable.cs, rebuilds the
table the way the C# does, and diffs it against build_table().

    python -m tools.check_unity_table
"""
import itertools
import os
import re

import numpy as np

from alloc.config import build_table
from sim.tiered import phase7_table

CS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "unity", "Parity3D", "Assets", "Scripts", "Runtime", "ParityTable.cs")


def _consts(src):
    out = {}
    for m in re.finditer(r"const double\s+(.+?);", src, re.S):
        for part in m.group(1).split(","):
            k, v = part.split("=")
            out[k.strip()] = float(v)
    return out


def _matrix(src, name, consts):
    m = re.search(rf"double\[,\] {name} =\s*\{{(.*?)\n        \}};", src, re.S)
    assert m, f"{name} not found in {CS}"
    rows = re.findall(r"\{([^{}]*)\}", m.group(1))
    out = []
    for r in rows:
        vals = []
        for tok in r.split(","):
            tok = tok.split("//")[0].strip()
            if not tok:
                continue
            for k, v in consts.items():
                tok = re.sub(rf"\b{k}\b", repr(v), tok)
            vals.append(float(eval(tok, {"__builtins__": {}})))  # arithmetic only, no names left
        out.append(vals)
    return np.array(out, np.float64)


def main():
    src = open(CS, encoding="utf-8").read()
    consts = _consts(src)
    q_ax = _matrix(src, "AxisQuality", consts)
    assert q_ax.shape == (4, 4), q_ax.shape
    # the error column follows sim/tiered.py::phase7_table: only a behaviour-tier-3 surrogate
    # diverges, at the measured admission rate
    m = re.search(r"PlazaEMax = ([0-9.]+)", src)
    assert m, "PlazaEMax not found"
    e_max = float(m.group(1))
    assert "return behTier == 3 ? eMax : 0.0" in src, "RowErr changed shape"

    # Allowed(): the same two coupling rules, transcribed from the C#
    body = re.search(r"static bool Allowed\(.*?\n        \}", src, re.S).group(0)
    assert "g == Impostor && a == FullIk" in body, "geometry/animation coupling changed"
    assert "(n == 2 || n == 3) && (b == 0 || b == 1)" in body, "nav/behaviour coupling changed"

    def allowed(t):
        b, n, a, g = t
        if g == 3 and a == 0:
            return False
        if n in (2, 3) and b in (0, 1):
            return False
        return True

    rows = [t for t in itertools.product(range(4), repeat=4) if allowed(t)]
    tiers = np.array(rows, np.int8)
    ax = np.arange(4)
    quality = q_ax[ax, tiers].mean(1)
    err = np.where(tiers[:, 0] == 3, e_max, 0.0)

    py = phase7_table(e_max)
    py_full = build_table()
    ok = True
    for label, a, b in (("m", len(rows), py.m),):
        if a != b:
            print(f"MISMATCH {label}: C# {a} vs Python {b}")
            ok = False
    if not np.allclose(quality, py_full.quality):
        print("MISMATCH quality vs alloc/config.py build_table()")
        ok = False
    if not np.array_equal(tiers, py.tiers):
        print("MISMATCH tiers:", int((tiers != py.tiers).sum()), "entries differ")
        ok = False
    for label, a, b in (("quality", quality, py.quality), ("err", err, py.err)):
        d = np.abs(a - b).max() if a.shape == b.shape else np.inf
        if d > 1e-12:
            print(f"MISMATCH {label}: max abs diff {d:.3e}")
            ok = False
    print(f"rows {len(rows)} (python {py.m}); quality {quality.min():.4f}-{quality.max():.4f}; "
          f"err {{{', '.join(str(v) for v in sorted(set(np.round(err, 5))))}}}  e_max {e_max}")
    print("C# table matches build_table() quality and phase7_table() err exactly" if ok
          else "C# TABLE HAS DRIFTED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
