"""Per-agent bounds do not bound the crowd. Measure how far apart the two quantities get.

Invariant 4 simulates the view-invariant core exactly for every agent at every tier, so agent
states are conditionally independent given the core and the crowd's joint-state KL from the
reference is the SUM of the per-agent divergences. That sum is what any crowd-level statistic
inherits its error from -- and nothing in PARITY bounds it.
"""
import numpy as np
from bench.camerapaths import camera
from sim.tiered import Run, calibrate

cal = calibrate("plaza", n=200, seed=0)
print(f"{'N':>5} {'max_i D_i':>10} {'cap':>6} {'sum_i D_i':>11} {'/ (N*cap)':>10} {'per-agent mean':>15}")
for n in (100, 200, 400, 800):
    r = Run("plaza", n, "parity", seed=0, cam=camera("plaza", "orbit", 600), calib=cal)
    for f in range(600):
        r.step(f)
    D = r.ledger.D
    tot = float(D.sum())
    print(f"{n:>5} {float(D.max()):>10.3f} {r.cap:>6.3f} {tot:>11.1f} "
          f"{100*tot/(n*r.cap):>9.1f}% {float(D.mean()):>15.3f}")
print("\nmax_i D_i is flat in N (the per-agent cap holds).")
print("sum_i D_i grows linearly in N and is bounded only by N*cap, i.e. not bounded at all.")
