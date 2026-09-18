Phase 3 — cost model and serial allocator. Correct before fast.

Deliver:
- `bench/telemetry.py`: per-tier timing harness
- `alloc/costmodel.py`: recursive-least-squares online fit of tier cost from measured
  per-frame timings
- `alloc/serial.py`: Lagrangian-bisection allocator with warm-started lambda
- `alloc/oracle.py`: brute-force optimal solver for small N
- the pruned configuration table with coupling constraints (an impostor-tier agent
  cannot run full IK; a field-navigated agent cannot execute gesture-bearing behaviour)

Acceptance:
- for N <= 200, Lagrangian utility is within 2% of brute-force optimal
- the selected assignment never exceeds the time budget, at any budget setting
- `pytest -q` covers both

Before coding, give me your bisection bounds on lambda, your convergence criterion,
and a short argument for why the returned assignment respects the budget. Wait for my
confirmation on that argument, then implement.
Finish by updating STATE.md and committing.
