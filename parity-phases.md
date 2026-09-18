# PARITY phase commands for Claude Code

Each phase is one session. Run `/clear` between them. These become slash
commands: `/phase1` … `/phase8`.

## Install

Run this once in the repo root. It creates `.claude/commands/` and writes all
eight files, plus the STATE.md seed.

```bash
mkdir -p .claude/commands bench/logs figures
python3 - <<'PY'
import os, re, pathlib, sys
src = pathlib.Path("parity-phases.md").read_text()
blocks = re.findall(r"<!-- FILE: (.+?) -->\n```text\n(.*?)```", src, re.S)
for name, body in blocks:
    p = pathlib.Path(name); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body.strip() + "\n")
    print("wrote", name)
PY
```

Then in Claude Code:

```
/model            # pin Fable (Claude Code 2.1.170 or later)
/phase1
```

---

<!-- FILE: STATE.md -->
```text
# STATE

## DONE
nothing yet

## IN PROGRESS
Phase 1

## DECISIONS
- Unity 6 + DOTS over Unreal Mass (documentation maturity for a solo build)
- Python/CuPy prototype before any engine work
- Threshold baseline reproduces MassLOD semantics, not "naive rendering"

## OPEN QUESTIONS
none

## NEXT
Phase 1: baseline and harness
```

---

<!-- FILE: .claude/commands/phase1.md -->
```text
Phase 1 — baseline and harness.

Build the standalone Python simulator and the condition-1 baseline.

Deliver:
- `sim/`: agent state as struct-of-arrays (numpy), fixed-timestep loop, seeded RNG
- three scenes: open plaza, transit hub with a queue, evacuation corridor with a bottleneck
- `sim/assign_threshold.py`: threshold-and-cap fidelity assigner reproducing MassLOD
  semantics — distance bands, frustum test, hysteresis on both distance and visibility,
  per-level count caps, visual and simulation tiers derived independently from one
  significance scalar
- `bench/log.py`: CSV logger keyed on (scene, agent_count, condition, frame)
- a scripted camera-path player so runs are reproducible

Acceptance:
- 2000 agents run headless and log per-frame tier histograms
- identical seed + identical camera path gives byte-identical logs
- `pytest -q` passes a determinism test asserting that

Do not build the allocator yet. Do not touch Unity.
Finish by updating STATE.md and committing.
```

---

<!-- FILE: .claude/commands/phase2.md -->
```text
Phase 2 — behaviour corpus and latent manifold. Runs on the T4.

Deliver:
- `latent/corpus.py`: prompt template generating ~5000 short behaviour descriptions
  (locomotion intent, social context, gesture tag) each with a scalar salience prior
- `latent/embed.py`: sentence-embedding pipeline, cached to disk
- `latent/train.py`: shallow autoencoder to 8–16 dims
- decoder head emitting: preferred-velocity modifier, gesture blend weights, gaze
  target, gait parameters
- `latent/truncation.py`: reconstruction error vs dimensions retained, for both the
  behaviour head and the animation head

Acceptance — this is the gate on invariant 2:
- truncating 16 → 4 dims degrades both heads MONOTONICALLY and smoothly
- if it does not, say so immediately and stop. We fall back to a discrete primitive
  set with a hand-defined pairwise distance matrix, and invariant 2 is dropped.

Report the truncation curve as a small table in STATE.md, not as a plot.
Finish by updating STATE.md and committing.
```

---

<!-- FILE: .claude/commands/phase3.md -->
```text
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
```

---

<!-- FILE: .claude/commands/phase4.md -->
```text
Phase 4 — parallel allocator. CS372IA core.

Four implementations of the Phase 3 allocator, all agreeing with the serial oracle.

Deliver:
- `alloc/threaded.py`: multithreaded CPU (shared-memory reference point, Unit II)
- `alloc/cuda/`: CUDA via Numba-CUDA or CuPy. Explicit thread/block/grid mapping,
  configuration table in constant memory, block reduction for aggregate cost,
  device-side bisection loop so lambda never round-trips to host
- `alloc/openacc/`: directive-based port of the same kernel
- `bench/speedup.py`: speedup, efficiency, and allocator share of frame budget
  (Amdahl framing) at N = 1k, 10k, 100k

Acceptance:
- all four produce bit-identical tier choices to the serial oracle for a fixed seed
- occupancy reasoning is stated explicitly against T4 limits (64 KB constant, 64 KB
  shared per SM, 32-thread warps, sm_75)

Also record: lines of code and rough development time, CUDA vs OpenACC, same kernel.
Do not run anything larger than N = 10k during development.
Finish by updating STATE.md and committing.
```

---

<!-- FILE: .claude/commands/phase5.md -->
```text
Phase 5 — ordering study. CS372IA core.

Characterise the coalescing-versus-coherence Pareto front in agent ordering.

Deliver:
- `order/key.py`: composite sort key combining morton(x,y) with (tier, behaviour_class),
  weighted by w in [0,1]
- `order/sweep.py`: sweep w, at three agent densities (sparse plaza, mixed, dense queue)
- measured per configuration: achieved bandwidth, warp execution efficiency, kernel
  time, bytes moved per agent per frame
- `order/adaptive.py`: policy choosing w per frame from density statistics

Acceptance:
- a reproducible Pareto front in which pure-Morton (w=1) and pure-state (w=0) are each
  dominated somewhere in the density sweep
- the adaptive policy is no worse than the better fixed w at every density

Nsight Compute counters are unavailable. Before implementing, tell me how you will
measure warp execution efficiency without them, and wait for my confirmation.
Finish by updating STATE.md and committing.
```

---

<!-- FILE: .claude/commands/phase6.md -->
```text
Phase 6 — INT8 tensor-core scoring with error bounds.

Deliver:
- `alloc/cuda/score_int8.py`: symmetric INT8 quantisation of the latent basis; scoring
  as an INT8 GEMM on the T4's tensor cores
- a per-candidate quantisation error interval, derived explicitly
- FP32 re-ranking of only those candidates whose intervals overlap the top-1 score
- `bench/quant.py`: fraction of decisions provably identical to FP32, plus a bound on
  deviation for the rest

Acceptance:
- the error interval is derived, not asserted. Show the derivation first and wait for
  my confirmation before implementing.
- exact-match fraction is reported across three latent dimensionalities

Finish by updating STATE.md and committing.
```

---

<!-- FILE: .claude/commands/phase7.md -->
```text
Phase 7 — error ledger, reconciliation, invariant core. The two guarantees.

Deliver:
- per-agent accumulated-divergence value and its cap as a constraint inside the
  allocator (invariant 3)
- `sim/surrogate.py`: compact Markov surrogate over latent regions, fitted so its
  stationary distribution matches a full-fidelity reference run
- `sim/reconcile.py`: on promotion, reconstruct gait phase, social context and intent
  continuity consistent with the surrogate's trajectory
- `sim/invariant.py`: the view-invariant / view-dependent state partition (invariant 4)
- `bench/camerapaths.py`: five distinct scripted camera paths over identical seeds

Acceptance:
- (a) KL divergence from a full-fidelity reference is bounded over a 10-minute run,
  where the threshold baseline's grows without bound
- (b) egress time distribution varies by less than a stated tolerance across the five
  camera paths, where the baseline does not

Before implementing, give me your surrogate state space and your definition of the
invariant core, and wait for confirmation. These two choices decide both results.
Finish by updating STATE.md and committing.
```

---

<!-- FILE: .claude/commands/phase8.md -->
```text
Phase 8 — engine port spec and figures. Design only; I will implement the Unity side
with a cheaper model.

Deliver:
- `unity/PORT_SPEC.md`: HLSL compute-shader port of the validated allocator — buffer
  layout, dispatch shape, thread group size, AsyncGPUReadback path, DOTS integration
  points, and exactly which Phase 4 decomposition choices carry over and which do not
- `figures/SPEC.md`: precise specifications for the four primary figures
  1. behavioural divergence vs elapsed time, one line per condition
  2. outcome variance across camera paths, baseline vs PARITY
  3. sustained agent count at fixed target frame time, with 1% lows
  4. ordering Pareto front plus the four-way speedup chart
- `bench/sweep.py`: the automated sweep over 5 conditions x 3 scenes x 6 agent counts

Do not write Unity C#. Do not write plotting code. Do not run the sweep.
Finish by updating STATE.md and committing.
```
