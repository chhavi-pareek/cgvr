# PARITY

Dual-budget cross-layer fidelity allocation for real-time crowds with bounded
behavioural error. Research implementation for two courses: CGVR (CS373IA) and
Parallel Architecture and GPU Programming (CS372IA).

## What the system does

PARITY assigns per-agent fidelity in a crowd across four axes (behaviour source,
navigation solver, animation tier, geometry tier) by solving a multiple-choice
knapsack every frame: maximise `sum(salience_i * quality(c_i))` subject to a
**global measured frame-time budget** and a **per-agent accumulated-error
budget**, solved by Lagrangian relaxation on the GPU.

The baseline we must beat is threshold-and-cap fidelity assignment reproducing
Unreal Engine 5 MassLOD semantics: distance bands, frustum test, hysteresis,
per-level entity count caps, independently derived visual and simulation tiers.

## Five invariants

These are the research claims. If an implementation choice would violate one,
**stop and tell me** rather than proceeding.

1. Tier costs are **estimated from measured telemetry** (recursive least
   squares), never authored as distance thresholds.
2. Behaviour parameters and animation blend weights decode from **one learned
   latent**, so their errors share a unit. No hand-weighted per-axis costs in
   the main path (only in the ablation condition).
3. Every agent carries an accumulated-divergence value. The allocator may not
   select a configuration that exceeds the cap, which forces periodic
   restoration and prevents unbounded drift.
4. A **view-invariant state core** (goal, route commitment, exit choice, coarse
   progress budget, queue membership) is simulated at fixed cost and is never
   allocated.
5. The allocator's own tier output feeds the **next frame's composite sort key**.

## Hard constraints

- Software only. No hardware, no new data collection, no human subjects.
- GPU target: one free-tier NVIDIA T4 (Turing, sm_75, 16 GB). Assume Nsight
  Compute hardware counters are **unavailable**. Use CUDA events, the CUDA
  Occupancy API, and computed bytes-moved.
- Engine: Unity 6 + URP + DOTS/Burst. **Never** propose or attempt a native CUDA
  plugin bridge into the engine. CUDA and OpenACC live in a standalone module;
  only the validated decomposition is ported to an HLSL compute shader.
- Python prototype first, engine second. Every GPU kernel must have a serial CPU
  reference that acts as its correctness oracle.

## Prior art — do not reinvent, do not claim as novel

ORCA and GPU-parallel ORCA (Charlton et al. 2019); impostor and VAT crowd
rendering; behavioural LOD (Kistler et al. 2010); knapsack-formulated LOD
selection (Funkhouser and Sequin 1993, Mason and Blake 1997); state-sorting for
warp-divergence reduction (Kofler et al.); UE5 MassLOD. Cite these. Adapt them.
Our contribution is the joint formulation, the commensurable error metric, the
error ledger, and the invariance guarantee.

## Repo layout

```
parity/
  sim/          standalone Python simulator, scenes, agent state (SoA)
  alloc/        allocator: serial, threaded, cuda/, openacc/
  latent/       corpus generation, embedding, autoencoder, decoder
  order/        composite-key sorting study
  bench/        harness, sweep runner, CSV logs (gitignored)
  unity/        Unity project; HLSL compute shaders
  figures/      plotting scripts and output
  STATE.md      session handoff, maintained by you
```

## Working rules

**Context economy. I am on a tight usage budget. These are not optional.**

- Read only files relevant to the current task. Do not scan or summarise the
  repo unprompted. Prefer targeted `rg` with a match limit over reading whole
  files.
- Never run a command whose stdout could exceed ~100 lines. Redirect to
  `bench/logs/` and print a short tail or a computed summary instead.
- **Never run the full benchmark sweep** unless I say `RUN SWEEP` explicitly. It
  takes minutes and floods context. During development use N <= 200 and a single
  scene.
- Before any change touching more than three files, output a short plan and wait
  for my confirmation.
- Edits over rewrites. Never reprint an unchanged file.
- Do not add dependencies, create README or docs files, add comments explaining
  obvious code, or refactor adjacent code, unless I ask.
- No preamble, no restating my request, no summary of what you are about to do.
  At most one clarifying question, and only if it blocks you; otherwise state
  your assumption in one line and continue.

## Session protocol

- Each numbered phase is one session. Run `/clear` between phases.
- At the end of every phase, update `STATE.md` with exactly these sections:
  `DONE`, `IN PROGRESS`, `DECISIONS`, `OPEN QUESTIONS`, `NEXT`. Then
  `git add -A && git commit` with a one-line message. Then stop.
- If context runs low mid-phase, update `STATE.md` first, then tell me to
  `/compact`.

## Commands

```
python -m sim.run --scene plaza --agents 200 --condition baseline
python -m bench.sweep            # only on explicit RUN SWEEP
pytest -q tests/                 # oracle agreement tests
```
