# STATE

## DONE
- Phase 1: standalone simulator (`sim/`), three scenes (plaza, hub with queue, corridor with bottleneck), scripted camera paths, MassLOD-style threshold-and-cap assigner, CSV logger (`bench/log.py`), determinism tests.
- Acceptance: 2000 agents headless (plaza, 200 frames, ~21 s) log per-frame visual/sim tier histograms; two runs with the same seed and camera path are byte-identical (`cmp`); `pytest -q` 7 passed.
- Repo initialised (`git init`), `pytest` installed, `.gitignore` and `pytest.ini` added.
- Phase 2 (built and run locally on CPU, not the T4): `latent/corpus.py` (5000 unique template-grammar descriptions, salience prior, synthetic decoder targets), `latent/embed.py` (all-MiniLM-L6-v2, cached in `latent/cache/`), `latent/train.py` (384 -> 16-dim AE, four decoder heads), `latent/truncation.py`. `torch` and `sentence-transformers` installed with your OK. Full training of 6 models took about 49 s.
- Phase 2 gate on invariant 2: **CLEARED (accepted after a 10-seed rerun).** Held-out normalised MSE (1.0 = predict the mean), mean of 10 seeds, k = latent dims kept:

  | k | nested beh | nested anim | plain beh | plain anim |
  |---|---|---|---|---|
  | 16 | 0.132 | 0.196 | 0.109 | 0.162 |
  | 12 | 0.146 | 0.246 | 0.336 | 0.416 |
  | 10 | 0.172 | 0.327 | 0.445 | 0.552 |
  | 8 | 0.213 | 0.435 | 0.563 | 0.680 |
  | 6 | 0.283 | 0.584 | 0.683 | 0.824 |
  | 5 | 0.361 | 0.664 | 0.742 | 0.862 |
  | 4 | 0.449 | 0.746 | 0.792 | 0.897 |

  Nested = ordered-truncation (nested-dropout) training, the main path. Plain = ordinary AE, dims ordered by latent variance, dropped dims set to the training mean.
  - Nested, checked at every k from 16 to 4 (13 points): monotone on the mean curve and on all 10 seeds, both heads (0 of 20 failures). Smoothness (no single k-step carries more than 35% of the total 16->4 rise) holds on both mean curves. Per seed it fails once in 20: behaviour seed 2 (max step share 0.36). The other behaviour seeds range 0.26-0.35, so that head sits close to the 35% line and the threshold is tight for it. Animation seeds range 0.17-0.25.
  - Accept rule, fixed before the rerun: 0 monotone failures and at most 1 of 10 smooth failures per head. Met.
  - Plain: monotone on the mean curve. Per seed, monotone fails on 1 of 10 behaviour and 2 of 10 animation seeds.
  - At k=4 the nested animation head keeps only about a quarter of the variance (error 0.75), the behaviour head about 55% (error 0.45). Degradation is smooth, but the animation head is close to unusable at 4 dims.

- Phase 3: `alloc/config.py` (4 axes x 4 tiers, 256 -> 180 rows after coupling; behaviour and animation quality = 1 - phase 2 nested truncation nMSE at k=16/8/4, core/none = 0), `alloc/costmodel.py` (RLS, forgetting 0.98, 13 features: per-agent core + 12 incremental axis-tier counts, tier 3 is the zero reference), `bench/telemetry.py` (stand-in numpy per-tier workloads timed with perf_counter, per-group ground truth + total; RLS fits from total only), `alloc/serial.py` (Lagrangian bisection, warm-started lambda, greedy fill of the integrality gap), `alloc/oracle.py` (brute force N<=8; exact DP over integer-tick costs at N=200), `tests/test_alloc.py`.
- Phase 3 acceptance (N=200, m=180, 6 seeds x 3 budgets, exact DP oracle on quantised costs): utility gap max 0.019% with fill, 0.057% without (bound: 2%). Budget sweep 0 -> 1.2 T(0) in 60 steps, all returned costs <= budget. Cold start 22 Lagrangian evaluations mean (max 28); warm start fewer, tested. Serial allocate ~2 ms at N=200. `pytest -q` 26 passed (7 phase 1 + 19 phase 3).
- Telemetry (N=200, 300 random-histogram frames, `python -m bench.telemetry`): last-50 relative RMS error of predicted total 6%. Geometry (250/62/16 us per agent) and animation (22/20 us) recovered to within a few us; navigation (9/2 us) is below the noise floor of geometry and is misattributed between tiers.

- Phase 4 (built and validated locally on CPU; nothing has run on the T4 yet): `alloc/common.py` (shared host prep: lam_max, headroom feasibility, utility/cost readout; `drive()` mirrors the serial bracketing/bisection), `alloc/threaded.py` (numba `prange`, OpenMP-style fork/join per Lagrangian evaluation, bisection on the host), `alloc/cuda/` (Numba-CUDA: one thread per agent, 256-thread blocks, quality/err table in constant memory, per-frame cost vector staged in shared memory, block tree reduction, last-block-done pattern advances a device-resident bracketing/bisection state machine so lambda never round-trips; host launches in batches of 8 and reads only a done flag; warm-start lambda lives on the device between frames), `alloc/openacc/` (`alloc_acc.c` same kernel with `#pragma acc parallel loop reduction`, host-driven; ctypes wrapper; `make serial` for a host-only build, `make acc` for nvc/T4), `bench/speedup.py`, `tests/test_parallel.py`.
- Phase 4 acceptance, bit-identical tier choices against `SerialAllocator` (plus identical lambda, eval count, fill steps, cost and utility) over warm-started budget sweeps (cold start, expand-up, expand-down, bisection), with and without headroom masks, integer-tick and float costs, degenerate cases (budget below floor, all-equal costs): threaded at N=1000/500/300 (10 tests), OpenACC C compiled serially with clang `-ffp-contract=off` at N=1000/500/200 (4 tests), CUDA under numba's CUDA simulator at N=40 with 32-thread blocks so the cross-block last-block reduction runs (3 tests, ~90 s of the suite). `pytest -q tests/` 43 passed in 116 s.
- Speedup on this M3 (8 cores, 4P+4E), N=1k / 10k, budget 0.5 T(0), warm-started frames, median of 6 (`bench/logs/speedup.csv`):

  | N | serial (numpy) | threaded x1 | x2 | x4 | x8 | openacc (serial-C build) |
  |---|---|---|---|---|---|---|
  | 1k | 7.1 ms | 5.4 ms (1.3x) | 3.1 ms (2.3x) | 3.0 ms (2.4x) | 2.6 ms (2.7x) | 3.7 ms (1.9x) |
  | 10k | 288 ms | 79 ms (3.6x) | 41 ms (7.0x) | 26.5 ms (10.9x) | 19.6 ms (14.7x) | 63 ms (4.6x) |

  Efficiency against the same JIT code on one thread at 10k: x2 0.96, x4 0.75, x8 0.51 (E-cores). The x1 vs numpy gap is the N x m score matrix the serial version materialises. Allocator share of a 16.7 ms frame at 10k: serial >100%, best CPU 117%; Amdahl frame-speedup bound with p = allocator share equals the allocator speedup there because the allocator is the whole frame. At 1k: 43% serial, 16% best, bound 1.37x. 100k not run (development cap).
- Lines of code (non-blank, non-comment), same kernel: CUDA device code + kernels 283 (Python), host class 95; OpenACC C 153 including its host driver and fill. Rough development time: CUDA ~25 min (device state machine, one phase-fallthrough bug caught in review, FMA-contraction workaround, simulator tuning), OpenACC ~10 min (compiled and matched the oracle first run). Whole phase ~45 min wall clock.

## IN PROGRESS
- Nothing.

## DECISIONS
- Repo root is `cgvr/`; `sim/`, `bench/`, `tests/` sit at root.
- Tier index: 0=HIGH, 1=MED, 2=LOW, 3=OFF.
- One significance scalar per agent: 3D distance to camera, multiplied by 2.0 when out of view (after visibility hysteresis). Visual and sim tiers each apply their own band table, hysteresis state, and per-level caps to that scalar. Visual is forced OFF when out of view; sim is not.
- Distance hysteresis is 10% on band edges. Visibility hysteresis is a 15-frame hold after leaving the frustum. Caps demote lowest-significance agents first; hysteresis state stores the pre-cap tier.
- Tiers are recorded only. Motion is full-rate reference dynamics (goal-seek plus O(N^2) chunked separation, no ORCA) for every agent regardless of tier.
- Log holds integer histograms only (no floats), so byte-identity is exact. Fixed dt 1/30, single `default_rng(seed)`.
- Phase 2 heads: behaviour head = preferred-velocity modifier (3) + gaze target (3); animation head = gesture blend weights (8, softmax) + gait parameters (5). All decode from one 16-dim latent. Training loss is the equal-weight sum of embedding reconstruction and the four per-group variance-normalised MSEs.
- Phase 2 targets are synthetic: seeded random loadings per categorical field, one intent x social interaction, tanh squashing, plus noise 0.05. The corpus has no real behaviour or animation ground truth, so the gate tests whether the latent degrades gracefully under truncation. It does not test that real behaviour decodes from text.
- Phase 2 gate criteria were fixed before running: monotone within 1e-3 at every k from 16 to 4, and no single step above 35% of the total rise. Not retuned after the 3-seed result; the 10-seed rerun used the same criteria plus a pre-set accept rule (see DONE).
- Hub queue is a scripted service window (front agent released every 45 frames, queuers rejoin at the tail). Corridor door is a hard clamp on a 1.6 m gap at x=30.

- Phase 3 cost model: `frame_ms = theta_core * N + sum theta[axis,tier<3] * n[axis,tier]`. Tier 3 of each axis is the zero-cost reference so the fit is identifiable; `theta_core` absorbs the invariant-core fixed cost plus lowest-tier costs. RLS recursion is unclamped (clamping inside the recursion diverged); costs read out are clamped >= 0. P-trace cap at 1e4 per feature guards against blow-up under forgetting.
- Allocator: rows sorted (cost asc, quality desc) so argmax tie-break prefers the cheaper row. lambda_lo = 0, lambda_hi = max U / min positive cost gap (every agent at min cost). Stop when hi-lo <= 1e-4 hi or budget slack <= 1e-3 budget, max 64 steps. Returned assignment is always the evaluated feasible end `hi`, then greedy fill by utility-gain/cost ratio while upgrades fit. If min-cost assignment exceeds the budget, return it with `infeasible=True`; never violate the error headroom mask.
- Config quality q(c) = unweighted mean of four per-axis fidelities in [0,1]. Error rate e(c) = behaviour nMSE + navigation deviation (visual axes contribute zero divergence). Couplings: impostor geometry excludes IK animation; field/core navigation excludes gesture-bearing behaviour (latent16, latent8). Dominance pruning is a runtime helper (`prune_dominated`), not applied by default.
- Oracle: brute force validates the DP at N=6 (5-row subtables); the DP is the N=200 oracle. Tests quantise costs to integer ticks and hand the same quantised costs to both solvers, so the comparison is exact.

- Phase 4 tie-break: parallel versions scan the unsorted table with an explicit comparator (score desc, cost asc, quality desc, index asc); this equals first-argmax over the serial's lexsorted rows, so the per-frame sort is gone and the fixed table can sit in constant memory.
- FMA contraction breaks bit-identity of score = (s*q) - (lam*c). NVVM contracts by default and numba exposes no flag, so real-hardware CUDA uses libdevice `dmul_rn`/`dadd_rn` (never contracted); the simulator uses plain ops. C builds use `-ffp-contract=off` (clang) / `-Mnofma` (nvc). Numba CPU does not contract without fastmath.
- Aggregate cost: threaded sums `c[a]` with numpy in agent order, the oracle's own order, so it matches for float costs too. CUDA sums per-block partials in fixed order (deterministic run to run) but not in numpy's order; with integer-tick costs the sum is exact, with float costs a last-bit difference could flip a knife-edge bisection decision. Acceptance is on ticks plus float instances that happened to agree.
- Numba binds `cuda.const.array_like` at compile time, so only quality and err (fixed) go in constant memory; the per-frame RLS cost vector is staged into shared memory by each block (1.4 KB). A CUDA-C port would `cudaMemcpyToSymbol` it. Kernels are compiled per table via a closure factory.
- CUDA host loop: launch 8 kernels, read the done flag (one 64 B D2H), repeat; early-exit launches cost a few us each. A cooperative-groups persistent kernel would remove even that but cannot run under the simulator, so it was not written.
- Occupancy statement for `k_eval` on T4 (sm_75, 40 SMs, 64 KB shared/SM, 64K regs/SM, 32 warps/SM, 16 blocks/SM, 64 KB constant): TPB 256 = 8 warps; static shared 4.1 KB (eval) / 6.1 KB (fill) per block, so shared allows 15 blocks/SM; constant 4 KB of 64 KB; threads allow 4 blocks/SM. Registers decide: <= 64 regs/thread gives 4 blocks = 32/32 warps (100%), 65-80 regs gives 3 blocks (75%). `occupancy_report(alloc)` prints regs/thread and the driver's active-blocks figure on hardware. fp64 runs at 1/32 rate on Turing: the inner loop is 2 mul + 1 sub + compare per (agent, row), ~54M fp64 ops at N=100k, of order 0.2 ms per evaluation, so per-launch overhead and the ~15 launches per frame matter more than ALU.
- Greedy fill is inherently sequential (one global argmax per step). Each step is a full N x m pass in every implementation; at N=10k the benchmark instance took 18 fill steps, more than the 13 Lagrangian evaluations. Kept for oracle identity; a capped or parallel-batch fill is a phase 5/8 decision.
- Threaded uses numba `prange` rather than Python threads; `threads=k` sets `numba.set_num_threads`. Numba could not analyse a reduction on a variable assigned in the inner loop, hence the array write plus host sum.
- `tests/conftest.py` and `alloc/cuda/__init__.py` set `NUMBA_ENABLE_CUDASIM=1` when no `nvidia-smi` is on PATH (or on macOS). On the T4 leave it unset. `numba` was installed (phase spec names Numba-CUDA); no other dependency added.

## OPEN QUESTIONS
- Should MassLOD sim tiers throttle tick rate or freeze agents in the baseline? Currently they do not. This decides how "behavioural error" is measured against the baseline in phase 3.
- At N=2000 the sim histogram sits on the caps (80/400/1200/320) in the plaza. Cap sizes are placeholders; tune before treating the baseline as a fair opponent.
- Nested-dropout training makes the dimension ordering hold by construction. The plain AE, which is the un-engineered evidence, is also monotone on the mean curve but not on every seed (3 of 20 seed/head curves fail). Both rest on synthetic targets (see DECISIONS).
- The phase spec says Phase 2 runs on the T4; this was run on CPU. Rerun `python -m latent.train && python -m latent.truncation` on the T4 before treating the numbers as final (you asked for a T4 reminder after all phases).
- Stray directory in repo root with a multi-line garbled name (from a failed setup script paste). Empty, untracked by git, not touched.

- Navigation and geometry per-axis fidelities in `alloc/config.py` are authored placeholders (1/0.9/0.6/0.3 and 1/0.75/0.5/0.25) and so is the navigation error rate. They are not in the latent, so invariant 2 covers only behaviour and animation. Replace with measured trajectory deviation (nav) and screen-space geometric error (geometry) once tiers execute.
- Telemetry times stand-in numpy kernels, not the simulator (phase 1 records tiers without executing tiered work). RLS fits genuine measured time, but the cost figures are not the crowd's. Navigation costs are unidentifiable under geometry noise at N=200; expect the same on the T4 unless nav work is real.
- Budget means predicted cost under the RLS model. `resid_var` is tracked; subtracting a k-sigma safety margin from the budget is not yet done.

- Nothing in phase 4 has executed on a GPU. On the T4: `pytest -q tests/test_parallel.py -k cuda` (kernels compile, libdevice path, constant memory from closure arrays, last-block pattern under real scheduling), `cd alloc/openacc && make acc` then `-k openacc`, then `python -m bench.speedup --sizes 1000 10000 100000` with CUDA-event timing if the perf_counter + synchronize numbers look launch-bound. `occupancy_report(CudaAllocator(table))` for the measured regs/thread.
- OpenACC with nvc: `#pragma acc update self(bj[besti:1], abuf[off+besti:1])` inside the fill and the `present(...)` subarray lookups have only been compiled as plain C. The `reduction(max:bestr)` with an `if` update is legal but check `-Minfo=accel` output.
- CUDA float-cost totals are summed in a different order from the oracle (see DECISIONS); if the T4 shows a mismatch on float costs at a knife edge, quantise costs to ticks inside `prepare()` for all implementations including the serial.

## NEXT
- Phase 5: ordering study (composite-key sort, warp-divergence). Invariant 5 (allocator tier output feeds next frame's sort key) hooks in at the assignment readback in `CudaAllocator.allocate`.
- First T4 session: run the phase 4 GPU checks listed under OPEN QUESTIONS before the phase 8 reminder.
