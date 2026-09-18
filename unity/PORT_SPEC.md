# PARITY allocator: HLSL compute-shader port specification

Target: Unity 6, URP, DOTS/Burst, D3D12 or Vulkan on one NVIDIA T4 (sm_75). No native
plugin. The validated decomposition is `alloc/cuda/allocator.py` (phase 4, oracle-identical
to `alloc/serial.py`); the ledger and headroom mask are `alloc/ledger.py` (phase 7); the
sort key is `order/key.py` (phase 5). Everything in this document is design; nothing here
has been compiled.

## 0. What is ported and what is not

| Component | Where it runs in the engine | Source of truth |
|---|---|---|
| Lagrangian evaluation kernel (`k_eval`) | compute shader `Alloc.compute`, kernel `Eval` | `alloc/cuda/allocator.py::k_eval` |
| Bracketing / bisection state machine (`_advance`) | same shader, thread 0 of the last group | `alloc/common.py::drive`, `_advance` |
| Greedy fill (`k_fill`) | kernel `Fill`, capped at `FILL_MAX` dispatches | `k_fill` |
| Host prep (`prepare`: lambda_max, headroom feasibility) | Burst job, CPU | `alloc/common.py::prepare` |
| RLS cost model | Burst job, CPU, 13 features | `alloc/costmodel.py` |
| Error ledger, headroom mask, reconciliation | Burst jobs, CPU | `alloc/ledger.py`, `sim/reconcile.py` |
| Invariant core | Burst job, CPU, every agent every frame | `sim/invariant.py::Core` |
| Composite sort key | Burst job (key) + GPU radix sort or Burst sort (permutation) | `order/key.py` |
| INT8 tensor-core scoring (phase 6) | **not ported** | stays a standalone CUDA result |
| OpenACC allocator | **not ported** | standalone |

Phase 6 is out because HLSL on Turing has no tensor-core intrinsic (`dot4add_i8packed`,
SM 6.4, is packed INT8 on the ALUs, not IMMA) and the allocator is launch-bound below
N = 50k anyway. The core, ledger and reconciliation stay on the CPU because they own the
guarantees (invariants 3 and 4) and their per-agent work is a few flops.

## 1. Numeric contract (this is the one Phase 4 choice that does not carry over)

Phase 4 achieved bit identity with the fp64 serial oracle by routing `s*q - lam*c` through
libdevice `dmul_rn`/`dadd_rn` so NVVM could not contract to FMA. In the engine:

- **Scores are fp32**, declared `precise` on the score expression (DXC then emits no FMA
  and keeps IEEE evaluation order). fp64 in HLSL is optional per device, absent on Metal,
  1/32 rate on Turing, and the bisection only needs `rtol = 1e-4` relative on lambda.
- **Costs are integer ticks** (`uint`, 1 tick = 0.1 us of predicted frame time), produced
  by the RLS job with `round()`; the budget is in the same ticks. Every cost sum is then an
  exact integer, so (a) the cross-group reduction is order-independent and the phase 4
  float-order caveat disappears, (b) `slack` in the fill is exact. The score uses
  `float(c_j)`; a tick count below 2^24 converts exactly.
- **Comparator** carries over verbatim: score desc, cost asc, quality desc, row index asc,
  scanning the unsorted table in the same `j` order in every thread.
- **Oracle**: bit identity is claimed against a **Burst fp32 reference** of the same kernel
  (`FloatMode.Strict`, same statement order, same comparator, integer ticks), not against
  the Python fp64 oracle. Python validates the decomposition; the Burst job validates the
  port. The test asserts identical `assign`, `lambda`, evaluation count, fill steps and total
  cost over the phase 4 test vectors (cold, expand-up, expand-down, bisection, headroom
  masks, degenerate budgets), exported from `tests/test_parallel.py` as binary blobs.

## 2. Buffers

Sizes at N = 100 000 agents, m <= 256 rows (`MAXM`, the phase 4 constant; the table has 180).

| Name | HLSL type | Elements | Bytes | Written by | Lifetime |
|---|---|---|---|---|---|
| `salience` | `StructuredBuffer<float>` | N | 400 KB | CPU each frame (`LockBufferForWrite`) | double-buffered |
| `headroom` | `StructuredBuffer<float>` | N | 400 KB | CPU each frame | double-buffered |
| `assign` | `RWStructuredBuffer<uint>` | 3 x N | 1.2 MB | GPU | persistent |
| `partials` | `globallycoherent RWStructuredBuffer<uint>` | G | 1.6 KB | GPU | persistent |
| `fillR/fillI/fillJ` | `globallycoherent RWStructuredBuffer<float/uint/int>` | G each | 4.7 KB | GPU | persistent |
| `st` | `globallycoherent RWStructuredBuffer<float>` | 16 | 64 B | CPU at frame start (lam_max, budget, tolerances); GPU state | persistent |
| `ist` | `globallycoherent RWStructuredBuffer<int>` | 16 | 64 B | CPU at frame start; GPU state | persistent |
| `warm` | `RWStructuredBuffer<float>` | 1 | 4 B | GPU only | persistent across frames, never read back |
| `args` | `RWStructuredBuffer<uint>` on a `GraphicsBuffer` with `Target.IndirectArguments \| Target.Structured` | 3 | 12 B | CPU sets (G,1,1) at frame start; GPU zeroes on finish | persistent |
| `TableCB` | `cbuffer` | 2 x 256 floats packed as `float4[64]` | 2 KB | CPU once | static |
| `CostCB` | `cbuffer` | 256 `uint` packed as `uint4[64]` | 1 KB | CPU each frame (`SetConstantBuffer`) | per frame |

G = ceil(N / 256) = 391 groups at 100k. `assign` is indexed `[slot * N + i]` with slots
`cur/lob/hib` rotated through `ist[I_CUR..I_HIB]` exactly as in phase 4. The `st`/`ist`
layouts are the phase 4 index constants (`S_LAM..S_TF`, `I_PHASE..I_NODC`) plus
`ist[15] = FILL_COUNT`.

The per-frame cost vector goes in a **constant buffer**, which is the `cudaMemcpyToSymbol`
the phase 4 notes asked for: HLSL cbuffer reads with a warp-uniform index are broadcast
loads, so the phase 4 shared-memory staging of `cost` (1.4 KB per group, plus a barrier)
is dropped. The fixed table (quality, err) is a second cbuffer. cbuffer arrays are
16-byte strided, hence the `float4[64]` packing with `q[j] = Q4[j >> 2][j & 3]`.

`globallycoherent` on `partials`, the fill buffers, `st` and `ist` is required: without it
the last group may read stale L1 lines when it sums the other groups' partials. It is the
HLSL replacement for phase 4's `cuda.threadfence()`.

## 3. Kernels and dispatch shape

`[numthreads(256,1,1)]`, 1-D groups, one thread per agent, `i = groupID.x * 256 + tid`.
Groups count G is read from `args[0]` inside the kernel (that is the last-group ticket
target, phase 4's `gridDim.x`).

### `Eval` (phase 4 `k_eval`)

1. `if (ist[I_DONE]) return;` (kept as a guard; normally unreachable, see dispatch chain)
2. Per agent: scan `j = 0..m-1`, skip rows with `err[j] > headroom[i]`, `precise float sc =
   s*q[j] - lam*float(c[j])`, comparator as above, write `assign[cur*N + i] = bj`, keep
   `bc` (ticks) as this thread's contribution.
3. Group reduction of `bc` in `groupshared uint part[256]` (tree, 8 steps,
   `GroupMemoryBarrierWithGroupSync`). Thread 0 writes `partials[g]`, then
   `DeviceMemoryBarrier()`, then `InterlockedAdd(ist[I_TICKET], 1, ticket)`, sets
   `groupshared flag = (ticket == G-1)`.
4. Last group: `DeviceMemoryBarrier()`, strided sum of `partials[0..G)` into `part[]` (uint,
   exact), tree-reduce, thread 0 resets the ticket and runs `Advance(total)`.
5. `Advance` is `_advance` line for line (phases 0, 1, 2, 3, 5, 6, cold path `ph = -1`).
   `Finish` additionally writes `args[0] = 0` and, if the fill is enabled and the result is
   feasible, `argsFill[0] = G`.

### `Fill` (phase 4 `k_fill`)

Same shape. Per agent: best upgrade ratio `du/dc` over feasible rows with `0 < dc <=
slack`, `du = s*q[j] - s*q[a_i]` in `precise` fp32, `dc` in ticks. Group argmax on `(r desc,
i asc)` in `groupshared float shr[256]; uint shi[256]; int shj[256]` (3 KB). Last group
applies the single upgrade, `slack -= dc`, `ist[I_FSTEPS]++`; when no upgrade exists or
`ist[I_FSTEPS] == FILL_MAX` it writes `argsFill[0] = 0`.

### `ForceFinish`

`[numthreads(1,1,1)]`, one group. If `!ist[I_DONE]`: `Finish(hib, hi, t_hi)` (the
bracketing always leaves `hib` feasible after phase 1 has been evaluated; before that,
i.e. after only the lambda = 0 evaluation, the result is `cur` with `infeasible = 1`,
mirroring phase 4's phase-0/phase-1 exits). Also zeroes both args buffers.

### Dispatch chain (replaces phase 4's batch-of-8 plus synchronous done-flag readback)

Unity has no synchronous readback in a frame budget and `AsyncGPUReadback` lands frames
later, so the CPU cannot poll `ist[I_DONE]` between launches. Instead the whole per-frame
allocation is a fixed command list recorded once:

```
SetConstantBuffer(CostCB); SetBuffer(...)             // per frame
args = (G,1,1); argsFill = (0,1,1); st/ist init       // 2 tiny uploads
repeat EVAL_MAX times:  DispatchIndirect(Eval, args)
DispatchIndirect(ForceFinish, argsOne)                // always 1 group
repeat FILL_MAX times:  DispatchIndirect(Fill, argsFill)
AsyncGPUReadback.Request(assign, slot = ist[I_RES])   // see section 4
```

Once `Finish` zeroes `args[0]`, every remaining `DispatchIndirect` launches zero groups,
which costs a command-processor read of 12 bytes and no SM time. Phase 4 measured the
early-exit launches at a few us each because every group still launched and returned;
the indirect form removes that.

`EVAL_MAX = 40`: phase 3 measured 22 evaluations mean, 28 max cold, fewer warm; the
bisection is capped at `max_iter = 64` on the device, so `EVAL_MAX` binds first and
`ForceFinish` returns the feasible `hib` bracket end with the phase 4 stopping rule not
yet met. The result is feasible and the utility loss is bounded by the bracket width; the
CSV logs `evals` and a `forced` flag so the frequency of that path is reported, not
hidden.

`FILL_MAX = 8`: phase 4 left "capped or parallel-batch fill" open. Decision: cap at 8
sequential steps (each is a full N x m pass and a global argmax; at N = 10k phase 4 saw 18
steps, at 100k the pass costs about the same as one evaluation). The Burst oracle applies
the same cap, so identity holds; the residual slack after 8 steps is logged. Parallel-batch
fill (k upgrades per pass) is rejected because it changes the assignment and would need a
new oracle.

The result slot `ist[I_RES]` is only known on the GPU, so the CPU cannot issue a
`CopyBuffer` of just that slot. Two readback requests are issued back to back: `ist` (64 B)
and the whole `assign` buffer (3 x N uint, 1.2 MB at 100k); the CPU picks the slot from
the first when both have landed. 1.2 MB per frame over PCIe is ~0.2 ms of copy-engine time
off the critical path and is negligible below 30k agents. Above that, a `Pack` kernel
(`[numthreads(256,1,1)]`, one extra dispatch after the fill chain) writes the result
slot's config indices as `uint8` into a 100 KB buffer (index < 256 always holds for
`m <= 256`) and that buffer is read back instead.

## 4. AsyncGPUReadback path and the latency margin

Timeline for frame f (CPU frame index):

| Step | Frame | Owner |
|---|---|---|
| Core step, salience, ledger accrue, headroom | f | Burst jobs |
| Upload salience/headroom/CostCB/state, record chain | f | allocator dispatch system |
| Chain executes | GPU frame f | GPU |
| `AsyncGPUReadback` completes | f + L, L in {1, 2} measured | readback system polls `request.done` |
| Tiers applied, promote/demote, reconciliation, ledger reset for promoted | f + L | Burst jobs |
| Composite sort key built from the applied tiers | f + L | sort key system (invariant 5) |

The ledger therefore accrues `L` extra frames at the old tier after the allocator chose
the new one from `headroom(f)`. To keep invariant 3 exact under that latency, the headroom
sent to the GPU is

```
headroom_i(f) = max(cap - D_i(f) - L_max * e_max, 0)
```

with `L_max = 2` and `e_max` the table's worst behaviour-tier-3 rate (the same
conservative rate the phase 7 table already uses). The CPU asserts `D_i <= cap` every frame
after accrual, as `sim/tiered.py` does, and counts violations; the guarantee is then
"never exceeds the cap" with the margin absorbing the readback delay, rather than a
post-hoc "exceeds by at most L x e".

Readback requests are issued every frame and consumed out of order is not allowed: the
readback system keeps a ring of `L_max + 1` requests and applies the newest completed one,
dropping older ones (a dropped result is just a stale allocation; the ledger margin
already covers two frames). If a request errors (device lost, resize), tiers hold and the
ledger keeps accruing, which forces restorations through the mask on the next successful
frame, so the error bound is preserved through a dropped frame.

Never call `request.WaitForCompletion()` in the frame; it is the synchronous readback
phase 4's host loop used and it stalls the GPU pipeline in the engine.

## 5. DOTS integration points

Component data (all `IComponentData`, unmanaged, SoA in chunks):

| Component | Fields | Allocated? |
|---|---|---|
| `CoreState` | route waypoints (3 x float2), segLen, L, s, chokeS, queued, slot, ctx, exitChoice | no (invariant 4): `partition()` core set |
| `ViewState` | latent d (index into the 1000-point corpus subset or 16 floats), region, pos, vel, gaitPhase, gesture weights | yes: `partition()` view set |
| `Salience` | float sig, visibility hold counter, distance-band hysteresis (phase 1 assigner state) | no |
| `Ledger` | float D, uint lastRestoreFrame | no |
| `Fidelity` | byte configIndex, byte4 tiers (beh, nav, anim, geo), byte prevBeh | written by readback system |
| `SortKey` | ulong key | rebuilt every frame from `Fidelity.tiers` and position |

Systems, in `SimulationSystemGroup` order:

1. `CoreStepSystem` (Burst, `IJobEntity`, all agents): `Core.step` at `mbar[ctx]`,
   chokepoint token bucket, arrival from `s >= L`, respawn/exits from the world rng in
   completion order. Fixed cost, tier-independent, the only writer of `CoreState`.
2. `SalienceSystem` (Burst): camera distance, frustum, 15-frame visibility hold, 10%
   band hysteresis, per-level caps; emits `sig` and the baseline's MassLOD tiers when the
   condition is `baseline`. Salience `s = 1/(1 + sig/20)` as in phase 7.
3. `LedgerSystem` (Burst): `D += e_rate[ctx]` for tier-3 agents (surrogate) using the
   tier **in effect**, then `headroom` with the latency margin of section 4.
4. `AllocatorPrepareSystem` (Burst): `prepare()`: `lam_max`, `qmax/qmin` per agent under
   the mask, feasibility check (`headroom < min err` is impossible since rate-0 rows exist),
   writes `salience`/`headroom` NativeArrays into the locked `GraphicsBuffer` ranges.
5. `AllocatorDispatchSystem` (main thread, `CommandBuffer` through the URP
   `ScriptableRenderContext` or `Graphics.ExecuteCommandBuffer` before rendering): section
   3 chain plus `GraphicsBuffer`-timed `BeginSample/EndSample` around the chain for the
   allocator's own GPU time.
6. `AllocatorReadbackSystem`: consumes completed requests; writes `Fidelity`; computes
   promote (`prevBeh == 3 && beh < 3`) and demote sets; schedules
   `ReconcileJob` (Burst): promote = draw `d ~ pi_ref(d | R, ctx)` from the 16 x 64
   conditional table, gait phase from core distance / stride, lateral slot at the core
   point, tangent velocity; demote = region from the live latent; `Ledger.D = 0` for
   promoted; `D_meas` reset for the measurement path.
7. `SortKeySystem` (Burst): composite key `[top k Morton bits][tier 2b][class 2b][remaining
   Morton bits][agent id]`, `k = round(20 w)`, `w` from the adaptive policy (`order/
   adaptive.py` lookup on `log1p(rho)` and `geff`, calibrated on the T4 sweep). The tier
   field is the tier the allocator just applied (invariant 5). The key sorts an
   `NativeArray<int>` permutation consumed by the tiered step; ECS chunks are not
   reordered.
8. Tiered behaviour / animation / geometry step (the user's side): consumes the permutation
   and `Fidelity`. The surrogate is a blob asset: 4 contexts x 16 x 16 coarse kernels, the
   16 x 64 sticky-lift conditional, the 16 x 19 decode table.
9. `TelemetrySystem` (Burst): `FrameTimingManager.GetLatestTimings` (`gpuFrameTime`,
   `cpuMainThreadFrameTime`, both with a 2-frame delay) plus the per-axis tier histogram
   from `Fidelity` two frames back, fed to the RLS update with the same 13 features
   (`theta_core * N + sum theta[axis, tier < 3] * n[axis, tier]`, forgetting 0.98, P-trace
   cap 1e4). Output: next frame's `CostCB` in ticks and the budget in ticks (target frame
   time minus the non-crowd measured share). Invariant 1 holds because nothing authored
   enters the cost vector.

Order 3 -> 4 -> 5 -> 6 within one frame means the readback consumed at step 6 is from
frame `f - L`; step 3 already used the tier in effect, so the ledger never sees a tier it
did not pay for.

## 6. Occupancy and timing on the T4

Same shape as phase 4 (256 threads, 8 warps per group): `groupshared` is 1 KB (Eval) and
3 KB (Fill), constant buffers 3 KB of 64 KB, so registers decide: <= 64 VGPRs gives 4
groups per SM (32/32 warps), 65-80 gives 3. Unity exposes no occupancy API and Nsight
counters are unavailable, so the register count is taken from the phase 4
`occupancy_report()` on the same kernel body (the HLSL and PTX inner loops are 2 mul, 1 sub,
1 compare per (agent, row) in fp32 now, so the count is expected lower than the fp64
figure). Timing: `CommandBuffer.BeginSample/EndSample` around the chain with
`Recorder.gpuElapsedNanoseconds`, median of 60 frames, N in {1k, 10k, 100k}; this is the
engine column of figure 4b. Per-evaluation timing is not available inside the chain and
is not needed: the CUDA-event numbers from `bench/speedup.py` on the T4 give per-launch
cost, and the chain's total minus `evals x` that gives the indirect-launch overhead.

## 7. Phase 4 decomposition choices: carry-over table

| Phase 4 choice | Engine | Reason |
|---|---|---|
| One thread per agent, 256-thread blocks, 1-D grid | carries over | same occupancy argument, same reduction width |
| Warp-uniform table scan, fixed `j` order | carries over | broadcast loads from cbuffer |
| Explicit comparator over the unsorted table | carries over | no per-frame sort; table static |
| Quality/err table in constant memory | carries over (cbuffer) | 2 KB |
| Cost vector staged in shared memory | **dropped** | cbuffer is the `cudaMemcpyToSymbol` numba lacked |
| Block tree reduction + last-block ticket | carries over | `InterlockedAdd` + `DeviceMemoryBarrier` + `globallycoherent` |
| Device-resident bracketing/bisection state machine | carries over verbatim | lambda never leaves the GPU |
| Warm-start lambda on the device | carries over | `warm` buffer persists |
| fp64 + libdevice non-contracted ops | **dropped** | fp32 `precise` + integer ticks; oracle becomes a Burst fp32 job |
| Float partial sums in fixed block order | **dropped** | uint ticks, exact in any order |
| Host batch of 8 launches + sync done-flag readback | **dropped** | `DispatchIndirect` chain with GPU-zeroed args, `ForceFinish` |
| Unbounded greedy fill | **dropped** | `FILL_MAX = 8`, oracle applies the same cap |
| Kernel compiled per table via closure | **dropped** | table is data (cbuffer), `MAXM = 256` compile-time |
| Host `prepare()` (lam_max, feasibility) | carries over | Burst job, O(N) reduction |
| `abuf[3, N]` slot rotation (`cur/lob/hib`) | carries over | `assign[3N]` |
| Result readback per frame | changed | async, L-frame latency, ledger margin `L_max * e_max` |
