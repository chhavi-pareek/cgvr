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
