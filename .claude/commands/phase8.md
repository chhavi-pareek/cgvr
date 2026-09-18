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
