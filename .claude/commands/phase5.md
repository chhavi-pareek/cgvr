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
