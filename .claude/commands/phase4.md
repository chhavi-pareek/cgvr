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
