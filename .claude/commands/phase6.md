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
