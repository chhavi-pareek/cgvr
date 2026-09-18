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
