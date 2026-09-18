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
