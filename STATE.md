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

## OPEN QUESTIONS
- Should MassLOD sim tiers throttle tick rate or freeze agents in the baseline? Currently they do not. This decides how "behavioural error" is measured against the baseline in phase 3.
- At N=2000 the sim histogram sits on the caps (80/400/1200/320) in the plaza. Cap sizes are placeholders; tune before treating the baseline as a fair opponent.
- Nested-dropout training makes the dimension ordering hold by construction. The plain AE, which is the un-engineered evidence, is also monotone on the mean curve but not on every seed (3 of 20 seed/head curves fail). Both rest on synthetic targets (see DECISIONS).
- The phase spec says Phase 2 runs on the T4; this was run on CPU. Rerun `python -m latent.train && python -m latent.truncation` on the T4 before treating the numbers as final (you asked for a T4 reminder after all phases).
- Stray directory in repo root with a multi-line garbled name (from a failed setup script paste). Empty, untracked by git, not touched.

## NEXT
- Phase 3: cost model and serial allocator. Latent dimension for the decoder is 16; the truncation curve above says how much fidelity each retained-dim count costs.
