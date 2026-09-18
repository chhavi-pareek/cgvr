# STATE

## DONE
- Phase 1: standalone simulator (`sim/`), three scenes (plaza, hub with queue, corridor with bottleneck), scripted camera paths, MassLOD-style threshold-and-cap assigner, CSV logger (`bench/log.py`), determinism tests.
- Acceptance: 2000 agents headless (plaza, 200 frames, ~21 s) log per-frame visual/sim tier histograms; two runs with the same seed and camera path are byte-identical (`cmp`); `pytest -q` 7 passed.
- Repo initialised (`git init`), `pytest` installed, `.gitignore` and `pytest.ini` added.

## IN PROGRESS
- Nothing.

## DECISIONS
- Repo root is `cgvr/`; `sim/`, `bench/`, `tests/` sit at root.
- Tier index: 0=HIGH, 1=MED, 2=LOW, 3=OFF.
- One significance scalar per agent: 3D distance to camera, multiplied by 2.0 when out of view (after visibility hysteresis). Visual and sim tiers each apply their own band table, hysteresis state, and per-level caps to that scalar. Visual is forced OFF when out of view; sim is not.
- Distance hysteresis is 10% on band edges. Visibility hysteresis is a 15-frame hold after leaving the frustum. Caps demote lowest-significance agents first; hysteresis state stores the pre-cap tier.
- Tiers are recorded only. Motion is full-rate reference dynamics (goal-seek plus O(N^2) chunked separation, no ORCA) for every agent regardless of tier.
- Log holds integer histograms only (no floats), so byte-identity is exact. Fixed dt 1/30, single `default_rng(seed)`.
- Hub queue is a scripted service window (front agent released every 45 frames, queuers rejoin at the tail). Corridor door is a hard clamp on a 1.6 m gap at x=30.

## OPEN QUESTIONS
- Should MassLOD sim tiers throttle tick rate or freeze agents in the baseline? Currently they do not. This decides how "behavioural error" is measured against the baseline in phase 3.
- At N=2000 the sim histogram sits on the caps (80/400/1200/320) in the plaza. Cap sizes are placeholders; tune before treating the baseline as a fair opponent.
- Stray directory in repo root with a multi-line garbled name (from a failed setup script paste). Empty, untracked by git, not touched.

## NEXT
- Phase 2: behaviour corpus and latent manifold (`/phase2`, on the T4).
