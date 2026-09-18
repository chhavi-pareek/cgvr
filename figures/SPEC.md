# PARITY figures: specification

Four primary figures. Each entry gives the input files and columns, the exact reduction
from rows to marks, the axes, the encoding, and the acceptance check the plotting script
must run before it draws (refuse to plot, with the missing column named, rather than draw a
partial figure). No plotting code here.

## Common conventions

- Output `figures/out/fig{1..4}_{name}.pdf` and `.png` at 300 dpi. Widths: 3.35 in
  (single column) or 7.0 in (double column) as stated per figure; heights as stated.
- Font: sans (Helvetica or DejaVu Sans), 8 pt body, 7 pt tick labels, 9 pt panel letters
  bold at the top-left outside the axes. No titles inside the figure; the caption carries
  them. Axes labels carry the unit in parentheses.
- Line width 1.2 pt for primary series, 0.7 pt for secondary. Markers 3.5 pt.
- Colour by **condition**, fixed across every figure (Okabe-Ito, colour-blind safe):

  | condition | hex | name |
  |---|---|---|
  | `parity` | `#0072B2` | blue |
  | `baseline` | `#D55E00` | vermillion |
  | `reference` | `#000000` | black |
  | `parity_nocap` | `#E69F00` | orange |
  | `parity_authored` | `#009E73` | bluish green |
  | secondary / annotation | `#999999` | grey |

  Camera paths are distinguished by marker, never by colour: orbit `o`, flythrough `s`,
  static_wide `^`, static_choke `v`, sweep `D`.
- Condition display names: "PARITY", "MassLOD baseline", "Reference (full fidelity)",
  "PARITY, no error cap", "PARITY, authored per-axis weights".
- Elapsed time in seconds is `frame * DT` with `DT = 1/30` (`sim.tiered.DT`). Never plot frames.
- Scenes are panels in the order plaza, hub, corridor, sharing the y axis within a figure.
- Every figure prints, to stdout, the numbers it annotated (so the caption can quote them).

## Figure 1: behavioural divergence vs elapsed time

**Input.** `bench/logs/sweep_kl.csv` (from `python -m bench.sweep`), columns `scene, cond, cam,
n, seed, frame, kl_p95, kl_max, cap`. Filter `n == 200`, `cam == orbit` (the main grid),
all seeds. Fallback for the long horizon: `bench/logs/phase7_kl_{hub,corridor}.csv` (18000
frames, conditions `parity`/`baseline` only, five cams).

**Reduction.** For each (scene, cond): x = `frame * DT` in s. Primary line y = `kl_p95`
(accumulated KL in nats, per agent since its last reconciliation for PARITY, since t = 0
for the baseline and the no-cap ablation), mean over seeds. Secondary line y = `kl_max`
(worst agent), same colour at 0.7 pt and 50% alpha. If more than one seed, band = min to
max over seeds of `kl_p95` at 20% alpha. The `reference` condition has KL = 0 by
construction (it runs the reference process for every agent) and is **not** drawn as a
line; the caption states this.

**Axes.** Layout 7.0 x 2.4 in, three panels (plaza, hub, corridor). x linear, 0 to the run
length (100 s for the sweep default of 3000 frames; 600 s for the phase 7 files). y
log10, shared, from 0.1 to the next decade above the largest `kl_max`; ticks 0.1, 1, 10,
100, 1000. Horizontal dashed grey line at `cap` (the PARITY cap for that scene, taken from
the `parity` rows), labelled "cap" at the right edge. y label "accumulated divergence
(nats)", x label "elapsed time (s)".

**Encoding.** One line per condition (four: PARITY, baseline, no-cap, authored). Legend in
the first panel only, lower right, no frame. Annotate on the last panel, in grey 7 pt, the
fitted slope of `kl_max` over the second half of the run for `baseline` and
`parity_nocap` ("+13 nats / 1000 frames"), from `kl_slope_per_1k` in `sweep_cells.csv`.

**Check.** Assert every drawn condition has rows at the final frame in every scene; assert
`parity` `kl_max` never exceeds `2 * cap` (the phase 7 result was 1.8x) and print the max
ratio; assert `baseline` `kl_max` at the end exceeds `parity` `kl_max` at the end in every
scene.

**Reads as.** PARITY and its authored-weights variant sit flat near the cap; the no-cap
ablation and the baseline climb without bound, the baseline faster. The gap between
`parity_nocap` and `parity` is the ledger (invariant 3); the gap between `parity_nocap` and
`baseline` is the surrogate's matched stationary law.

## Figure 2: outcome variance across camera paths, baseline vs PARITY

**Input.** `bench/logs/sweep_egress.csv` from `python -m bench.sweep --cam-sweep`, columns
`scene, cond, n, cam, seed, agent, trip, frame`, filtered to `n == 200`, seed 0, `cond in
{parity, baseline}`, all five cams. Fallback: `bench/logs/phase7_egress_{hub,corridor}.csv`.

**Reduction.** For each (scene, cond): the reference camera is `orbit`. Join trips on
`(agent, trip)` between `orbit` and each other cam `c`; for every matched pair
`delta = (frame_c - frame_orbit) * DT` in seconds. Trips present under one cam and absent
under another are counted separately as `unmatched`. Per (scene, cond, cam) compute:
`n_trips`, `n_matched`, `max |delta|`, `W1` = 1-Wasserstein distance in seconds between the
two egress-time samples (`scipy.stats.wasserstein_distance` on the full frame lists, not
only matched pairs).

**Layout.** 7.0 x 2.6 in, three scene panels, each with two sub-axes:
- **Main (left 70%)**: ECDF of `|delta|` (s) pooled over the four non-reference cams, x
  on a symlog scale with linear threshold one frame (1/30 s), range 0 to the max over
  conditions; y from 0 to 1, "fraction of trips". One step line per condition. PARITY's
  ECDF is a vertical step at 0 by the phase 7 result; draw it as the step plus a
  filled marker at (0, 1) labelled "max |Δ| = 0.00 s". Vertical dashed grey line at
  `BAND / 1 m/s = 2.0 s` (the guarantee's ceiling on visible lag) labelled "band".
- **Inset (right 30%)**: trips completed per camera path, x = five cams as markers on a
  categorical axis (marker per cam per the convention, coloured by condition), y = `n_trips`;
  PARITY's five values coincide (draw one marker row and a horizontal line); baseline's spread
  is the point.

**Encoding.** Two conditions only, both colours. Legend in the first panel, lower right.
Under each panel, one grey 7 pt line: "baseline: max |Δ| = 94.7 s, W1 ≤ 30.4 s; PARITY:
0 / 0" with the computed numbers.

**Check.** Assert PARITY `n_trips` is identical across all five cams and `max |delta| == 0`
in every scene (this is acceptance (b) of phase 7 and the figure must fail loudly if the
sweep broke it); print the baseline `max |delta|` and `W1` per scene and cam.

**Reads as.** The same crowd, watched from five cameras, exits at byte-identical times
under PARITY and at times that differ by tens of seconds under the baseline.

## Figure 3: sustained agent count at a fixed target frame time, with 1% lows

**Input.** `bench/logs/sweep_cells.csv`, columns `scene, cond, n, cam, seed, ft_mean_ms,
ft_p99_ms, ft_low1_ms, alloc_ms_mean, alloc_share`. `ft_low1_ms` is the mean of the slowest
1% of frames after a 30-frame warm-up, the game-benchmark "1% low" expressed as a frame
time. Filter `cam == orbit`. When the engine numbers exist they are a second CSV with the
same columns (`bench/logs/unity_cells.csv`, `ft_*` from `FrameTimingManager`), drawn with
the same script and a `--source` switch; the prototype and engine figures are never mixed
on one axis.

**Target frame time** `T*` is a script parameter. Engine: 16.7 ms (60 Hz), with 33.3 ms as a
second dashed line. Prototype: choose the smallest of {33.3, 100, 333} ms such that at
least three conditions cross it inside the agent-count range, and print the choice; the
prototype's absolute frame times are Python and mean nothing against 60 Hz, only their
ratios between conditions do.

**Reduction.** For each (scene, cond): the six agent counts sorted ascending with
`ft_mean_ms` and `ft_low1_ms` (mean over seeds). **Sustained N** = the largest N with
`ft_low1_ms <= T*`, refined by log-log linear interpolation between the last passing and
first failing count; if the smallest count already fails, sustained N = 0 with a marker
"< N_min"; if the largest count passes, sustained N is reported as ">= N_max" (an open
bar with a hatched top). Compute the same with `ft_mean_ms` for the lighter bar.

**Layout.** 7.0 x 2.6 in, two panels.
- **(a) left, 55%**: frame time vs agent count. x log2 with ticks at the six counts, y log10
  in ms. One line per condition through `ft_mean_ms` (solid) and `ft_low1_ms` (dashed,
  same colour), the region between them filled at 15% alpha. Horizontal grey dashed line at
  `T*` labelled with its value. Vertical ticks at each condition's sustained N on the `T*`
  line. Scenes: one panel per scene stacked vertically if space allows, otherwise the hub
  scene alone in (a) and all three scenes in (b).
- **(b) right, 45%**: sustained agent count, grouped bars: groups = scenes, bars = conditions
  in the fixed order reference, baseline, parity, parity_nocap, parity_authored. Each bar
  is the 1% low sustained N (full colour) with a thin outline bar behind it at the mean
  sustained N (same colour at 35% alpha). y linear in agents, label "agents sustained at
  T* (1% low)". Numeric value above each full bar, 7 pt.

**Encoding.** Additionally print, per (scene, cond), `alloc_share` at the largest N so the
caption can state the allocator's fraction of the frame (the prototype runs the serial
allocator by default; `--allocator threaded` is oracle-identical and the CSV records which).

**Check.** Assert each (scene, cond) has all six counts; assert `ft_low1_ms >= ft_mean_ms`
in every row; warn (not fail) if `reference` sustains more agents than `parity` anywhere,
because that means the allocator overhead exceeded its saving at that N and the caption must
say so.

**Reads as.** At the target frame time, PARITY sustains more agents than the reference and
the ablations; the 1% low bar, not the mean, is the number quoted in the text.

## Figure 4: ordering Pareto front and the four-way speedup chart

**Input.**
- (a) `bench/logs/order_sweep.csv`, columns `density, n, seed, frame, w, bytes_per_agent,
  wee, wee_meas, t_us, source`, `n == 50000`, seed 0, mean over frames. Rows with `w == -1.0`
  (`order/sweep.py::BLIND`) are the spatially blind reference.
- (b) `bench/logs/speedup.csv`, columns `n, impl, threads, ms, evals, fill, match`, `impl in
  {serial, threaded, cuda, openacc[...]}`, `n in {1000, 10000, 100000}`. `openacc[...]`
  rows carry the backend in brackets; strip it for the label and print it in the caption.

**Reduction (a).** Per density (sparse, mixed, dense): points `(bytes_per_agent, wee)` for
each `w` in the 11-value grid, the blind point, and the Pareto-optimal subset under
(minimise bytes, maximise WEE) computed by the script (must reproduce `order/sweep.py::
pareto`: assert the same set). On hardware rows (`source == measured`) also compute
`t_us` and mark the modelled-time or measured-time minimiser.

**Reduction (b).** Speedup = `ms_serial / ms_impl` at the same `n`; for `threaded` take the
best thread count and also keep `threads == 1`. Four bars per `n`: serial (1.0 by
definition), threaded best, OpenACC, CUDA. Missing CUDA or 100k rows (not yet run on the T4)
are drawn as hollow bars with the text "T4 pending" and the script prints which are missing.

**Layout.** 7.0 x 2.8 in, two panels.
- **(a) left, 50%**: x = bytes moved per agent per frame (B), log10; y = warp execution
  efficiency, linear 0.5 to 0.9. One marker family per density (sparse `o`, mixed `s`,
  dense `^`), colour = `w` on a sequential colormap (viridis, 0 to 1) with a slim colourbar
  labelled "spatial weight w"; blind reference as a hollow grey marker of the same shape.
  The Pareto-optimal points of each density connected by a grey step line. Text labels
  "w=0" and "w=1" at the two extremes of each density. Annotate the dense density's
  modelled time at the two extremes ("17.0 us vs 19.5 us") from `t_us` in grey.
- **(b) right, 50%**: grouped bars, groups = N (1k, 10k, 100k), bars = serial, threaded
  (best, with the x1 result as a thin outline bar behind it), OpenACC, CUDA. y = speedup
  over serial, log10 from 0.5 to 100. Bar colours by implementation, not by condition
  (this is the one figure without conditions): serial `#999999`, threaded `#56B4E9`,
  OpenACC `#CC79A7`, CUDA `#0072B2`. Absolute ms printed above each bar, 7 pt. Horizontal
  dotted grey line at the speedup that brings the serial allocator under 16.7 ms at that N
  (`ms_serial / 16.7`), labelled "frame budget".

**Check.** (a) assert the recomputed Pareto set equals the file's; assert `wee_meas` equals
`wee` wherever `wee_meas` is not NaN (hardware run consistency). (b) assert every non-serial
row has `match == True` (oracle identity) before it is drawn; print the T4 rows present.

**Reads as.** (a) The spatial and the state orderings are each Pareto-optimal; the blind
state ordering is dominated everywhere; which extreme wins in time depends on density. (b)
The same allocator, four implementations, the GPU columns filled in from the T4 session.

## Not primary, but produced by the same script for the appendix

- Tier-mix stacked area per condition over time from `sweep_kl.csv` (`t0..t3`).
- Ledger mean/max vs cap from `sweep_kl.csv` (`ledger_mean`, `ledger_max`, `cap`).
- Restorations per 1000 frames per condition from `sweep_cells.csv`.
- INT8 provable fraction vs lambda per d from `bench/logs/quant.csv` (phase 6).
