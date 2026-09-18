# Level 2.3 ablation summary

**All ATE: ATE vs nominal GT polyline (uncorrected image frame).** Map px primary; metres = px x nominal 0.5 m/px (unverified georeference). 'Tracked' = frames the run localized; 'all' = every frame. Commits are parsed from each run log; Δ commits = symmetric difference of commit frames vs control.

**Inertness and determinism:** 2 control runs, frame-log sha256 identical (`63e3e4bb11db…`, expected `63e3e4bb11db…`).

## Per run

| run | LSR | coverage | mean px | median px | p95 px | RMSE px | RMSE m | ΔRMSE vs control | all-frames RMSE px | commits | Δ commits | scale / pair |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control_r1 | 100.0% | 100.0% | 39.3 | 24.0 | 116.1 | 53.1 | 26.55 | +0% | 53.1 | 15 | 0 | 0.6266 [11, 24] |
| control_r2 | 100.0% | 100.0% | 39.3 | 24.0 | 116.1 | 53.1 | 26.55 | +0% | 53.1 | 15 | 0 | 0.6266 [11, 24] |
| A1_no_ransac | 100.0% | 100.0% | 69.8 | 55.2 | 172.4 | 89.0 | 44.51 | +68% | 89.0 | 15 | 0 | 0.6266 [11, 24] |
| A2a_topsis_shape | 100.0% | 100.0% | 61.6 | 52.2 | 175.7 | 79.5 | 39.76 | +50% | 79.5 | 15 | 0 | 0.6067 [11, 24] |
| A2b_topsis_ratio | 100.0% | 100.0% | 51.7 | 42.2 | 133.3 | 66.7 | 33.33 | +26% | 66.7 | 15 | 0 | 0.6266 [11, 24] |
| A3b_nodes_only | 100.0% | 100.0% | 127.9 | 99.9 | 312.9 | 158.6 | 79.28 | +199% | 158.6 | 10 | 15 | 0.5318 [4, 19] |
| A4_drop30_s1 | 100.0% | 100.0% | 60.1 | 49.6 | 143.4 | 73.8 | 36.88 | +39% | 73.8 | 15 | 20 | 0.6266 [11, 24] |
| A4_drop30_s2 | 100.0% | 100.0% | 65.0 | 55.0 | 125.2 | 75.5 | 37.77 | +42% | 75.5 | 15 | 24 | 0.6266 [11, 24] |
| A4_drop30_s3 | 100.0% | 100.0% | 78.7 | 57.8 | 176.5 | 94.9 | 47.44 | +79% | 94.9 | 17 | 24 | 0.6266 [11, 24] |
| A4_drop30_s4 | 100.0% | 100.0% | 48.3 | 47.0 | 103.0 | 59.0 | 29.50 | +11% | 59.0 | 14 | 17 | 0.6266 [11, 24] |
| A4_drop30_s5 | 100.0% | 100.0% | 100.8 | 87.8 | 239.8 | 122.6 | 61.31 | +131% | 122.6 | 15 | 26 | 0.6266 [11, 24] |
| A4_drop50_s1 | 100.0% | 100.0% | 187.7 | 148.2 | 513.2 | 243.7 | 121.86 | +359% | 243.7 | 18 | 23 | 0.6266 [11, 24] |
| A4_drop50_s2 | 100.0% | 100.0% | 82.2 | 47.1 | 232.4 | 117.3 | 58.66 | +121% | 117.3 | 17 | 28 | 0.6266 [11, 24] |
| A4_drop50_s3 | 100.0% | 100.0% | 85.2 | 69.6 | 222.8 | 105.7 | 52.87 | +99% | 105.7 | 16 | 27 | 0.6266 [11, 24] |
| A4_drop50_s4 | 100.0% | 100.0% | 159.2 | 175.4 | 263.5 | 178.8 | 89.42 | +237% | 178.8 | 15 | 22 | 0.6266 [11, 24] |
| A4_drop50_s5 | 100.0% | 100.0% | 214.7 | 244.3 | 453.5 | 266.2 | 133.10 | +401% | 266.2 | 17 | 22 | 0.6266 [11, 24] |
| A4_drop70_s1 | 100.0% | 100.0% | 94.7 | 97.8 | 184.7 | 109.8 | 54.91 | +107% | 109.8 | 25 | 34 | 0.6266 [11, 24] |
| A4_drop70_s2 | 100.0% | 100.0% | 62.6 | 57.0 | 138.1 | 74.2 | 37.09 | +40% | 74.2 | 23 | 32 | 0.6266 [11, 24] |
| A4_drop70_s3 | 100.0% | 100.0% | 102.1 | 101.7 | 191.6 | 117.6 | 58.79 | +121% | 117.6 | 24 | 33 | 0.6266 [11, 24] |
| A4_drop70_s4 | 100.0% | 100.0% | 55.8 | 51.3 | 112.6 | 63.8 | 31.89 | +20% | 63.8 | 26 | 33 | 0.6266 [11, 24] |
| A4_drop70_s5 | 100.0% | 100.0% | 80.7 | 71.4 | 157.4 | 93.5 | 46.73 | +76% | 93.5 | 24 | 27 | 0.6266 [11, 24] |
| A_coast | 94.5% | 100.0% | 62.2 | 61.9 | 126.5 | 74.2 | 37.09 | +40% | 74.2 | 15 | 0 | 0.6266 [11, 24] |
| A_prior_static_heading | 100.0% | 100.0% | 89.0 | 51.0 | 270.0 | 126.6 | 63.31 | +138% | 126.6 | 15 | 0 | 0.6266 [11, 24] |

## A4 dropout — distribution over seeds (median [min – max])

| group | seeds | RMSE px | ΔRMSE vs control | LSR | coverage | commits | Δ commits |
|---|---|---|---|---|---|---|---|
| A4_drop30 | 5 | 75.5 [59.0 – 122.6] | +42% [+11% – +131%] | 100.0% [100.0% – 100.0%] | 100.0% [100.0% – 100.0%] | 15 [14 – 17] | 24 [17 – 26] |
| A4_drop50 | 5 | 178.8 [105.7 – 266.2] | +237% [+99% – +401%] | 100.0% [100.0% – 100.0%] | 100.0% [100.0% – 100.0%] | 17 [15 – 18] | 23 [22 – 28] |
| A4_drop70 | 5 | 93.5 [63.8 – 117.6] | +76% [+20% – +121%] | 100.0% [100.0% – 100.0%] | 100.0% [100.0% – 100.0%] | 24 [23 – 26] | 33 [27 – 34] |

## Theses tested

- **control** — Production pipeline, all experiment switches off. Reference for every delta; two repeats prove determinism.
- **A1_no_ransac** — RANSAC consensus protects the position solve from wrong landmark bindings. With an infinite inlier radius every binding votes and the solve is their plain mean.
- **A2a_topsis_shape** — Multi-criteria TOPSIS beats any single criterion. Here candidates are ranked by route shape (internal-angle error) alone.
- **A2b_topsis_ratio** — Multi-criteria TOPSIS beats any single criterion. Here candidates are ranked by edge-length ratio error alone.
- **A3b_nodes_only** — Graph topology (edges) carries localization value. Relocalization uses node positions and classes only, at the current nav scale; edge-length ratios, which are scale-invariant, are unavailable.
- **A4_drop30** — Graceful degradation with sparse landmarks: 30% of detections dropped at random (boot frame exempt).
- **A4_drop50** — Graceful degradation with sparse landmarks: 50% of detections dropped at random (boot frame exempt).
- **A4_drop70** — Graceful degradation with sparse landmarks: 70% of detections dropped at random (boot frame exempt).
- **A_coast** — Coasted tracks voting in RANSAC (since 2026-05-11) cause the high-inlier false consensus seen in Variant B. Only tracks detected this frame vote in the continuation solve; the handoff trigger and scale lock are unchanged.
- **A_prior_static_heading** — How much tracking depends on the flight-plan prior. The leg headings from the FC-log replay (which drive the flashlight, kinematic clamp and 1D gate) are replaced by one static 225 deg heading. The waypoint-0 start prior is unchanged.

## Cross-track RMSE by mission leg (px) — where each ablation does its damage

| run | leg 0 (f1–254; static heading off by 8.2°) | leg 1 (f255–352; off by 5.0°) | leg 2 (f353–401; off by 28.2°) | coasting frames |
|---|---|---|---|---|
| control | 64.6 | 24.1 | 17.3 | 0 |
| A1_no_ransac | 63.8 | 134.0 | 88.8 | 0 |
| A2a_topsis_shape | 96.5 | 37.3 | 26.9 | 0 |
| A2b_topsis_ratio | 57.2 | 92.7 | 46.7 | 0 |
| A3b_nodes_only | 86.3 | 185.8 | 313.1 | 0 |
| A_coast | 51.6 | 90.1 | 122.5 | 22 |
| A_prior_static_heading | 61.5 | 147.9 | 260.5 | 0 |

## Interpretation

**How to read the deltas.** Single sequence, deterministic runs, so each non-A4 row is one sample. Rows with **Δ commits = 0** (A1, A2a, A2b, A_coast, A_prior) kept the control's exact commit schedule, so their deltas are not commit-rescheduling artifacts. A3b and every A4 run rescheduled commits. A4's seed spread (30% dropout: 59–123 px RMSE for the same condition) shows how far rescheduling alone can move RMSE, roughly ±50%. No ablation improved RMSE: every component tested is load-bearing on this sequence.

- **A1 — RANSAC consensus: supported.** +68%, same commit schedule. Leg 0 is unchanged; the damage comes after the turns (leg 1: 134 vs 24 px), where wrong bindings from later relocalizations would otherwise be voted out.
- **A2 — multi-criteria TOPSIS: supported, with a confound on A2a.** Both single criteria are worse, and the commit schedule is unchanged: shape-only +50%, ratio-only +26%. A2a's boot commit bound one map node differently ((1858, 3525) vs (1857, 3520)) for the same video pair [11, 24], so the scale locked at 0.6067 instead of 0.6266 (−3.2%). Part of A2a's degradation, concentrated on leg 0, is scale poisoning rather than worse selection.
- **A3b — graph edges: supported; the value is scale-invariant binding.** +199%, with every commit rescheduled (10 commits, 15 differ). At boot the node-only matcher works at the 0.5 scale prior and bound a different pair ([4, 19]), locking scale 0.5318 (−15%). The unit test predicted this failure mode (0/7 correct bindings at the prior). Edge-length ratios do not need the scale, so they can bind correctly before it is known. "Edges" and "correct scale lock" cannot be separated here, because that coupling is the mechanism.
- **A4 — sparse landmarks: coverage is robust; accuracy degrades but not monotonically.** LSR and coverage stay 100% even at 70% dropout, so **LSR is not a correctness metric** in this pipeline. Median commits rise monotonically (15 → 17 → 24); that is the robust signal (more relocalization as landmarks thin out). RMSE medians (+42%, +237%, +76%) are not monotonic in dropout rate and every run rescheduled commits, so one sequence cannot establish an ATE dose-response.
- **A_coast — the audit hypothesis as a fix: refuted for the production pipeline.** +40%, same commit schedule, 22 coasting frames (LSR 94.5%). Excluding coasted votes helps on leg 0 (51.6 vs 64.6 px) but thins the voter set after the turns (legs 1–2: 90 / 123 vs 24 / 17 px). Coasted votes are net load-bearing. The Variant B mechanism (coasted votes carrying a filter-driven velocity) is untested: this run used production velocities.
- **A_prior — flight-plan heading prior: strongly load-bearing.** +138%, same commit schedule. Leg 0, where the static heading is only 8° off, is unchanged (61.5 vs 64.6 px). After the first turn the error jumps to 148 and 261 px, worst on the 28°-off leg. Caveat: 225° is a *wrong* prior, not an absent one, and the waypoint-0 start prior is still in place. This measures sensitivity to heading-prior quality, and confirms that a large share of the 26.5 m baseline depends on flight-plan information rather than vision alone.
