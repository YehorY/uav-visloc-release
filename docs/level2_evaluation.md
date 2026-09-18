# Level 2 — Evidence Base

Reproducible statistics for the UAV-VisLoc pipeline on the frozen GeoTest1 sequence (401 frames): extended
trajectory-error statistics, a classical feature-matching baseline, a controlled ablation suite, and annotated
failure cases. It ends with the response to the 2024 reviewer remark about statistics in the application.

Branch `level-2-evidence-base`. Commits: `40b759a` (2.1), `fd60ddd` (2.2), `f6e0ef5` (2.3), and 2.4 with this
document.

---

## 1. Ground truth and metric caveats

Every number in this document must be read with these limits.

- **Label.** All ATE-style errors are **vs the nominal GT polyline, in the uncorrected image frame.**
- **What the ground truth is.** Four GPS waypoints joined into a polyline. There is no per-frame GPS. The
  per-frame error is the distance to the *nearest* point on the polyline, so it is **cross-track only**.
  Along-track error is measured separately at the two recorded waypoint passages (f255, f353).
- **Units.** Map pixels are primary. Metres are **map px × nominal 0.5 m/px**, and that GSD is unverified.
  Waypoints reach map pixels by stretching the waypoints' own lat/lon box (+10% pad) over the whole map image.
  The map has no stored georeference. With square pixels that assumption is inconsistent: it implies 0.398 m/px
  east-west but 0.483 m/px north-south, a ~21% anisotropy.
- **The GT frame disagrees with the map image.** SIFT/ORB trajectories (§3) sit ~196 px from waypoint 0 at
  frame 1. One similarity transform (scale 0.766, rotation −6.6°) maps them onto the polyline better than our
  pipeline fits it.
- **The pipeline has GT-derived priors.** It starts *at* waypoint 0. Its heading prior comes from the waypoint
  legs (flight-controller log replay), which drives the relocalization flashlight, the kinematic clamp and the
  backward-relock gate. §4 (A_prior) measures how much that matters.
- **LSR is not a correctness metric here.** LSR counts frames not declared lost. It stays 100% even with 70% of
  detections removed (§4).
- **Single sequence.** Every result is one flight. Deterministic runs remove noise between repeats, but not
  sensitivity to small perturbations (§4, A4 seed spread).

## 2. Extended ATE statistics (Stage 2.1)

**What was added.**
- A `track_state` column in the frame log: 0 lost, 1 tracking, 2 coasting.
- `core/metrics.py` with mean / median / p95 / max / RMSE, computed over *tracked* frames (the run's own
  convention) and over *all* frames (coverage-honest). It also reports coverage, along-track error at waypoint
  passages, and commit-schedule divergence.

**Validation.** From the frame log alone the evaluator reproduces the pipeline's printed ATE:
**53.028294 px × 0.5 = 26.514147 m** (printed 26.51 m). The trajectory history is seeded with the boot pose, a
zero-error term, and that changes RMSE by −0.124%. The 28 pre-existing log columns stayed cell-identical to the
Level 1 baseline, and two runs are byte-identical (reference sha256 `63e3e4bb…`).

**Production baseline (control run):**

| | mean | median | p95 | max | RMSE |
|---|---|---|---|---|---|
| map px | 39.3 | 24.0 | 116.1 | 126.5 | **53.1** |
| metres (nominal) | 19.67 | 11.99 | 58.06 | 63.25 | **26.55** |

- LSR 100%, coverage 100%. RMSE over all frames equals the tracked RMSE (no lost frames).
- Along-track at the recorded passages: **+178 px** at wp1 (the estimate crosses 28 frames early) and **+305 px**
  at wp2 (29 frames early).
- 15 relocalization commits, all pre-emptive handoffs.

## 3. Classical baseline: SIFT / ORB + MAGSAC homography (Stage 2.2)

`scripts/classical_baseline.py` matches every frame to the map image and writes a frame log scored by the same
evaluator. Global mode uses no prior. Windowed mode starts at waypoint 0, the prior our pipeline gets. There is no
scale prior. The script is deterministic in every column except wall-clock time.

| System | LSR | mean px | median px | p95 px | RMSE px (m) | along-track wp1 / wp2 | CPU time |
|---|---|---|---|---|---|---|---|
| **our pipeline** (GPU) | 100% | 39.3 | 24.0 | 116.1 | **53.1 (26.5)** | +178 / +305 px | — |
| SIFT windowed | 100% | 200.2 | 174.5 | 349.0 | 214.5 (107.3) | +146 / +348 px | 72 s |
| SIFT global | 100% | 200.2 | 174.4 | 349.0 | 214.5 (107.3) | +146 / +349 px | 114 s |
| ORB windowed | 100% | 200.1 | 174.1 | 348.6 | 214.5 (107.3) | +146 / +346 px | 60 s |
| ORB global | 100% | 200.1 | 174.1 | 348.6 | 214.5 (107.3) | +146 / +346 px | 98 s |

**Why the 4× gap is not evidence that graph matching is needed:**

1. **The classical result is a precise measurement.** The four variants agree per frame to a median of
   0.15–0.8 px (max 3.5 px). Even global matching with no prior localizes all 401 frames.
2. **It disagrees with the GT frame, not with reality.** Fitting one transform from the SIFT trajectory to the
   polyline, constrained by the two recorded passages so it cannot slide or shrink, gives:

   | Transform applied to SIFT | cross-track RMSE | along-track wp1 / wp2 |
   |---|---|---|
   | none | 214.5 px | +146 / +348 px |
   | similarity (scale 0.766, rotation −6.6°) | **43.6 px** | **+19 / +7 px** |
   | axis-aligned anisotropic (0.934 x / 0.732 y) | 51.0 px | +25 / +9 px |

   Caveat: four fitted parameters on one sequence, so this is optimistic.
3. **The pipeline has priors the baseline lacks** (§1). The gap between pipeline and SIFT grows from 196 px at f1
   to 461 px at f401; it is not a constant offset.
4. **The scale is not constant.** The SIFT homography scale falls from 0.644 to 0.586 map px per video px over the
   flight (−9.8%). The pipeline locks 0.6266 once. §5 Case 2b shows this does not, however, accumulate into error.

## 4. Ablation suite (Stage 2.3)

`scripts/run_ablations.py` runs each configuration in `configs/ablations.yaml` as a fresh subprocess, changing
one variable against a control from the same batch. Experimental code lives in `experiments/` behind an
`experiments:` config section, off by default. Both control runs are byte-identical to the reference, so the
hooks are inert.

| Run | Variable changed | RMSE px (m) | ΔRMSE | commits | Δ commits vs control | boot scale / pair |
|---|---|---|---|---|---|---|
| **control** | — | **53.1 (26.55)** | — | 15 | — | 0.6266 / [11, 24] |
| A1 | no RANSAC outlier rejection (∞ inlier radius) | 89.0 (44.51) | **+68%** | 15 | **0** | 0.6266 / [11, 24] |
| A2a | TOPSIS: route shape only | 79.5 (39.76) | **+50%** | 15 | **0** | **0.6067** / [11, 24] |
| A2b | TOPSIS: edge-length ratio only | 66.7 (33.33) | **+26%** | 15 | **0** | 0.6266 / [11, 24] |
| A3b | no graph edges (node-only matcher) | 158.6 (79.28) | **+199%** | 10 | 15 | **0.5318 / [4, 19]** |
| A_coast | coasted tracks don't vote in the solve | 74.2 (37.09) | **+40%** | 15 | **0** | 0.6266 / [11, 24] |
| A_prior | static 225° heading instead of leg headings | 126.6 (63.31) | **+138%** | 15 | **0** | 0.6266 / [11, 24] |

Detection dropout (A4), 5 seeds per rate, boot frame exempt. Median [min – max]:

| Dropout | RMSE px | ΔRMSE | LSR | commits | Δ commits |
|---|---|---|---|---|---|
| 30% | 75.5 [59.0 – 122.6] | +42% [+11 – +131%] | 100% | 15 [14 – 17] | 24 [17 – 26] |
| 50% | 178.8 [105.7 – 266.2] | +237% [+99 – +401%] | 100% | 17 [15 – 18] | 23 [22 – 28] |
| 70% | 93.5 [63.8 – 117.6] | +76% [+20 – +121%] | 100% | 24 [23 – 26] | 33 [27 – 34] |

Cross-track RMSE by mission leg (px):

| Run | leg 0 (f1–254) | leg 1 (f255–352) | leg 2 (f353–401) |
|---|---|---|---|
| control | 64.6 | 24.1 | 17.3 |
| A1 | 63.8 | 134.0 | 88.8 |
| A2a | 96.5 | 37.3 | 26.9 |
| A2b | 57.2 | 92.7 | 46.7 |
| A3b | 86.3 | 185.8 | 313.1 |
| A_coast | 51.6 | 90.1 | 122.5 |
| A_prior | 61.5 | 147.9 | 260.5 |

**Interpretation.** Rows with Δ commits = 0 kept exactly the control's relocalization schedule, so their deltas
are real attributions, not rescheduling artifacts. A3b and every A4 run rescheduled commits. The 30% dropout
condition alone spans 59–123 px across seeds, which shows how far rescheduling can move RMSE. **No ablation
improved RMSE: every component tested is load-bearing on this sequence.**

- **A1, RANSAC.** Leg 0 is unchanged. After the turns error rises to 134 px vs 24, where later relocalizations
  bind wrong landmarks that consensus would otherwise vote out.
- **A2, multi-criteria TOPSIS.** Both single criteria are worse. A2a has a confound: its boot commit bound one map
  node 5 px differently for the same video pair, locking scale 0.6067 (−3.2%). Part of A2a's degradation is scale
  poisoning.
- **A3b, graph edges.** The largest effect. Without edges the boot match runs at the 0.5 scale prior and binds the
  wrong pair (§5 Case 2a). Edge-length ratios are scale-invariant, so graph matching can bind correctly before the
  scale is known. "Edges" and "correct scale lock" cannot be separated; that coupling is the mechanism.
- **A4, sparsity.** LSR stays 100%. Median commits rise with sparsity (15 → 17 → 24), which is the robust signal.
  RMSE is not monotonic in dropout rate and every run rescheduled, so one sequence cannot give an ATE
  dose-response.
- **A_coast.** Excluding coasted votes helps on leg 0 (51.6 vs 64.6 px) but thins the voters after the turns, and
  adds 22 coasting frames. This refutes it as a fix for the production pipeline.
- **A_prior.** Leg 0, where 225° is only 8° from the true leg, is unchanged. After the first turn error reaches
  148 and 261 px. A large share of the baseline accuracy depends on flight-plan heading. Caveat: 225° is a
  *wrong* prior, not an absent one, and the waypoint-0 start prior remains.

## 5. Failure cases (Stage 2.4)

Rendered by `scripts/render_failure_cases.py` into `out/failure_cases/`. Every number below is recomputed from the
logs and stored in the matching `.json`.

### Case 1 — sparse landmarks: the solve rests on one inlier (A4 50% dropout, seed 5, f131)

![case1](../out/failure_cases/case1_sparse_f131.png)

- At f131 YOLO finds 57 landmarks and dropout keeps 33 (control: 57).
- From ~f101 to f154 exactly **3 bound pairs** vote, one above the ≤2 handoff trigger, so **no search fires**.
  Only **1** of them is an inlier (56 frames in f100–f180).
- Cross-track error climbs at **exactly 8.0 px/frame, the lateral clamp cap**: 141.5 px (f123) → 205.5 (f131)
  → 269.5 (f139). Control over the same frames: 103.0 → 108.3 → 113.5 px. The clamp is the only thing bounding the
  drift.
- The run never recovers: 17 commits in total (control 15). Eight of them fall in f249–f339, where error still
  ranges 223–479 px (peak 479.3 px at f265).
- Illustrative of a mechanism, not a production failure rate: this is an ablation run with a rescheduled commit
  schedule.

### Case 1b — production run: stale bindings after a handoff in a detection trough (control, f80–f180)

![case1b](../out/failure_cases/case1b_stale_binding_f159.png)

- The handoffs at f88 and f98 happen at 27 and 26 detections, the sparsest point of the first half.
- For the next ~60 frames the solve keeps the same bindings: 8 pairs, 5 inliers. Meanwhile detections rise to 99
  (f159).
- Cross-track error climbs 87.1 px (f98) → **126.5 px (f159), the worst of the flight.** The trigger never fires
  because the stale pairs stay alive.
- Recovery begins at f160 while all 8 bindings are still voting: their consensus shifts. From f164 the error falls
  at **exactly 8.0 px/frame, the lateral clamp cap**, so the clamp paces the correction as well as the drift. The
  bound-pair count only drops (6 → 5 → 3) at f169–f171, and the f172 re-bind completes the recovery (48.4 px).
- The failure is in trigger design (pair count, not binding quality), not a lack of landmarks.

### Case 2a — boot scale poisoning without edges (A3b nodes-only, f1)

![case2a](../out/failure_cases/case2a_scale_poison_f1.png)

- The node-only matcher, at the 0.5 scale prior, binds landmark pair **[4, 19]**: video (567,342)↔(54,129), map
  (2040,3622)↔(1772,3499). It locks **scale 0.5318**.
- Control binds [11, 24] and locks 0.6266. SIFT measures 0.70 at f1 (median 0.644 over f1–f50).
- The lock is one-time, so the error persists for the whole flight: RMSE 158.6 vs 53.1 px, 10 commits vs 15.

### Case 2b — negative finding: the scale excess does not accumulate (control vs SIFT)

![case2b](../out/failure_cases/case2b_scale_nonaccumulation.png)

**Hypothesis tested:** the static scale lock drives error as the true scale drifts. If so, inside each
commit-free segment the pipeline's path length should exceed SIFT's by `0.6266 / measured scale`.

| Segment | pipeline / SIFT path | predicted by scale | image speed px/frame |
|---|---|---|---|
| f2–87 | 1.002 | 0.987 | 7.41 |
| f99–171 | 1.065 | 1.012 | 7.52 |
| f196–249 | 1.039 | 1.045 | 7.34 |
| f287–320 | **0.938** | 1.070 | 11.70 |
| f329–354 | **0.968** | 1.067 | 14.66 |
| f361–373 | **0.960** | 1.051 | 15.42 |
| f375–398 | **0.938** | 1.066 | **16.08** |

(Short segments and the f267–f285 zig-zag outlier, 1.656, are in the JSON.)

- **Not supported.** Late in the flight the pipeline covers *less* distance than the image, although its scale is
  6–7% too high (f340: measured 0.5906 vs 0.6266, +6.1%).
- In the last three segments image speed reaches 14.7–16.1 px/frame, at or above the **15 px/frame forward
  clamp**. The clamp caps motion and absorbs the scale excess: in f375–f398, 15 / 16.08 = 0.933, against a measured
  0.938.
- **f287–f320 is not explained by the clamp** (11.7 px/frame, still 0.938). It overlaps the Case 3 heading
  divergence; its cause is not established.
- This refutes the Stage 2.2 suggestion that the stale scale lock explains the along-track "too fast" bias. SIFT
  shows the same lead against GT, which places that bias in the ground truth.

### Case 3 — heading prior vs image motion at the waypoint-2 turn (control, f352)

![case3](../out/failure_cases/case3_heading_prior_f352.png)

| Frame | SIFT direction (image) | pipeline direction |
|---|---|---|
| f304 | −111.7° | −112.0° |
| f316 | −108.5° | **−140.5°** |
| f328 | −106.7° | −110.9° |
| f344 | −105.2° | −110.2° |
| **f352** | −104.8° | **−131.6°** |
| f360 | −104.3° | −112.9° |

- In the imagery the aircraft eases through **+7.5° of turn over f300–f368**. The heading prior, derived from the
  distorted waypoint geometry, holds the old leg (−130°) and then **jumps +23.1° at f353**.
- The clamp decomposes motion along that prior leg direction and pulls the estimate's direction away from the
  image motion, by up to **35.4° (f319)**. The clamp is active on 20 frames of f300–f372.
- Commits pull the direction back (f321/f328, then f355/f360). **Commit f355 moves the pipeline–SIFT gap by
  +23.6 px; no other commit moves it by more than 10.0 px.**
- Cross-track error stays moderate (window max 49.9 px at f313). The failure is mostly in direction and
  along-track position, which ATE cannot see.
- Caveat: SIFT directions are image-frame, ~6.6° rotated from GT. The comparison uses direction *changes* and
  pipeline-vs-image differences, which that rotation does not affect.

## 6. How to reproduce

```bash
python run.py --config configs/demo.yaml                                  # production run -> out/frame_log.csv
python scripts/evaluate.py --config configs/demo.yaml --log pipeline=out/frame_log.csv --out out/eval/pipeline
python scripts/classical_baseline.py --config configs/demo.yaml --detector sift --mode windowed --out out/classical/sift_windowed.csv
python scripts/run_ablations.py --matrix configs/ablations.yaml           # ~25 min on RTX 5060
python scripts/render_failure_cases.py --config configs/demo.yaml --out out/failure_cases
```

## 7. Response to the 2024 reviewer remark

> *«бажано було б додати необхідну статистику у програмний застосунок»*
> ("it would be desirable to add the necessary statistics to the software application")

The application now produces its statistics itself, reproducibly, from a single configuration:

| Statistic | Implementation | Output / command | What it established |
|---|---|---|---|
| Per-frame telemetry: detections, graph size, candidates, TOPSIS best and runner-up scores, inliers, residual, landmark spread, estimate vs GT, scale, heading, gating, tracker health, clamp activity, tracking state | `core/diagnostics.py` `FrameLogger` (Level 0) | `out/frame_log.csv`, 29 columns, one row per frame | Every other statistic here is computed from this log |
| Per-stage timing distribution (mean / median / p95 / max / share) | `StageProfiler` (Level 0) | `out/profile.csv` + table printed at end of run | On GPU YOLO is ~35% of runtime; relocalization candidate generation ~22% |
| Determinism and reproducibility | Global seeding; SHA256 data manifest; byte-identical run checks | `scripts/verify_data.py`; repeated runs | Identical frame logs across runs; every refactor verified identical to a reference |
| Extended ATE statistics | `core/metrics.py`, `scripts/evaluate.py` (2.1) | mean / median / p95 / max / RMSE over tracked and all frames; coverage; along-track at passages; commit schedule | Reproduces the printed ATE exactly; exposed the coverage artifact behind earlier low-ATE claims |
| Comparison with classical methods | `scripts/classical_baseline.py` (2.2) | SIFT and ORB, global and windowed | Classical matching localizes this sequence; the GT frame disagrees with the map image |
| Contribution of each component | `scripts/run_ablations.py`, `experiments/` (2.3) | 23 controlled runs, summary tables | RANSAC, TOPSIS, graph edges and the heading prior are each load-bearing |
| Failure analysis | `scripts/render_failure_cases.py` (2.4) | annotated figures + JSON | Four mechanisms documented; one hypothesis (scale accumulation) refuted |

These statistics did more than decorate the results: they overturned several of them. A historical "~10 m"
result turned out to be a coverage artifact. LSR proved not to measure correctness. The ground truth disagrees with
the map image. Part of the accuracy comes from flight-plan priors. Stating these limits with numbers is the
purpose of the statistics the reviewer asked for.

## 8. Limitations

- One flight, one map tile. No claim here generalizes without more sequences.
- Ground truth is an uncorrected waypoint polyline with an unverified georeference. Absolute metres are nominal.
- The pipeline starts at waypoint 0 and uses leg headings derived from the same waypoints. A_prior replaces the
  heading with a wrong static one; it does not remove the prior entirely.
- Ablations other than A4 are single deterministic runs. Runs that reschedule relocalization commits carry
  roughly ±50% RMSE sensitivity.
