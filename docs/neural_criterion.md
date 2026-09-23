# Siamese similarity as a 4th TOPSIS criterion

Status: **implemented, measured, off by default.** Enabling it costs accuracy on GeoTest1 at every weight tried.

## What was wired

- `core/neural_comparator.py` (a `SiameseNetwork`: MLP 32 → 128 → 256 → 128 → 64, with BatchNorm) existed since commit `0c9c78e` but nothing imported it, and its checkpoint was missing from the working tree.
- The checkpoint was recovered from commit `8759359` (2026-04-19), blob `1fd9407`, 329 KB, restored to `data/models/siamese_drone_v1.pth`. Its tensor shapes match the architecture exactly, so no retraining was needed and `tools/generate_dataset.py` / `tools/train_siamese.py` were not run.
- `EnsembleNavigator` now takes `neural_cfg`. When enabled, the embedding distance between the camera route and each map candidate becomes a 4th **cost** column in the TOPSIS decision matrix, alongside `angle_error`, `ratio_error` and constellation affinity.
- Geometric weights are rescaled by `1 − weight` and the neural column takes `weight`, so the vector still sums to 1 and the geometric balance is unchanged.

## Encoding caveat

The net was trained on **16-point** paths (32 inputs) sampled as consecutive map points, with noise, ±15° rotation and 0.9–1.1 scale (`tools/generate_dataset.py`). Production routes have at most 10 vertices (`relocalization.photo_route_max_vertices`), so `resample_path()` resamples each route to 16 points by arc length. This is deterministic but **off the training distribution**: a resampled 10-vertex route is not the same object as 16 consecutive map points. That mismatch is the most likely reason the criterion hurts rather than helps.

## Measurement (GeoTest1, one variable changed, same seed)

| Neural weight | ATE | ΔATE | LSR | Commits |
|---|---|---|---|---|
| off (control) | 26.51 m | — | 100% | 15 |
| 0.1 | 27.08 m | +2% | 100% | 15 |
| 0.2 | 33.33 m | +26% | 100% | 15 |
| 0.35 | 51.95 m | +96% | 100% | 15 |
| 0.5 | 48.55 m | +83% | 100% | 15 |

The commit schedule is unchanged (15 relocalization commits in every run), so the deltas reflect candidate selection, not a rescheduled search. Degradation grows with the weight, which is what a criterion carrying no usable signal looks like once it outvotes the geometric ones.

With `enabled: false` the frame log is byte-identical to the geometric baseline: sha256 `63e3e4bb11db52a4ff898e8ce314bdc260d5296be3fab8b328b83c4e52c987c6`.

## What would make it usable

1. **Retrain on the real distribution:** routes of the length the pipeline actually produces, built from map landmarks, with the pipeline's own detection noise, rather than synthetic 16-point paths.
2. **Validate the embedding before wiring it in:** on held-out pairs, check that the distance separates the correct map route from decoys. That is a scoring test, not a pipeline run.
3. **Re-run this weight sweep.** The criterion earns a place only if some weight beats 26.51 m with an unchanged commit schedule.

Reproduce: set `neural_comparator.enabled: true` and a `weight` in `configs/demo.yaml`, then `python run.py --config configs/demo.yaml`.
