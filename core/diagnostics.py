"""
core/diagnostics.py
===================
LEVEL 0 — Measurement Infrastructure for UAV-VisLoc.

Modular, dependency-light instrumentation:
  * set_global_determinism(seed)  — freezes every stochastic source used by the pipeline.
  * StageProfiler                 — per-frame, per-stage millisecond timing (context manager)
                                    + end-of-run time-distribution table + profile CSV.
  * FrameLogger                   — strict per-frame diagnostic CSV with a fixed schema.
  * closest_point_on_polyline()   — GT reference helper (per-frame GT does not exist in this
                                    dataset; the nearest point on the GT waypoint polyline is
                                    the honest available reference, matching the ATE metric).

Design rule: NOTHING in this module mutates pipeline state. The pipeline pushes values in;
this module only records. Timing columns live in a SEPARATE csv from the deterministic
frame log, so the determinism acceptance test (two runs -> identical frame_log.csv) is not
polluted by wall-clock noise.
"""

import csv
import math
import os
import random
import time


# ---------------------------------------------------------------------------
# 1. DETERMINISM
# ---------------------------------------------------------------------------

def set_global_determinism(seed: int = 42) -> None:
    """
    Freeze all stochastic sources at the very start of the pipeline.
    Covers: python random, numpy, torch (CPU/GPU), OpenCV RNG.
    (The pipeline's deterministic RANSAC is exhaustive — seed-free by design — but any
    library-level sampler now draws from a frozen stream.)
    """
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import cv2
        cv2.setRNGSeed(seed)
    except ImportError:
        pass
    print(f"[LEVEL0] Global determinism frozen (seed={seed}).")


# ---------------------------------------------------------------------------
# 2. PER-STAGE TIMING
# ---------------------------------------------------------------------------

class _StageTimer:
    """Context manager handed out by StageProfiler.stage(name)."""

    def __init__(self, profiler, name):
        self._p = profiler
        self._name = name
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._p._add(self._name, dt_ms)
        return False


class StageProfiler:
    """
    Per-frame, per-stage timing in milliseconds.

    Usage in the pipeline loop:
        profiler.new_frame(frame_id)
        with profiler.stage("t_yolo"):
            ...
        profiler.end_frame(t_total_ms)

    A stage may be entered multiple times per frame (e.g., several RANSAC solves);
    durations accumulate. Stages not entered in a frame are logged as 0.0.
    """

    STAGES = ["t_yolo", "t_graph_build", "t_candidate_gen",
              "t_topsis", "t_ransac", "t_transform_fit", "t_track", "t_total"]

    def __init__(self):
        self._rows = []            # list of dicts (one per frame)
        self._current = None
        self._frame_id = None

    def new_frame(self, frame_id):
        self._frame_id = frame_id
        self._current = {s: 0.0 for s in self.STAGES}

    def stage(self, name):
        if name not in self.STAGES:
            raise KeyError(f"Unknown stage '{name}'")
        return _StageTimer(self, name)

    def _add(self, name, dt_ms):
        if self._current is not None:
            self._current[name] += dt_ms

    def add_stage_ms(self, name, dt_ms):
        """Record an already-measured duration. For blocks that cannot be wrapped in the
        `stage()` context manager without re-indenting unrelated pipeline code."""
        if name not in self.STAGES:
            raise KeyError(f"Unknown stage '{name}'")
        self._add(name, float(dt_ms))

    def end_frame(self, t_total_ms):
        if self._current is None:
            return
        self._current["t_total"] = float(t_total_ms)
        row = {"frame_id": self._frame_id}
        row.update(self._current)
        self._rows.append(row)
        self._current = None

    # -- outputs ------------------------------------------------------------

    def save_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame_id"] + self.STAGES)
            w.writeheader()
            for r in self._rows:
                w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"[LEVEL0] Profile written: {path} ({len(self._rows)} frames)")

    def distribution_table(self) -> str:
        """End-of-run time distribution table (mean / median / p95 / max / share of total)."""
        if not self._rows:
            return "[LEVEL0] No profiling data collected."
        n = len(self._rows)
        total_sum = sum(r["t_total"] for r in self._rows) or 1e-9
        lines = ["", "=" * 74,
                 "LEVEL 0 — PER-STAGE TIME DISTRIBUTION (ms/frame)",
                 "=" * 74,
                 f"{'stage':<16}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}{'% of total':>12}"]
        for s in self.STAGES:
            vals = sorted(r[s] for r in self._rows)
            mean = sum(vals) / n
            median = vals[n // 2]
            p95 = vals[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
            vmax = vals[-1]
            share = 100.0 * sum(vals) / total_sum
            lines.append(f"{s:<16}{mean:>9.2f}{median:>9.2f}{p95:>9.2f}{vmax:>9.2f}{share:>11.1f}%")
        lines.append("=" * 74)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. PER-FRAME DIAGNOSTIC CSV
# ---------------------------------------------------------------------------

class FrameLogger:
    """
    Strict-schema per-frame CSV. All fields are pipeline-pushed; missing values default to
    NaN (floats) / 0 (counts). Deterministic by construction: contains no wall-clock data
    (timestamp_capture comes from the VIDEO stream, which is deterministic).
    """

    COLUMNS = ["frame_id", "timestamp_capture",
               "n_detections", "n_graph_nodes", "n_graph_edges", "n_candidates",
               "topsis_score_best", "topsis_score_2nd",
               "n_inliers", "inlier_ratio", "residual_rms",
               "landmark_spread_x", "landmark_spread_y",
               "est_x", "est_y", "gt_x", "gt_y",
               "scale", "heading_used", "low_confidence",
               # STEP 1 / MODULE 1a — tracker telemetry (read-only, no behavior)
               "mean_det_conf", "n_confirmed", "J_px", "n_jpx_tracks",
               "id_switch_proxy", "n_bound_hits", "prior_resid_med",
               # clamp_hit: the Phase-8.2 asymmetric clamp bound the continuation delta this frame.
               "clamp_hit",
               # LEVEL 2 — track_state: 0 lost / 1 tracking / 2 coasting. Without it a frame log cannot
               # separate localized frames from lost ones (the "7 m at 17% LSR" coverage artifact).
               "track_state"]

    _COUNT_FIELDS = {"frame_id", "n_detections", "n_graph_nodes", "n_graph_edges",
                     "n_candidates", "n_inliers", "low_confidence",
                     "n_confirmed", "n_jpx_tracks", "id_switch_proxy", "n_bound_hits",
                     "clamp_hit", "track_state"}

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.COLUMNS)
        self._w.writeheader()
        self._n = 0

    def new_row(self, frame_id):
        """Fresh row dict with schema defaults; the pipeline fills what it knows."""
        row = {}
        for c in self.COLUMNS:
            row[c] = 0 if c in self._COUNT_FIELDS else float("nan")
        row["frame_id"] = frame_id
        return row

    def log(self, row: dict):
        out = {}
        for c in self.COLUMNS:
            v = row.get(c)
            if isinstance(v, float):
                out[c] = f"{v:.4f}" if v == v else "nan"   # v==v is False for NaN
            else:
                out[c] = v
        self._w.writerow(out)
        self._n += 1

    def close(self):
        self._f.close()
        print(f"[LEVEL0] Frame log written: {self._path} ({self._n} frames)")


# ---------------------------------------------------------------------------
# 4. GT REFERENCE HELPER
# ---------------------------------------------------------------------------

def closest_point_on_polyline(pt, polyline):
    """
    Nearest point on the GT waypoint polyline to `pt` (map px). Per-frame GT positions do
    not exist in this dataset; this is the same reference geometry the ATE metric uses.
    Returns (gt_x, gt_y).
    """
    px, py = float(pt[0]), float(pt[1])
    best = (float("nan"), float("nan"))
    best_d2 = float("inf")
    for i in range(len(polyline) - 1):
        ax, ay = float(polyline[i][0]), float(polyline[i][1])
        bx, by = float(polyline[i + 1][0]), float(polyline[i + 1][1])
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
        cx, cy = ax + t * vx, ay + t * vy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best_d2:
            best_d2, best = d2, (cx, cy)
    return best
