"""
core/metrics.py
===============
LEVEL 2 — evaluation metrics computed from a frame log (one row per video frame).

Pure functions: no pipeline, video, model or OpenCV imports. Every trajectory — our pipeline, the
classical baselines, every ablation — is scored by this one module, so numbers are comparable by
construction.

GROUND TRUTH (state this with every number)
  * Four GPS waypoints joined into a polyline; there is no per-frame GPS.
  * `gt_x, gt_y` in the log is the NEAREST point on that polyline, so the per-frame error is
    CROSS-TRACK only. Along-track error is measured separately at the two recorded waypoint
    passages (`along_track_at_passages`).
  * Waypoints reach map pixels through `CoordinateToolkit.gps_to_map_pixel`, which stretches the
    waypoints' own lat/lon box (+10% pad each side) over the whole image, rounding down to whole
    pixels. The map has no stored georeference. With square pixels that assumption is inconsistent:
    it implies 0.398 m/px east-west but 0.483 m/px north-south (~21% anisotropy). Primary unit is
    therefore MAP PIXELS; metres are nominal (x 0.5 m/px) and unverified.
  * Our pipeline starts from waypoint 0 and takes leg headings from the flight-controller log.

TWO ATE CONVENTIONS
  * `tracked`: frames the system localized (track_state 1 or 2). What a run reports about itself.
    Blind to coverage: a run that tracks 17% of frames can score a tiny ATE.
  * `all`: every frame, using the position the system published even while lost. Coverage-honest.
  * `pipeline_ate_px` reproduces run_video_test.py's printed ATE exactly: tracked frames plus the
    boot pose that seeds the trajectory history before frame 1 (a zero-error term).
"""

import csv
import math

import numpy as np

TRACK_LOST, TRACK_TRACKING, TRACK_COASTING = 0, 1, 2
NOMINAL_GSD_M_PER_PX = 0.5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _num(v):
    if v in (None, "", "nan"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_frame_log(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: empty frame log")
    for col in ("frame_id", "est_x", "est_y", "gt_x", "gt_y", "track_state"):
        if col not in rows[0]:
            raise ValueError(f"{path}: missing column {col!r}")
    return rows


# ---------------------------------------------------------------------------
# Per-frame errors and statistics
# ---------------------------------------------------------------------------

def cross_track_vectors_px(rows):
    """(N, 2) array of est - gt in map px; gt is the nearest polyline point."""
    return np.array([[_num(r["est_x"]) - _num(r["gt_x"]), _num(r["est_y"]) - _num(r["gt_y"])]
                     for r in rows], dtype=float)


def track_states(rows):
    return np.array([int(_num(r["track_state"])) for r in rows], dtype=int)


def error_stats(errors):
    """mean / median / p95 / max / RMSE of a 1-D error sample. p95 uses linear interpolation."""
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan"),
                "max": float("nan"), "rmse": float("nan")}
    return {"n": int(e.size), "mean": float(e.mean()), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "max": float(e.max()),
            "rmse": float(math.sqrt(float(np.mean(e * e))))}


def ate_stats(rows):
    """Cross-track statistics in map px for both conventions (`tracked`, `all`)."""
    v = cross_track_vectors_px(rows)
    err = np.hypot(v[:, 0], v[:, 1])
    st = track_states(rows)
    return {"tracked": error_stats(err[st != TRACK_LOST]), "all": error_stats(err)}


def pipeline_ate_px(rows):
    """run_video_test.py's printed ATE, in map px: RMSE over [boot pose] + tracked frames.
    The boot pose is waypoint 0, which lies on the polyline, so its term is exactly 0."""
    v = cross_track_vectors_px(rows)
    err = np.hypot(v[:, 0], v[:, 1])[track_states(rows) != TRACK_LOST]
    terms = np.concatenate([[0.0], err])
    return float(math.sqrt(float(np.mean(terms * terms))))


def to_metres(stats_px, gsd=NOMINAL_GSD_M_PER_PX):
    return {k: (v * gsd if k != "n" else v) for k, v in stats_px.items()}


def implied_gsd_from_waypoint_box(waypoints_latlon, map_w, map_h):
    """GSD (m/px) per axis implied by gps_to_map_pixel's box-stretch assumption."""
    la = [p[0] for p in waypoints_latlon]
    lo = [p[1] for p in waypoints_latlon]
    lat_span = (max(la) - min(la)) * 1.2
    lon_span = (max(lo) - min(lo)) * 1.2
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians(sum(la) / len(la)))
    return lon_span * m_lon / map_w, lat_span * m_lat / map_h


def ate_stats_implied_metres(rows, gsd_x, gsd_y):
    """Cross-track error converted per axis with the box-implied (anisotropic) GSD."""
    v = cross_track_vectors_px(rows)
    err = np.hypot(v[:, 0] * gsd_x, v[:, 1] * gsd_y)
    st = track_states(rows)
    return {"tracked": error_stats(err[st != TRACK_LOST]), "all": error_stats(err)}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(rows):
    st = track_states(rows)
    n = len(st)
    longest, run = 0, 0
    for s in st:
        run = run + 1 if s == TRACK_LOST else 0
        longest = max(longest, run)
    return {"frames": n,
            "lsr": float(np.mean(st == TRACK_TRACKING)) if n else float("nan"),
            "coverage": float(np.mean(st != TRACK_LOST)) if n else float("nan"),
            "coasting_frames": int(np.sum(st == TRACK_COASTING)),
            "lost_frames": int(np.sum(st == TRACK_LOST)),
            "longest_lost_run": int(longest)}


# ---------------------------------------------------------------------------
# Along-track error at the recorded waypoint passages
# ---------------------------------------------------------------------------

def along_track_at_passages(rows, waypoints_px, passage_frames):
    """
    For interior waypoint i+1 passed at recorded frame F:
      signed_px     = (est(F) - wp) . u_leg   (negative: estimate behind the waypoint)
      crossing_delta = first frame whose along-leg projection reaches the waypoint, minus F
    Uses the logged estimate on every frame (tracked or not).
    """
    est = {int(_num(r["frame_id"])): np.array([_num(r["est_x"]), _num(r["est_y"])]) for r in rows}
    wps = [np.asarray(p, dtype=float) for p in waypoints_px]
    out = []
    for i, f in enumerate(sorted(int(x) for x in passage_frames)):
        if i + 1 >= len(wps):
            break
        leg = wps[i + 1] - wps[i]
        u = leg / np.linalg.norm(leg)
        s_wp = float(np.dot(wps[i + 1], u))
        signed = float(np.dot(est[f], u) - s_wp) if f in est and np.all(np.isfinite(est[f])) else float("nan")
        cross = next((k for k in sorted(est) if np.all(np.isfinite(est[k])) and float(np.dot(est[k], u)) >= s_wp), None)
        out.append({"waypoint": i + 1, "passage_frame": f, "signed_px": signed,
                    "crossing_frame": cross, "crossing_delta_frames": (cross - f) if cross is not None else None})
    return out


# ---------------------------------------------------------------------------
# Relocalization schedule (the commit-rescheduling butterfly effect)
# ---------------------------------------------------------------------------

def reloc_search_frames(rows):
    """Frames where a relocalization search produced candidates (n_candidates > 0). A search frame is
    where a commit CAN happen; commits themselves are not logged separately."""
    if "n_candidates" not in rows[0]:
        return []
    return [int(_num(r["frame_id"])) for r in rows if _num(r["n_candidates"]) > 0]


def schedule_divergence(rows, reference_rows):
    a, b = set(reloc_search_frames(rows)), set(reloc_search_frames(reference_rows))
    return {"search_frames": len(a), "reference_search_frames": len(b), "shared": len(a & b),
            "symmetric_difference": len(a ^ b)}
