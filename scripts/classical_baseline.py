"""
scripts/classical_baseline.py — LEVEL 2.2 classical baseline: SIFT or ORB features + MAGSAC homography,
video frame -> satellite map. Writes a frame log scored by scripts/evaluate.py with the same code as
our pipeline.

    python scripts/classical_baseline.py --config configs/demo.yaml --detector sift --mode windowed \
        --out out/classical/sift_windowed.csv

Modes
  global    match each frame against the WHOLE map. No prior at all.
  windowed  match only map features inside a window around the previous estimate, starting at
            waypoint 0 — the same start prior our pipeline gets. The window grows 1.25x per
            consecutive failure (capped at the full map) and resets on success.

No scale prior: frame and map are both downscaled 0.5x; the homography recovers scale itself, so
our pipeline's calibrated 0.6266 never leaks in. Deterministic: brute-force matching (FLANN's
randomized trees are not reproducible) and cv2.setRNGSeed per frame before MAGSAC.

A frame counts as localized (track_state 1) only if the homography has >= MIN_INLIERS inliers and
is physically plausible: positive determinant, scale in [0.1, 3] map px per video px, anisotropy
<= 1.5, negligible perspective. Otherwise track_state 0 and the last estimate is held.
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from core.diagnostics import closest_point_on_polyline   # noqa: E402
from evaluate import waypoints_px                         # noqa: E402

MAP_DS, FRAME_DS = 0.5, 0.5
MAP_FEATURES, FRAME_FEATURES = 20000, 4000
SIFT_RATIO = 0.75
MIN_MATCHES, MIN_INLIERS = 15, 15
REPROJ_PX = 5.0                      # MAGSAC threshold, full-resolution map px
WINDOW_HALF_PX, WINDOW_GROWTH = 1200.0, 1.25
SCALE_RANGE, MAX_ANISOTROPY, MAX_PERSPECTIVE = (0.1, 3.0), 1.5, 1e-3


def make_detector(name, n):
    if name == "sift":
        return cv2.SIFT_create(nfeatures=n)
    return cv2.ORB_create(nfeatures=n, scaleFactor=1.2, nlevels=8)


def features(det, gray, ds):
    small = cv2.resize(gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    kp, des = det.detectAndCompute(small, None)
    pts = np.array([k.pt for k in kp], dtype=np.float32) / ds if kp else np.zeros((0, 2), np.float32)
    return pts, des


def match(detector, des_frame, des_map):
    if des_frame is None or des_map is None or len(des_frame) < 2 or len(des_map) < 2:
        return []
    if detector == "sift":
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_frame, des_map, k=2)
        return [(m.queryIdx, m.trainIdx) for m, n in (p for p in knn if len(p) == 2) if m.distance < SIFT_RATIO * n.distance]
    return [(m.queryIdx, m.trainIdx) for m in cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_frame, des_map)]


def plausible(H):
    if H is None or abs(H[2, 2]) < 1e-12:
        return False, float("nan")
    Hn = H / H[2, 2]
    A = Hn[:2, :2]
    if np.linalg.det(A) <= 0:
        return False, float("nan")
    s = np.linalg.svd(A, compute_uv=False)
    scale = float(np.sqrt(s[0] * s[1]))
    ok = (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1] and s[0] / s[1] <= MAX_ANISOTROPY
          and abs(Hn[2, 0]) <= MAX_PERSPECTIVE and abs(Hn[2, 1]) <= MAX_PERSPECTIVE)
    return ok, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--detector", choices=["sift", "orb"], required=True)
    ap.add_argument("--mode", choices=["global", "windowed"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)

    wps, map_w, map_h = waypoints_px(cfg)
    map_gray = cv2.imread(cfg["data"]["map_image"], cv2.IMREAD_GRAYSCALE)
    det_map = make_detector(args.detector, MAP_FEATURES)
    det_frame = make_detector(args.detector, FRAME_FEATURES)
    t0 = time.perf_counter()
    map_pts, map_des = features(det_map, map_gray, MAP_DS)
    print(f"[BASELINE] {args.detector}/{args.mode}: {len(map_pts)} map features in {time.perf_counter()-t0:.1f}s")

    cap = cv2.VideoCapture(cfg["data"]["video"])
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    centre = np.array([[[vw / 2.0, vh / 2.0]]], dtype=np.float64)
    prior = np.array(wps[0], dtype=float)
    est = prior.copy()
    half = WINDOW_HALF_PX
    rows, frame_id = [], 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if args.max_frames and frame_id > args.max_frames:
            break
        tf = time.perf_counter()
        fpts, fdes = features(det_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), FRAME_DS)

        if args.mode == "windowed":
            sel = np.nonzero((np.abs(map_pts[:, 0] - prior[0]) <= half) & (np.abs(map_pts[:, 1] - prior[1]) <= half))[0]
        else:
            sel = np.arange(len(map_pts))
        pairs = match(args.detector, fdes, map_des[sel] if len(sel) else None)

        state, n_inl, scale = 0, 0, float("nan")
        if len(pairs) >= MIN_MATCHES:
            src = np.float32([fpts[q] for q, _ in pairs]).reshape(-1, 1, 2)
            dst = np.float32([map_pts[sel[t]] for _, t in pairs]).reshape(-1, 1, 2)
            cv2.setRNGSeed(seed * 100003 + frame_id)
            H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, REPROJ_PX, maxIters=10000, confidence=0.999)
            n_inl = int(mask.sum()) if mask is not None else 0
            good, scale = plausible(H)
            if good and n_inl >= MIN_INLIERS:
                est = cv2.perspectiveTransform(centre, H)[0, 0].astype(float)
                state = 1

        if state == 1:
            prior, half = est.copy(), WINDOW_HALF_PX
        else:
            half = min(half * WINDOW_GROWTH, float(max(map_w, map_h)))

        gx, gy = closest_point_on_polyline(est, wps)
        rows.append({"frame_id": frame_id, "est_x": f"{est[0]:.4f}", "est_y": f"{est[1]:.4f}",
                     "gt_x": f"{gx:.4f}", "gt_y": f"{gy:.4f}", "track_state": state, "n_candidates": 0,
                     "n_matches": len(pairs), "n_inliers": n_inl,
                     "inlier_ratio": f"{(n_inl / len(pairs)) if pairs else 0.0:.4f}",
                     "h_scale": f"{scale:.4f}" if scale == scale else "nan",
                     "ms": f"{(time.perf_counter() - tf) * 1000.0:.1f}"})
        if frame_id % 50 == 0:
            loc = sum(1 for r in rows if r["track_state"] == 1)
            print(f"[BASELINE] frame {frame_id}: localized {loc}/{frame_id}")
    cap.release()

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    loc = sum(1 for r in rows if r["track_state"] == 1)
    print(f"[BASELINE] wrote {args.out}: {len(rows)} frames, localized {loc} ({100*loc/len(rows):.1f}%), "
          f"{time.perf_counter()-t0:.0f}s total")


if __name__ == "__main__":
    main()
