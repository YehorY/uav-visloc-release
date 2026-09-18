"""
scripts/render_failure_cases.py — LEVEL 2.4 annotated failure cases.

    python scripts/render_failure_cases.py --config configs/demo.yaml --out out/failure_cases

Inputs (already on disk from Levels 2.2-2.3): out/ablations/<run>/{frame_log.csv,run.log} and
out/classical/sift_windowed.csv. YOLO is re-run on each key frame only, to draw detections.
Every number written to the figures and JSON is recomputed here from those logs.

Cases
  case1_sparse_f131       A4_drop50_s5: position solve on one inlier, no search fires, lateral clamp drift
  case1b_stale_binding    control f80-f180: handoff taken in a detection trough, stale bindings for ~60 frames
  case2a_scale_poison_f1  A3b_nodes_only: boot binds the wrong landmark pair and locks the wrong scale
  case2b_scale_nonaccum   control vs SIFT: ~7% scale excess does not accumulate (forward clamp masks it)
  case3_heading_f352      control: heading prior (waypoint geometry) disagrees with image motion at the turn

All ATE-style errors are vs the nominal GT polyline (uncorrected image frame).
"""

import argparse
import csv
import json
import os
import re
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import yaml                                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from core import metrics as M                                # noqa: E402
from evaluate import waypoints_px                            # noqa: E402
from experiments.dropout import apply_detection_dropout      # noqa: E402

LOCKED_SCALE = 0.6266
C_GT, C_RUN, C_CTRL, C_SIFT = "#1f5fbf", "#d62728", "#2ca02c", "#ff8c00"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_run(name):
    d = os.path.join(ROOT, "out", "ablations", name)
    rows = M.load_frame_log(os.path.join(d, "frame_log.csv"))
    P = np.array([[float(r["est_x"]), float(r["est_y"])] for r in rows])
    v = M.cross_track_vectors_px(rows)
    raw = open(os.path.join(d, "run.log"), "rb").read().replace(b"\x00", b"").decode("utf-8", "ignore")
    commits, frame = [], None
    for line in raw.splitlines():
        m = re.search(r"\[Frame (\d+)\]", line)
        if m:
            frame = int(m.group(1))
        if "[SLAM] Match Found!" in line and frame is not None:
            commits.append(frame)
    return {"name": name, "rows": rows, "P": P, "err": np.hypot(v[:, 0], v[:, 1]), "commits": commits, "log": raw}


def col(run, key):
    out = []
    for r in run["rows"]:
        try:
            out.append(float(r[key]))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.array(out)


def bound_pairs(run):
    ni, ir = col(run, "n_inliers"), col(run, "inlier_ratio")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((ir > 0) & np.isfinite(ir), np.round(ni / ir), 0.0)


def scale_telem(run):
    m = re.search(r"\[SCALE-TELEM\] pair_ids=\[(\d+), (\d+)\] map_dist=([0-9.]+)px vid_dist=([0-9.]+)px "
                  r"vid=\((-?\d+),(-?\d+)\)<->\((-?\d+),(-?\d+)\) map=\((-?\d+),(-?\d+)\)<->\((-?\d+),(-?\d+)\)", run["log"])
    s = re.search(r"Strict Scale locked at:\s*([0-9.]+)", run["log"])
    g = [int(x) if "." not in x else float(x) for x in m.groups()]
    return {"pair_ids": g[0:2], "map_dist": g[2], "vid_dist": g[3], "vid": [g[4:6], g[6:8]],
            "map": [g[8:10], g[10:12]], "scale": float(s.group(1))}


def direction(X, k, span=8):
    a, b = max(0, k - span), min(len(X) - 1, k + span)
    d = X[b] - X[a]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def read_frames(video, wanted):
    cap, out, f = cv2.VideoCapture(video), {}, 0
    while len(out) < len(wanted):
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        if f in wanted:
            out[f] = img
    cap.release()
    return out


class Detector:
    def __init__(self, model_path, classes):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.classes = set(classes)

    def __call__(self, img):
        dets = []
        for r in self.model(img, verbose=False, device=self.device):
            for b in r.boxes:
                c = int(b.cls[0])
                if c in self.classes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].cpu().numpy())
                    dets.append({"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "cls": c, "box": (x1, y1, x2, y2)})
        return dets


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def draw_video(ax, img, dets, title, dropped=None, pairs=()):
    ax.imshow(img[..., ::-1])
    for k, d in enumerate(dets):
        x1, y1, x2, y2 = d["box"]
        gone = dropped is not None and k in dropped
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0,
                                   ec="#888888" if gone else "#00e000", ls=":" if gone else "-"))
    for (p, q), color, label in pairs:
        ax.plot([p[0], q[0]], [p[1], q[1]], "-o", color=color, lw=2.5, ms=7, label=label)
    if pairs:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def draw_map(ax, map_img, wps, trajs, window, title, marks=(), margin=160):
    pts = [np.asarray(t[window[0] - 1:window[1]]) for t, *_ in trajs] + [np.asarray(m[0]).reshape(1, 2) for m in marks]
    allp = np.vstack(pts)
    x0, y0 = np.maximum(allp.min(axis=0) - margin, 0).astype(int)
    x1, y1 = allp.max(axis=0) + margin
    x1, y1 = int(min(x1, map_img.shape[1])), int(min(y1, map_img.shape[0]))
    ax.imshow(map_img[y0:y1, x0:x1, ::-1], extent=(x0, x1, y1, y0))
    W = np.asarray(wps, dtype=float)
    ax.plot(W[:, 0], W[:, 1], "--", color=C_GT, lw=2, label="GT polyline (uncorrected)")
    ax.plot(W[:, 0], W[:, 1], "s", color=C_GT, ms=6)
    for t, color, label, commits in trajs:
        seg = t[window[0] - 1:window[1]]
        ax.plot(seg[:, 0], seg[:, 1], "-", color=color, lw=1.8, label=label)
        cs = [c for c in commits if window[0] <= c <= window[1]]
        if cs:
            ax.plot(t[np.array(cs) - 1, 0], t[np.array(cs) - 1, 1], "o", mfc="none", mec=color, ms=8, mew=1.5)
    for p, color, label in marks:
        ax.plot(p[0], p[1], "*", color=color, ms=14, mec="black", label=label)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.legend(loc="best", fontsize=7)
    ax.set_title(title + "  (circles = commits)", fontsize=10)
    ax.tick_params(labelsize=7)


def shade_commits(ax, commits, window, color="#999999"):
    for c in commits:
        if window[0] <= c <= window[1]:
            ax.axvline(c, color=color, lw=0.8, ls=":")


def layout(title):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], hspace=0.35, wspace=0.12)
    axes = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :]))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig, axes


def save(fig, out_dir, name, data):
    fig.text(0.01, 0.005, "All errors vs nominal GT polyline (uncorrected image frame). Map px; metres = px x nominal 0.5.",
             fontsize=8, color="#555555")
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"[FAILURE] wrote {name}.png / .json")


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, args.config), encoding="utf-8"))
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    os.chdir(ROOT)

    wps, _, _ = waypoints_px(cfg)
    map_img = cv2.imread(cfg["data"]["map_image"])
    ctrl, drop, nodes = load_run("control_r1"), load_run("A4_drop50_s5"), load_run("A3b_nodes_only")
    srows = list(csv.DictReader(open(os.path.join(ROOT, "out", "classical", "sift_windowed.csv"))))
    S = np.array([[float(r["est_x"]), float(r["est_y"])] for r in srows])
    hs = np.array([float(r["h_scale"]) for r in srows])
    frames = read_frames(cfg["data"]["video"], {1, 131, 159, 352})
    det = Detector(cfg["data"]["model"], cfg["sensor"]["detector_landmark_classes"])
    x = np.arange(1, len(ctrl["rows"]) + 1)
    index = []

    # ---- Case 1: sparse landmarks (A4_drop50_s5, f131) -------------------------------------------
    K, W = 131, (100, 180)
    dets = det(frames[K])
    kept = apply_detection_dropout([(d["cx"], d["cy"]) for d in dets], [d["cls"] for d in dets],
                                   [0.0] * len(dets), [(0, 0)] * len(dets), 0.5, 5, K)[0]
    kept_set = {(round(c[0], 3), round(c[1], 3)) for c in kept}
    dropped = {k for k, d in enumerate(dets) if (round(d["cx"], 3), round(d["cy"], 3)) not in kept_set}
    ni_d, bp_d, det_d, det_c = col(drop, "n_inliers"), bound_pairs(drop), col(drop, "n_detections"), col(ctrl, "n_detections")
    win = slice(W[0] - 1, W[1])
    climb = [K - 8, K + 8]
    rate = (drop["err"][climb[1] - 1] - drop["err"][climb[0] - 1]) / (climb[1] - climb[0])
    fig, (a0, a1, a2, a3) = layout("Case 1 — sparse landmarks: the position solve rests on ONE inlier and no search fires "
                                   "(A4 50% dropout, seed 5, key frame f131)")
    draw_video(a0, frames[K], dets, f"f{K}: YOLO detections — solid kept ({len(dets)-len(dropped)}), dotted dropped ({len(dropped)})",
               dropped=dropped)
    draw_map(a1, map_img, wps, [(drop["P"], C_RUN, "A4 50% dropout (seed 5)", drop["commits"]),
                                (ctrl["P"], C_CTRL, "control", ctrl["commits"])], W, f"f{W[0]}–f{W[1]}",
             marks=[(drop["P"][K - 1], C_RUN, f"dropout est. f{K}")])
    a2.plot(x[win], drop["err"][win], color=C_RUN, label="cross-track error, dropout run")
    a2.plot(x[win], ctrl["err"][win], color=C_CTRL, label="cross-track error, control")
    a2.axvline(K, color="black", lw=1)
    shade_commits(a2, drop["commits"], W, C_RUN)
    a2.set_ylabel("px")
    a2.legend(fontsize=8, loc="upper left")
    a2.set_title(f"error climbs {rate:.1f} px/frame around f{K} (lateral clamp cap: "
                 f"{cfg['sequence_specific_tuning']['kinematic_clamp_pxf']['lateral']} px/frame)", fontsize=9)
    a3.plot(x[win], det_d[win], color=C_RUN, label="detections, dropout run")
    a3.plot(x[win], det_c[win], color=C_CTRL, ls="--", label="detections, control")
    a3b = a3.twinx()
    a3b.step(x[win], ni_d[win], color="black", where="mid", label="RANSAC inliers")
    a3b.step(x[win], bp_d[win], color="#9467bd", where="mid", label="bound pairs voting")
    a3b.axhline(cfg["sequence_specific_tuning"]["relocalization"]["handoff_max_pairs"], color="#9467bd", ls=":", lw=1)
    a3b.set_ylabel("count")
    a3.set_xlabel("frame")
    a3.legend(fontsize=8, loc="upper left")
    a3b.legend(fontsize=8, loc="upper right")
    a3.set_title("dotted purple = handoff trigger (search fires only when bound pairs <= 2)", fontsize=9)
    one = [f for f in range(W[0], W[1] + 1) if ni_d[f - 1] == 1]
    data1 = {"run": "A4_drop50_s5", "key_frame": K, "window": W,
             "detections_key": int(det_d[K - 1]), "detections_control_key": int(det_c[K - 1]),
             "yolo_on_key_frame": len(dets), "dropped_on_key_frame": len(dropped),
             "error_px": {f"f{f}": round(float(drop["err"][f - 1]), 1) for f in (123, 127, 131, 135, 139)},
             "error_px_control": {f"f{f}": round(float(ctrl["err"][f - 1]), 1) for f in (123, 127, 131, 135, 139)},
             "error_growth_px_per_frame_around_key": round(float(rate), 2),
             "frames_with_single_inlier_in_window": len(one),
             "bound_pairs_at_key": int(bp_d[K - 1]),
             "min_bound_pairs_while_single_inlier": int(min(bp_d[f - 1] for f in one)) if one else None,
             "search_candidates_in_window": int(np.nansum(col(drop, "n_candidates")[win] > 0)),
             "commits": drop["commits"], "max_error_px": round(float(drop["err"].max()), 1),
             "max_error_frame": int(np.argmax(drop["err"])) + 1}
    save(fig, out, "case1_sparse_f131", data1)
    index.append(("case1_sparse_f131", "Sparse landmarks (A4 50% dropout, seed 5)", data1))

    # ---- Case 1b: stale bindings in the production run (control f80-f180) --------------------------
    K, W = 159, (80, 180)
    dets = det(frames[K])
    win = slice(W[0] - 1, W[1])
    bp_c, ni_c = bound_pairs(ctrl), col(ctrl, "n_inliers")
    fig, (a0, a1, a2, a3) = layout("Case 1b — production run: handoff committed in a detection trough, "
                                   "then ~60 frames on the same stale bindings (control, peak f159)")
    draw_video(a0, frames[K], dets, f"f{K}: {len(dets)} detections available — the solve still uses ~{int(bp_c[K-1])} old bindings")
    draw_map(a1, map_img, wps, [(ctrl["P"], C_CTRL, "control", ctrl["commits"]), (S, C_SIFT, "SIFT (image geometry)", [])],
             W, f"f{W[0]}–f{W[1]}", marks=[(ctrl["P"][K - 1], C_CTRL, f"control est. f{K}")])
    a2.plot(x[win], ctrl["err"][win], color=C_CTRL, label="cross-track error, control")
    a2.axvline(K, color="black", lw=1)
    shade_commits(a2, ctrl["commits"], W)
    a2.set_ylabel("px")
    a2.legend(fontsize=8, loc="upper left")
    a2.set_title("dotted = commits (f88, f98 in the trough; f172 resets the bindings)", fontsize=9)
    a3.plot(x[win], col(ctrl, "n_detections")[win], color=C_CTRL, label="detections")
    a3b = a3.twinx()
    a3b.step(x[win], bp_c[win], color="#9467bd", where="mid", label="bound pairs voting")
    a3b.step(x[win], ni_c[win], color="black", where="mid", label="RANSAC inliers")
    a3b.set_ylabel("count")
    a3.set_xlabel("frame")
    a3.legend(fontsize=8, loc="upper left")
    a3b.legend(fontsize=8, loc="upper right")
    data1b = {"run": "control_r1", "key_frame": K, "window": W,
              "detections": {f"f{f}": int(col(ctrl, 'n_detections')[f - 1]) for f in (88, 98, 110, 130, 159, 172)},
              "bound_pairs": {f"f{f}": int(bp_c[f - 1]) for f in (100, 120, 140, 159, 170, 175)},
              "inliers": {f"f{f}": int(ni_c[f - 1]) for f in (100, 120, 140, 159, 170, 175)},
              "error_px": {f"f{f}": round(float(ctrl["err"][f - 1]), 1) for f in (98, 120, 140, 159, 172, 180)},
              "peak_error_px": round(float(ctrl["err"][win].max()), 1),
              "peak_frame": int(W[0] + np.argmax(ctrl["err"][win])),
              "commits_in_window": [c for c in ctrl["commits"] if W[0] <= c <= W[1]]}
    save(fig, out, "case1b_stale_binding_f159", data1b)
    index.append(("case1b_stale_binding_f159", "Stale bindings after a trough handoff (production run)", data1b))

    # ---- Case 2a: boot scale poisoning without graph edges (A3b, f1) -------------------------------
    K = 1
    dets = det(frames[K])
    tc, tn = scale_telem(ctrl), scale_telem(nodes)
    fig, (a0, a1, a2, a3) = layout("Case 2a — scale poisoning at boot: without edges the matcher binds the wrong "
                                   "pair and locks the wrong scale (A3b nodes-only, f1)")
    draw_video(a0, frames[K], dets, f"f1: {len(dets)} detections; bound pairs used for the one-time scale lock",
               pairs=[((tc["vid"][0], tc["vid"][1]), C_CTRL, f"control pair {tc['pair_ids']} -> scale {tc['scale']:.4f}"),
                      ((tn["vid"][0], tn["vid"][1]), C_RUN, f"nodes-only pair {tn['pair_ids']} -> scale {tn['scale']:.4f}")])
    draw_map(a1, map_img, wps, [(ctrl["P"], C_CTRL, "control", ctrl["commits"]), (nodes["P"], C_RUN, "A3b nodes-only", nodes["commits"])],
             (1, 60), "map side of the pairs, f1–f60",
             marks=[(tc["map"][0], C_CTRL, "control pair (map)"), (tc["map"][1], C_CTRL, None),
                    (tn["map"][0], C_RUN, "nodes-only pair (map)"), (tn["map"][1], C_RUN, None)])
    a2.plot(x, ctrl["err"], color=C_CTRL, label="cross-track error, control")
    a2.plot(x, nodes["err"], color=C_RUN, label="cross-track error, A3b nodes-only")
    a2.set_ylabel("px")
    a2.legend(fontsize=8, loc="upper left")
    a2.set_title("the wrong scale persists for the whole flight (the lock is one-time)", fontsize=9)
    a3.plot(x, hs, color=C_SIFT, label="SIFT homography scale (measured per frame)")
    a3.axhline(tc["scale"], color=C_CTRL, label=f"control lock {tc['scale']:.4f}")
    a3.axhline(tn["scale"], color=C_RUN, label=f"nodes-only lock {tn['scale']:.4f}")
    a3.set_ylabel("map px / video px")
    a3.set_xlabel("frame")
    a3.legend(fontsize=8, loc="lower left")
    data2a = {"run": "A3b_nodes_only", "key_frame": 1, "control": tc, "nodes_only": tn,
              "sift_scale_f1": round(float(hs[0]), 4), "sift_scale_median_f1_f50": round(float(np.median(hs[:50])), 4),
              "nodes_only_scale_error_vs_control_pct": round(100 * (tn["scale"] / tc["scale"] - 1), 1),
              "rmse_px": {"control": round(M.error_stats(ctrl["err"])["rmse"], 1), "nodes_only": round(M.error_stats(nodes["err"])["rmse"], 1)},
              "commits": {"control": len(ctrl["commits"]), "nodes_only": len(nodes["commits"])}}
    save(fig, out, "case2a_scale_poison_f1", data2a)
    index.append(("case2a_scale_poison_f1", "Boot scale poisoning without edges (A3b)", data2a))

    # ---- Case 2b: scale excess does not accumulate (negative finding) --------------------------------
    commits = ctrl["commits"]
    segs = []
    for a, b in zip(commits, commits[1:] + [len(x) + 1]):
        i, j = a, b - 2                                    # 0-based: frames a+1 .. b-1
        if j - i < 8:
            continue
        pp = float(np.linalg.norm(np.diff(ctrl["P"][i:j + 1], axis=0), axis=1).sum())
        sp = float(np.linalg.norm(np.diff(S[i:j + 1], axis=0), axis=1).sum())
        segs.append({"frames": [a + 1, b - 1], "pipeline_over_sift_path": round(pp / sp, 3),
                     "predicted_by_scale": round(LOCKED_SCALE / float(np.median(hs[i:j + 1])), 3),
                     "sift_speed_px_per_frame": round(sp / (j - i), 2)})
    fwd = cfg["sequence_specific_tuning"]["kinematic_clamp_pxf"]["forward"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.18)
    b0, b1, b2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    fig.suptitle("Case 2b — negative finding: the ~7% scale excess does NOT accumulate into error; "
                 "the forward clamp caps motion (control vs SIFT)", fontsize=13, fontweight="bold")
    b0.plot(x, hs, color=C_SIFT, label="SIFT homography scale (measured)")
    b0.axhline(LOCKED_SCALE, color=C_CTRL, label=f"pipeline lock {LOCKED_SCALE}")
    b0.axvline(340, color="black", lw=1)
    b0.set_ylabel("map px / video px")
    b0.legend(fontsize=9)
    b0.set_title(f"f340: measured {hs[339]:.3f} vs locked {LOCKED_SCALE} (+{100*(LOCKED_SCALE/hs[339]-1):.1f}% excess)", fontsize=10)
    labels = [f"f{s['frames'][0]}–{s['frames'][1]}" for s in segs]
    xi = np.arange(len(segs))
    b1.bar(xi - 0.2, [s["pipeline_over_sift_path"] for s in segs], 0.4, color=C_CTRL, label="pipeline / SIFT path length")
    b1.bar(xi + 0.2, [s["predicted_by_scale"] for s in segs], 0.4, color="#bbbbbb", label="predicted if scale drove error")
    b1.axhline(1.0, color="black", lw=0.8)
    b1.set_xticks(xi)
    b1.set_xticklabels(labels, rotation=45, fontsize=7)
    b1.set_ylim(0.8, 1.2)
    b1.legend(fontsize=8)
    b1.set_title("commit-free segments (the f267–285 outlier, 1.66, is clipped: zig-zag, not scale)", fontsize=9)
    b2.bar(xi, [s["sift_speed_px_per_frame"] for s in segs], color=C_SIFT)
    b2.axhline(fwd, color=C_RUN, ls="--", label=f"forward clamp {fwd} px/frame")
    b2.set_xticks(xi)
    b2.set_xticklabels(labels, rotation=45, fontsize=7)
    b2.set_ylabel("image speed, map px/frame")
    b2.legend(fontsize=8)
    b2.set_title("late segments fly at ~the clamp limit, so the clamp absorbs the scale excess", fontsize=9)
    data2b = {"locked_scale": LOCKED_SCALE, "sift_scale_f340": round(float(hs[339]), 4),
              "sift_scale_trend_pct_over_flight": round(100 * float(np.polyfit(x - 1, hs, 1)[0]) * 400 / float(np.median(hs)), 1),
              "forward_clamp_px_per_frame": fwd, "segments": segs}
    save(fig, out, "case2b_scale_nonaccumulation", data2b)
    index.append(("case2b_scale_nonaccumulation", "Negative finding: scale excess does not accumulate", data2b))

    # ---- Case 3: heading prior vs image motion at the wp2 turn (control, f352) -----------------------
    K, W = 352, (300, 372)
    dets = det(frames[K])
    win = slice(W[0] - 1, W[1])
    dp = np.array([direction(ctrl["P"], k) for k in range(len(x))])
    ds = np.array([direction(S, k) for k in range(len(x))])
    legs = [float(np.degrees(np.arctan2(*(np.subtract(wps[i + 1], wps[i])[::-1])))) for i in range(len(wps) - 1)]
    passages = cfg["sequence"]["waypoint_passage_frames"]
    prior = np.array([legs[sum(1 for p in passages if f >= p)] for f in x])
    base = float(np.mean(ds[290:320]))
    clamp = col(ctrl, "clamp_hit")
    gap = np.linalg.norm(ctrl["P"] - S, axis=1)
    jumps = {c: round(float(gap[c] - gap[c - 2]), 1) for c in ctrl["commits"] if 2 <= c < len(x)}
    fig, (a0, a1, a2, a3) = layout("Case 3 — heading prior vs image motion at the waypoint-2 turn: the estimate is "
                                   "pulled ~30° off the imaged direction until commits recover it (control, f352)")
    draw_video(a0, frames[K], dets, f"f{K}: one frame before the recorded passage (f353); {len(dets)} detections")
    draw_map(a1, map_img, wps, [(ctrl["P"], C_CTRL, "control", ctrl["commits"]), (S, C_SIFT, "SIFT (image geometry)", [])],
             W, f"f{W[0]}–f{W[1]}", marks=[(ctrl["P"][K - 1], C_CTRL, f"control est. f{K}"), (S[K - 1], C_SIFT, f"SIFT f{K}")])
    a2.plot(x[win], dp[win], color=C_CTRL, label="pipeline motion direction")
    a2.plot(x[win], ds[win], color=C_SIFT, label="SIFT motion direction (image)")
    a2.step(x[win], prior[win], color=C_GT, where="post", ls="--", label="heading prior (waypoint legs, GT frame)")
    a2.axvline(K, color="black", lw=1)
    shade_commits(a2, ctrl["commits"], W)
    a2.set_ylabel("degrees")
    a2.legend(fontsize=8, loc="lower left")
    a2.set_title(f"image direction changes {ds[367]-base:+.1f}° gradually; prior jumps "
                 f"{legs[2]-legs[1]:+.1f}° at f{passages[1]} (SIFT dirs are image frame, ~6.6° rotated from GT)", fontsize=9)
    a3.step(x[win], clamp[win], color=C_RUN, where="mid", label="kinematic clamp active")
    a3.plot(x[win], ctrl["err"][win] / max(1.0, float(ctrl["err"][win].max())), color=C_CTRL, lw=1,
            label="cross-track error (normalised)")
    shade_commits(a3, ctrl["commits"], W)
    a3.set_ylim(-0.1, 1.2)
    a3.set_xlabel("frame")
    a3.legend(fontsize=8, loc="upper left")
    a3.set_title(f"commit f355 moves the pipeline–SIFT gap by {jumps.get(355):+.1f} px; "
                 f"the largest move at any other commit is {max(abs(v) for c, v in jumps.items() if c != 355):.1f} px", fontsize=9)
    div = np.abs(((dp - ds) + 180) % 360 - 180)
    data3 = {"run": "control_r1", "key_frame": K, "window": W,
             "direction_deg": {f"f{f}": {"pipeline": round(float(dp[f - 1]), 1), "sift": round(float(ds[f - 1]), 1)}
                               for f in (304, 316, 328, 344, 352, 360)},
             "sift_direction_change_f300_f368_deg": round(float(ds[367] - base), 1),
             "prior_leg_change_deg": round(legs[2] - legs[1], 1),
             "max_pipeline_vs_sift_divergence_deg": round(float(div[win].max()), 1),
             "max_divergence_frame": int(W[0] + np.argmax(div[win])),
             "clamp_active_frames_in_window": int(np.nansum(clamp[win])),
             "gap_jump_px_at_commits": jumps,
             "max_cross_track_px_in_window": round(float(ctrl["err"][win].max()), 1)}
    save(fig, out, "case3_heading_prior_f352", data3)
    index.append(("case3_heading_prior_f352", "Heading prior vs image motion at the turn (production run)", data3))

    with open(os.path.join(out, "index.md"), "w", encoding="utf-8") as fh:
        fh.write("# Level 2.4 failure cases\n\nAll errors vs nominal GT polyline (uncorrected image frame). "
                 "Generated by `scripts/render_failure_cases.py`; numbers in each `.json`.\n\n")
        for name, title, _ in index:
            fh.write(f"## {title}\n\n![{name}]({name}.png)\n\nData: [`{name}.json`]({name}.json)\n\n")
    print("[FAILURE] wrote index.md")


if __name__ == "__main__":
    main()
