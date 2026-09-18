"""
scripts/evaluate.py — LEVEL 2 evaluator. Scores one or more frame logs with core/metrics.py.

    python scripts/evaluate.py --config configs/demo.yaml \
        --log baseline=out/frame_log.csv --log sift_windowed=out/classical/sift_windowed.csv \
        --out out/eval/level2

The FIRST --log is the reference for relocalization-schedule divergence. Writes <out>.json and
<out>.md. Primary unit is map pixels; metres are nominal (0.5 m/px) and unverified — see the
ground-truth notes in core/metrics.py.
"""

import argparse
import json
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import metrics as M                                   # noqa: E402
from tools.coordinate_toolkit import CoordinateToolkit          # noqa: E402


def waypoints_px(config):
    """Project waypoints to map pixels exactly as run_video_test.py does."""
    wps = config["sequence"]["gps_waypoints"]
    img = cv2.imread(config["data"]["map_image"])
    if img is None:
        raise FileNotFoundError(config["data"]["map_image"])
    map_h, map_w = img.shape[:2]
    tk = CoordinateToolkit(gsd=config["sensor"]["map_gsd_m_per_px"])
    la = [p[0] for p in wps]
    lo = [p[1] for p in wps]
    pts = [tk.gps_to_map_pixel(a, o, min(la), max(la), min(lo), max(lo), map_w, map_h) for a, o in wps]
    return pts, map_w, map_h


def evaluate(config, logs):
    wps_px, map_w, map_h = waypoints_px(config)
    gsd_x, gsd_y = M.implied_gsd_from_waypoint_box(config["sequence"]["gps_waypoints"], map_w, map_h)
    passages = config["sequence"]["waypoint_passage_frames"]
    ref_rows = M.load_frame_log(logs[0][1])
    results = []
    for label, path in logs:
        rows = M.load_frame_log(path)
        ate = M.ate_stats(rows)
        results.append({
            "label": label, "log": path,
            "coverage": M.coverage(rows),
            "ate_px": ate,
            "ate_m_nominal": {k: M.to_metres(v) for k, v in ate.items()},
            "ate_m_box_implied": M.ate_stats_implied_metres(rows, gsd_x, gsd_y),
            "pipeline_ate_px": M.pipeline_ate_px(rows),
            "along_track": M.along_track_at_passages(rows, wps_px, passages),
            "schedule_vs_reference": M.schedule_divergence(rows, ref_rows),
        })
    return {"reference": logs[0][0], "gsd_nominal": M.NOMINAL_GSD_M_PER_PX,
            "gsd_box_implied_xy": [gsd_x, gsd_y], "waypoints_px": [list(map(int, p)) for p in wps_px],
            "results": results}


def to_markdown(report):
    L = []
    L.append("### ATE, cross-track, map px (primary unit)\n")
    L.append("| run | LSR | coverage | tracked mean | median | p95 | max | RMSE | all-frames RMSE | all p95 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in report["results"]:
        t, a, c = r["ate_px"]["tracked"], r["ate_px"]["all"], r["coverage"]
        L.append(f"| {r['label']} | {100*c['lsr']:.1f}% | {100*c['coverage']:.1f}% | {t['mean']:.1f} | "
                 f"{t['median']:.1f} | {t['p95']:.1f} | {t['max']:.1f} | {t['rmse']:.1f} | {a['rmse']:.1f} | {a['p95']:.1f} |")
    L.append("\n### Same, in metres (NOMINAL 0.5 m/px — unverified georeference)\n")
    L.append("| run | tracked mean | median | p95 | max | RMSE | all-frames RMSE | pipeline-convention ATE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in report["results"]:
        t, a = r["ate_m_nominal"]["tracked"], r["ate_m_nominal"]["all"]
        L.append(f"| {r['label']} | {t['mean']:.2f} | {t['median']:.2f} | {t['p95']:.2f} | {t['max']:.2f} | "
                 f"{t['rmse']:.2f} | {a['rmse']:.2f} | {r['pipeline_ate_px'] * M.NOMINAL_GSD_M_PER_PX:.2f} |")
    L.append("\n### Along-track error at recorded waypoint passages, and relocalization schedule\n")
    L.append("| run | wp1 @f255: px / crossing Δframes | wp2 @f353: px / crossing Δframes | search frames | Δ vs reference |")
    L.append("|---|---|---|---|---|")
    for r in report["results"]:
        cells = []
        for p in r["along_track"]:
            d = p["crossing_delta_frames"]
            cells.append(f"{p['signed_px']:+.0f} / {('%+d' % d) if d is not None else 'never'}")
        s = r["schedule_vs_reference"]
        L.append(f"| {r['label']} | {cells[0]} | {cells[1]} | {s['search_frames']} | {s['symmetric_difference']} |")
    gx, gy = report["gsd_box_implied_xy"]
    L.append(f"\nGround truth: 4 GPS waypoints as a polyline; error is cross-track to the nearest point. "
             f"Waypoint-box-implied GSD {gx:.3f} (x) / {gy:.3f} (y) m/px vs nominal 0.5 — the map georeference "
             f"is unverified. Reference for schedule divergence: `{report['reference']}`.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--log", action="append", required=True, help="label=path/to/frame_log.csv")
    ap.add_argument("--out", required=True, help="output path prefix (writes .json and .md)")
    args = ap.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    logs = [tuple(x.split("=", 1)) if "=" in x else (os.path.basename(x), x) for x in args.log]
    report = evaluate(config, logs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    md = to_markdown(report)
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
