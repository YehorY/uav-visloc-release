"""
scripts/render_demo.py — LEVEL 3.2 two-panel demonstration renderer (offline; the pipeline is not touched).

    python scripts/render_demo.py --config configs/demo.yaml \
        --frame-log out/l3_hook_on.csv --profile out/l3_hook_on_profile.csv \
        --track-log out/demo/track_log.jsonl --sift out/classical/sift_windowed.csv --out out/demo

Outputs
  geotest1_diagnostic.mp4   1920x720, real time, pipeline + GT + SIFT image-matching reference, full HUD
  geotest1_hero.gif         README hero: pipeline + GT only, ~20 s, <= --gif-max-mb. Rendered in "hero" mode:
                            larger type, simplified HUD, background imagery softened (Gaussian) so the
                            GIF fits the size budget; overlays are drawn after softening and stay crisp.
  still_f{N}.jpg            diagnostic-mode stills (unsoftened) for documentation
  demo_manifest.json        sizes, frame counts, playback settings, GIF budget search, map-scale check

Layout 1920x720: camera 1280x720 left (detections, track roles, HUD), map 640x720 right (satellite imagery
at ONE uniform scale centred on the estimate, GT polyline, trajectory, camera footprint, bound landmarks,
mini-map, scale bar, north arrow). An honesty caption spans the bottom.
"""

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from evaluate import waypoints_px   # noqa: E402

W, H, CAM_W, MAP_W = 1920, 720, 1280, 640
MAP_WIN_W, MAP_WIN_H = 800, 900              # map px shown in the 640x720 panel
MAP_S = MAP_W / MAP_WIN_W                     # panel px per map px (uniform)
assert abs(MAP_W / MAP_WIN_W - H / MAP_WIN_H) < 1e-9, "map panel must use one uniform scale"
EVENT_HOLD = 8                                # slow-loop badge held so frame-skipping GIFs cannot miss it
SRC_FPS = 30.0
CAPTION = "GeoTest1: Google Earth render, ~30° curved trajectory, 16% scale variation, nominal 0.5m/px"
CAPTION_NOTES = "errors vs uncorrected GT polyline  ·  camera assumed north-up nadir"

# BGR colours
C_TRAJ, C_SIFT, C_GT = (90, 230, 120), (0, 150, 255), (255, 190, 70)
C_INLIER, C_OUTLIER, C_UNSOLVED, C_UNBOUND, C_COAST = (80, 220, 80), (60, 60, 240), (0, 220, 255), (230, 200, 90), (235, 235, 235)
C_FAST, C_SLOW, C_COMMIT = (70, 170, 40), (0, 140, 230), (230, 80, 230)


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Style:
    """Diagnostic mode is viewed at 1920x720; hero mode is viewed at ~800x300 (x0.42), so everything scales up."""
    def __init__(self, hero):
        self.hero = hero
        m = 2.2 if hero else 1.0
        self.f_sm, self.f_md, self.f_mdb, self.f_lgb = font(int(15 * m)), font(int(18 * m)), font(int(18 * m), True), font(int(22 * m), True)
        self.lw = 2 if hero else 1                       # line-width multiplier
        self.r = 1.7 if hero else 1.0                     # marker-radius multiplier
        self.footer_h = 56 if hero else 30
        self.top_h = 132 if hero else 78
        self.blur = 2.6 if hero else 0.0                  # background softening sigma (panel px)
        self.mini = (200, 250) if hero else (150, 190)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_inputs(args):
    rows = list(csv.DictReader(open(args.frame_log, newline="")))
    prof = [fnum(r["t_total"]) for r in csv.DictReader(open(args.profile, newline=""))]
    tl = [json.loads(line) for line in open(args.track_log, encoding="utf-8")]
    assert len(rows) == len(tl) == len(prof), (len(rows), len(tl), len(prof))
    sift = np.array([[fnum(r["est_x"]), fnum(r["est_y"])] for r in csv.DictReader(open(args.sift, newline=""))])
    assert len(sift) == len(rows), (len(sift), len(rows))
    return rows, prof, tl, sift


def dashed(img, p, q, color, thick=2, dash=12, gap=8):
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    if L < 1:
        return
    u = (q - p) / L
    s = 0.0
    while s < L:
        a, b = p + u * s, p + u * min(s + dash, L)
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, thick, cv2.LINE_AA)
        s += dash + gap


def blend_rect(img, x0, y0, x1, y1, color, alpha):
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def loop_state(k, tl):
    """(label, detail, colour) for frame index k (0-based), holding slow-loop events EVENT_HOLD frames."""
    for j in range(k, max(-1, k - EVENT_HOLD), -1):
        if tl[j]["search"]:
            f = tl[j]["f"]
            detail = f"relocalization at f{f}: " + ("COMMIT, new bindings" if tl[j]["commit"] else "no commit")
            return "SLOW LOOP: RELOCALIZATION", detail, C_SLOW
    last = next((tl[j]["f"] for j in range(k, -1, -1) if tl[j]["commit"]), None)
    return "FAST LOOP", (f"continuation on bindings from f{last}" if last else "continuation"), C_FAST


class MapPanel:
    def __init__(self, map_img, wps, traj, sift):
        self.map = map_img
        self.wps = np.asarray(wps, dtype=float)
        self.traj, self.sift = traj, sift
        pts = np.vstack([self.wps, traj])
        self.lo, self.hi = pts.min(axis=0) - 150, pts.max(axis=0) + 150
        self._mini_cache = {}

    def mini_base(self, mw, mh):
        """Whole-route overview at one uniform scale, letterboxed into mw x mh."""
        if (mw, mh) not in self._mini_cache:
            s = min(mw / (self.hi[0] - self.lo[0]), mh / (self.hi[1] - self.lo[1]))
            lo = (self.lo + self.hi) / 2 - np.array([mw, mh]) / (2 * s)
            M = np.array([[s, 0, -lo[0] * s], [0, s, -lo[1] * s]], np.float32)
            base = cv2.warpAffine(self.map, M, (mw, mh), flags=cv2.INTER_AREA, borderValue=(30, 30, 30))
            self._mini_cache[(mw, mh)] = ((base * 0.6).astype(np.uint8), s, lo)
        return self._mini_cache[(mw, mh)]

    def render(self, k, row, tl_row, st, show_sift, commits_so_far, in_slow):
        est = np.array([fnum(row["est_x"]), fnum(row["est_y"])])
        x0, y0 = int(round(est[0] - MAP_WIN_W / 2)), int(round(est[1] - MAP_WIN_H / 2))
        mh, mw = self.map.shape[:2]
        crop = np.full((MAP_WIN_H, MAP_WIN_W, 3), 40, np.uint8)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + MAP_WIN_W, mw), min(y0 + MAP_WIN_H, mh)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.map[sy0:sy1, sx0:sx1]
        panel = cv2.resize(crop, (MAP_W, H), interpolation=cv2.INTER_AREA)
        if st.blur:
            panel = cv2.GaussianBlur(panel, (0, 0), st.blur)
        panel = (panel * 0.82).astype(np.uint8)
        origin = np.array([x0, y0], float)
        P = lambda p: tuple(np.round((np.asarray(p, float) - origin) * MAP_S).astype(int))
        lw = st.lw

        for a, b in zip(self.wps[:-1], self.wps[1:]):
            dashed(panel, P(a), P(b), C_GT, 2 * lw, 12 * lw, 8 * lw)
        for wpt in self.wps:
            q = P(wpt)
            d = int(5 * st.r)
            cv2.rectangle(panel, (q[0] - d, q[1] - d), (q[0] + d, q[1] + d), C_GT, 2 * lw, cv2.LINE_AA)
        if show_sift:
            cv2.polylines(panel, [np.array([P(p) for p in self.sift[:k + 1]], np.int32)], False, C_SIFT, 2 * lw, cv2.LINE_AA)
        cv2.polylines(panel, [np.array([P(p) for p in self.traj[:k + 1]], np.int32)], False, C_TRAJ, 3 * lw, cv2.LINE_AA)
        for f in commits_so_far:
            cv2.circle(panel, P(self.traj[f - 1]), int(6 * st.r), C_COMMIT, 2 * lw, cv2.LINE_AA)
        # scale-true camera footprint (camera frame is 1920x1080 px, 1 camera px = `scale` map px)
        scale = fnum(row["scale"])
        q = P(est)
        fw, fh = 1920 * scale * MAP_S / 2, 1080 * scale * MAP_S / 2
        dashed(panel, (q[0] - fw, q[1] - fh), (q[0] + fw, q[1] - fh), (245, 245, 245), lw, 6 * lw, 6 * lw)
        dashed(panel, (q[0] - fw, q[1] + fh), (q[0] + fw, q[1] + fh), (245, 245, 245), lw, 6 * lw, 6 * lw)
        if in_slow:
            hd = math.radians(fnum(row["heading_used"]))
            r, half = 220 * MAP_S, math.radians(15)
            poly = [q] + [(int(q[0] + r * math.cos(hd + t)), int(q[1] + r * math.sin(hd + t))) for t in np.linspace(-half, half, 12)]
            over = panel.copy()
            cv2.fillPoly(over, [np.array(poly, np.int32)], C_SLOW)
            cv2.addWeighted(over, 0.40, panel, 0.60, 0, panel)
        for t in tl_row["tracks"]:
            if t["map"] is None:
                continue
            col = {"bound_inlier": C_INLIER, "bound_outlier": C_OUTLIER, "bound_unsolved": C_UNSOLVED}.get(t["role"], C_COAST)
            cv2.circle(panel, P(t["map"]), int(6 * st.r), col, 2 * lw, cv2.LINE_AA)
        if show_sift:
            cv2.circle(panel, P(self.sift[k]), 7, C_SIFT, -1, cv2.LINE_AA)
        cv2.circle(panel, q, int(8 * st.r), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, q, int(8 * st.r), (40, 40, 220), 3 * lw, cv2.LINE_AA)

        # mini-map: whole route, uniform scale
        mw_, mh_ = st.mini
        base, s, lo = self.mini_base(mw_, mh_)
        mini = base.copy()
        Mf = lambda p: tuple(np.round((np.asarray(p, float) - lo) * s).astype(int))
        cv2.polylines(mini, [np.array([Mf(p) for p in self.wps], np.int32)], False, C_GT, lw, cv2.LINE_AA)
        cv2.polylines(mini, [np.array([Mf(p) for p in self.traj[:k + 1]], np.int32)], False, C_TRAJ, 2 * lw, cv2.LINE_AA)
        cv2.rectangle(mini, Mf(origin), Mf(origin + [MAP_WIN_W, MAP_WIN_H]), (255, 255, 255), 1)
        cv2.circle(mini, Mf(est), 3 * lw, (40, 40, 220), -1)
        mx0, my0 = MAP_W - mw_ - 12, st.top_h // 2 + 12
        panel[my0:my0 + mh_, mx0:mx0 + mw_] = mini
        cv2.rectangle(panel, (mx0 - 1, my0 - 1), (mx0 + mw_, my0 + mh_), (200, 200, 200), lw)

        # scale bar (200 map px = 100 m nominal) + north arrow
        bar = int(round(200 * MAP_S))
        by = H - st.footer_h - (22 if not st.hero else 30)
        box_top = by - (32 if not st.hero else 58)
        blend_rect(panel, 10, box_top, 30 + bar + 10, by + 14, (0, 0, 0), 0.5)
        cv2.line(panel, (20, by), (20 + bar, by), (255, 255, 255), 3 * lw)
        for xx in (20, 20 + bar):
            cv2.line(panel, (xx, by - 6 * lw), (xx, by + 6 * lw), (255, 255, 255), 2 * lw)
        ay0 = st.top_h // 2 + 12
        cv2.arrowedLine(panel, (34, ay0 + 60 * lw), (34, ay0 + 8), (255, 255, 255), 3 * lw, cv2.LINE_AA, tipLength=0.35)
        return panel, (by, box_top, ay0)


def draw_track(cam, t, st, counts):
    s = CAM_W / 1920.0
    p = (int(t["x"] * s), int(t["y"] * s))
    role, R, lw = t["role"], st.r, st.lw
    counts[role] = counts.get(role, 0) + 1
    if role == "bound_inlier":
        cv2.circle(cam, p, int(11 * R), C_INLIER, 3 * lw, cv2.LINE_AA)
        cv2.circle(cam, p, int(4 * R), C_INLIER, -1, cv2.LINE_AA)
    elif role == "bound_outlier":
        cv2.circle(cam, p, int(11 * R), C_OUTLIER, 3 * lw, cv2.LINE_AA)
    elif role == "bound_unsolved":
        cv2.circle(cam, p, int(11 * R), C_UNSOLVED, 2 * lw, cv2.LINE_AA)
    elif role == "coasting" and t["map"] is not None:
        # bound but not detected this frame: its predicted position still votes in RANSAC
        counts["coasting"] -= 1
        counts["bound_coasting"] = counts.get("bound_coasting", 0) + 1
        cv2.circle(cam, p, int(11 * R), C_COAST, 2 * lw, cv2.LINE_AA)
    elif role == "coasting":
        cv2.circle(cam, p, int(4 * R), C_COAST, lw, cv2.LINE_AA)
    else:
        cv2.circle(cam, p, int(3 * R), C_UNBOUND, -1, cv2.LINE_AA)


def compose(k, frame, rows, prof, tl, mp, st, show_sift, playback):
    row, tr = rows[k], tl[k]
    canvas = np.zeros((H, W, 3), np.uint8)
    cam = cv2.resize(frame, (CAM_W, H), interpolation=cv2.INTER_AREA)
    if st.blur:
        cam = cv2.GaussianBlur(cam, (0, 0), st.blur)
    s = CAM_W / 1920.0
    if not st.hero:
        for cx, cy, bw, bh, _ in tr["dets"]:
            cv2.rectangle(cam, (int((cx - bw / 2) * s), int((cy - bh / 2) * s)),
                          (int((cx + bw / 2) * s), int((cy + bh / 2) * s)), (210, 210, 210), 1, cv2.LINE_AA)
    counts = {}
    order = {"unbound": 0, "coasting": 1, "bound_unsolved": 2, "bound_outlier": 3, "bound_inlier": 4}
    for t in sorted(tr["tracks"], key=lambda t: order[t["role"]]):
        draw_track(cam, t, st, counts)
    label, detail, lcol = loop_state(k, tl)
    commits = [t["f"] for t in tl[:k + 1] if t["commit"]]
    panel, (by, box_top, ay0) = mp.render(k, row, tr, st, show_sift, commits, label.startswith("SLOW"))
    canvas[:, :CAM_W] = cam
    canvas[:, CAM_W:] = panel

    TOP, FH = st.top_h, st.footer_h
    blend_rect(canvas, 0, 0, CAM_W, TOP, (0, 0, 0), 0.62)
    blend_rect(canvas, CAM_W, 0, W, TOP // 2, (0, 0, 0), 0.62)
    blend_rect(canvas, 0, H - FH, W, H, (0, 0, 0), 0.75)
    cv2.line(canvas, (CAM_W, 0), (CAM_W, H), (230, 230, 230), 2 * st.lw)

    # legend box: bottom-right of the camera panel (bottom-left holds the Google Earth watermark)
    legend = [("bound, RANSAC inlier", C_INLIER, "bound_inlier"), ("bound, RANSAC outlier", C_OUTLIER, "bound_outlier"),
              ("bound, no solve this frame", C_UNSOLVED, "bound_unsolved"), ("tracked, not bound", C_UNBOUND, "unbound"),
              ("bound, coasting (votes on prediction)", C_COAST, "bound_coasting"),
              ("unbound, coasting", C_COAST, "coasting")]
    if st.hero:
        legend = [("bound, RANSAC inlier", C_INLIER, ""), ("bound, RANSAC outlier", C_OUTLIER, ""),
                  ("bound, coasting (votes)", C_COAST, ""), ("tracked, not bound", C_UNBOUND, "")]
    row_h = int(22 * (1.9 if st.hero else 1.0))
    lg_w = 470 if st.hero else 310
    lg_h = int(34 * (1.9 if st.hero else 1.0)) + row_h * len(legend)
    lg_x0, lg_y1 = CAM_W - lg_w - 10, H - FH - 10
    blend_rect(canvas, lg_x0, lg_y1 - lg_h, CAM_W - 10, lg_y1, (0, 0, 0), 0.58)
    # loop badge
    badge_h = 64 if st.hero else 32
    bx1, by0 = CAM_W - 14, 12
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    badge_font = st.f_lgb if st.hero else st.f_mdb
    bw = int(tmp.textlength(label, font=badge_font)) + 24
    bx0 = bx1 - bw
    cv2.rectangle(canvas, (bx0, by0), (bx1, by0 + badge_h), lcol, -1, cv2.LINE_AA)

    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    f = int(row["frame_id"])
    ni, ir = fnum(row["n_inliers"]), fnum(row["inlier_ratio"])
    pairs = int(round(ni / ir)) if ir > 0 else 0
    xt = math.hypot(fnum(row["est_x"]) - fnum(row["gt_x"]), fnum(row["est_y"]) - fnum(row["gt_y"]))
    fps = 1000.0 / max(1e-6, float(np.mean(prof[max(0, k - 14):k + 1])))
    white, soft = (255, 255, 255), (232, 232, 232)
    if st.hero:
        d.text((16, 8), "UAV-VisLoc", font=st.f_lgb, fill=white)
        d.text((16, 70), f"frame {f}/{len(rows)}  ·  inliers {int(ni)}/{pairs}  ·  cross-track {xt * 0.5:4.1f} m",
               font=st.f_md, fill=soft)
        d.text((bx0 + 12, by0 + 8), label, font=badge_font, fill=white)
    else:
        d.text((16, 10), "UAV-VisLoc", font=st.f_lgb, fill=white)
        d.text((150, 14), f"frame {f}/{len(rows)}   ·   t = {f / SRC_FPS:5.2f} s   ·   playback {playback}   ·   "
                          f"pipeline {fps:4.1f} FPS (RTX 5060)", font=st.f_md, fill=soft)
        d.text((16, 46), f"inliers {int(ni)}/{pairs}   ·   locked scale {fnum(row['scale']):.4f}   ·   heading prior "
                         f"{fnum(row['heading_used']):+.1f}°   ·   cross-track {xt:5.1f} px ({xt * 0.5:4.1f} m nominal)",
               font=st.f_md, fill=soft)
        d.text((bx0 + 12, by0 + 4), label, font=badge_font, fill=white)
        d.text((bx1, by0 + 40), detail, font=st.f_sm, fill=(240, 240, 240), anchor="ra")
    lx, ly = lg_x0 + 12, lg_y1 - lg_h + 6
    d.text((lx, ly), "Landmark roles", font=st.f_mdb, fill=white)
    dot = int(12 * (1.9 if st.hero else 1.0))
    for i, (txt, col, key) in enumerate(legend):
        yy = ly + int(28 * (1.9 if st.hero else 1.0)) + i * row_h
        d.ellipse((lx, yy + 4, lx + dot, yy + 4 + dot), outline=col[::-1], width=3 * st.lw)
        d.text((lx + dot + 10, yy), txt if st.hero else f"{txt}  ({counts.get(key, 0)})", font=st.f_sm, fill=soft)
    # map-panel text
    title = "Satellite map · uniform scale" if st.hero else \
        "Map: satellite, north-up, uniform scale" + ("  ·  orange = SIFT reference" if show_sift else "")
    d.text((CAM_W + 12, 6 if st.hero else 7), title, font=st.f_sm, fill=soft)
    d.text((CAM_W + 34 + 14 * st.lw, ay0 + 4), "N", font=st.f_mdb, fill=white)
    d.text((CAM_W + 20, box_top + 2), "100 m (nominal)", font=st.f_sm, fill=white)
    if not st.hero:
        d.text((CAM_W + 170, H - FH - 50), "— trajectory   - - GT polyline   ○ commit   white dashes: footprint",
               font=st.f_sm, fill=soft)
    if st.hero:
        d.text((W // 2, H - FH + 8), CAPTION, font=st.f_md, fill=white, anchor="ma")
    else:
        d.text((W // 2, H - FH + 6), CAPTION + "   ·   " + CAPTION_NOTES, font=st.f_sm, fill=white, anchor="ma")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


OVERLAY_RGB = [c[::-1] for c in (C_TRAJ, C_GT, C_INLIER, C_OUTLIER, C_UNSOLVED, C_UNBOUND, C_COAST, C_FAST, C_SLOW,
                                  C_COMMIT, (40, 40, 220), (255, 255, 255), (0, 0, 0))]


def quantize_reserved(fr, colors):
    """Adaptive per-frame palette for the imagery plus exact reserved overlay colours, so role colours survive
    small palettes (a plain median cut maps inlier green to teal and outlier red to orange at 32 colours)."""
    im = Image.fromarray(fr)
    n_adapt = colors - len(OVERLAY_RGB)
    adapt = im.quantize(colors=n_adapt, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = adapt.getpalette()[:3 * n_adapt] + [v for c in OVERLAY_RGB for v in c]
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(pal + [0] * (768 - len(pal)))
    return im.quantize(palette=pal_img, dither=Image.Dither.NONE)


def encode_gif(frames_rgb, path, colors, frame_ms):
    q = [quantize_reserved(fr, colors) for fr in frames_rgb]
    q[0].save(path, save_all=True, append_images=q[1:], duration=frame_ms, loop=0, optimize=True, disposal=1)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frame-log", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--track-log", required=True)
    ap.add_argument("--sift", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif-max-mb", type=float, default=10.0)
    ap.add_argument("--gif-seconds", type=float, default=20.0)
    ap.add_argument("--gif-width", type=int, default=800)
    ap.add_argument("--stills", default="1,98,159,255,352,401")
    ap.add_argument("--skip-mp4", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    rows, prof, tl, sift = load_inputs(args)
    wps, _, _ = waypoints_px(cfg)
    traj = np.array([[fnum(r["est_x"]), fnum(r["est_y"])] for r in rows])
    mp = MapPanel(cv2.imread(cfg["data"]["map_image"]), wps, traj, sift)
    stills = {int(x) for x in args.stills.split(",")}
    diag_st, hero_st = Style(False), Style(True)
    n = len(rows)
    playback = n / SRC_FPS / args.gif_seconds          # < 1 means slower than real time
    gif_h = int(round(args.gif_width * H / W))

    mp4_path = os.path.join(args.out, "geotest1_diagnostic.mp4")
    writer = None if args.skip_mp4 else cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), SRC_FPS, (W, H))
    hero_frames = []                                  # every source frame, downscaled RGB; subsampled at encode time
    cap = cv2.VideoCapture(cfg["data"]["video"])
    k = 0
    while k < n:
        ok, frame = cap.read()
        if not ok:
            break
        f = k + 1
        if writer is not None or f in stills:
            diag = compose(k, frame, rows, prof, tl, mp, diag_st, show_sift=True, playback="1.00× (real time)")
            if writer is not None:
                writer.write(diag)
            if f in stills:
                cv2.imwrite(os.path.join(args.out, f"still_f{f}.jpg"), diag, [cv2.IMWRITE_JPEG_QUALITY, 90])
        hero = compose(k, frame, rows, prof, tl, mp, hero_st, show_sift=False, playback="")
        hero_frames.append(cv2.cvtColor(cv2.resize(hero, (args.gif_width, gif_h), interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB))
        if f % 50 == 0:
            print(f"[DEMO] frame {f}/{n}")
        k += 1
    cap.release()
    if writer is not None:
        writer.release()
    assert k == n, f"video ended at {k} frames, logs have {n}"

    # GIF budget search: highest frame rate first, then fewer colours; duration held at --gif-seconds
    gif_path = os.path.join(args.out, "geotest1_hero.gif")
    tried, chosen = [], None
    for step, colors in ((2, 64), (3, 64), (3, 48), (4, 48), (4, 40), (4, 32), (5, 32), (6, 32)):
        sub = hero_frames[::step]
        frame_ms = int(round(args.gif_seconds * 1000.0 / len(sub) / 10.0)) * 10   # GIF delays are centiseconds
        size = encode_gif(sub, gif_path, colors, frame_ms)
        rec = {"source_step": step, "colors": colors, "frames": len(sub), "frame_ms": frame_ms,
               "gif_fps": round(1000.0 / frame_ms, 2), "bytes": size}
        tried.append(rec)
        print(f"[DEMO] GIF step={step} colors={colors} frames={len(sub)} {frame_ms} ms: {size / 1e6:.2f} MB")
        if size <= args.gif_max_mb * 1e6:
            chosen = rec
            break
    manifest = {
        "gif": {"path": gif_path, "dims": [args.gif_width, gif_h], "chosen": chosen, "tried": tried,
                "duration_s": round(chosen["frames"] * chosen["frame_ms"] / 1000.0, 2) if chosen else None,
                "playback_speed_vs_real_time": round(playback, 3), "background_blur_sigma_panel_px": hero_st.blur,
                "within_limit": chosen is not None, "max_mb": args.gif_max_mb},
        "diagnostic_mp4": None if args.skip_mp4 else
        {"path": mp4_path, "bytes": os.path.getsize(mp4_path), "dims": [W, H], "frames": k, "fps": SRC_FPS, "codec": "mp4v"},
        "stills": sorted(stills),
        "map_panel": {"window_map_px": [MAP_WIN_W, MAP_WIN_H], "panel_px": [MAP_W, H],
                      "scale_x": MAP_W / MAP_WIN_W, "scale_y": H / MAP_WIN_H, "uniform": MAP_W / MAP_WIN_W == H / MAP_WIN_H},
        "caption": CAPTION, "caption_notes": CAPTION_NOTES,
    }
    with open(os.path.join(args.out, "demo_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if chosen is None:
        sys.exit("GIF exceeds budget at every tried setting")


if __name__ == "__main__":
    main()
