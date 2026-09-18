"""
scripts/run_ablations.py — LEVEL 2.3 ablation runner.

    python scripts/run_ablations.py --matrix configs/ablations.yaml [--only A1_no_ransac,A_coast] [--resume]

For each run: copy base_config, apply common_overrides and the run's overrides (dotted keys), write
out/ablations/<run>/config.yaml, and execute run.py in a FRESH SUBPROCESS so no state carries over.
The control runs first and must hash to expect_control_sha256, otherwise the suite aborts. Then
every frame log is scored with core/metrics.py and out/ablations/summary.{md,json} is written.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from core import metrics as M              # noqa: E402
from evaluate import waypoints_px          # noqa: E402

LABEL = "ATE vs nominal GT polyline (uncorrected image frame)"


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def expand(matrix):
    """Yield (run_id, group_id, thesis, overrides) for every concrete run, control first."""
    for spec in matrix["runs"]:
        ov = dict(spec.get("overrides") or {})
        if spec.get("seeds"):
            for s in spec["seeds"]:
                o = dict(ov)
                o["experiments.detection_dropout.seed"] = int(s)
                yield f"{spec['id']}_s{s}", spec["id"], spec["thesis"], o
        else:
            for r in range(int(spec.get("repeats", 1))):
                rid = spec["id"] if int(spec.get("repeats", 1)) == 1 else f"{spec['id']}_r{r + 1}"
                yield rid, spec["id"], spec["thesis"], ov


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parse_log(path):
    raw = open(path, "rb").read().replace(b"\x00", b"").decode("utf-8", errors="ignore")
    commits, frame = [], None
    for line in raw.splitlines():
        m = re.search(r"\[Frame (\d+)\]", line)
        if m:
            frame = int(m.group(1))
        if "[SLAM] Match Found!" in line and frame is not None:
            commits.append(frame)
    scale = re.search(r"Strict Scale locked at:\s*([0-9.]+)", raw)
    pair = re.search(r"\[SCALE-TELEM\] pair_ids=(\[[^\]]*\])", raw)
    ate = re.search(r"Absolute Trajectory Error \(RMSE\):\s*([0-9.]+)", raw)
    return {"commit_frames": commits,
            "scale": float(scale.group(1)) if scale else None,
            "pair": pair.group(1) if pair else None,
            "printed_ate_m": float(ate.group(1)) if ate else None,
            "traceback": "Traceback" in raw}


def run_one(run_id, overrides, matrix, base, args):
    d = os.path.join(ROOT, matrix["out_dir"], run_id)
    os.makedirs(d, exist_ok=True)
    flog, prof, cfg_path, log_path = (os.path.join(d, n) for n in ("frame_log.csv", "profile.csv", "config.yaml", "run.log"))
    if args.resume and os.path.exists(flog) and os.path.exists(log_path):
        print(f"[ABLATIONS] {run_id}: resume, reusing existing output")
        return {"run_id": run_id, "frame_log": flog, "log": log_path, "wall_s": None, "returncode": 0}
    cfg = copy.deepcopy(base)
    for k, v in {**(matrix.get("common_overrides") or {}), **overrides}.items():
        set_dotted(cfg, k, v)
    set_dotted(cfg, "output.frame_log", flog)
    set_dotted(cfg, "output.profile_log", prof)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    with open(log_path, "wb") as fh:
        rc = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), "--config", cfg_path],
                            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    wall = time.perf_counter() - t0
    print(f"[ABLATIONS] {run_id}: exit {rc} in {wall:.0f}s")
    return {"run_id": run_id, "frame_log": flog, "log": log_path, "wall_s": wall, "returncode": rc}


def fmt(x, nd=1):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--only", default="", help="comma-separated group ids to run (control always runs)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    matrix = yaml.safe_load(open(os.path.join(ROOT, args.matrix), encoding="utf-8"))
    base = yaml.safe_load(open(os.path.join(ROOT, matrix["base_config"]), encoding="utf-8"))
    only = {s for s in args.only.split(",") if s}

    runs = list(expand(matrix))
    records = []
    for rid, gid, thesis, ov in runs:
        if gid != "control" and only and gid not in only:
            continue
        rec = run_one(rid, ov, matrix, base, args)
        rec.update({"group": gid, "thesis": thesis, "overrides": ov})
        if rec["returncode"] != 0 or not os.path.exists(rec["frame_log"]):
            sys.exit(f"[ABLATIONS] ABORT: {rid} failed (exit {rec['returncode']}); see {rec['log']}")
        rec["sha256"] = sha256(rec["frame_log"])
        if gid == "control" and matrix.get("expect_control_sha256") and rec["sha256"] != matrix["expect_control_sha256"]:
            sys.exit(f"[ABLATIONS] ABORT: control {rid} frame log sha256 {rec['sha256'][:12]} != expected "
                     f"{matrix['expect_control_sha256'][:12]}: experiment hooks are NOT inert.")
        records.append(rec)

    # ---------------- scoring ----------------
    wps, map_w, map_h = waypoints_px(base)
    passages = base["sequence"]["waypoint_passage_frames"]
    control = next(r for r in records if r["group"] == "control")
    ctrl_rows = M.load_frame_log(control["frame_log"])
    ctrl_log = parse_log(control["log"])
    ctrl_rmse = M.ate_stats(ctrl_rows)["tracked"]["rmse"]
    for r in records:
        rows = M.load_frame_log(r["frame_log"])
        lg = parse_log(r["log"])
        ate = M.ate_stats(rows)
        a, b = set(lg["commit_frames"]), set(ctrl_log["commit_frames"])
        r.update({"coverage": M.coverage(rows), "ate_px": ate, "log_info": lg,
                  "along_track": M.along_track_at_passages(rows, wps, passages),
                  "commits": len(lg["commit_frames"]), "commit_symdiff": len(a ^ b),
                  "delta_rmse_pct": 100.0 * (ate["tracked"]["rmse"] / ctrl_rmse - 1.0)})

    out_dir = os.path.join(ROOT, matrix["out_dir"])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, default=str)

    L = [f"# Level 2.3 ablation summary\n", f"**All ATE: {LABEL}.** Map px primary; metres = px x nominal 0.5 m/px "
         "(unverified georeference). 'Tracked' = frames the run localized; 'all' = every frame. Commits are parsed "
         "from each run log; Δ commits = symmetric difference of commit frames vs control.\n"]
    ctrl_runs = [r for r in records if r["group"] == "control"]
    shas = {r["sha256"] for r in ctrl_runs}
    L.append(f"**Inertness and determinism:** {len(ctrl_runs)} control runs, frame-log sha256 "
             f"{'identical' if len(shas) == 1 else 'DIFFERENT'} "
             f"(`{ctrl_runs[0]['sha256'][:12]}…`, expected `{str(matrix.get('expect_control_sha256'))[:12]}…`).\n")
    L.append("## Per run\n")
    L.append("| run | LSR | coverage | mean px | median px | p95 px | RMSE px | RMSE m | ΔRMSE vs control | all-frames RMSE px | commits | Δ commits | scale / pair |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        t, al, c, lg = r["ate_px"]["tracked"], r["ate_px"]["all"], r["coverage"], r["log_info"]
        L.append(f"| {r['run_id']} | {100*c['lsr']:.1f}% | {100*c['coverage']:.1f}% | {fmt(t['mean'])} | {fmt(t['median'])} | "
                 f"{fmt(t['p95'])} | {fmt(t['rmse'])} | {fmt(t['rmse']*0.5, 2)} | {r['delta_rmse_pct']:+.0f}% | "
                 f"{fmt(al['rmse'])} | {r['commits']} | {r['commit_symdiff']} | {fmt(lg['scale'], 4)} {lg['pair'] or ''} |")

    groups = [g for g in dict.fromkeys(r["group"] for r in records) if g.startswith("A4_")]
    if groups:
        L.append("\n## A4 dropout — distribution over seeds (median [min – max])\n")
        L.append("| group | seeds | RMSE px | ΔRMSE vs control | LSR | coverage | commits | Δ commits |")
        L.append("|---|---|---|---|---|---|---|---|")
        for g in groups:
            rs = [r for r in records if r["group"] == g]

            def rng(vals, f=lambda v: f"{v:.1f}"):
                return f"{f(statistics.median(vals))} [{f(min(vals))} – {f(max(vals))}]"
            L.append(f"| {g} | {len(rs)} | {rng([r['ate_px']['tracked']['rmse'] for r in rs])} | "
                     f"{rng([r['delta_rmse_pct'] for r in rs], lambda v: f'{v:+.0f}%')} | "
                     f"{rng([100*r['coverage']['lsr'] for r in rs], lambda v: f'{v:.1f}%')} | "
                     f"{rng([100*r['coverage']['coverage'] for r in rs], lambda v: f'{v:.1f}%')} | "
                     f"{rng([r['commits'] for r in rs], lambda v: f'{v:.0f}')} | "
                     f"{rng([r['commit_symdiff'] for r in rs], lambda v: f'{v:.0f}')} |")

    L.append("\n## Theses tested\n")
    for g in dict.fromkeys(r["group"] for r in records):
        L.append(f"- **{g}** — {next(r['thesis'] for r in records if r['group'] == g)}")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
