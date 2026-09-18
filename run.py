"""
run.py — LEVEL 0 canonical entry point.

    python run.py --config configs/demo.yaml

Loads the YAML config, freezes global determinism BEFORE any pipeline import executes
stochastic work, then hands the config to the pipeline.
"""

import argparse

import yaml

from core.diagnostics import set_global_determinism


def main():
    ap = argparse.ArgumentParser(description="UAV-VisLoc pipeline runner (Level 0)")
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/demo.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Determinism FIRST — before the pipeline module touches any RNG.
    set_global_determinism(int(cfg.get("seed", 42)))

    import run_video_test as pipeline
    pipeline.run_video_test(config=cfg)


if __name__ == "__main__":
    main()
