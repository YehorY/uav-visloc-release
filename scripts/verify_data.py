"""
scripts/verify_data.py — LEVEL 0 frozen-test-sequence fetch/mount mechanism.

The benchmark inputs are a FROZEN triplet (video / weights / map). This script:
  1. verifies presence + SHA256 of each file against the manifest below;
  2. if a file is missing and DATA_SOURCE_DIR is set, copies ("mounts") it from there;
  3. exits non-zero on any mismatch — CI-gateable.

Usage:
    python scripts/verify_data.py                 # verify only
    DATA_SOURCE_DIR=/mnt/share python scripts/verify_data.py   # fetch missing, then verify
    python scripts/verify_data.py --write-manifest            # (maintainers) refresh hashes
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

DATA_FILES = ["data/GeoTest1.mp4", "data/best.pt", "data/datamap_4k.jpeg"]
MANIFEST = os.path.join(os.path.dirname(__file__), "data_manifest.json")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true",
                    help="Record current file hashes as the frozen reference.")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    if args.write_manifest:
        manifest = {p: sha256(p) for p in DATA_FILES}
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[DATA] Manifest written: {MANIFEST}")
        return 0

    if not os.path.exists(MANIFEST):
        print(f"[DATA] ERROR: manifest missing ({MANIFEST}). Run with --write-manifest first.")
        return 2
    with open(MANIFEST) as f:
        manifest = json.load(f)

    src_dir = os.environ.get("DATA_SOURCE_DIR")
    ok = True
    for p in DATA_FILES:
        if not os.path.exists(p):
            if src_dir and os.path.exists(os.path.join(src_dir, os.path.basename(p))):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                shutil.copy2(os.path.join(src_dir, os.path.basename(p)), p)
                print(f"[DATA] fetched {p} from {src_dir}")
            else:
                print(f"[DATA] MISSING: {p} (set DATA_SOURCE_DIR to fetch)")
                ok = False
                continue
        h = sha256(p)
        if h != manifest.get(p):
            print(f"[DATA] HASH MISMATCH: {p}\n       expected {manifest.get(p)}\n       actual   {h}")
            ok = False
        else:
            print(f"[DATA] OK: {p}")
    print("[DATA] Frozen sequence verified." if ok else "[DATA] VERIFICATION FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
