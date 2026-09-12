#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF cache repair: rebuild broken snapshot files from blobs
=========================================================
Symptom this fixes
------------------
On some Windows setups, `huggingface_hub` fails to materialise the
symlinks under
    ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<rev>/<file>
and leaves **0-byte regular files** there instead.  The real payload is
still intact under `blobs/<sha256>`, but `transformers` then dies with
    OSError: ... '<file>' is not a valid JSON file
or
    SafetensorError / empty weights.

The script re-materialises every snapshot entry by copying the matching
blob (matched on the repo's reported blob id), so **no re-download** is
needed.  Idempotent: already-correct files are skipped.

Usage
-----
    python scripts/tools/hf_repair_cache.py google/siglip-base-patch16-224
    python scripts/tools/hf_repair_cache.py Qwen/Qwen3-Embedding-0.6B
    python scripts/tools/hf_repair_cache.py --all     # every model in the cache
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def hub_dir() -> Path:
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env) / "hub"
    hf_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache)
    return Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"


def repo_folder(repo_id: str) -> str:
    prefix = "datasets--" if repo_id.startswith("datasets/") else "models--"
    return prefix + repo_id.replace("/", "--")


def repair(repo_id: str, api, revision: str = "main", dry_run: bool = False) -> int:
    base = hub_dir() / repo_folder(repo_id)
    if not base.exists():
        print(f"[skip] {repo_id}: no cache entry at {base}")
        return 0

    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    sha = info.sha
    snap = base / "snapshots" / sha
    blobs = base / "blobs"
    if not snap.exists():
        print(f"[skip] {repo_id}: snapshot {sha} not materialised")
        return 0

    fixed = ok = missing = 0
    for sib in info.siblings or []:
        name = sib.rfilename
        want = sib.size
        blob_id = getattr(sib, "blob_id", None)
        if not blob_id and getattr(sib, "lfs", None):
            blob_id = sib.lfs.sha256
        dst = snap / name

        # already fine?
        if dst.exists():
            got = dst.stat().st_size
            if want is None or got == want:
                ok += 1
                continue

        # locate blob
        src = None
        if blob_id:
            cand = blobs / blob_id
            if cand.exists():
                src = cand
        if src is None:                       # fallback: match by exact size
            if want is not None:
                for b in blobs.iterdir():
                    if b.is_file() and b.stat().st_size == want:
                        src = b
                        break
        if src is None:
            print(f"  [MISS] {name}: no blob found (want {want} bytes)")
            missing += 1
            continue

        action = "would fix" if dry_run else "fixed"
        print(f"  [{action}] {name}: {src.stat().st_size:,} bytes")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            shutil.copyfile(src, dst)
        fixed += 1

    print(f"[done] {repo_id}@{sha[:8]}  repaired={fixed} already_ok={ok} missing={missing}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", nargs="?", help="e.g. google/siglip-base-patch16-224")
    ap.add_argument("--all", action="store_true", help="repair every models-- entry in the cache")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    if args.all:
        root = hub_dir()
        repos = []
        for d in sorted(root.glob("models--*")):
            repos.append(d.name.replace("models--", "").replace("--", "/"))
        if not repos:
            print(f"no models found under {root}")
            return 0
    elif args.repo_id:
        repos = [args.repo_id]
    else:
        ap.error("pass a repo_id or --all")

    bad = 0
    for r in repos:
        try:
            bad += repair(r, api, args.revision, args.dry_run)
        except Exception as e:                                  # noqa: BLE001
            print(f"[error] {r}: {type(e).__name__}: {e}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
