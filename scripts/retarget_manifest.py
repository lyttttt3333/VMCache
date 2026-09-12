#!/usr/bin/env python3
"""Rewrite the absolute data paths in manifests so they point at this cluster.

The manifests carry absolute paths because the rollout jobs that wrote them did.
Moving the corpus to another cluster is therefore a rewrite, not just a copy.

The default mapping drops the `artifacts/` level: on cw the data sits at
`<repo>/artifacts/X`, and `sync_heavy_condition_cache_to_hsg.sbatch` lands it
flat at `<DATA_ROOT>/X`, which is what the hsg launcher reads.

Dry run by default. `--apply` writes; `--verify` additionally requires that every
rewritten path exists, which is the point of running this before spending a GPU
allocation.

    python3 scripts/retarget_manifest.py \
        --to-root /lustre/fsw/portfolios/nvr/users/yitongl/ref2va_data \
        --apply --verify \
        ref2va_decoupled_distill_train_manifest.json \
        covebench_teacher_rollout_index.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FROM_ROOT = REPO / "artifacts"

# Keys whose values are filesystem paths. Rewriting by key rather than by
# "any string that looks like a path" keeps prompts and ids untouched.
PATH_KEYS = frozenset(
    {
        "path",
        "source_path",
        "teacher_latent_path",
        "teacher_video_path",
        "rollout_root",
        "teacher_latent_root",
        "teacher_video_root",
        "cases_dir",
    }
)
PATH_LIST_KEYS = frozenset({"teacher_latent_roots"})


def _swap(value: str, old: str, new: str, hits: list[str], misses: list[str]) -> str:
    if value.startswith(old):
        swapped = new + value[len(old):]
        hits.append(swapped)
        return swapped
    misses.append(value)
    return value


def rewrite(node: Any, old: str, new: str, hits: list[str], misses: list[str]) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in PATH_KEYS and isinstance(value, str):
                out[key] = _swap(value, old, new, hits, misses)
            elif key in PATH_LIST_KEYS and isinstance(value, list):
                out[key] = [
                    _swap(item, old, new, hits, misses) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                out[key] = rewrite(value, old, new, hits, misses)
        return out
    if isinstance(node, list):
        return [rewrite(item, old, new, hits, misses) for item in node]
    return node


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument(
        "--from-root",
        default=str(DEFAULT_FROM_ROOT),
        help=f"path prefix to replace (default: {DEFAULT_FROM_ROOT})",
    )
    parser.add_argument("--to-root", required=True, help="prefix to replace it with")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    old = args.from_root.rstrip("/")
    new = args.to_root.rstrip("/")
    print(f"{old}\n  -> {new}\n")

    failed = False
    for name in args.manifests:
        manifest = Path(name)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        hits: list[str] = []
        misses: list[str] = []
        rewritten = rewrite(payload, old, new, hits, misses)

        print(f"{manifest}: {len(hits)} rewritten, {len(misses)} untouched")
        if hits:
            print(f"  e.g. {hits[0]}")
        if misses:
            # Not necessarily wrong -- a manifest can legitimately reference
            # paths outside the data root -- but worth seeing.
            print(f"  untouched e.g. {misses[0]}")

        if args.verify:
            missing = [path for path in hits if not os.path.exists(path)]
            print(f"  existing: {len(hits) - len(missing)}/{len(hits)}")
            if missing:
                failed = True
                print(f"  FATAL: {len(missing)} rewritten paths do not exist", file=sys.stderr)
                for path in missing[:5]:
                    print(f"    {path}", file=sys.stderr)

        if args.apply and not failed:
            manifest.write_text(
                json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  written: {manifest}")
        elif not args.apply:
            print("  dry run; pass --apply to write")
        print()

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
