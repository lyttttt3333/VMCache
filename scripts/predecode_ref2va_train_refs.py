#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=int(
            os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("SLURM_PROCID", "0"))
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(
            os.environ.get("SLURM_ARRAY_TASK_COUNT", os.environ.get("SLURM_NTASKS", "1"))
        ),
    )
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "RAVEN"))
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINIMAX_H3_REF_DECODE_CACHE"] = str(cache_dir)

    from projects.minimax_h3.modeling.ref2va_reference import (  # noqa: PLC0415
        _decode_video,
        parse_ref_block_plan,
    )

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cases = manifest["cases"]
    decoded = 0
    for case_index, case in enumerate(cases):
        if case_index % args.num_shards != args.shard_index:
            continue
        target = dict(case.get("target", manifest["target"]))
        specs = parse_ref_block_plan(
            case["conditions"],
            case["media_probes"],
            resolved_shapes=case["resolved_shapes"],
            target_frame_count=int(target["num_frames"]),
        )
        for spec in specs:
            if spec.kind not in ("video", "video_audio"):
                continue
            _decode_video(spec)
            decoded += 1
            print(f"decoded {case['id']}: {spec.path}", flush=True)
    print(
        f"done decoded={decoded} shard={args.shard_index}/{args.num_shards} "
        f"cache_dir={cache_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
