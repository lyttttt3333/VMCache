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
    parser.add_argument("--rollout-index", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--target-frames", type=int, default=124)
    parser.add_argument(
        "--shard-index", type=int, default=int(os.environ.get("SLURM_PROCID", "0"))
    )
    parser.add_argument(
        "--num-shards", type=int, default=int(os.environ.get("SLURM_NTASKS", "1"))
    )
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "RAVEN"))
    os.environ["MINIMAX_H3_REF_DECODE_CACHE"] = str(Path(args.cache_dir).resolve())

    from projects.minimax_h3.modeling.ref2va_reference import (  # noqa: PLC0415
        _decode_video,
        parse_ref_block_plan,
    )

    final_cases = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rollout_index = json.loads(Path(args.rollout_index).read_text(encoding="utf-8"))
    rollout_by_id = {
        str(case["id"]): case
        for case in rollout_index.get("cases", [])
        if isinstance(case, dict) and "id" in case
    }
    reference_root = Path(args.reference_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    decoded = 0
    for item_index, item in enumerate(final_cases):
        if item_index % args.num_shards != args.shard_index:
            continue
        case_id = str(item["id"])
        source = rollout_by_id[case_id]
        conditions = [
            {
                **dict(source["conditions"][0]),
                "path": str(reference_root / case_id / "reference_video.mp4"),
                "source_path": str(reference_root / case_id / "reference_video.mp4"),
            }
        ]
        specs = parse_ref_block_plan(
            conditions,
            source["media_probes"],
            resolved_shapes=source["resolved_shapes"],
            target_frame_count=args.target_frames,
        )
        for spec in specs:
            if spec.kind in ("video", "video_audio"):
                _decode_video(spec)
                decoded += 1
                print(f"decoded {case_id}: {spec.path}", flush=True)
    print(
        f"done decoded={decoded} shard={args.shard_index}/{args.num_shards} "
        f"cache_dir={cache_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
