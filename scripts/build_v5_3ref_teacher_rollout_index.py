#!/usr/bin/env python3
"""Build a strict teacher-rollout index from the curated V5 3-reference set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("heavy_ref2va_v5_32_organic_pexels_3video_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v5_3ref_teacher_rollout_index.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/v5_3ref_teacher_rollout_teacache_2x"),
    )
    return parser.parse_args()


def normalized_condition(condition: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(condition["path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "type": condition["type"],
        "role": condition.get("role", "reference video"),
        "source_id": condition.get("source_id"),
        "path": str(path),
        "source_path": str(path),
        "start_time_seconds": float(condition.get("start_time_seconds", 0.0)),
    }


def main() -> None:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))

    cases = []
    seen_paths: set[str] = set()
    for index, raw_case in enumerate(source["cases"]):
        raw_conditions = list(raw_case["conditions"])
        if len(raw_conditions) != 3:
            raise ValueError(
                f"case {raw_case['id']} has {len(raw_conditions)} conditions; expected 3"
            )
        conditions = [normalized_condition(item) for item in raw_conditions]
        for condition in conditions:
            path = condition["path"]
            if path in seen_paths:
                raise ValueError(f"reference media is reused across cases: {path}")
            seen_paths.add(path)

        media_probes = [dict(item["video_probe"]) for item in raw_conditions]
        resolved_shapes = [dict(item["resolved_shape"]) for item in raw_conditions]
        if len(media_probes) != 3 or len(resolved_shapes) != 3:
            raise ValueError(f"case {raw_case['id']} has incomplete media metadata")

        case_id = str(raw_case["id"])
        cases.append(
            {
                "id": case_id,
                "index": index,
                "seed": int(raw_case["seed"]),
                "ref_noise_seed": int(raw_case.get("ref_noise_seed", raw_case["seed"])),
                "prompt": str(raw_case["prompt"]),
                "theme": raw_case.get("theme"),
                "semantic_merge": raw_case.get("semantic_merge"),
                "conditions": conditions,
                "media_probes": media_probes,
                "resolved_shapes": resolved_shapes,
                "teacher_latent_path": str(
                    output_root
                    / "latents"
                    / case_id
                    / "validation"
                    / "backbone"
                    / "prompt0000.pt"
                ),
                "teacher_video_path": str(
                    output_root
                    / "videos"
                    / case_id
                    / "validation"
                    / "backbone"
                    / "prompt0000.mp4"
                ),
            }
        )

    if len(cases) != 32 or len(seen_paths) != 96:
        raise ValueError(
            f"expected 32 cases and 96 unique references, got {len(cases)} and "
            f"{len(seen_paths)}"
        )

    payload = {
        "name": "v5_3ref_teacher_rollout_teacache_2x",
        "purpose": "coupled MiniMax-H3 teacher targets for multi-reference distillation",
        "source_manifest": str(source_path),
        "case_count": len(cases),
        "unique_reference_video_count": len(seen_paths),
        "target": dict(source["target"]),
        "teacher_sampler": {
            "num_denoising_steps": 50,
            "teacache_compute_steps": [*range(0, 48, 2), 49],
        },
        "teacher_latent_root": str(output_root / "latents"),
        "teacher_video_root": str(output_root / "videos"),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output_path}: {len(cases)} cases, "
        f"{len(seen_paths)} unique reference videos"
    )


if __name__ == "__main__":
    main()
