#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("H3_CONDITION_CACHE_ROOT", Path(__file__).resolve().parent))
SRC = ROOT / "artifacts/heavy_condition_asset_dataset_v6_8_organic_pexels_hotmatch_32aligned/dataset_manifest.json"
OUT = ROOT / "heavy_ref2va_v6_8_organic_pexels_hotmatch_3video_cases.json"


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    cases = []
    for case in src["cases"]:
        conditions = []
        for ref in case["references"]:
            width = int(ref["processed_width"])
            height = int(ref["processed_height"])
            conditions.append(
                {
                    "type": "video",
                    "role": ref["role"],
                    "source_id": ref["id"],
                    "path": ref["clip_path"],
                    "start_time_seconds": 0.0,
                    "video_probe": {
                        "width": width,
                        "height": height,
                        "has_audio": False,
                        "audio_sample_rate": None,
                        "duration_seconds": 5.0,
                    },
                    "resolved_shape": {
                        "resolved_width": width,
                        "resolved_height": height,
                        "frame_count": 107,
                    },
                }
            )
        cases.append(
            {
                "id": case["id"],
                "seed": 2026083100 + int(case["case_index"]),
                "prompt": case["prompt"],
                "theme": case["theme"],
                "semantic_merge": case["semantic_merge"],
                "conditions": conditions,
            }
        )

    manifest = {
        "name": "heavy_ref2va_v6_8_organic_pexels_hotmatch_3video_cases",
        "target": {
            "width": 1376,
            "height": 768,
            "fps": 24,
            "num_frames": 107,
            "duration_seconds": 107 / 24,
        },
        "source_dataset": str(SRC),
        "note": "Generated from v6 8-case organic Pexels dataset for hot-speed-matched TC+CC versus TC-only comparison.",
        "cases": cases,
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(OUT)
    print(f"cases={len(cases)} refs={sum(len(c['conditions']) for c in cases)}")


if __name__ == "__main__":
    main()
