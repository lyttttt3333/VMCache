#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "heavy_ref2va_v6_8_organic_pexels_hotmatch_3video_cases.json"
OUT = ROOT / "refcacheblend_poc_manifests"
CASE_INDEX = 0


def write_manifest(name: str, base: dict, case: dict, conditions: list[dict]) -> Path:
    out = {
        "name": f"refcacheblend_poc_{name}",
        "target": base["target"],
        "source_dataset": base.get("source_dataset"),
        "note": (
            "RefCacheBlend POC manifest. ABC is the full three-reference request; "
            "A/B/C are single-asset cache-source requests."
        ),
        "cases": [
            {
                "id": f"{case['id']}_{name}",
                "seed": case["seed"],
                "prompt": case["prompt"],
                "theme": case.get("theme", ""),
                "semantic_merge": case.get("semantic_merge", ""),
                "conditions": conditions,
            }
        ],
    }
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(SOURCE.read_text(encoding="utf-8"))
    case = base["cases"][CASE_INDEX]
    conditions = case["conditions"]
    if len(conditions) != 3:
        raise SystemExit(f"expected three conditions, got {len(conditions)}")
    paths = [
        write_manifest("abc", base, case, conditions),
        write_manifest("a", base, case, [conditions[0]]),
        write_manifest("b", base, case, [conditions[1]]),
        write_manifest("c", base, case, [conditions[2]]),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
