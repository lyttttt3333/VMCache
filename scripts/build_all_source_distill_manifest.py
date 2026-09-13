#!/usr/bin/env python3
"""Build the phase-2 distillation manifest without touching an active run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LATENT_SUFFIX = Path("validation") / "backbone" / "prompt0000.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--cove-index", type=Path, required=True)
    parser.add_argument("--cove-latent-root", type=Path, required=True)
    parser.add_argument("--v5-index", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v5-weight", type=float, default=6.0)
    parser.add_argument("--allow-incomplete-v5", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def excluded_ids(path: Path) -> set[str]:
    payload = load_json(path)
    if isinstance(payload, list):
        return {
            str(item["id"] if isinstance(item, dict) else item)
            for item in payload
        }
    ids = {str(item) for item in payload.get("ids", [])}
    ids.update(
        str(item["id"])
        for item in payload.get("cases", [])
        if isinstance(item, dict) and "id" in item
    )
    return ids


def normalized_case(
    source: dict[str, Any],
    *,
    case_id: str,
    source_name: str,
    target: dict[str, Any],
    latent_path: Path,
    sampling_weight: float,
) -> dict[str, Any]:
    case = dict(source)
    case.update(
        {
            "id": case_id,
            "original_id": str(source["id"]),
            "dataset_source": source_name,
            "target": dict(source.get("target", target)),
            "teacher_latent_path": str(latent_path.expanduser().resolve()),
            "teacher_latent_exists": latent_path.is_file(),
            "sampling_weight": float(sampling_weight),
        }
    )
    return case


def require_latents(cases: list[dict[str, Any]], *, source_name: str) -> None:
    missing = [case["teacher_latent_path"] for case in cases if not case["teacher_latent_exists"]]
    if missing:
        raise FileNotFoundError(
            f"{source_name} has {len(missing)} missing teacher latents; first: {missing[0]}"
        )


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.v5_weight) or args.v5_weight <= 0:
        raise ValueError("--v5-weight must be finite and positive")

    base = load_json(args.base_manifest)
    cove = load_json(args.cove_index)
    v5 = load_json(args.v5_index)
    held_out = excluded_ids(args.exclude_ids)

    base_cases = [
        normalized_case(
            source,
            case_id=str(source["id"]),
            source_name="five_v2v_600",
            target=base["target"],
            latent_path=Path(source["teacher_latent_path"]),
            sampling_weight=float(source.get("sampling_weight", 1.0)),
        )
        for source in base["cases"]
    ]
    require_latents(base_cases, source_name="five_v2v_600")

    cove_root = args.cove_latent_root.expanduser().resolve()
    cove_sources = [
        source for source in cove["cases"] if str(source["id"]) not in held_out
    ]
    cove_cases = [
        normalized_case(
            source,
            case_id=f"cove__{source['id']}",
            source_name="covebench_train",
            target=cove["target"],
            latent_path=cove_root / str(source["id"]) / LATENT_SUFFIX,
            sampling_weight=1.0,
        )
        for source in cove_sources
    ]
    require_latents(cove_cases, source_name="covebench_train")

    v5_cases = [
        normalized_case(
            source,
            case_id=f"v5_3ref__{source['id']}",
            source_name="v5_3ref",
            target=v5["target"],
            latent_path=Path(source["teacher_latent_path"]),
            sampling_weight=args.v5_weight,
        )
        for source in v5["cases"]
    ]
    if not args.allow_incomplete_v5:
        require_latents(v5_cases, source_name="v5_3ref")
    else:
        v5_cases = [case for case in v5_cases if case["teacher_latent_exists"]]

    cases = base_cases + cove_cases + v5_cases
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("combined manifest contains duplicate case ids")
    for index, case in enumerate(cases):
        case["index"] = index

    weighted_total = sum(float(case["sampling_weight"]) for case in cases)
    payload = {
        "name": "ref2va_decoupled_distill_all_sources_phase2",
        "purpose": (
            "Train the decoupled-reference student on held-out-safe single-reference "
            "editing data and weighted three-reference composition data."
        ),
        "teacher_attention_mode": "coupled_native_self_attention",
        "student_attention_mode": (
            "decoupled_reference_self_attention_plus_generation_reads_references"
        ),
        "teacher_rollout_sampler": "teacache_2x",
        "target": dict(base["target"]),
        "per_case_target_enabled": True,
        "case_count": len(cases),
        "complete_teacher_latent_count": sum(
            bool(case["teacher_latent_exists"]) for case in cases
        ),
        "missing_teacher_latent_count": sum(
            not bool(case["teacher_latent_exists"]) for case in cases
        ),
        "sources": {
            "five_v2v_600": {"case_count": len(base_cases), "sampling_weight": 1.0},
            "covebench_train": {
                "case_count": len(cove_cases),
                "excluded_validation_ids": len(held_out),
                "sampling_weight": 1.0,
            },
            "v5_3ref": {
                "case_count": len(v5_cases),
                "sampling_weight": float(args.v5_weight),
                "sampling_probability": (
                    len(v5_cases) * float(args.v5_weight) / weighted_total
                ),
            },
        },
        "source_paths": {
            "base_manifest": str(args.base_manifest.expanduser().resolve()),
            "cove_index": str(args.cove_index.expanduser().resolve()),
            "cove_latent_root": str(cove_root),
            "v5_index": str(args.v5_index.expanduser().resolve()),
            "excluded_validation_ids": str(args.exclude_ids.expanduser().resolve()),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {len(base_cases)} five + {len(cove_cases)} cove + "
        f"{len(v5_cases)} three-reference; total={len(cases)}; "
        f"three-reference sampling={payload['sources']['v5_3ref']['sampling_probability']:.3f}"
    )


if __name__ == "__main__":
    main()
