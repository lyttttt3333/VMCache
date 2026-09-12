#!/usr/bin/env python3
"""Construct the distillation dataset and pull one sample, CPU only.

Cheap precondition check before spending a GPU allocation: it proves the
manifest resolves, every teacher latent the dataset will ask for is on disk,
and the validation ids are actually excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.data import WorkerResumeContext
from projects.minimax_h3.data.ref2va_distill import Ref2VATeacherRolloutDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exclude-ids", required=True)
    parser.add_argument("--seed", type=int, default=2026091201)
    args = parser.parse_args()

    ctx = WorkerResumeContext(
        rank=0,
        world_size=1,
        num_workers=1,
        next_logical_worker_id=0,
        committed_states={},
    )
    dataset = Ref2VATeacherRolloutDataset(
        seed=args.seed,
        resume_context=ctx,
        manifest_path=args.manifest,
        require_teacher_latents=True,
        shuffle=True,
        exclude_ids_path=args.exclude_ids,
    )
    print(f"dataset built: {len(dataset.cases)} cases, {len(dataset.exclude_ids)} excluded")

    envelope = next(iter(dataset))
    case = envelope.batch
    latent = Path(case["teacher_latent_path"])
    print(f"first case: {case['id']}")
    print(f"  latent:    {latent}  ({latent.stat().st_size / 1e6:.2f} MB)")
    print(f"  target:    {case['target']['num_frames']}f "
          f"{case['target']['width']}x{case['target']['height']} @{case['target']['fps']}fps")
    for condition in case["conditions"]:
        path = Path(condition["path"])
        print(f"  condition: {condition['type']} {path.name} exists={path.exists()}")


if __name__ == "__main__":
    main()
