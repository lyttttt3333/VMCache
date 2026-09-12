# SPDX-License-Identifier: Apache-2.0
"""Batched teacher rollout worker for MiniMax-H3 Ref2VA.

This keeps the released/native Ref2VA path intact, but moves the outer loop
from one Slurm array element per case into one validation call that can process
many cases after a single model load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from common import media
from common.distributed import ops
from common.logging import get_logger
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, combine_seed
from projects.minimax_h3.meta_models.minimax_h3_base import _WallTimer
from projects.minimax_h3.meta_models.minimax_h3_ref2va import MiniMaxH3Ref2VABase
from projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS

logger = get_logger()


def _case_output_paths(
    case_id: str,
    *,
    latent_root: Path,
    video_root: Path,
    variant_name: str,
) -> tuple[Path, Path]:
    latent_path = latent_root / case_id / "validation" / variant_name / "prompt0000.pt"
    video_path = video_root / case_id / "validation" / variant_name / "prompt0000.mp4"
    return latent_path, video_path


class MiniMaxH3Ref2VATeacherRolloutWorker(MiniMaxH3Ref2VABase):
    """Run many native/coupled Ref2VA teacher rollouts in one loaded process."""

    @execution_phase(ExecutionPhase.VALIDATION)
    @torch.no_grad()
    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        config = ctx["config"]
        models = ctx["models"]
        validation = config.validation

        index_path = Path(str(validation.rollout_index)).expanduser()
        with index_path.open(encoding="utf-8") as f:
            rollout_index = json.load(f)
        cases = list(rollout_index["cases"])
        target = dict(rollout_index["target"])
        # The rollout index owns the output geometry. Do not retain a packer
        # built for a previous validation invocation or inherited config shape.
        self._validation_packer_cache = None

        raw_indices = validation.get("case_indices", None)
        if raw_indices is not None and str(raw_indices).strip():
            if isinstance(raw_indices, str):
                indices = [
                    int(part.strip())
                    for part in raw_indices.replace(":", ",").split(",")
                    if part.strip()
                ]
            else:
                indices = [int(item) for item in raw_indices]
            selected_entries = [(index, cases[index]) for index in indices]
        else:
            start = int(validation.get("case_start", 0))
            requested_count = validation.get("case_count", None)
            if requested_count is None:
                stop = len(cases)
            else:
                stop = min(len(cases), start + int(requested_count))
            selected_entries = list(enumerate(cases[start:stop], start=start))
        if not selected_entries:
            logger.info("teacher rollout worker found no selected cases")
            return ctx

        rank = ops.get_rank()
        sp_size = self._sp_size()
        is_group_main = rank % sp_size == 0
        if ops.get_world_size() // sp_size != 1:
            raise ValueError("teacher rollout worker currently expects one SP group")

        variants = [("backbone", models["backbone"])]
        if validation.get("validate_ema", False):
            variants.append(("backbone_ema", models["backbone_ema"]))

        latent_root = Path(
            str(validation.get("latent_root", rollout_index["teacher_latent_root"]))
        ).expanduser()
        video_root = Path(
            str(validation.get("video_root", rollout_index["teacher_video_root"]))
        ).expanduser()
        save_latent = bool(validation.get("save_latent", True))
        save_video = bool(validation.get("save_video", False))
        force = bool(validation.get("force", False))
        fps = int(target.get("fps", validation.get("fps", MINIMAX_H3_SUPPORTED_FPS)))

        original_training = [(model, model.training) for _, model in variants]
        saved_items: list[tuple[str, str, str]] = []
        try:
            for _, model in variants:
                model.eval()

            for absolute_index, case in selected_entries:
                case_id = str(case["id"])
                skip = False
                if rank == 0 and not force:
                    requested_paths: list[Path] = []
                    for variant_name, _ in variants:
                        latent_path, video_path = _case_output_paths(
                            case_id,
                            latent_root=latent_root,
                            video_root=video_root,
                            variant_name=variant_name,
                        )
                        if save_latent:
                            requested_paths.append(latent_path)
                        if save_video:
                            requested_paths.append(video_path)
                    skip = bool(requested_paths) and all(path.is_file() for path in requested_paths)
                skip = bool(ops.all_gather_object(skip)[0])
                if skip:
                    logger.info("skip completed teacher rollout case %s", case_id)
                    continue

                validation.conditions = case["conditions"]
                validation.media_probes = case["media_probes"]
                validation.resolved_shapes = case["resolved_shapes"]
                validation.prompts = [case["prompt"]]
                validation.num_prompts = 1
                validation.batch_size = 1
                validation.seed = int(case["seed"])
                validation.ref_noise_seed = int(case.get("ref_noise_seed", case["seed"]))
                validation.width = int(target["width"])
                validation.height = int(target["height"])
                validation.fps = int(target["fps"])
                validation.num_frames = int(target["num_frames"])

                logger.info(
                    "teacher rollout case %d/%d: %s",
                    absolute_index + 1,
                    len(cases),
                    case_id,
                )
                with _WallTimer("prepare_inputs", case_id=case_id):
                    inputs = self._validation_inputs(config, models, [case["prompt"]])

                for variant_name, model in variants:
                    sample_rng = RandomState(
                        combine_seed(int(validation.seed), "validation", 0, variant_name)
                    )
                    with _WallTimer("sample_latents", case_id=case_id, variant=variant_name):
                        latents = self._sample_latents(
                            model, inputs, sample_rng, rngs=[sample_rng]
                        )

                    if is_group_main:
                        latent_path, video_path = _case_output_paths(
                            case_id,
                            latent_root=latent_root,
                            video_root=video_root,
                            variant_name=variant_name,
                        )
                        if save_latent:
                            latent_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp = latent_path.with_name(latent_path.name + ".tmp")
                            torch.save(
                                {
                                    "video": latents.video[0].to(torch.bfloat16).cpu(),
                                    "audio": latents.audio[0].to(torch.bfloat16).cpu(),
                                    "prompt": case["prompt"],
                                    "prompt_idx": 0,
                                    "case_id": case_id,
                                    "source_index": absolute_index,
                                },
                                tmp,
                            )
                            tmp.replace(latent_path)
                            saved_items.append((case_id, "latent", str(latent_path)))

                        if save_video:
                            with _WallTimer(
                                "decode_latents",
                                case_id=case_id,
                                variant=variant_name,
                            ):
                                frames, waveform = self._decode_latents(
                                    models, latents, index=0
                                )
                            video_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp_video = video_path.with_name(video_path.name + ".tmp.mp4")
                            with _WallTimer("save_video", case_id=case_id, variant=variant_name):
                                media.save_video(
                                    frames,
                                    tmp_video,
                                    fps=fps,
                                    nrow=int(validation.get("nrow", 1)),
                                    crf=int(validation.get("crf", 18)),
                                    audio_tensor=waveform,
                                    audio_sample_rate=int(models["audio_vae"].sample_rate),
                                    normalize=True,
                                    value_range=(0, 1),
                                )
                            tmp_video.replace(video_path)
                            saved_items.append((case_id, "video", str(video_path)))
                            del frames, waveform
                    del latents
                del inputs
                ops.barrier()
        finally:
            for model, training in original_training:
                model.train(training)

        gathered = ops.all_gather_object(saved_items)
        if rank == 0:
            flat_items = [item for rank_items in gathered for item in rank_items]
            logger.info("teacher rollout worker saved %d artifacts", len(flat_items))
            for case_id, kind, path in flat_items:
                logger.info("saved %s for %s: %s", kind, case_id, path)
        return ctx
