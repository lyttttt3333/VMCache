# SPDX-License-Identifier: Apache-2.0
"""Teacher-student denoising distillation for decoupled MiniMax-H3 Ref2VA.

The teacher path is MiniMax-H3's native coupled self-attention over
text/reference/target rows. The student path keeps text and noisy generation
tokens in the active dense sequence, but forces video/image reference assets to
self-attend only within their own asset span. This trains the model behavior
needed for reusable reference KV caches without treating text as reusable
condition state.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import media
from common.data import WorkerResumeContext
from common.diffusion import build_diffusion
from common.diffusion.schedule import PredictionType
from common.distributed import ops
from common.distributed.ops import get_device
from common.distributed.unified_parallel import (
    get_unified_parallel_group,
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from common.logging import get_logger
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, combine_seed, local_seed, yield_seed
from projects.minimax_h3.meta_models.minimax_h3_base import _WallTimer
from projects.minimax_h3.meta_models.minimax_h3_ref2va import MiniMaxH3Ref2VABase
from projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS

logger = get_logger()


@contextlib.contextmanager
def _node_local_video_save_lock() -> Iterator[None]:
    lock_path = Path("/tmp") / (
        f"h3_ref2va_validation_save_{os.environ.get('SLURM_JOB_ID', 'nojid')}.lock"
    )
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class MiniMaxH3Ref2VADecoupledDistill(MiniMaxH3Ref2VABase):
    """Distill native Ref2VA denoising into the decoupled reference attention path."""

    _COVEBENCH_REFERENCE_ROOT = Path(
        "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/"
        "heavy_condition_cache/artifacts/covebench_teacher_rollout/cases"
    )
    _COVEBENCH_TEACACHE_TEACHER_ROOT = Path(
        "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/"
        "heavy_condition_cache/artifacts/covebench_teacher_rollout_teacache_2x/videos"
    )

    def __init__(self, config: Any) -> None:
        self.config = config
        mm = config.meta_model
        self.audio_loss_weight = float(mm.get("audio_loss_weight", 1.0))
        self._validation_packer_cache: Any = None

        diffusion = build_diffusion(config.diffusion)
        self.training_timesteps = diffusion["training_timesteps"]
        self.sampling_timesteps = diffusion["sampling_timesteps"]
        self.audio_sampling_timesteps = diffusion["audio_sampling_timesteps"]
        self.schedule = diffusion["schedule"]
        self.sampler = diffusion["sampler"]
        if not hasattr(self.training_timesteps, "sample_pair"):
            raise TypeError(
                "MiniMax-H3 Ref2VA distillation requires a paired training "
                "timestep node with sample_pair(), e.g. PairedLogitNormalTrainingTimesteps"
            )
        assert self.schedule.pred_type == PredictionType.x_0, (
            "MiniMax-H3 model nodes return x0; distillation target must be x0, "
            f"got {self.schedule.pred_type}"
        )

    def _validation_packer(self, config: Any) -> Any:
        """Use a layout packer separate from the manifest-backed training dataset."""
        if self._validation_packer_cache is not None:
            return self._validation_packer_cache
        packer_cfg = config.meta_model.get("packer_data", None)
        if packer_cfg is None:
            raise ValueError(
                "meta_model.packer_data is required because config.data is the "
                "teacher-rollout manifest dataset, not the Ref2VA layout packer"
            )
        module = importlib.import_module(packer_cfg.module)
        dataset_cls = getattr(module, packer_cfg.class_name)
        args = dict(packer_cfg.get("args", {}))
        validation = config.validation
        args.update(
            prompts=[str(validation.get("packer_prompt", "packer"))],
            paths=None,
            height=int(validation.height),
            width=int(validation.width),
            num_frames=int(validation.num_frames),
        )
        self._validation_packer_cache = dataset_cls(
            seed=int(validation.seed),
            resume_context=WorkerResumeContext(
                rank=0,
                world_size=1,
                num_workers=1,
                next_logical_worker_id=0,
                committed_states={},
            ),
            **args,
        )
        return self._validation_packer_cache

    def _prepare_case(self, ctx: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
        config = ctx["config"]
        models = ctx["models"]
        target = dict(case["target"])
        validation = config.validation
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
        self._reset_packer_if_shape_changed(
            int(validation.width),
            int(validation.height),
            int(validation.num_frames),
        )

        inputs = self._validation_inputs(config, models, [case["prompt"]])
        latent_path = Path(str(case["teacher_latent_path"])).expanduser()
        if not latent_path.is_file():
            raise FileNotFoundError(f"missing teacher rollout latent: {latent_path}")
        payload = torch.load(latent_path, map_location="cpu")
        video = payload["video"].to(device=get_device(), dtype=torch.float32)
        audio = payload["audio"].to(device=get_device(), dtype=torch.float32)
        if tuple(video.shape) != tuple(inputs.layouts[0].latent_shape):
            raise ValueError(
                f"teacher video latent shape {tuple(video.shape)} does not match "
                f"target layout {inputs.layouts[0].latent_shape} for case {case['id']}"
            )
        if tuple(audio.shape) != tuple(inputs.layouts[0].audio_shape):
            raise ValueError(
                f"teacher audio latent shape {tuple(audio.shape)} does not match "
                f"target layout {inputs.layouts[0].audio_shape} for case {case['id']}"
            )
        next_ctx = dict(ctx)
        next_ctx["batch"] = case
        next_ctx["inputs"] = inputs
        next_ctx["clean_latents"] = ([video], [audio])
        next_ctx["case_id"] = str(case["id"])
        return next_ctx

    def _reset_packer_if_shape_changed(
        self,
        width: int,
        height: int,
        num_frames: int,
    ) -> None:
        shape_key = (int(width), int(height), int(num_frames))
        if getattr(self, "_ref2va_packer_shape_key", None) != shape_key:
            self._validation_packer_cache = None
            self._ref2va_packer_shape_key = shape_key

    def _covebench_validation_cases(self, validation: Any) -> list[dict[str, Any]]:
        manifest = Path(str(validation.covebench_manifest)).expanduser()
        final_cases = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(final_cases, list):
            raise ValueError(f"{manifest} must contain a list of validation cases")
        rollout_index_path = Path(str(validation.covebench_rollout_index)).expanduser()
        rollout_index = json.loads(rollout_index_path.read_text(encoding="utf-8"))
        rollout_by_id = {
            str(case["id"]): case
            for case in rollout_index.get("cases", [])
            if isinstance(case, dict) and "id" in case
        }
        reference_root = Path(
            str(validation.get("covebench_reference_root", self._COVEBENCH_REFERENCE_ROOT))
        ).expanduser()
        teacher_root = Path(
            str(
                validation.get(
                    "covebench_teacher_video_root",
                    self._COVEBENCH_TEACACHE_TEACHER_ROOT,
                )
            )
        ).expanduser()
        cases: list[dict[str, Any]] = []
        for rank, item in enumerate(final_cases):
            case_id = str(item["id"])
            source = rollout_by_id.get(case_id)
            if source is None:
                raise KeyError(f"CoVeBench validation id {case_id!r} missing from rollout index")
            reference_path = reference_root / case_id / "reference_video.mp4"
            teacher_video_path = (
                teacher_root / case_id / "validation" / "backbone" / "prompt0000.mp4"
            )
            case = {
                **source,
                "id": case_id,
                "prompt": str(item["prompt"]),
                "final32_rank": rank,
                "tier": item.get("tier"),
                "tier_key": item.get("tier_key"),
                "rank_in_tier": item.get("rank_in_tier"),
                "edit_zh": item.get("edit_zh"),
                "teacher_similarity_score": item.get("similarity_score"),
                "teacher_psnr": item.get("psnr"),
                "teacher_ssim": item.get("ssim"),
                "teacher_lpips": item.get("lpips"),
                "teacher_video_path": str(teacher_video_path),
            }
            case["conditions"] = [
                {
                    **dict(source["conditions"][0]),
                    "path": str(reference_path),
                    "source_path": str(reference_path),
                }
            ]
            cases.append(case)
        return cases

    @staticmethod
    def _split_cases_for_group(
        pending: list[tuple[int, dict[str, Any]]],
        group_index: int,
        num_groups: int,
    ) -> list[dict[str, Any]]:
        if not pending:
            return []
        per_group = (len(pending) + num_groups - 1) // num_groups
        padded = pending + [pending[-1]] * (per_group * num_groups - len(pending))
        start = group_index * per_group
        entries = []
        for offset, (case_index, case) in enumerate(padded[start : start + per_group]):
            entries.append(
                {
                    "case_index": case_index,
                    "case": case,
                    "should_save": start + offset < len(pending),
                }
            )
        return entries

    @execution_phase(ExecutionPhase.VALIDATION)
    @torch.no_grad()
    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        validation = ctx["config"].validation
        if not validation.get("covebench_manifest", None):
            return super().validate(ctx)
        return self._validate_covebench_final32(ctx)

    def _validate_covebench_final32(self, ctx: dict[str, Any]) -> dict[str, Any]:
        config = ctx["config"]
        models = ctx["models"]
        validation = config.validation
        step = int(ctx["step"])
        cases = self._covebench_validation_cases(validation)
        tcache_enabled = os.environ.get("H3_REF2VA_TCACHE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        sampling_policy = {
            "name": "teacache_2x" if tcache_enabled else "full_compute",
            "num_denoising_steps": int(self.sampling_timesteps.num_sampling_steps),
            "tcache_enabled": tcache_enabled,
            "tcache_compute_steps": (
                os.environ.get("H3_REF2VA_TCACHE_COMPUTE_STEPS", "")
                if tcache_enabled
                else None
            ),
            "environment": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith("H3_REF2VA_TCACHE")
                or key.startswith("H3_REF2VA_CONDITION_KV_CACHE")
            },
        }
        adapter = config.models.backbone.get("adapter", {})
        manifest_path = Path(str(validation.covebench_manifest)).expanduser()
        rollout_index_path = Path(str(validation.covebench_rollout_index)).expanduser()
        case_inputs = []
        for case in cases:
            condition_inputs = []
            for condition in case["conditions"]:
                condition_path = Path(str(condition["path"])).expanduser().resolve()
                stat = condition_path.stat()
                condition_inputs.append(
                    {
                        "type": condition.get("type"),
                        "role": condition.get("role"),
                        "path": str(condition_path),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "start_time_seconds": condition.get("start_time_seconds"),
                    }
                )
            case_inputs.append(
                {
                    "id": str(case["id"]),
                    "prompt": str(case["prompt"]),
                    "seed": int(case.get("seed", validation.get("seed", 2026091201))),
                    "ref_noise_seed": int(
                        case.get(
                            "ref_noise_seed",
                            case.get("seed", validation.get("seed", 2026091201)),
                        )
                    ),
                    "conditions": condition_inputs,
                    "media_probes": case.get("media_probes"),
                    "resolved_shapes": case.get("resolved_shapes"),
                }
            )
        fingerprint_target = dict(validation.get("target", {}))
        fingerprint_payload = {
            "student_attention": "decoupled_reference",
            "seed": int(config.engine.seed),
            "adapter": {
                "r": adapter.get("r"),
                "lora_alpha": adapter.get("lora_alpha"),
                "target_modules": list(adapter.get("target_modules", [])),
            },
            "sampling_policy": sampling_policy,
            "target": {
                "width": int(fingerprint_target.get("width", validation.get("width", 1344))),
                "height": int(fingerprint_target.get("height", validation.get("height", 768))),
                "fps": int(
                    fingerprint_target.get(
                        "fps", validation.get("fps", MINIMAX_H3_SUPPORTED_FPS)
                    )
                ),
                "num_frames": int(
                    fingerprint_target.get(
                        "num_frames", validation.get("num_frames", 124)
                    )
                ),
                "crf": int(validation.get("crf", 18)),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            "rollout_index": {
                "path": str(rollout_index_path.resolve()),
                "sha256": hashlib.sha256(rollout_index_path.read_bytes()).hexdigest(),
            },
            "cases": case_inputs,
        }
        validation_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        rank = ops.get_rank()
        sp_size = self._sp_size()
        group_index = rank // sp_size
        num_groups = ops.get_world_size() // sp_size
        is_group_main = rank % sp_size == 0
        media_dir = logger.iter_dir("media", step)
        out_dir = media_dir / "covebench_final32" / "backbone"
        completed_local: set[str] = set()
        for video_path in out_dir.glob("*.mp4"):
            meta_path = video_path.with_suffix(".json")
            if video_path.stat().st_size <= 0 or not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                int(meta.get("step", -1)) == step
                and meta.get("validation_fingerprint") == validation_fingerprint
            ):
                completed_local.add(video_path.stem)
        completed_views = ops.all_gather_object(completed_local)
        completed = set().union(*completed_views)
        pending = [
            (index, case)
            for index, case in enumerate(cases)
            if str(case["id"]) not in completed
        ]
        entries = self._split_cases_for_group(pending, group_index, num_groups)
        fps = int(validation.get("fps", MINIMAX_H3_SUPPORTED_FPS))
        saved_items: list[dict[str, Any]] = []
        original_training = models["backbone"].training
        previous_force = getattr(self, "_ref2va_force_decoupled_ref_attention", False)
        self._reset_packer_if_shape_changed(
            int(validation.get("width", 1344)),
            int(validation.get("height", 768)),
            int(validation.get("num_frames", 124)),
        )
        try:
            models["backbone"].eval()
            self._ref2va_force_decoupled_ref_attention = True
            for entry in entries:
                case = entry["case"]
                case_id = str(case["id"])
                target = dict(validation.get("target", {}))
                if not target:
                    target = {"width": 1344, "height": 768, "fps": 24, "num_frames": 124}
                validation.conditions = case["conditions"]
                validation.media_probes = case["media_probes"]
                validation.resolved_shapes = case["resolved_shapes"]
                validation.prompts = [case["prompt"]]
                validation.num_prompts = 1
                validation.batch_size = 1
                validation.seed = int(case.get("seed", validation.get("seed", 2026091201)))
                validation.ref_noise_seed = int(case.get("ref_noise_seed", validation.seed))
                validation.width = int(target.get("width", validation.get("width", 1344)))
                validation.height = int(target.get("height", validation.get("height", 768)))
                validation.fps = int(target.get("fps", validation.get("fps", 24)))
                validation.num_frames = int(
                    target.get("num_frames", validation.get("num_frames", 124))
                )
                self._reset_packer_if_shape_changed(
                    int(validation.width),
                    int(validation.height),
                    int(validation.num_frames),
                )

                with _WallTimer("covebench_prepare_inputs", case_id=case_id):
                    inputs = self._validation_inputs(config, models, [case["prompt"]])
                sample_rng = RandomState(
                    combine_seed(int(validation.seed), "covebench_final32", case_id)
                )
                with _WallTimer("covebench_sample_latents", case_id=case_id):
                    latents = self._sample_latents(
                        models["backbone"], inputs, sample_rng, rngs=[sample_rng]
                    )
                save_error = None
                if is_group_main and entry["should_save"]:
                    try:
                        with _WallTimer("covebench_decode_latents", case_id=case_id):
                            frames, waveform = self._decode_latents(models, latents, index=0)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        video_path = out_dir / f"{case_id}.mp4"
                        tmp_video_path = video_path.with_name(f"{case_id}.tmp.mp4")
                        with _WallTimer("covebench_save_video", case_id=case_id):
                            with _node_local_video_save_lock():
                                media.save_video(
                                    frames,
                                    tmp_video_path,
                                    fps=fps,
                                    nrow=1,
                                    crf=int(validation.get("crf", 18)),
                                    audio_tensor=waveform,
                                    audio_sample_rate=int(models["audio_vae"].sample_rate),
                                    normalize=True,
                                    value_range=(0, 1),
                                )
                            if (
                                not tmp_video_path.is_file()
                                or tmp_video_path.stat().st_size == 0
                            ):
                                raise RuntimeError(
                                    f"failed to encode validation video {video_path}"
                                )
                            tmp_video_path.replace(video_path)
                        meta = {
                            "id": case_id,
                            "step": step,
                            "validation_fingerprint": validation_fingerprint,
                            "prompt": case["prompt"],
                            "edit_zh": case.get("edit_zh"),
                            "tier": case.get("tier"),
                            "tier_key": case.get("tier_key"),
                            "rank_in_tier": case.get("rank_in_tier"),
                            "reference_video_path": case["conditions"][0]["path"],
                            "teacher_video_path": case.get("teacher_video_path"),
                            "student_video_path": str(video_path),
                            "teacher_similarity_score": case.get(
                                "teacher_similarity_score"
                            ),
                            "teacher_psnr": case.get("teacher_psnr"),
                            "teacher_ssim": case.get("teacher_ssim"),
                            "teacher_lpips": case.get("teacher_lpips"),
                            "sampling_policy": sampling_policy,
                        }
                        meta_path = out_dir / f"{case_id}.json"
                        meta_path.write_text(
                            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        saved_items.append(meta)
                        del frames, waveform
                    except Exception as exc:  # noqa: BLE001 - synchronize rank-local I/O errors
                        save_error = f"rank={rank} case={case_id}: {exc!r}"
                save_errors = ops.all_gather_object(save_error)
                failures = [error for error in save_errors if error is not None]
                if failures:
                    raise RuntimeError(
                        "CoVeBench validation decode/save failed: " + "; ".join(failures)
                    )
                del latents, inputs
            gathered = ops.all_gather_object(saved_items)
            if rank == 0:
                newly_saved = {
                    str(item["id"]): item
                    for rank_items in gathered
                    for item in rank_items
                }
                flat = []
                for case in cases:
                    case_id = str(case["id"])
                    meta_path = out_dir / f"{case_id}.json"
                    if meta_path.is_file():
                        flat.append(json.loads(meta_path.read_text(encoding="utf-8")))
                    elif case_id in newly_saved:
                        flat.append(newly_saved[case_id])
                summary_path = media_dir / "covebench_final32" / "summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(
                    json.dumps(
                        {
                            "step": step,
                            "case_count": len(cases),
                            "saved_count": len(flat),
                            "manifest": str(validation.covebench_manifest),
                            "rollout_index": str(validation.covebench_rollout_index),
                            "sampling_policy": sampling_policy,
                            "validation_fingerprint": validation_fingerprint,
                            "items": flat,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                logger.info(
                    "CoVeBench validation step=%d complete: %d/%d videos",
                    step,
                    len(flat),
                    len(cases),
                )
                log_video = getattr(logger, "log_video", None)
                if validation.get("log_video", False) and log_video is not None and flat:
                    log_video(
                        {
                            f"{item['id']}.mp4": Path(item["student_video_path"])
                            for item in flat
                        },
                        step=step,
                        collection="validation/covebench_final32/backbone",
                        captions={
                            f"{item['id']}.mp4": item["prompt"]
                            for item in flat
                        },
                        fps=fps,
                    )
        finally:
            self._ref2va_force_decoupled_ref_attention = previous_force
            models["backbone"].train(original_training)
            # Object collectives use the already-initialized CPU/Gloo backend.
            # A CUDA barrier here can fail after long validation runs when the
            # node PID budget cannot accommodate another NCCL helper thread.
            ops.all_gather_object(None)
        return ctx

    @execution_phase(ExecutionPhase.PREPARE)
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: batch, models. Writes: inputs, clean_latents, case_id."""
        if is_unified_parallel_initialized() and get_unified_parallel_world_size() > 1:
            next_ctx = dict(ctx)
            next_ctx["batch"] = dict(ctx["batch"])
            return next_ctx
        return self._prepare_case(ctx, dict(ctx["batch"]))

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Broadcast one raw case per SP group, then rebuild inputs on each rank."""
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("unified parallel is initialized but torch.distributed is not")
        group = get_unified_parallel_group()
        local_rank = get_unified_parallel_rank()
        group_size = get_unified_parallel_world_size()
        for source_rank in range(group_size):
            src = (
                dist.get_global_rank(group, source_rank)
                if hasattr(dist, "get_global_rank")
                else source_rank
            )
            payload = [ctx["batch"] if local_rank == source_rank else None]
            dist.broadcast_object_list(payload, src=src, group=group)
            yield self._prepare_case(ctx, dict(payload[0]))

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_timesteps(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: inputs, rng. Writes: train_timesteps."""
        inputs = ctx["inputs"]
        rng = ctx["rng"]
        with local_seed(rng.seed % 2**31):
            video_t, audio_t = self.training_timesteps.sample_pair(
                (inputs.batch_size,),
                inputs.seqlens,
                get_device(),
            )
        rng.seed = yield_seed(rng.seed)
        ctx["train_timesteps"] = (video_t.to(torch.float32), audio_t.to(torch.float32))
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: clean_latents, train_timesteps, rng. Writes: noisy_latents."""
        clean_video, clean_audio = ctx["clean_latents"]
        video_t, audio_t = ctx["train_timesteps"]
        rng = ctx["rng"]
        video_noise = [
            torch.empty_like(tensor, device=get_device()).normal_(generator=self._generator(rng))
            for tensor in clean_video
        ]
        audio_noise = [
            torch.empty_like(tensor, device=get_device()).normal_(generator=self._generator(rng))
            for tensor in clean_audio
        ]
        rng.seed = yield_seed(rng.seed)
        ctx["noisy_latents"] = (
            self.schedule.forward(clean_video, video_noise, video_t),
            self.schedule.forward(clean_audio, audio_noise, audio_t),
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Run coupled teacher and decoupled-reference student at the same x_t."""
        inputs = ctx["inputs"]
        video_xts, audio_xts = ctx["noisy_latents"]
        video_t, audio_t = ctx["train_timesteps"]
        teacher = (
            ctx["models"].get("teacher_backbone")
            or ctx["models"].get("backbone_ema")
            or ctx["models"]["backbone"]
        )
        was_training = teacher.training
        teacher.eval()
        try:
            with torch.no_grad():
                ctx["teacher_pred"] = self._bidirectional_forward(
                    teacher,
                    inputs,
                    video_xts=video_xts,
                    audio_xts=audio_xts,
                    video_timesteps=video_t,
                    audio_timesteps=audio_t,
                )
        finally:
            teacher.train(was_training)

        ctx["student_pred"] = self._bidirectional_forward(
            ctx["models"]["backbone"],
            inputs,
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_timesteps=video_t,
            audio_timesteps=audio_t,
            decoupled_ref_attention=True,
        )
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: teacher_pred, student_pred. Writes: loss, metrics."""
        student_video, student_audio = ctx["student_pred"]
        teacher_video, teacher_audio = ctx["teacher_pred"]
        video_losses = [
            F.mse_loss(student.float(), teacher.float().detach())
            for student, teacher in zip(student_video, teacher_video, strict=True)
        ]
        audio_losses = [
            F.mse_loss(student.float(), teacher.float().detach())
            for student, teacher in zip(student_audio, teacher_audio, strict=True)
        ]
        video_loss = torch.stack(video_losses).mean()
        audio_loss = torch.stack(audio_losses).mean()
        loss = video_loss + self.audio_loss_weight * audio_loss
        ctx["loss"] = loss
        ctx["metrics"] = {
            "train_losses/video_teacher_student": float(video_loss.detach().item()),
            "train_losses/audio_teacher_student": float(audio_loss.detach().item()),
            "train_losses/total": float(loss.detach().item()),
        }
        return ctx


__all__ = ["MiniMaxH3Ref2VADecoupledDistill"]
