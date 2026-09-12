"""MiniMaxH3Base: bidirectional multi-step inference with the released MiniMax-H3 checkpoint.

Sampling is the global, non-causal loop: every step is one full-sequence dense
forward over ``[text | all audio | all video]``. There is no cache, no chunk loop
and no clean/noise cache roles -- causality is a generation-time constraint of the
*student*, not of the released bidirectional model. No training logic is attached.

The class subclasses ``CausalMiniMaxH3DMD`` purely to reuse the shared primitives
-- packed layout derivation (``_build_inputs``), text encoding
(``_encode_prompts``), the dense bidirectional forward (``_bidirectional_forward``,
the very call the DMD critics and teacher use), VAE decoding (``_decode_latents``)
and the validation bookkeeping (``_validation_*``, resume, collective split,
mp4/grid logging). It deliberately does NOT call the parent ``__init__``, because
the DMD constructor asserts training knobs (``fake_loss_type``, ``dmd_loss``, ...)
that an inference config has no business carrying.

An inference trial therefore needs only the sampling-side config:
``diffusion.sampling_timesteps`` and ``diffusion.audio_sampling_timesteps`` (the
same step count, with the video and audio shifts of the released pipeline),
``diffusion.schedule`` with ``pred_type: x_0``, ``diffusion.sampler``, the
``backbone``/``text_encoder``/``video_vae``/``audio_vae`` model nodes, a
``validation`` block, and a ``data`` node whose dataset class is instantiated only
as the layout packer. There is no CFG: the released checkpoint is
guidance-distilled, so there is a single positive denoise branch.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

import torch

from common import media
from common.diffusion import build_diffusion
from common.diffusion.schedule import PredictionType
from common.distributed import ops
from common.distributed.ops import get_device
from common.logging import get_logger
from common.phase import ExecutionPhase, execution_phase
from common.seed import RandomState, combine_seed

from projects.minimax_h3.meta_models.causal_minimax_h3_dmd import (
    CausalMiniMaxH3DMD,
    _RolloutX0s,
)
from projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS

logger = get_logger()


def _hot_timing_enabled() -> bool:
    return os.environ.get("H3_REF2VA_HOT_TIMING", "").lower() in {"1", "true", "yes", "on"}


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _hot_timing_path() -> Path:
    configured = os.environ.get("H3_REF2VA_HOT_TIMING_PATH")
    if configured:
        return Path(configured)
    return Path(logger.config.persistence.run_dir) / "metrics" / "hot_timing_rank0.jsonl"


def _log_hot_timing(event: dict[str, Any]) -> None:
    if not _hot_timing_enabled() or ops.get_rank() != 0:
        return
    path = _hot_timing_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": time.time(), **event}
    with path.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


class _WallTimer:
    def __init__(self, name: str, **fields: Any) -> None:
        self.name = name
        self.fields = fields
        self.start = 0.0

    def __enter__(self) -> "_WallTimer":
        if _hot_timing_enabled():
            _sync_cuda()
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not _hot_timing_enabled():
            return
        _sync_cuda()
        _log_hot_timing(
            {
                "event": self.name,
                "elapsed_ms": (time.perf_counter() - self.start) * 1000.0,
                **self.fields,
            }
        )


class MiniMaxH3Base(CausalMiniMaxH3DMD):
    """Bidirectional multi-step T2AV sampling; no training primitives.

    Reuses the parent's layout/forward/decode/validation helpers but owns its
    own ``__init__`` (sampling diffusion nodes only) and ``validate`` (global
    dense sampling instead of the chunk-causal rollout).
    """

    def __init__(self, config: Any) -> None:
        # Deliberately not super().__init__: the DMD constructor requires
        # training-only knobs this config does not carry.
        self.config = config
        diffusion = build_diffusion(config.diffusion)
        self.sampling_timesteps = diffusion["sampling_timesteps"]
        self.audio_sampling_timesteps = diffusion["audio_sampling_timesteps"]
        self.schedule = diffusion["schedule"]
        self.sampler = diffusion["sampler"]
        # Every node is a MiniMaxH3X0Model: the prediction already IS x0, and
        # the sampler converts through this schedule on every step.
        assert self.schedule.pred_type == PredictionType.x_0, (
            f"diffusion.schedule.pred_type must be x_0 because every MiniMax-H3 "
            f"model node returns x0, got {self.schedule.pred_type}"
        )
        # Built lazily on the first validation, exactly like the DMD parent:
        # the dataset owns the packed layout, so validation packs its prompts
        # through the very same class rather than re-deriving it here.
        self._validation_packer_cache: Any = None

    # ------------------------------------------------------------ sampling --
    def _condition_cache_refresh_steps(self, num_steps: int) -> set[int]:
        raw_steps = os.environ.get("H3_REF2VA_CONDITION_KV_CACHE_REFRESH_STEPS", "")
        if raw_steps.strip():
            steps = {
                int(part.strip())
                for part in re.split(r"[,:\s]+", raw_steps)
                if part.strip()
            }
            return {step for step in steps if 0 <= step < num_steps}

        raw_count = os.environ.get("H3_REF2VA_CONDITION_KV_CACHE_REFRESH_COUNT", "")
        if not raw_count.strip():
            return set()
        count = max(1, int(raw_count))
        if count == 1:
            return {0}
        return {
            round(index * (num_steps - 1) / (count - 1))
            for index in range(count)
        }

    def _tcache_compute_steps(self, num_steps: int, force_steps: set[int]) -> set[int]:
        enabled = os.environ.get("H3_REF2VA_TCACHE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return set(range(num_steps))

        raw_steps = os.environ.get("H3_REF2VA_TCACHE_COMPUTE_STEPS", "")
        if raw_steps.strip():
            compute_steps = {
                int(part.strip())
                for part in re.split(r"[,:\s]+", raw_steps)
                if part.strip()
            }
            compute_steps = {step for step in compute_steps if 0 <= step < num_steps}
        else:
            stride = max(1, int(os.environ.get("H3_REF2VA_TCACHE_STRIDE", "2")))
            start = max(1, int(os.environ.get("H3_REF2VA_TCACHE_START", "2")))
            end = int(os.environ.get("H3_REF2VA_TCACHE_END", str(num_steps - 2)))
            end = min(num_steps - 2, max(start, end))
            compute_steps = {
                step
                for step in range(num_steps)
                if step < start or step > end or (step - start) % stride == 0
            }

        compute_steps.add(0)
        compute_steps.add(num_steps - 1)
        compute_steps.update(force_steps)
        return {step for step in compute_steps if 0 <= step < num_steps}

    @torch.no_grad()
    def _sample_latents(
        self, model: Any, inputs: Any, rng: Any,
        *, rngs: Sequence[Any] | None = None,
    ) -> _RolloutX0s:
        """Global multi-step flow sampling; every step is one dense forward.

        One shared step index drives both grids, each with its own shift --
        H3's native "steps aligned, sigmas not" pairing, so the pair the model
        sees at every step is one its own sampler produces. The final step
        lands on the clean timestep 0: the grid's last ``s`` is the bound, and
        the eta=0 step collapses to the predicted x0.

        ``rngs`` (optional, one stream per sample) gives every sample its own
        deterministic stream inside ONE batched sampling pass -- the
        validation pattern (per-prompt seeds, batch is just a shared forward,
        so changing batch_size never perturbs a prompt's noise).
        """
        device = get_device()
        sample_rngs = rngs if rngs is not None else [rng] * inputs.batch_size
        assert len(sample_rngs) == inputs.batch_size, (
            f"rngs must have one stream per sample: {len(sample_rngs)} vs "
            f"batch_size {inputs.batch_size}"
        )
        video_noises = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        audio_noises = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]

        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        video_grid = self.sampling_timesteps.timesteps
        audio_grid = self.audio_sampling_timesteps.timesteps
        assert video_grid.dim() == 1 and audio_grid.dim() == 1, (
            "dynamic-shift sampling grids are not supported: sampling advances "
            "the video and audio grids by a shared step index"
        )
        assert video_grid.numel() == audio_grid.numel(), (
            "sampling_timesteps and audio_sampling_timesteps must have the same "
            f"number of steps, got {video_grid.numel()} and {audio_grid.numel()}"
        )
        num_steps = int(video_grid.numel())
        # The sampler natively accepts a per-sample rng list.
        step_rng = rngs if rngs is not None else rng

        video_xts, audio_xts = video_noises, audio_noises
        refresh_steps = self._condition_cache_refresh_steps(num_steps)
        if refresh_steps and ops.get_rank() == 0:
            logger.info("condition cache refresh steps: %s", sorted(refresh_steps))
        tcache_compute_steps = self._tcache_compute_steps(num_steps, refresh_steps)
        tcache_enabled = len(tcache_compute_steps) < num_steps
        if tcache_enabled and ops.get_rank() == 0:
            logger.info(
                "tcache compute steps: compute=%s skip=%s steps=%s",
                len(tcache_compute_steps),
                num_steps - len(tcache_compute_steps),
                sorted(tcache_compute_steps),
            )
        cached_video_pred: list[torch.Tensor] | None = None
        cached_audio_pred: list[torch.Tensor] | None = None
        tcache_compute_count = 0
        tcache_skip_count = 0
        with _WallTimer(
            "sample_loop",
            cache_enabled=bool(getattr(self, "_condition_kv_cache_enabled", lambda: False)()),
            tcache_enabled=tcache_enabled,
            num_steps=num_steps,
        ):
            for step in range(num_steps):
                self._ref2va_condition_cache_force_refresh = step in refresh_steps
                if self._ref2va_condition_cache_force_refresh:
                    cache_state = getattr(self, "_ref2va_condition_cache_state", None)
                    if isinstance(cache_state, dict):
                        old_kv = cache_state.pop("kv", None)
                        cache_state["ready"] = False
                        del old_kv
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                video_t = video_grid[step].expand(inputs.batch_size).to(device)
                audio_t = audio_grid[step].expand(inputs.batch_size).to(device)
                video_s = self.sampling_timesteps.get_next_timesteps(video_t)
                audio_s = self.audio_sampling_timesteps.get_next_timesteps(audio_t)

                self._h3_hot_timing_step = step
                self._h3_hot_timing_num_steps = num_steps
                if (
                    step in tcache_compute_steps
                    or cached_video_pred is None
                    or cached_audio_pred is None
                ):
                    video_pred, audio_pred = self._bidirectional_forward(
                        model,
                        inputs,
                        video_xts=video_xts,
                        audio_xts=audio_xts,
                        video_timesteps=video_t,
                        audio_timesteps=audio_t,
                    )
                    cached_video_pred = video_pred
                    cached_audio_pred = audio_pred
                    tcache_compute_count += 1
                else:
                    video_pred = cached_video_pred
                    audio_pred = cached_audio_pred
                    tcache_skip_count += 1
                video_xts = self.sampler.step_to(
                    pred=video_pred, x_t=video_xts, t=video_t, s=video_s,
                    rng=step_rng, seqlens=inputs.seqlens,
                )
                audio_xts = self.sampler.step_to(
                    pred=audio_pred, x_t=audio_xts, t=audio_t, s=audio_s,
                    rng=step_rng, seqlens=inputs.seqlens,
                )
        cache_state = getattr(self, "_ref2va_condition_cache_state", None)
        if isinstance(cache_state, dict) and ops.get_rank() == 0:
            logger.info(
                "condition cache summary: fill_count=%s reuse_count=%s ready=%s last_mode=%s",
                cache_state.get("fill_count", 0),
                cache_state.get("reuse_count", 0),
                cache_state.get("ready", False),
                cache_state.get("last_mode", None),
            )
        if tcache_enabled and ops.get_rank() == 0:
            logger.info(
                "tcache summary: compute_count=%s skip_count=%s",
                tcache_compute_count,
                tcache_skip_count,
            )
        self._ref2va_condition_cache_force_refresh = False

        return _RolloutX0s(
            video=video_xts, audio=audio_xts, video_eps=None, audio_eps=None
        )

    # ---------------------------------------------------------- validation --
    @execution_phase(ExecutionPhase.VALIDATION)
    @torch.no_grad()
    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """One bidirectional sample per prompt, then both VAEs and an muxed mp4.

        Same contract as the DMD parent's validate -- every rank walks the same
        number of entries (the split pads by repeating the last one with
        ``should_save=False``), because the sampling is collective under FSDP
        and sequence parallelism; the only difference is ``_sample_latents``
        replacing the chunk-causal rollout.
        """
        config = ctx["config"]
        models = ctx["models"]
        step = int(ctx["step"])
        validation = config.validation
        prompts = self._validation_prompts(validation)
        assert prompts, "validation requires at least one prompt"

        rank = ops.get_rank()
        sp_size = self._sp_size()
        group_index = rank // sp_size
        num_groups = ops.get_world_size() // sp_size
        is_group_main = rank % sp_size == 0

        variants = [("backbone", models["backbone"])]
        if validation.get("validate_ema", False):
            variants.append(("backbone_ema", models["backbone_ema"]))
        media_dir = logger.iter_dir("media", step)
        # Latent mode writes the sampled x0 straight to disk instead of decoding
        # it. That is what a distillation corpus consumes, and it drops both VAEs
        # and the mp4 encode from the per-sample cost.
        save_latent = bool(validation.get("save_latent", False))
        # A corpus does not belong under runs/<exp>/media/<step>/: that path
        # carries the step, so a dump resumed at a different one would scan an
        # empty directory and re-sample all of it. latent_dir pins the location.
        out_dir = media_dir
        if save_latent and validation.get("latent_dir", None):
            out_dir = Path(validation.latent_dir).expanduser()

        # Resume: drop prompts already on disk, and adopt rank 0's view of which
        # those are. A divergent view gives ranks different per-rank counts and
        # desynchronises the collective forwards.
        completed = ops.all_gather_object(
            self._completed_prompt_ids(
                out_dir,
                [name for name, _ in variants],
                suffix="pt" if save_latent else "mp4",
            )
        )[0]
        pending = [(index, prompt) for index, prompt in enumerate(prompts) if index not in completed]
        entries = self._split_validation_prompts(pending, group_index, num_groups)

        original_training = [(model, model.training) for _, model in variants]
        saved_items: list[tuple[str, str, str, str]] = []
        fps = int(validation.get("fps", MINIMAX_H3_SUPPORTED_FPS))
        try:
            for _, model in variants:
                model.eval()
            # Validation batch_size packs that many prompts into ONE sampling
            # forward (training parity: the dataset packs the same way under
            # its max_seqlen budget). 1 keeps the per-prompt path, including
            # the exact per-prompt rng seed.
            batch_size = int(validation.get("batch_size", 1))
            assert batch_size >= 1, f"validation.batch_size must be >= 1, got {batch_size}"
            for batch_start in range(0, len(entries), batch_size):
                batch_entries = entries[batch_start : batch_start + batch_size]
                with _WallTimer("prepare_inputs", batch_size=len(batch_entries)):
                    inputs = self._validation_inputs(
                        config, models, [entry["prompt"] for entry in batch_entries]
                    )
                for variant_name, model in variants:
                    # Per-prompt seeds (the wan/crafter pattern): each prompt
                    # keeps its own deterministic stream inside the batched
                    # sampling, so changing batch_size never perturbs a
                    # prompt's noise. With batch_size 1 this is the historical
                    # per-prompt rng. shift_seed: false pins every prompt to
                    # the same seed (ltx2's option); true (default) shifts by
                    # prompt index.
                    shift_seed = bool(validation.get("shift_seed", True))
                    sample_rngs = [
                        RandomState(
                            combine_seed(
                                int(validation.seed),
                                "validation",
                                entry["prompt_idx"] if shift_seed else 0,
                                variant_name,
                            )
                        )
                        for entry in batch_entries
                    ]
                    latents = self._sample_latents(
                        model, inputs, sample_rngs[0], rngs=sample_rngs
                    )
                    for index, entry in enumerate(batch_entries):
                        if not (is_group_main and entry["should_save"]):
                            continue
                        stem = f"validation/{variant_name}/prompt{entry['prompt_idx']:04d}"
                        if save_latent:
                            name = f"{stem}.pt"
                            path = out_dir / name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            # Write-then-rename: a run killed mid-save would
                            # otherwise leave a truncated .pt that resume counts
                            # as complete, silently seeding the corpus with a
                            # corrupt sample.
                            tmp = path.with_name(path.name + ".tmp")
                            torch.save(
                                {
                                    "video": latents.video[index].to(torch.bfloat16).cpu(),
                                    "audio": latents.audio[index].to(torch.bfloat16).cpu(),
                                    "prompt": entry["prompt"],
                                    "prompt_idx": entry["prompt_idx"],
                                },
                                tmp,
                            )
                            tmp.replace(path)
                            saved_items.append((variant_name, name, str(path), ""))
                            continue
                        with _WallTimer(
                            "decode_latents",
                            variant=variant_name,
                            prompt_idx=entry["prompt_idx"],
                        ):
                            frames, waveform = self._decode_latents(models, latents, index=index)
                        name = f"{stem}.mp4"
                        path = media_dir / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with _WallTimer(
                            "save_video",
                            variant=variant_name,
                            prompt_idx=entry["prompt_idx"],
                        ):
                            media.save_video(
                                frames,
                                path,
                                fps=fps,
                                nrow=int(validation.get("nrow", 1)),
                                crf=int(validation.get("crf", 18)),
                                audio_tensor=waveform,
                                audio_sample_rate=int(models["audio_vae"].sample_rate),
                                normalize=True,
                                # revert_tensor already clamped the decoded pixels.
                                value_range=(0, 1),
                            )
                        saved_items.append(
                            (variant_name, name, str(path), f"[{variant_name}] {entry['prompt']}")
                        )
                        del frames, waveform
                    del latents
                del inputs

            gathered = ops.all_gather_object(saved_items)
            try:
                if rank == 0 and save_latent:
                    # Nothing to preview: no frames were decoded, and a grid of
                    # latents is not a thing. The count is the useful signal.
                    total = sum(len(rank_items) for rank_items in gathered)
                    logger.info(
                        "saved %d latents under %s", total, out_dir / "validation"
                    )
                elif rank == 0:
                    all_items = [item for rank_items in gathered for item in rank_items]
                    for variant_name, _ in variants:
                        variant_items = [item for item in all_items if item[0] == variant_name]
                        if not variant_items:
                            continue
                        log_video = getattr(logger, "log_video", None)
                        if log_video is not None:
                            log_video(
                                {name: Path(path) for _, name, path, _ in variant_items},
                                step=step,
                                collection=f"validation/{variant_name}/videos",
                                captions={name: caption for _, name, _, caption in variant_items},
                                fps=fps,
                            )
                        if not validation.get("save_grid", True):
                            continue
                        grid_name = f"validation/{variant_name}/grid.mp4"
                        grid_path = media_dir / grid_name
                        try:
                            self._save_grid(
                                [Path(path) for _, _, path, _ in variant_items],
                                grid_path,
                                fps=fps,
                                nrow=int(validation.get("nrow", 4)),
                                crf=int(validation.get("crf", 18)),
                            )
                            log_video = getattr(logger, "log_video", None)
                            if log_video is not None:
                                log_video(
                                    {grid_name: grid_path},
                                    step=step,
                                    collection=f"validation/{variant_name}/grid",
                                    fps=fps,
                                )
                        except Exception as exc:
                            # The grid is a preview; the per-prompt files are the
                            # real artifact. Losing a run at step N because a
                            # convenience encode hiccuped is the worse failure.
                            logger.warning("validation grid build failed: %s", exc)
            finally:
                ops.barrier()
        finally:
            for model, training in original_training:
                model.train(training)
        return ctx
