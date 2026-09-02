# SPDX-License-Identifier: Apache-2.0
"""Native bidirectional Ref2VA validation for MiniMax-H3."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from common.distributed.ops import get_device
from projects.minimax_h3.meta_models.minimax_h3_base import MiniMaxH3Base
from projects.minimax_h3.modeling.packing import (
    minimax_h3_packed_sequence_ref2va_blocks,
)
from projects.minimax_h3.modeling.ref2va_encoder import (
    MINIMAX_H3_QWEN3VL_HIDDEN_DIM,
    MiniMaxH3Ref2VAPresentationProcessor,
    Ref2VAPresentation,
)
from projects.minimax_h3.modeling.ref2va_reference import (
    EncodedReferencePlan,
    encode_ref_block_plan,
    parse_ref_block_plan,
)

_AUDIO_CHANNELS = 2
_VIDEO_PATCH_SIZE = (1, 2, 2)


@dataclass(frozen=True)
class _Ref2VALayout:
    latent_shape: tuple[int, int, int, int]
    audio_shape: tuple[int, int, int]

    @property
    def video_patch_grid(self) -> tuple[int, int, int]:
        _, latent_t, latent_h, latent_w = self.latent_shape
        return latent_t, latent_h // _VIDEO_PATCH_SIZE[1], latent_w // _VIDEO_PATCH_SIZE[2]


@dataclass(frozen=True)
class _Ref2VAInputs:
    batch_size: int
    prompt_embeds: list[torch.Tensor]
    text_lens: list[int]
    seqlens: torch.Tensor
    layouts: list[_Ref2VALayout]
    native: list[dict[str, Any]]
    reference_plans: list[EncodedReferencePlan]
    token_tags: torch.Tensor


def _processor_path(validation: Any) -> str:
    configured = validation.get("processor_path", None)
    if configured is None:
        raise ValueError("validation.processor_path must be supplied explicitly")
    return str(configured)


def _strict_suffix_mask(
    mask: torch.Tensor,
    *,
    target_rows: int,
    name: str,
) -> None:
    assert mask.dtype == torch.bool and mask.ndim == 1, (
        f"{name} must be a one-dimensional bool tensor"
    )
    assert int(mask.sum().item()) == int(target_rows), (
        f"{name} has {int(mask.sum().item())} target rows but expected {target_rows}"
    )
    prefix = int(mask.numel()) - int(target_rows)
    assert not bool(mask[:prefix].any()), f"{name} reference prefix must be false"
    assert bool(mask[prefix:].all()), f"{name} target suffix must be true"


def _target_row_counts(layout: _Ref2VALayout) -> tuple[int, int]:
    latent_t, patch_h, patch_w = layout.video_patch_grid
    return latent_t * patch_h * patch_w, layout.audio_shape[2] * _AUDIO_CHANNELS


def _native_with_qwen_tags(
    *,
    presentation: Ref2VAPresentation,
    plan: EncodedReferencePlan,
    layout: _Ref2VALayout,
    device: torch.device,
) -> dict[str, Any]:
    latent_channels, latent_t, latent_h, latent_w = layout.latent_shape
    audio_channels, _, audio_t = layout.audio_shape
    assert latent_channels * _VIDEO_PATCH_SIZE[1] * _VIDEO_PATCH_SIZE[2] == 96
    assert audio_channels == _AUDIO_CHANNELS

    text_len = int(presentation.input_ids.numel())
    native = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        ref_blocks=plan.ref_blocks,
        audio_channel=_AUDIO_CHANNELS,
    )
    text_pos = native["text_pos"].to(torch.long)
    qwen_tags = presentation.text_token_tags.to(dtype=torch.long, device="cpu")
    assert list(qwen_tags.shape) == [text_len]
    native["token_tags"][text_pos] = qwen_tags
    native = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in native.items()
    }

    target_visual_rows, target_audio_rows = _target_row_counts(layout)
    _strict_suffix_mask(
        native["update_mask"],
        target_rows=target_visual_rows,
        name="update_mask",
    )
    _strict_suffix_mask(
        native["audio_update_mask"],
        target_rows=target_audio_rows,
        name="audio_update_mask",
    )
    assert int(native["img_pos"].numel()) == (
        int(plan.visual_rows.shape[0]) + target_visual_rows
    )
    assert int(native["audio_pos"].numel()) == (
        int(plan.audio_rows.shape[0]) + target_audio_rows
    )
    return native


class MiniMaxH3Ref2VABase(MiniMaxH3Base):
    """Thin native Ref2VA specialization of the released dense H3 sampler."""

    def _condition_kv_cache_enabled(self) -> bool:
        override = getattr(self, "_ref2va_condition_cache_override", None)
        if override is not None:
            return bool(override)
        value = os.environ.get("H3_REF2VA_CONDITION_KV_CACHE", "")
        return value.lower() in {"1", "true", "yes", "on"}

    def _condition_kv_cache_force_refresh(self) -> bool:
        return bool(getattr(self, "_ref2va_condition_cache_force_refresh", False))

    def _validation_inputs(
        self,
        config: Any,
        models: dict[str, Any],
        prompts: Sequence[str],
    ) -> _Ref2VAInputs:
        validation = config.validation
        if self._condition_kv_cache_enabled():
            self._ref2va_condition_cache_state: dict[str, Any] = {}
        assert validation.get("batch_size", None) == 1, (
            "native Ref2VA validation requires validation.batch_size to be "
            "explicitly set to 1"
        )
        assert len(prompts) == 1, (
            f"native Ref2VA validation accepts one prompt per gate, got {len(prompts)}"
        )
        if validation.get("media_probes", None) is None:
            raise ValueError("validation.media_probes must be supplied explicitly")
        if validation.get("resolved_shapes", None) is None:
            raise ValueError("validation.resolved_shapes must be supplied explicitly")

        packer = self._validation_packer(config)
        latent_shape = tuple(int(value) for value in packer.latent_shape)
        audio_shape = tuple(int(value) for value in packer.audio_shape)
        if len(latent_shape) != 4 or len(audio_shape) != 3:
            raise ValueError(
                f"unexpected target geometry {latent_shape!r} and {audio_shape!r}"
            )
        layout = _Ref2VALayout(latent_shape=latent_shape, audio_shape=audio_shape)

        specs = parse_ref_block_plan(
            validation.conditions,
            validation.media_probes,
            resolved_shapes=validation.resolved_shapes,
            target_frame_count=int(validation.num_frames),
        )
        plan = encode_ref_block_plan(
            specs,
            video_vae=models["video_vae"],
            audio_vae=models["audio_vae"],
            target_latent_t=latent_shape[1],
            noise_seed=int(validation.get("ref_noise_seed", validation.seed)),
            visual_anchor=float(
                validation.get("visual_anchor", validation.get(
                    "imgvid_cond_noise_aug_for_inference", 0.999
                ))
            ),
            audio_anchor=float(validation.get("audio_anchor", 1.0)),
        )

        processor = MiniMaxH3Ref2VAPresentationProcessor.from_pretrained(
            _processor_path(validation)
        )
        presentations = [
            processor.build(str(prompt), plan.qwen_media) for prompt in prompts
        ]
        text_lens = [int(item.input_ids.numel()) for item in presentations]
        prompt_embeds = self._encode_prompts(models, presentations, text_lens)

        device = get_device()
        native = [
            _native_with_qwen_tags(
                presentation=presentations[0],
                plan=plan,
                layout=layout,
                device=device,
            )
        ]
        seq_len = int(native[0]["seq_len"])
        assert seq_len <= int(packer.max_seqlen), (
            f"native Ref2VA sequence needs {seq_len} rows but target packer budget "
            f"is {int(packer.max_seqlen)}"
        )
        return _Ref2VAInputs(
            batch_size=1,
            prompt_embeds=[embedding.to(device) for embedding in prompt_embeds],
            text_lens=text_lens,
            seqlens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            layouts=[layout],
            native=native,
            reference_plans=[plan],
            token_tags=torch.cat([entry["token_tags"] for entry in native]),
        )

    def _encode_prompts(
        self,
        models: dict[str, Any],
        text_input_ids: Sequence[Any],
        text_lens: Sequence[int],
    ) -> list[torch.Tensor]:
        assert len(text_input_ids) == len(text_lens)
        encoded: list[torch.Tensor] = []
        for presentation, text_len in zip(text_input_ids, text_lens):
            if not isinstance(presentation, Ref2VAPresentation):
                raise TypeError("native Ref2VA prompt encoding requires presentations")
            assert int(presentation.input_ids.numel()) == int(text_len)
            hidden = models["text_encoder"].encode(presentation)
            assert list(hidden.shape) == [
                int(presentation.input_ids.numel()),
                MINIMAX_H3_QWEN3VL_HIDDEN_DIM,
            ], f"unexpected Ref2VA Qwen hidden shape {list(hidden.shape)}"
            encoded.append(hidden)
        return encoded

    def _bidirectional_forward(
        self,
        model: Any,
        inputs: _Ref2VAInputs,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        device = inputs.token_tags.device
        assert len(video_xts) == len(audio_xts) == inputs.batch_size
        assert len(inputs.native) == len(inputs.reference_plans) == inputs.batch_size
        video_timesteps = video_timesteps.to(device=device, dtype=torch.float32)
        audio_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)

        video_blocks: list[torch.Tensor] = []
        audio_blocks: list[torch.Tensor] = []
        row_timesteps: list[torch.Tensor] = []
        img_pos: list[torch.Tensor] = []
        target_img_pos: list[torch.Tensor] = []
        audio_pos: list[torch.Tensor] = []
        text_pos: list[torch.Tensor] = []
        update_masks: list[torch.Tensor] = []
        audio_update_masks: list[torch.Tensor] = []
        condition_prefix_lens: list[int] = []
        condition_text_lens: list[int] = []
        condition_asset_spans: list[tuple[int, int]] = []
        cu_host: list[int] = [0]
        offset = 0

        for index, (_layout, native, plan) in enumerate(
            zip(inputs.layouts, inputs.native, inputs.reference_plans)
        ):
            seq_len = int(native["seq_len"])
            sample_img = native["img_pos"].to(torch.long)
            sample_audio = native["audio_pos"].to(torch.long)
            visual_mask = native["update_mask"]
            audio_mask = native["audio_update_mask"]
            target_visual_rows = int(visual_mask.sum().item())
            target_audio_rows = int(audio_mask.sum().item())
            used_rows = int(native["cu_seqlens"][1].item())
            condition_prefix = used_rows - target_visual_rows - target_audio_rows
            assert condition_prefix > 0, (
                f"native[{index}] condition prefix must be positive, got "
                f"{condition_prefix}"
            )
            condition_prefix_lens.append(condition_prefix)
            text_len = int(native["text_pos"].numel())
            condition_text_lens.append(text_len)
            asset_cursor = text_len
            for block in plan.blocks:
                block_rows = int(block.audio_rows.shape[0]) + int(block.visual_rows.shape[0])
                if block_rows <= 0:
                    continue
                condition_asset_spans.append(
                    (offset + asset_cursor, offset + asset_cursor + block_rows)
                )
                asset_cursor += block_rows
            assert asset_cursor == condition_prefix, (
                f"native[{index}] RefCacheBlend asset spans cover {asset_cursor} "
                f"rows but condition prefix is {condition_prefix}"
            )
            _strict_suffix_mask(
                visual_mask,
                target_rows=target_visual_rows,
                name=f"native[{index}].update_mask",
            )
            _strict_suffix_mask(
                audio_mask,
                target_rows=target_audio_rows,
                name=f"native[{index}].audio_update_mask",
            )

            target_video = self._video_rows(
                video_xts[index], 0, video_xts[index].shape[1]
            )
            target_audio = self._audio_rows(
                audio_xts[index], 0, audio_xts[index].shape[2]
            )
            assert int(target_video.shape[0]) == target_visual_rows
            assert int(target_audio.shape[0]) == target_audio_rows
            visual_rows = torch.cat(
                [
                    plan.visual_rows.to(
                        device=device, dtype=target_video.dtype
                    ),
                    target_video,
                ],
                dim=0,
            )
            audio_rows = torch.cat(
                [
                    plan.audio_rows.to(
                        device=device, dtype=target_audio.dtype
                    ),
                    target_audio,
                ],
                dim=0,
            )
            assert int(visual_rows.shape[0]) == int(sample_img.numel())
            assert int(audio_rows.shape[0]) == int(sample_audio.numel())

            video_blocks.append(
                target_video.new_zeros((seq_len, int(target_video.shape[1]))).index_copy(
                    0, sample_img, visual_rows
                )
            )
            audio_blocks.append(
                target_audio.new_zeros((seq_len, int(target_audio.shape[1]))).index_copy(
                    0, sample_audio, audio_rows
                )
            )

            video_t = video_timesteps[index]
            audio_t = audio_timesteps[index]
            sample_t = video_t.expand(seq_len).clone()
            ref_visual_rows = int(plan.visual_rows.shape[0])
            ref_audio_rows = int(plan.audio_rows.shape[0])
            if ref_visual_rows:
                visual_anchors = plan.visual_row_anchors.to(device=device)
                assert list(visual_anchors.shape) == [ref_visual_rows]
                sample_t[sample_img[:ref_visual_rows]] = torch.minimum(
                    video_t.expand(ref_visual_rows), 1.0 - visual_anchors
                )
            sample_t[sample_img[ref_visual_rows:]] = video_t
            if ref_audio_rows:
                audio_anchors = plan.audio_row_anchors.to(device=device)
                assert list(audio_anchors.shape) == [ref_audio_rows]
                sample_t[sample_audio[:ref_audio_rows]] = torch.minimum(
                    audio_t.expand(ref_audio_rows), 1.0 - audio_anchors
                )
            sample_t[sample_audio[ref_audio_rows:]] = audio_t
            row_timesteps.append(sample_t)

            img_pos.append(sample_img + offset)
            target_img_pos.append(sample_img[visual_mask] + offset)
            audio_pos.append(sample_audio + offset)
            text_pos.append(native["text_pos"].to(torch.long) + offset)
            update_masks.append(visual_mask)
            audio_update_masks.append(audio_mask)
            cu_host.extend(
                int(value) + offset for value in native["cu_seqlens"][1:].tolist()
            )
            offset += seq_len

        packed_cu_host = tuple(cu_host)
        packed_cu = torch.tensor(packed_cu_host, dtype=torch.int32, device=device)
        max_seqlen = max(
            stop - start
            for start, stop in zip(packed_cu_host[:-1], packed_cu_host[1:])
        )
        all_img_pos = torch.cat(img_pos)
        infer_out_pos = torch.cat(target_img_pos)
        packed_update_mask = torch.cat(update_masks)
        assert int(infer_out_pos.numel()) == int(packed_update_mask.sum().item()), (
            "target-only infer_out_pos must contain exactly the update_mask target rows"
        )
        assert int(infer_out_pos.numel()) != int(packed_update_mask.numel()), (
            "target-only infer_out_pos must differ from the full golden-builder "
            "update_mask when Ref2VA reference rows are present"
        )
        all_audio_pos = torch.cat(audio_pos)
        x = torch.cat(video_blocks).unsqueeze(0).to(dtype=model.param_dtype)
        audio_x = torch.cat(audio_blocks).unsqueeze(0).to(dtype=model.param_dtype)
        kwargs = self._common_kwargs(
            inputs,
            x=x,
            audio_x=audio_x,
            eps=torch.zeros_like(x),
            audio_eps=torch.zeros_like(audio_x),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=torch.cat(
                [entry["img_position_ids"] for entry in inputs.native], dim=0
            ),
            token_tags=torch.cat(
                [entry["token_tags"] for entry in inputs.native], dim=0
            ),
            img_pos=all_img_pos,
            audio_pos=all_audio_pos,
            text_pos=torch.cat(text_pos),
            infer_out_pos=infer_out_pos,
        )
        kwargs.update(
            update_mask=packed_update_mask,
            update_audio_mask=torch.cat(audio_update_masks),
            # Target-only infer_out_pos has a different length from the full
            # golden-builder update_mask and therefore requires skip masking.
            skip_mask_out_condition=True,
            clean_timesteps_by_tag=(1.0, 0.999, 1.0),
            packed_seq_params={
                "cu_seqlens_q": packed_cu,
                "cu_seqlens_q_host": packed_cu_host,
                "max_seqlen_q": max_seqlen,
            },
        )
        if self._condition_kv_cache_enabled():
            cache_state = getattr(self, "_ref2va_condition_cache_state", None)
            if cache_state is None:
                cache_state = {}
                self._ref2va_condition_cache_state = cache_state
            kwargs.update(
                condition_cache_state=cache_state,
                condition_cache_prefix_lens=torch.tensor(
                    condition_prefix_lens,
                    dtype=torch.long,
                    device=device,
                ),
                condition_cache_text_len=condition_text_lens[0],
                condition_cache_asset_spans=torch.tensor(
                    condition_asset_spans,
                    dtype=torch.long,
                    device=device,
                ),
                condition_cache_force_refresh=self._condition_kv_cache_force_refresh(),
            )
        video_logits, audio_logits = model(**kwargs)

        video_out: list[torch.Tensor] = []
        audio_out: list[torch.Tensor] = []
        video_cursor = 0
        audio_cursor = 0
        for index, (layout, native) in enumerate(zip(inputs.layouts, inputs.native)):
            visual_mask = native["update_mask"]
            audio_mask = native["audio_update_mask"]
            target_video_rows = int(visual_mask.sum().item())
            target_audio_rows = int(audio_mask.sum().item())
            _strict_suffix_mask(
                visual_mask,
                target_rows=target_video_rows,
                name=f"output[{index}].update_mask",
            )
            _strict_suffix_mask(
                audio_mask,
                target_rows=target_audio_rows,
                name=f"output[{index}].audio_update_mask",
            )
            video_rows = video_logits[
                video_cursor : video_cursor + target_video_rows
            ]
            sample_audio_rows = int(audio_mask.numel())
            all_sample_audio = audio_logits[
                audio_cursor : audio_cursor + sample_audio_rows
            ]
            target_audio = all_sample_audio[
                sample_audio_rows - target_audio_rows :
            ]
            assert int(video_rows.shape[0]) == target_video_rows
            assert int(target_audio.shape[0]) == target_audio_rows
            video_out.append(self._video_from_rows(video_rows, layout))
            audio_out.append(self._audio_from_rows(target_audio, layout))
            video_cursor += target_video_rows
            audio_cursor += sample_audio_rows

        assert video_cursor == int(video_logits.shape[0])
        assert audio_cursor == int(audio_logits.shape[0])
        return video_out, audio_out


__all__ = ["MiniMaxH3Ref2VABase"]
