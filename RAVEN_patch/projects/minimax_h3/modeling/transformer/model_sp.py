# SPDX-License-Identifier: Apache-2.0
"""Ulysses sequence parallelism for the bidirectional MiniMax H3 DiT.

``model.py`` was ported from SGLang with its Ulysses control flow left intact
and only the collectives removed: ``_ulysses_ctx()`` is hard-wired to
``(1, 0)`` and the exchange points import from ``sglang...usp``. This module
re-supplies those collectives from this repo's
``common/distributed/unified_parallel``. It builds the full replicated input
embedding, slices that embedding at the SP boundary, and reuses the parent's
row arithmetic for ``inverse_indices`` and ``token_tags``. Only the source of
``(world_size, rank)`` changes; ``_ulysses_ctx`` is never called here.

Two deliberate divergences from ``projects/wan_t2v/modeling/model_sp.py``:

* **RoPE runs before the exchange, on the rank-local rows.** ``rope_cache`` is
  built from ``img_position_ids[:, row_start:row_stop]``, so the rotation
  belongs where those rows still live. wan rotates after the exchange because
  its ``freqs`` are global; copying that here would rotate every row of the
  gathered sequence with this rank's slice of the positions.
* **No sequence padding.** The packer already rounds the packed sequence up to
  ``MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT`` (64), so any world size dividing 64
  splits the rows evenly. wan's ``represented_len``/pad-document machinery has
  nothing to do here, and the packed attention metadata (``cu_seqlens``,
  ``max_seqlen``) stays exactly as the caller built it.

Heads are not padded either: the exchange needs a head count divisible by the
world size, which is the contract ``model.py``'s
``_validate_sequence_parallel_config`` already states, so
``_ulysses_scatter_heads`` enforces it rather than working around it. The state
dict is untouched: the model subclass adds no parameters, buffers or
submodules, and the attention subclass is a pure ``forward`` override, which is
what makes the ``__class__`` swap legal.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from common.distributed.unified_parallel import (
    Gather,
    Slice,
    gather_heads_scatter_seq,
    gather_seq_scatter_heads_qkv,
    get_unified_parallel_group,
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)

from ..checkpointing import maybe_checkpoint
from .config import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)
from .model import (
    _BF16_DTYPE,
    _FORWARD_SUPPORTED_KWARGS,
    _MINIMAX_H3_FLASH_ATTENTION,
    _apply_qk_norm,
    _apply_rope_qk,
    _minimax_h3_attention_core_bcg,
    _minimax_h3_attention_core_impl,
    _required_kwarg,
    _rope_cos_sin_cache,
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
)

_SP_FORWARD_SUPPORTED_KWARGS = _FORWARD_SUPPORTED_KWARGS | {
    "condition_cache_asset_spans",
    "condition_cache_text_len",
}


def _unified_parallel_ctx() -> tuple[int, int]:
    """(world_size, rank) of this repo's unified-parallel group, or (1, 0).

    The seam ``MiniMaxH3DiTModel._sequence_parallel_ctx`` overrides in both SP
    models, so ``build_rope_cache`` slices by the same rank their forwards do
    instead of by the hard-stubbed ``_ulysses_ctx()``.
    """
    if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
        return 1, 0
    return get_unified_parallel_world_size(), get_unified_parallel_rank()


def _ulysses_scatter_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    up_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[T_local, n, d] row shards -> [T_global, n/up_size, d] head shards.

    ``n`` must divide ``up_size``, which is the contract
    ``_validate_sequence_parallel_config`` in ``model.py`` states. H3 has 56
    heads, so the head-divisible sizes are 1, 2, 4, 7, 8, 14, 28, 56; the rows
    are never padded either, so the world size must also divide
    ``MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT`` (64) and the usable sizes are
    1, 2, 4, 8.

    ``seq_dim=0`` because h3's packed sequence is 1-D thd rows with no batch
    dimension, unlike wan's bidirectional ``[B, S, ...]``.
    """
    total, n, d = q.shape
    if n % up_size:
        raise ValueError(
            f"attention heads {n} must be divisible by the unified-parallel "
            f"world size {up_size}. H3's 56 heads divide by 1, 2, 4, 7, 8, 14, "
            "28 and 56, and the packed sequence alignment 64 narrows that to "
            "1, 2, 4, 8."
        )
    inner_dim = n * d
    qkv = gather_seq_scatter_heads_qkv(
        torch.cat(
            [
                q.reshape(total, inner_dim),
                k.reshape(total, inner_dim),
                v.reshape(total, inner_dim),
            ],
            dim=-1,
        ),
        seq_dim=0,
    )
    q, k, v = qkv.split(inner_dim // up_size, dim=-1)
    local_heads = n // up_size
    global_total = q.shape[0]
    return (
        q.view(global_total, local_heads, d),
        k.view(global_total, local_heads, d),
        v.view(global_total, local_heads, d),
    )


def _gather_local_row_counts(local_count: int, *, device: torch.device) -> list[int]:
    group = get_unified_parallel_group()
    world = get_unified_parallel_world_size()
    count = torch.tensor([local_count], device=device, dtype=torch.int64)
    gathered = [torch.empty_like(count) for _ in range(world)]
    dist.all_gather(gathered, count, group=group)
    return [int(item.item()) for item in gathered]


def _ulysses_scatter_heads_varseq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    local_row_counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variable-row version of _ulysses_scatter_heads for target-only cache reads."""
    local_rows = int(q.shape[0])
    max_rows = max(local_row_counts)
    if local_rows < max_rows:
        pad = q.new_zeros((max_rows - local_rows, q.shape[1], q.shape[2]))
        q = torch.cat((q, pad), dim=0)
        k = torch.cat((k, pad), dim=0)
        v = torch.cat((v, pad), dim=0)

    q, k, v = _ulysses_scatter_heads(q, k, v, get_unified_parallel_world_size())
    if min(local_row_counts) == max_rows:
        return q, k, v

    valid_parts = [
        torch.arange(
            rank * max_rows,
            rank * max_rows + count,
            device=q.device,
            dtype=torch.long,
        )
        for rank, count in enumerate(local_row_counts)
        if count
    ]
    if not valid_parts:
        empty = q.new_empty((0, q.shape[1], q.shape[2]))
        return empty, empty, empty
    valid = torch.cat(valid_parts)
    return (
        q.index_select(0, valid),
        k.index_select(0, valid),
        v.index_select(0, valid),
    )


def _all_gather_varseq(x: torch.Tensor, *, local_row_counts: list[int]) -> torch.Tensor:
    group = get_unified_parallel_group()
    world = get_unified_parallel_world_size()
    local_rows = int(x.shape[0])
    max_rows = max(local_row_counts)
    if local_rows < max_rows:
        x = torch.cat(
            (x, x.new_zeros((max_rows - local_rows, *x.shape[1:]))),
            dim=0,
        )
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x.contiguous(), group=group)
    return torch.cat(
        [chunk[:count] for chunk, count in zip(gathered, local_row_counts) if count],
        dim=0,
    )


def _condition_cache_prefix_refresh_blocks(num_blocks: int) -> int:
    raw = os.environ.get("H3_REF2VA_CONDITION_KV_CACHE_PREFIX_REFRESH_BLOCKS", "")
    if not raw.strip():
        return 0
    return max(0, min(num_blocks, int(raw)))


def _split_env_paths(name: str) -> list[Path]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return []
    return [Path(part).expanduser() for part in raw.split(":") if part.strip()]


def _rank_cache_file(root: Path) -> Path:
    return root / f"rank{get_unified_parallel_rank():02d}.pt"


def _export_refcacheblend_parts(
    *,
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    text_len: int,
    asset_spans: torch.Tensor | None,
    condition_prefix_len: int,
) -> None:
    export_root_raw = os.environ.get("H3_REFCACHEBLEND_EXPORT_DIR", "")
    if not export_root_raw.strip():
        return
    if asset_spans is None:
        raise ValueError("RefCacheBlend export requires condition_cache_asset_spans")
    export_root = Path(export_root_raw).expanduser()
    export_root.mkdir(parents=True, exist_ok=True)
    spans = [(int(start), int(stop)) for start, stop in asset_spans.cpu().tolist()]
    if int(text_len) < 0 or int(text_len) > int(condition_prefix_len):
        raise ValueError(
            f"invalid condition_cache_text_len={text_len} for prefix {condition_prefix_len}"
        )
    for start, stop in spans:
        if start < text_len or stop > condition_prefix_len or stop <= start:
            raise ValueError(
                f"invalid RefCacheBlend asset span [{start}, {stop}) for "
                f"text_len={text_len} prefix={condition_prefix_len}"
            )
    layers = []
    for k, v in kv_cache:
        layers.append(
            {
                "text": (
                    k[:text_len].detach().cpu(),
                    v[:text_len].detach().cpu(),
                ),
                "assets": [
                    (
                        k[start:stop].detach().cpu(),
                        v[start:stop].detach().cpu(),
                    )
                    for start, stop in spans
                ],
                "full_prefix_rows": int(condition_prefix_len),
            }
        )
    payload = {
        "format": "h3_refcacheblend_parts_v1",
        "rank": get_unified_parallel_rank(),
        "world_size": get_unified_parallel_world_size(),
        "text_len": int(text_len),
        "asset_spans": spans,
        "condition_prefix_len": int(condition_prefix_len),
        "layers": layers,
    }
    tmp = _rank_cache_file(export_root).with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(_rank_cache_file(export_root))


def _load_refcacheblend_composed_kv(
    *,
    num_layers: int,
    text_len: int,
    asset_spans: torch.Tensor | None,
    condition_prefix_len: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    text_dirs = _split_env_paths("H3_REFCACHEBLEND_IMPORT_TEXT_DIR")
    asset_dirs = _split_env_paths("H3_REFCACHEBLEND_IMPORT_ASSET_DIRS")
    if not text_dirs and not asset_dirs:
        return None
    if len(text_dirs) != 1:
        raise ValueError("H3_REFCACHEBLEND_IMPORT_TEXT_DIR must contain exactly one path")
    if asset_spans is None:
        raise ValueError("RefCacheBlend import requires condition_cache_asset_spans")
    spans = [(int(start), int(stop)) for start, stop in asset_spans.cpu().tolist()]
    if len(asset_dirs) != len(spans):
        raise ValueError(
            "H3_REFCACHEBLEND_IMPORT_ASSET_DIRS count must match asset spans: "
            f"{len(asset_dirs)} != {len(spans)}"
        )

    text_payload = torch.load(
        _rank_cache_file(text_dirs[0]), map_location="cpu", weights_only=False
    )
    asset_payloads = [
        torch.load(_rank_cache_file(root), map_location="cpu", weights_only=False)
        for root in asset_dirs
    ]
    payloads = [text_payload, *asset_payloads]
    for payload in payloads:
        if payload.get("format") != "h3_refcacheblend_parts_v1":
            raise ValueError(f"unsupported RefCacheBlend payload: {payload.get('format')!r}")
        if int(payload.get("rank", -1)) != get_unified_parallel_rank():
            raise ValueError("RefCacheBlend payload rank does not match current rank")
        if int(payload.get("world_size", -1)) != get_unified_parallel_world_size():
            raise ValueError("RefCacheBlend payload world size does not match")
        if len(payload.get("layers", ())) != num_layers:
            raise ValueError("RefCacheBlend payload layer count mismatch")

    composed: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index in range(num_layers):
        text_k, text_v = text_payload["layers"][layer_index]["text"]
        if int(text_k.shape[0]) != int(text_len):
            raise ValueError(
                f"RefCacheBlend text rows mismatch at layer {layer_index}: "
                f"{int(text_k.shape[0])} != {text_len}"
            )
        k_parts = [text_k]
        v_parts = [text_v]
        for payload in asset_payloads:
            assets = payload["layers"][layer_index]["assets"]
            if len(assets) != 1:
                raise ValueError(
                    "single-asset RefCacheBlend import expects payloads with one asset"
                )
            asset_k, asset_v = assets[0]
            k_parts.append(asset_k)
            v_parts.append(asset_v)
        k = torch.cat(k_parts, dim=0)
        v = torch.cat(v_parts, dim=0)
        if int(k.shape[0]) != int(condition_prefix_len):
            raise ValueError(
                f"composed RefCacheBlend rows mismatch at layer {layer_index}: "
                f"{int(k.shape[0])} != {condition_prefix_len}"
            )
        composed.append((k.to(device=device), v.to(device=device)))
    return composed


def _condition_cache_indices(
    *,
    cu_seqlens_host: tuple[int, ...],
    condition_prefix_lens: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.is_tensor(condition_prefix_lens):
        prefixes = [int(value) for value in condition_prefix_lens.view(-1).tolist()]
    else:
        prefixes = [int(value) for value in condition_prefix_lens]
    # Ref2VA validation currently enforces batch_size=1. Keep this first
    # experiment strict so per-sample condition/target interleaving is explicit.
    if len(prefixes) != 1 or len(cu_seqlens_host) != 3:
        raise NotImplementedError(
            "condition KV cache currently supports one Ref2VA sample with one "
            "live segment plus its padding segment"
        )
    live_start, live_stop, _pad_stop = (int(value) for value in cu_seqlens_host)
    prefix = prefixes[0]
    if not 0 < prefix < live_stop - live_start:
        raise ValueError(
            f"condition prefix length {prefix} is invalid for live rows "
            f"{live_stop - live_start}"
        )
    condition = torch.arange(
        live_start, live_start + prefix, device=device, dtype=torch.long
    )
    target = torch.arange(live_start + prefix, live_stop, device=device, dtype=torch.long)
    q_lens = torch.tensor([int(target.numel())], device=device, dtype=torch.int32)
    k_lens = torch.tensor([live_stop - live_start], device=device, dtype=torch.int32)
    return condition, target, q_lens, k_lens, torch.tensor(prefixes, device=device)


def _project_attention_qkv(
    attention: MiniMaxH3Attention,
    x: torch.Tensor,
    *,
    rope_cache: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total, n, d = x.shape[0], attention.num_heads, attention.head_dim
    qkv, _ = attention.qkv_proj(x)
    q, k, v = qkv.split(attention.local_inner_dim, dim=-1)
    q = q.view(total, n, d)
    k = k.view(total, n, d)
    v = v.view(total, n, d)
    cos_sin_cache, positions = rope_cache
    q, k = _apply_qk_norm(q, k, attention.q_norm, attention.k_norm, attention.head_dim)
    q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)
    return q, k, v


_TOKEN_DYNAMICS_TRACE_STATE: dict[tuple[int, str, str], torch.Tensor] = {}
_TOKEN_DYNAMICS_TRACE_FILE: Path | None = None


def _parse_trace_layers(raw: str, num_blocks: int) -> set[int]:
    if not raw.strip():
        return {0, num_blocks // 2, num_blocks - 1}
    layers = set()
    for part in raw.replace(",", ":").split(":"):
        if not part.strip():
            continue
        index = int(part)
        if 0 <= index < num_blocks:
            layers.add(index)
    return layers


def _sample_global_indices(indices: torch.Tensor, max_rows: int) -> torch.Tensor:
    count = int(indices.numel())
    if count <= max_rows:
        return indices
    positions = torch.linspace(
        0,
        count - 1,
        steps=max_rows,
        device=indices.device,
        dtype=torch.float32,
    ).round().to(torch.long)
    return indices.index_select(0, positions)


def _token_dynamics_trace_file(trace_dir: str) -> Path | None:
    global _TOKEN_DYNAMICS_TRACE_FILE
    if _TOKEN_DYNAMICS_TRACE_FILE is not None:
        return _TOKEN_DYNAMICS_TRACE_FILE
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return None
    path = Path(trace_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    _TOKEN_DYNAMICS_TRACE_FILE = path / "rank0_token_dynamics.jsonl"
    return _TOKEN_DYNAMICS_TRACE_FILE


def _trace_token_dynamics(
    *,
    attention: MiniMaxH3Attention,
    q: torch.Tensor,
    v: torch.Tensor,
    condition_global_indices: torch.Tensor | None,
    cu_seqlens_host: tuple[int, ...] | None,
) -> None:
    trace_dir = os.environ.get("H3_REF2VA_TRACE_TOKEN_DYNAMICS_DIR", "").strip()
    if not trace_dir or condition_global_indices is None or cu_seqlens_host is None:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    layer = getattr(attention, "_h3_trace_layer", None)
    num_blocks = int(os.environ.get("H3_REF2VA_TRACE_NUM_BLOCKS", "50"))
    if layer is None or layer not in _parse_trace_layers(
        os.environ.get("H3_REF2VA_TRACE_LAYERS", ""), num_blocks
    ):
        return
    try:
        step = int(os.environ.get("H3_REF2VA_TRACE_STEP", "-1"))
    except ValueError:
        step = -1
    max_rows = max(1, int(os.environ.get("H3_REF2VA_TRACE_MAX_ROWS", "512")))
    live_start, live_stop = int(cu_seqlens_host[0]), int(cu_seqlens_host[1])
    condition_indices = condition_global_indices
    condition_stop = live_start + int(condition_indices.numel())
    target_indices = torch.arange(
        condition_stop,
        live_stop,
        device=condition_indices.device,
        dtype=torch.long,
    )
    if not int(condition_indices.numel()) or not int(target_indices.numel()):
        return

    row: dict[str, Any] = {
        "time": time.time(),
        "step": step,
        "layer": int(layer),
        "condition_rows": int(condition_indices.numel()),
        "target_rows": int(target_indices.numel()),
        "sample_rows": max_rows,
    }
    for tensor_name, tensor in (("q", q), ("v", v)):
        for token_group, indices in (
            ("condition", condition_indices),
            ("target", target_indices),
        ):
            sampled_indices = _sample_global_indices(indices, max_rows)
            sampled = tensor.index_select(0, sampled_indices).detach().to(
                dtype=torch.float32, device="cpu"
            )
            flat = sampled.reshape(-1)
            prefix = f"{token_group}_{tensor_name}"
            row[f"{prefix}_rms"] = float(torch.sqrt(torch.mean(flat * flat)).item())
            prev_key = (int(layer), token_group, tensor_name)
            prev = _TOKEN_DYNAMICS_TRACE_STATE.get(prev_key)
            if prev is not None and prev.shape == sampled.shape:
                delta = sampled - prev
                delta_flat = delta.reshape(-1)
                row[f"{prefix}_delta_rms"] = float(
                    torch.sqrt(torch.mean(delta_flat * delta_flat)).item()
                )
                denom = row[f"{prefix}_rms"] + 1e-8
                row[f"{prefix}_relative_delta"] = float(row[f"{prefix}_delta_rms"] / denom)
                row[f"{prefix}_cosine_distance"] = float(
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        flat, prev.reshape(-1), dim=0
                    ).item()
                )
            else:
                row[f"{prefix}_delta_rms"] = None
                row[f"{prefix}_relative_delta"] = None
                row[f"{prefix}_cosine_distance"] = None
            _TOKEN_DYNAMICS_TRACE_STATE[prev_key] = sampled

    trace_file = _token_dynamics_trace_file(trace_dir)
    if trace_file is not None:
        with trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _validate_local_embedding_layout(
    layout: dict[str, Any],
    *,
    local_seq_len: int,
) -> None:
    """Reject a ``local_embedding_layout`` that is not this rank's.

    ``_embed``'s trusted-layout branch slices the text rows with two plain ints
    and writes them at local rows ``[0, text_rows)``, so a layout built on one
    rank and broadcast to the others is read without complaint and silently
    gives every rank the first rank's text rows.
    """
    text_start = int(layout["text_source_start"])
    text_stop = int(layout["text_source_stop"])
    if not 0 <= text_start <= text_stop:
        raise ValueError(
            "local_embedding_layout text slice must satisfy "
            f"0 <= text_source_start <= text_source_stop, got "
            f"[{text_start}, {text_stop})"
        )
    text_rows = text_stop - text_start
    if text_rows > local_seq_len:
        raise ValueError(
            f"local_embedding_layout carries {text_rows} text rows, more than "
            f"this rank's {local_seq_len} rows. The layout must be built per "
            "rank under sequence parallelism, against that rank's row window."
        )
    for name in ("img_row_ids", "audio_row_ids"):
        row_ids = layout[name]
        if not row_ids.numel():
            continue
        low = int(row_ids.min())
        high = int(row_ids.max())
        # Text owns the leading text_rows local rows, so the latent rows start
        # after them; a layout from another rank lands outside that window.
        if low < text_rows or high >= local_seq_len:
            raise ValueError(
                f"local_embedding_layout {name} spans [{low}, {high}], outside "
                f"this rank's latent rows [{text_rows}, {local_seq_len}). The "
                "layout must be built per rank under sequence parallelism, "
                "against that rank's row window."
            )


class MiniMaxH3AttentionSP(MiniMaxH3Attention):
    """MiniMaxH3Attention with a sequence-to-heads Ulysses exchange.

    Adds no attribute of its own - it is a pure ``forward`` override, which is
    what lets ``MiniMaxH3DiTModelSP`` install it by ``__class__`` assignment.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
        ulysses_active: bool = False,
        condition_cache_mode: str | None = None,
        condition_global_indices: torch.Tensor | None = None,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        local_row_counts: list[int] | None = None,
        target_ordinals: torch.Tensor | None = None,
        q_lens: torch.Tensor | None = None,
        k_lens: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """x: [T_local, hidden] this rank's row shard -> [T_local, hidden].

        qkv projection, q/k RMSNorm and RoPE run on the local rows; the
        all-to-all then trades sequence for heads, so each rank attends the
        whole packed sequence with a slice of the heads and ``cu_seqlens``
        keeps its global packed-document semantics.
        """
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            if condition_cache_mode is not None:
                raise NotImplementedError(
                    "condition KV cache is implemented only for MiniMaxH3AttentionSP"
                )
            return super().forward(
                x,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ulysses_active=ulysses_active,
            )

        q, k, v = _project_attention_qkv(self, x, rope_cache=rope_cache)

        if condition_cache_mode == "reuse":
            if (
                cached_kv is None
                or local_row_counts is None
                or target_ordinals is None
                or q_lens is None
                or k_lens is None
            ):
                raise ValueError("condition cache reuse is missing routing tensors")
            q, k, v = _ulysses_scatter_heads_varseq(
                q,
                k,
                v,
                local_row_counts=local_row_counts,
            )
            cached_k, cached_v = cached_kv
            k = torch.cat((cached_k.to(device=k.device), k), dim=0)
            v = torch.cat((cached_v.to(device=v.device), v), dim=0)
            out = _MINIMAX_H3_FLASH_ATTENTION(
                q,
                k,
                v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )
            local_flat = out.flatten(1)
            gathered = [
                torch.empty_like(local_flat)
                for _ in range(get_unified_parallel_world_size())
            ]
            dist.all_gather(
                gathered, local_flat.contiguous(), group=get_unified_parallel_group()
            )
            full_flat = torch.cat(gathered, dim=-1)
            out = full_flat.index_select(0, target_ordinals)
            out, _ = self.out_proj(out)
            return out

        q, k, v = _ulysses_scatter_heads(q, k, v, get_unified_parallel_world_size())
        if condition_cache_mode == "fill":
            _trace_token_dynamics(
                attention=self,
                q=q,
                v=v,
                condition_global_indices=condition_global_indices,
                cu_seqlens_host=cu_seqlens_host,
            )
        cached_out = None
        if condition_cache_mode == "fill":
            if condition_global_indices is None:
                raise ValueError("condition cache fill is missing condition indices")
            cached_out = (
                k.index_select(0, condition_global_indices).detach(),
                v.index_select(0, condition_global_indices).detach(),
            )
        elif condition_cache_mode is not None:
            raise ValueError(f"unsupported condition cache mode {condition_cache_mode!r}")

        attention_core = (
            _minimax_h3_attention_core_bcg
            if self.bcg_breakpoint
            else _minimax_h3_attention_core_impl
        )
        out = attention_core(
            self,
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            # The exchange is done here with this repo's collectives; the
            # upstream sglang usp branch inside the core stays off.
            ulysses_active=False,
        )

        out = gather_heads_scatter_seq(out.flatten(1), head_dim=1, seq_dim=0)
        out, _ = self.out_proj(out)
        if cached_out is not None:
            return out, cached_out
        return out


class MiniMaxH3DiTModelSP(MiniMaxH3DiTModel):
    """MiniMaxH3DiTModel with the block stack sharded over packed rows.

    ``forward`` is a fork of the parent's for the same reason the causal model
    forks it: the parent keeps its packed forward in one method. It drops the
    branches that are statically dead in this build - the ring-degree guard
    and ``_resolve_attention_backend_once`` (both hard-stubbed), the batched
    block AdaLN and the output-column gathers (both require TP > 1, and
    ``get_tp_world_size`` hard-returns 1) - and replaces the removed sglang
    output all-gather with ``Gather``.
    """

    def __init__(
        self,
        config: Any,
        hf_config: dict[str, Any],
        quant_config: Any = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for block in self.blocks:
            if not isinstance(block.attn, MiniMaxH3Attention):
                raise TypeError(f"Unsupported attention type: {type(block.attn).__name__}")
            block.attn.__class__ = MiniMaxH3AttentionSP
        # The text refiner is deliberately left alone: _embed refines the whole
        # prompt on every rank, so its attention sees no row shard.

    def _sequence_parallel_ctx(self) -> tuple[int, int]:
        """The context ``forward`` shards by, so ``build_rope_cache`` matches it."""
        return _unified_parallel_ctx()

    def _condition_cache_block_stack(
        self,
        hidden: torch.Tensor,
        *,
        cache_state: dict[str, Any],
        signature: tuple[Any, ...],
        condition_global_indices: torch.Tensor,
        target_global_indices: torch.Tensor,
        local_target_positions: torch.Tensor,
        target_ordinals: torch.Tensor,
        local_row_counts: list[int],
        q_lens: torch.Tensor,
        k_lens: torch.Tensor,
        condition_cache_asset_spans: torch.Tensor | None,
        condition_cache_text_len: int,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...],
        max_seqlen: int,
        force_refresh: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        del target_global_indices
        prefix_refresh_blocks = _condition_cache_prefix_refresh_blocks(len(self.blocks))
        if cache_state.get("signature") != signature:
            cache_state.clear()
            cache_state["signature"] = signature
        elif force_refresh and (
            prefix_refresh_blocks <= 0 or prefix_refresh_blocks >= len(self.blocks)
        ):
            old_kv = cache_state.pop("kv", None)
            del old_kv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cache_state["ready"] = False

        cache_ready = bool(cache_state.get("ready", False))
        if not cache_ready:
            imported_kv = _load_refcacheblend_composed_kv(
                num_layers=len(self.blocks),
                text_len=int(condition_cache_text_len),
                asset_spans=condition_cache_asset_spans,
                condition_prefix_len=int(condition_global_indices.numel()),
                device=hidden.device,
            )
            if imported_kv is not None:
                cache_state["kv"] = imported_kv
                cache_state["ready"] = True
                cache_state["last_mode"] = "refcacheblend_import"
                cache_state["refcacheblend_import_count"] = (
                    int(cache_state.get("refcacheblend_import_count", 0)) + 1
                )
                cache_ready = True

        if not cache_ready:
            cache_state["last_mode"] = "fill"
            cache_state["fill_count"] = int(cache_state.get("fill_count", 0)) + 1
            kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
            for index, block in enumerate(self.blocks):
                setattr(block.attn, "_h3_trace_layer", index)
                hidden, cached_kv = block(
                    hidden,
                    adaln_input=adaln_input,
                    combined_indices=combined_indices,
                    rope_cache=rope_cache,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    ulysses_active=False,
                    adaln_params=None,
                    condition_cache_mode="fill",
                    condition_global_indices=condition_global_indices,
                )
                kv_cache.append(cached_kv)

            cache_state["kv"] = kv_cache
            cache_state["ready"] = True
            _export_refcacheblend_parts(
                kv_cache=kv_cache,
                text_len=int(condition_cache_text_len),
                asset_spans=condition_cache_asset_spans,
                condition_prefix_len=int(condition_global_indices.numel()),
            )
            return hidden, False

        kv_cache = cache_state.get("kv")
        if not isinstance(kv_cache, list) or len(kv_cache) != len(self.blocks):
            raise RuntimeError("condition cache is marked ready but has invalid layer KV")

        if force_refresh and 0 < prefix_refresh_blocks < len(self.blocks):
            cache_state["last_mode"] = f"prefix_refresh_{prefix_refresh_blocks}"
            cache_state["prefix_refresh_count"] = (
                int(cache_state.get("prefix_refresh_count", 0)) + 1
            )
            for index, block in enumerate(self.blocks[:prefix_refresh_blocks]):
                setattr(block.attn, "_h3_trace_layer", index)
                hidden, cached_kv = block(
                    hidden,
                    adaln_input=adaln_input,
                    combined_indices=combined_indices,
                    rope_cache=rope_cache,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_host=cu_seqlens_host,
                    max_seqlen=max_seqlen,
                    ulysses_active=False,
                    adaln_params=None,
                    condition_cache_mode="fill",
                    condition_global_indices=condition_global_indices,
                )
                kv_cache[index] = cached_kv
            block_iter = zip(
                self.blocks[prefix_refresh_blocks:],
                kv_cache[prefix_refresh_blocks:],
            )
        else:
            cache_state["last_mode"] = "reuse"
            cache_state["reuse_count"] = int(cache_state.get("reuse_count", 0)) + 1
            block_iter = zip(self.blocks, kv_cache)

        hidden = hidden.index_select(0, local_target_positions)
        combined_indices = combined_indices.index_select(0, local_target_positions)
        cos_sin_cache, _positions = rope_cache
        rope_cache = (
            cos_sin_cache.index_select(0, local_target_positions),
            torch.arange(
                int(local_target_positions.numel()),
                device=hidden.device,
                dtype=torch.long,
            ),
        )

        for block, cached_kv in block_iter:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ulysses_active=False,
                adaln_params=None,
                condition_cache_mode="reuse",
                cached_kv=cached_kv,
                local_row_counts=local_row_counts,
                target_ordinals=target_ordinals,
                q_lens=q_lens,
                k_lens=k_lens,
            )
        return hidden, True

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed forward over this rank's row shard.

        Returns the same global ``(video_logits, audio_logits)`` on every rank:
        the row shards are gathered before the output rows are selected.
        """
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            if kwargs.get("condition_cache_state") is not None and os.environ.get(
                "H3_REF2VA_CONDITION_KV_CACHE_DEBUG"
            ) and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    "[condition-cache-debug] fallback_to_parent_forward "
                    "because unified_parallel is not active",
                    flush=True,
                )
            return super().forward(**kwargs)

        unexpected = sorted(set(kwargs) - _SP_FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModelSP.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_SP_FORWARD_SUPPORTED_KWARGS)}"
            )

        tap_outputs = kwargs.get("tap_outputs")
        tap_blocks = self._validate_taps(kwargs.get("tap_blocks"), tap_outputs)
        classify_mode = bool(kwargs.get("classify_mode", False))

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = (
            _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        )
        update_mask = _required_kwarg(kwargs, "update_mask")
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")

        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(
            _required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_pos = self._pos_ids(
            _required_kwarg(kwargs, "text_pos_info"),
            "text_pos_info",
        )
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(
            torch.int32
        )
        raw_cu_seqlens_host = self._psp_optional_field(psp, "cu_seqlens_q_host")
        cu_seqlens_host = tuple(
            int(value)
            for value in (
                cu_seqlens.tolist()
                if raw_cu_seqlens_host is None
                else raw_cu_seqlens_host
            )
        )
        max_seqlen = int(self._psp_field(psp, "packed_seq_params", "max_seqlen_q"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(
            refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q"
        ).to(torch.int32)
        refiner_max = int(
            self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q")
        )

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(
                "token_tags must cover the full packed sequence "
                f"({seq_len}), got {token_tags.shape[0]}."
            )
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(
                f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}"
            )
        device = x.device

        sp_ws = get_unified_parallel_world_size()
        sp_rank = get_unified_parallel_rank()
        # The packer aligns the packed sequence to
        # MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT, so no padding is needed as long
        # as the world size divides it. Heads must divide it too -- see
        # _ulysses_scatter_heads, which rejects an indivisible head count rather
        # than padding it.
        if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % sp_ws:
            raise ValueError(
                "packed sequence alignment "
                f"{MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT} not divisible by "
                f"unified-parallel world size {sp_ws}"
            )
        if seq_len % sp_ws:
            raise ValueError(
                f"packed seq_len {seq_len} not divisible by unified-parallel "
                f"world size {sp_ws}"
            )
        local_seq_len = seq_len // sp_ws
        row_start = sp_rank * local_seq_len
        row_stop = row_start + local_seq_len

        # RoPE is row-local before Ulysses exchanges sequence for heads inside
        # attention. Input embeddings are instead built in full below and sliced
        # at the explicit SP boundary. Serving normally prepares the request-static
        # cache once; direct model callers use this fallback.
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids[:, row_start:row_stop]).to(device)
            rope_cache = (
                _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
                torch.arange(
                    local_seq_len,
                    device=device,
                    dtype=torch.long,
                ),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        local_embedding_layout = kwargs.get("local_embedding_layout")
        if local_embedding_layout is not None:
            _validate_local_embedding_layout(
                local_embedding_layout, local_seq_len=local_seq_len
            )

        # _embed sits outside the checkpointed block stack, and here it runs over
        # the FULL sequence on every rank -- the Slice below is what shards it.
        # Its two full-length internals, the [seq_len, hidden] zeros buffer and
        # the bf16 video rows, are 611.5 and 603.2 MiB at the smoke's shape and
        # were being saved for backward: DMD2 forwards this model three times per
        # FAKE step, so 3.56 GiB sat live for the whole backward (measured, job
        # 15440105: both sizes allocated 3x and freed 0x). Recomputing them costs
        # one more pass over projections and no attention.
        decoder_input, t_emb = maybe_checkpoint(
            self._embed,
            enabled=self.gradient_checkpointing,
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            row_start=0,
            row_stop=seq_len,
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            local_embedding_layout=None,
        )
        if decoder_input.shape[0] % sp_ws:
            raise ValueError(
                f"full embedding length {decoder_input.shape[0]} not divisible by "
                f"unified-parallel world size {sp_ws}"
            )
        decoder_input = Slice.apply(
            get_unified_parallel_group(), decoder_input, 0, True
        )
        # request-step AdaLN input shared by all blocks
        adaln_input = nn.functional.silu(t_emb).to(_BF16_DTYPE)
        inverse_indices = inverse_indices.to(device)
        block_inverse = inverse_indices[row_start:row_stop]
        if block_token_tags is None:
            assert token_tags is not None
            token_tags = token_tags.to(device)
            block_token_tags = token_tags[row_start:row_stop].clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != local_seq_len:
                raise ValueError(
                    "block_token_tags must cover the rank-local packed sequence "
                    f"({local_seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(
                block_token_tags,
                block_inverse,
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
        elif block_combined.shape[0] != local_seq_len:
            raise ValueError(
                "block_combined_indices must cover the rank-local packed "
                f"sequence ({local_seq_len}), got {block_combined.shape[0]}."
            )

        cu_seqlens = cu_seqlens.to(device)
        condition_cache_state = kwargs.get("condition_cache_state")
        condition_target_only = False
        local_target_positions = None
        target_global_indices = None
        local_row_counts = None
        if condition_cache_state is not None:
            internal_state = getattr(self, "_condition_cache_state_internal", None)
            if internal_state is None:
                internal_state = {}
                self._condition_cache_state_internal = internal_state
            condition_cache_state = internal_state
            if tap_blocks is not None or classify_mode:
                raise NotImplementedError(
                    "condition KV cache is an inference-only path without taps"
                )
            condition_global_indices, target_global_indices, q_lens, k_lens, prefixes = (
                _condition_cache_indices(
                    cu_seqlens_host=cu_seqlens_host,
                    condition_prefix_lens=_required_kwarg(
                        kwargs, "condition_cache_prefix_lens"
                    ),
                    device=device,
                )
            )
            local_target_mask = (target_global_indices >= row_start) & (
                target_global_indices < row_stop
            )
            local_target_global = target_global_indices[local_target_mask]
            local_target_positions = local_target_global - row_start
            target_ordinals = torch.searchsorted(target_global_indices, local_target_global)
            local_row_counts = _gather_local_row_counts(
                int(local_target_positions.numel()), device=device
            )
            signature = (
                cu_seqlens_host,
                tuple(int(value) for value in prefixes.tolist()),
                int(seq_len),
                int(sp_ws),
            )
            hidden, condition_target_only = self._condition_cache_block_stack(
                decoder_input,
                cache_state=condition_cache_state,
                signature=signature,
                condition_global_indices=condition_global_indices,
                target_global_indices=target_global_indices,
                local_target_positions=local_target_positions,
                target_ordinals=target_ordinals,
                local_row_counts=local_row_counts,
                q_lens=q_lens,
                k_lens=k_lens,
                condition_cache_asset_spans=kwargs.get("condition_cache_asset_spans"),
                condition_cache_text_len=int(kwargs.get("condition_cache_text_len", 0)),
                adaln_input=adaln_input,
                combined_indices=block_combined,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                force_refresh=bool(kwargs.get("condition_cache_force_refresh", False)),
            )
            if os.environ.get("H3_REF2VA_CONDITION_KV_CACHE_DEBUG") and (
                not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
            ):
                print(
                    "[condition-cache-debug] "
                    f"mode={condition_cache_state.get('last_mode')} "
                    f"fill_count={condition_cache_state.get('fill_count', 0)} "
                    f"reuse_count={condition_cache_state.get('reuse_count', 0)} "
                    f"prefix_refresh_count={condition_cache_state.get('prefix_refresh_count', 0)} "
                    f"ready={condition_cache_state.get('ready', False)} "
                    f"force_refresh={bool(kwargs.get('condition_cache_force_refresh', False))}",
                    flush=True,
                )
        else:
            hidden = self._checkpoint_block_stack(
                decoder_input,
                tap_blocks=tap_blocks,
                tap_outputs=tap_outputs,
                adaln_input=adaln_input,
                combined_indices=block_combined,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                # MiniMaxH3AttentionSP owns the exchange; this flag only selects the
                # unvendored sglang usp path inside the attention core.
                ulysses_active=False,
                adaln_params=None,
            )

        if classify_mode:
            # Returning before Gather leaves the taps row-sharded, exactly as the
            # discriminator's _ulysses_scatter_kv expects.
            #
            # They leave through the RETURN VALUE, and the `tap_outputs` list is
            # an internal detail of whoever calls this. A caller on the far side
            # of an FSDP boundary cannot use that list at all: FSDP2's
            # pre_forward tree_flattens and tree_unflattens args/kwargs, which
            # rebuilds a list into a NEW object, so the callee appends to the
            # copy and the caller's list stays empty. It only rebuilds when some
            # input tensor requires grad, which is why the FAKE phase (detached
            # latents, early return) worked and GEN (generator output, carries
            # grad) came back with zero taps -- job 15441216.
            #
            # t_emb rides along because the discriminator's AdaLN input is
            # derived from it, and reaching it through a forward hook on
            # time_embedder was the other half of that same reach-across.
            return tuple(tap_outputs or ()), t_emb

        if condition_target_only:
            assert local_target_positions is not None
            assert target_global_indices is not None
            assert local_row_counts is not None
            target_inverse = block_inverse.index_select(0, local_target_positions)
            video_logits, audio_logits = self.final_layer(
                hidden,
                adaln_input=adaln_input,
                inverse_indices=target_inverse,
            )
            video_width = video_logits.shape[-1]
            logits = _all_gather_varseq(
                torch.cat((video_logits, audio_logits), dim=-1),
                local_row_counts=local_row_counts,
            )
            video_logits, audio_logits = logits.split(
                (video_width, logits.shape[-1] - video_width), dim=-1
            )
            video_ordinals = torch.searchsorted(
                target_global_indices, infer_out_pos.to(device)
            )
            if not bool(
                torch.equal(
                    target_global_indices.index_select(0, video_ordinals),
                    infer_out_pos.to(device),
                )
            ):
                raise RuntimeError("condition cache target video rows are not aligned")
            video_logits = video_logits.index_select(0, video_ordinals)

            audio_pos_global = audio_pos.to(device)
            audio_ordinals = torch.searchsorted(target_global_indices, audio_pos_global)
            in_bounds = audio_ordinals < int(target_global_indices.numel())
            matched = torch.zeros_like(in_bounds)
            if bool(in_bounds.any()):
                matched[in_bounds] = (
                    target_global_indices.index_select(0, audio_ordinals[in_bounds])
                    == audio_pos_global[in_bounds]
                )
            full_audio = audio_logits.new_zeros(
                (int(audio_pos_global.numel()), audio_logits.shape[-1])
            )
            if bool(matched.any()):
                full_audio[matched] = audio_logits.index_select(
                    0, audio_ordinals[matched]
                )
            return video_logits, full_audio

        video_logits, audio_logits = self.final_layer(
            hidden,
            adaln_input=adaln_input,
            inverse_indices=block_inverse,
        )
        # One collective for both heads, and before the row selection below:
        # infer_out_pos and audio_pos are global row indexes.
        video_width = video_logits.shape[-1]
        logits = Gather.apply(
            get_unified_parallel_group(),
            torch.cat((video_logits, audio_logits), dim=-1),
            0,
            True,
        )
        video_logits, audio_logits = logits.split(
            (video_width, logits.shape[-1] - video_width), dim=-1
        )

        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(
                    "update_mask length mismatch: "
                    f"{update_mask.shape[0]} != {video_logits.shape[0]}"
                )
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


__all__ = [
    "MiniMaxH3AttentionSP",
    "MiniMaxH3DiTModelSP",
]
