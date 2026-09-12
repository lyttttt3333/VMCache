"""Stateful Ref2VA distillation dataset over teacher rollout manifests."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

from torch.utils.data import IterableDataset, get_worker_info

from common.data import (
    WorkerResumeContext,
    WorkerStateEnvelope,
    WorkerStateLoadError,
)
from common.seed import yield_seed


class Ref2VATeacherRolloutDataset(IterableDataset):
    """Infinite stream of Ref2VA cases with teacher x0 latent paths.

    The dataset deliberately does not pack tokens. Ref2VA packing depends on
    runtime model components: reference VAE encoding, Qwen presentation building,
    text encoding, and the exact target geometry. The meta model owns that work.
    """

    _STATE_SCHEMA = "h3_ref2va_teacher_rollout_worker"
    _STATE_VERSION = 1
    _INHERITED_PACKING_ARGS = {
        "audio_latent_channels",
        "height",
        "max_retries",
        "max_seqlen",
        "max_seqlen_per_sample",
        "num_frames",
        "spatial_vae_stride",
        "sync_group_size",
        "tokenizer_path",
        "video_latent_channels",
        "width",
    }

    def __init__(
        self,
        seed: int,
        resume_context: WorkerResumeContext,
        *,
        manifest_path: str,
        require_teacher_latents: bool = True,
        shuffle: bool = True,
        exclude_ids_path: str | None = None,
        exclude_ids: list[str] | None = None,
        **inherited_packing_args: Any,
    ) -> None:
        super().__init__()
        unknown_args = set(inherited_packing_args) - self._INHERITED_PACKING_ARGS
        if unknown_args:
            names = ", ".join(sorted(unknown_args))
            raise TypeError(f"unexpected Ref2VA distill dataset arguments: {names}")
        self.seed = int(seed)
        self.resume_context = resume_context
        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.target = payload["target"]
        excluded = set(str(case_id) for case_id in (exclude_ids or []))
        if exclude_ids_path is not None:
            raw_excluded = json.loads(Path(exclude_ids_path).read_text(encoding="utf-8"))
            if isinstance(raw_excluded, list):
                excluded.update(str(item["id"] if isinstance(item, dict) else item) for item in raw_excluded)
            else:
                excluded.update(str(item) for item in raw_excluded.get("ids", []))
                excluded.update(
                    str(item["id"])
                    for item in raw_excluded.get("cases", [])
                    if isinstance(item, dict) and "id" in item
                )
        self.exclude_ids = excluded
        self.cases = [
            case
            for case in payload["cases"]
            if str(case.get("id")) not in excluded
        ]
        self.shuffle = bool(shuffle)
        if not self.cases:
            raise ValueError(f"{self.manifest_path} contains no cases")
        if require_teacher_latents:
            missing = [
                case["teacher_latent_path"]
                for case in self.cases
                if not Path(case["teacher_latent_path"]).exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} teacher latents are missing; first: {missing[0]}"
                )
        self._decoded_worker_states: dict[int, int] = {}
        for logical_id, raw in resume_context.committed_states.items():
            self._decoded_worker_states[int(logical_id)] = self._decode_worker_state(raw)

    def initial_worker_snapshot(self, logical_worker_id: int) -> bytes:
        return self._encode_worker_state(self._initial_offset(logical_worker_id))

    def _initial_offset(self, logical_worker_id: int) -> int:
        return self.seed + self.resume_context.rank * 1_000_003 + int(logical_worker_id)

    def _encode_worker_state(self, offset: int) -> bytes:
        return pickle.dumps(
            {
                "schema": self._STATE_SCHEMA,
                "version": self._STATE_VERSION,
                "manifest_path": str(self.manifest_path),
                "offset": int(offset),
            }
        )

    def _decode_worker_state(self, raw: bytes) -> int:
        try:
            payload = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001 - normalize dataloader error type
            raise WorkerStateLoadError("invalid Ref2VA worker state") from exc
        if (payload.get("schema"), payload.get("version")) != (
            self._STATE_SCHEMA,
            self._STATE_VERSION,
        ):
            raise WorkerStateLoadError("unsupported Ref2VA worker state")
        if payload.get("manifest_path") != str(self.manifest_path):
            raise WorkerStateLoadError("Ref2VA distill manifest changed")
        return int(payload["offset"])

    def _case_for_offset(self, offset: int) -> dict[str, Any]:
        if self.shuffle:
            rng = random.Random(offset)
            index = rng.randrange(len(self.cases))
        else:
            index = offset % len(self.cases)
        case = dict(self.cases[index])
        case["dataset_index"] = index
        case["target"] = self.target
        return case

    def __iter__(self):
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info else 0
        physical_worker_count = worker_info.num_workers if worker_info else 1
        effective_workers = self.resume_context.num_workers or 1
        if physical_worker_count != effective_workers:
            raise ValueError(
                "worker topology mismatch: context expects "
                f"{effective_workers}, runtime has {physical_worker_count}"
            )
        logical_worker_id = (
            physical_worker_id + self.resume_context.next_logical_worker_id
        ) % physical_worker_count
        offset = self._decoded_worker_states.get(
            logical_worker_id, self._initial_offset(logical_worker_id)
        )
        while True:
            case = self._case_for_offset(offset)
            offset = yield_seed(offset)
            yield WorkerStateEnvelope(
                case,
                logical_worker_id,
                self._encode_worker_state(offset),
            )


__all__ = ["Ref2VATeacherRolloutDataset"]
