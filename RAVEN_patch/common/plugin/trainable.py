"""Runtime plugins for explicit trainable-parameter selection."""

from __future__ import annotations

from typing import Any

import torch

from ..logging import get_logger
from ..model.runtime import RuntimePlugin

logger = get_logger()


class LoraOnlyTrainable(RuntimePlugin):
    """Freeze a model except PEFT LoRA parameters.

    This is intentionally an after-runtime plugin: adapters are attached before
    runtime, and DefaultRuntime may have just applied a broad requires_grad_()
    setting. Re-applying the selection here leaves placement/optimizer seeing
    exactly the LoRA trainable set.
    """

    _DEFAULT_TOKENS = (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
    )

    def after_runtime(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        if not isinstance(model, torch.nn.Module):
            return state

        tokens = tuple(self.config.get("name_tokens", self._DEFAULT_TOKENS))
        allow_empty = bool(self.config.get("allow_empty", False))
        trainable = 0
        trainable_tensors = 0
        for name, parameter in model.named_parameters():
            enabled = any(token in name for token in tokens)
            parameter.requires_grad_(enabled)
            if enabled:
                trainable += int(parameter.numel())
                trainable_tensors += 1

        if trainable_tensors == 0 and not allow_empty:
            raise RuntimeError(
                f"[{state['name']}] LoraOnlyTrainable found no LoRA parameters; "
                "check the model adapter target_modules/custom_module_mapping"
            )
        logger.info(
            "[%s] LoraOnlyTrainable: %s tensors, %s params",
            state["name"],
            trainable_tensors,
            f"{trainable:,}",
        )
        return state


__all__ = ["LoraOnlyTrainable"]
