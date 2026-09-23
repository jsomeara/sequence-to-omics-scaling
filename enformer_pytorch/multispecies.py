from __future__ import annotations

import torch
from transformers import PreTrainedModel

from .config_enformer import EnformerConfig
from .modeling_enformer import Enformer


class MultiSpeciesEnformer(PreTrainedModel):
    """An Enformer trunk with separate human and mouse prediction heads."""

    config_class = EnformerConfig
    base_model_prefix = "enformer"
    main_input_name = "human_x"

    def __init__(self, config: EnformerConfig) -> None:
        super().__init__(config)
        self.enformer = Enformer(config)

    def forward(
        self,
        human_x: torch.Tensor,
        mouse_x: torch.Tensor,
        human_labels: torch.Tensor | None = None,
        mouse_labels: torch.Tensor | None = None,
    ) -> dict[str, object]:
        human_output = self.enformer(
            human_x,
            labels=human_labels,
            head="human",
            target_length=self.config.target_length,
        )
        mouse_output = self.enformer(
            mouse_x,
            labels=mouse_labels,
            head="mouse",
            target_length=self.config.target_length,
        )

        losses = [
            output.loss
            for output in (human_output, mouse_output)
            if output.loss is not None
        ]
        # The reference SPACE training wrapper adds the independent species
        # Poisson losses for its joint baseline objective.
        loss = torch.stack(losses).sum() if losses else None

        return {
            "loss": loss,
            "logits": (human_output.logits, mouse_output.logits),
        }
