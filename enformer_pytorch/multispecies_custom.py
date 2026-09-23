from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig

from .modeling_custom import CustomSeq2Func


class CustomModelConfig(PretrainedConfig):
    """
    Minimal Hugging Face-compatible config for CustomSeq2Func.

    This is NOT EnformerConfig. It exists only so Hugging Face Trainer and
    related utilities can serialize/read model metadata through model.config.
    """

    model_type = "custom_seq2func"

    def __init__(
        self,
        output_heads: dict[str, int] | None = None,
        target_length: int = 896,
        sequence_length: int = 131072,
        model_size: str = "full",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.output_heads = (
            {"human": 5313, "mouse": 1643}
            if output_heads is None
            else output_heads
        )
        self.target_length = target_length
        self.sequence_length = sequence_length
        self.model_size = model_size


def poisson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-20,
) -> torch.Tensor:
    """
    Enformer-style Poisson loss:

        mean(pred - target * log(pred))

    The custom model ends each species head in Softplus, so predictions
    are non-negative. The clamp is retained for numerical safety.
    """
    return (pred - target * torch.log(pred.clamp(min=eps))).mean()


class MultiSpeciesCustom(nn.Module):
    """
    Joint human/mouse training wrapper for CustomSeq2Func.

    The actual architecture lives entirely in modeling_custom.py.
    No EnformerConfig or old Enformer model is used.

    `config` is a small generic Hugging Face PretrainedConfig used only for
    compatibility with Trainer/checkpoint serialization and existing training
    code that reads model.config.output_heads / target_length.
    """

    main_input_name = "human_x"

    def __init__(self, model_size: str = "full") -> None:
        super().__init__()

        self.model = CustomSeq2Func(model_size=model_size)

        self.config = CustomModelConfig(
            output_heads={
                "human": self._head_output_features("human"),
                "mouse": self._head_output_features("mouse"),
            },
            target_length=self._target_length(),
            sequence_length=131072,
            model_size=model_size,
            model_dim=self.model.model_dim,
            transformer_blocks=1,
            attention_heads=12,
        )

    def set_track_indices(self, indices: dict[str, list[int]]) -> None:
        """Restrict supervision/metrics, retaining full heads and initialization.

        Nonpersistent buffers move with the model but leave checkpoint weight
        keys unchanged. The run/checkpoint manifest records their values.
        """
        for species in ("human", "mouse"):
            selected = indices[species]
            total = self.config.output_heads[species]
            if not selected or selected != sorted(set(selected)) or selected[0] < 0 or selected[-1] >= total:
                raise ValueError(f"Invalid {species} track indices.")
            value = None if selected == list(range(total)) else torch.tensor(selected, dtype=torch.long)
            self.register_buffer(f"_{species}_track_indices", value, persistent=False)

    def _head_output_features(self, species: str) -> int:
        head = self.model.heads[species]

        for module in reversed(head):
            if isinstance(module, nn.Linear):
                return module.out_features

        raise RuntimeError(
            f"Could not determine output dimension for {species!r} head."
        )

    def _target_length(self) -> int:
        if hasattr(self.model, "crop") and hasattr(self.model.crop, "target_length"):
            return int(self.model.crop.target_length)

        raise RuntimeError(
            "Could not determine target length from CustomSeq2Func.crop."
        )

    def forward(
        self,
        human_x: torch.Tensor,
        mouse_x: torch.Tensor,
        human_labels: torch.Tensor | None = None,
        mouse_labels: torch.Tensor | None = None,
    ) -> dict[str, object]:
        human_logits = self.model(human_x, species="human")
        mouse_logits = self.model(mouse_x, species="mouse")

        # Full-track runs take the original path without indexing or new RNG calls.
        human_indices = getattr(self, "_human_track_indices", None)
        mouse_indices = getattr(self, "_mouse_track_indices", None)
        if human_indices is not None:
            human_logits = human_logits.index_select(-1, human_indices)
            if human_labels is not None:
                human_labels = human_labels.index_select(-1, human_indices)
        if mouse_indices is not None:
            mouse_logits = mouse_logits.index_select(-1, mouse_indices)
            if mouse_labels is not None:
                mouse_labels = mouse_labels.index_select(-1, mouse_indices)

        losses: list[torch.Tensor] = []

        if human_labels is not None:
            if human_logits.shape != human_labels.shape:
                raise ValueError(
                    "Human prediction/label shape mismatch: "
                    f"{tuple(human_logits.shape)} vs "
                    f"{tuple(human_labels.shape)}"
                )
            losses.append(poisson_loss(human_logits, human_labels))

        if mouse_labels is not None:
            if mouse_logits.shape != mouse_labels.shape:
                raise ValueError(
                    "Mouse prediction/label shape mismatch: "
                    f"{tuple(mouse_logits.shape)} vs "
                    f"{tuple(mouse_labels.shape)}"
                )
            losses.append(poisson_loss(mouse_logits, mouse_labels))

        loss = torch.stack(losses).sum() if losses else None

        return {
            "loss": loss,
            "logits": (human_logits, mouse_logits),
        }


# Compatibility alias for code importing the historical class name.
MultiSpeciesEnformer = MultiSpeciesCustom
