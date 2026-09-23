from __future__ import annotations

from transformers import PretrainedConfig


class EnformerConfig(PretrainedConfig):
    """Local Enformer architecture defaults, matching the published config.

    No hosted config is needed. RoPE is an optional local extension; the
    published architecture uses relative positional features (rope=False).
    """

    model_type = "enformer"

    def __init__(
        self,
        dim: int = 1536,
        depth: int = 11,
        heads: int = 8,
        output_heads: dict[str, int] | None = None,
        target_length: int = 896,
        attn_dim_key: int = 64,
        dropout_rate: float = 0.4,
        attn_dropout: float = 0.05,
        pos_dropout: float = 0.01,
        use_checkpointing: bool = False,
        num_downsamples: int = 7,
        dim_divisible_by: int = 128,
        use_tf_gamma: bool = False,
        rope: bool = False,
        rope_theta: float = 10_000.0,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads or {"human": 5313, "mouse": 1643}
        self.target_length = target_length
        self.attn_dim_key = attn_dim_key
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.use_checkpointing = use_checkpointing
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.use_tf_gamma = use_tf_gamma
        self.rope = bool(rope)
        self.rope_theta = float(rope_theta)

        super().__init__(**kwargs)
