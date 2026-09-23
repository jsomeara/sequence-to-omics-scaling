import math
import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_LENGTH = 131_072
TARGET_LENGTH = 896
MODEL_DIM = 1536

# Width in eighths of the original architecture. All variants retain the same
# depth, 12 attention heads, sequence resolution, and output track counts.
MODEL_WIDTHS = {"tiny": 1, "small": 2, "medium": 4, "large": 6, "full": 8}


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class MultiScaleStem(nn.Module):
    """
    Original multi-scale motif stem.

    This is intentionally unchanged from the strong baseline so this ablation
    only tests tower gating / deepest regional context.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__()
        c_branch = out_channels // 4
        c_rem = out_channels - (c_branch * 3)

        self.branch9 = nn.Conv1d(
            in_channels, c_branch, kernel_size=9, padding=4, bias=False
        )
        self.branch15 = nn.Conv1d(
            in_channels, c_branch, kernel_size=15, padding=7, bias=False
        )
        self.branch25 = nn.Conv1d(
            in_channels, c_branch, kernel_size=25, padding=12, bias=False
        )
        self.branch1 = nn.Conv1d(
            in_channels, c_rem, kernel_size=1, bias=False
        )

        self.post = nn.Sequential(
            nn.SyncBatchNorm(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b9 = self.branch9(x)
        b15 = self.branch15(x)
        b25 = self.branch25(x)
        b1 = self.branch1(x)
        out = torch.cat([b9, b15, b25, b1], dim=1)
        return self.post(out)


class ConvBlock(nn.Module):
    """
    Ungated replacement for the baseline GatedConvBlock.

    The baseline expands dim -> 4*dim and then uses GLU to reduce to 2*dim.
    Here we go directly dim -> 2*dim and use GELU. This preserves:
      * the post-activation hidden width (2*dim)
      * the depthwise kernel and receptive field
      * the residual connection
      * the final projection back to dim

    It removes only the explicit GLU gate and the duplicated pre-gate channels.
    """

    def __init__(self, dim: int, kernel_size: int = 5, expansion: int = 2):
        super().__init__()
        hidden_dim = dim * expansion

        self.net = nn.Sequential(
            nn.SyncBatchNorm(dim),
            nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.SyncBatchNorm(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvTowerStage(nn.Module):
    """
    Baseline tower stage with the same dense Conv5 expansion and pooling,
    but with the refinement GLU removed.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.expand = nn.Sequential(
            nn.SyncBatchNorm(in_channels),
            nn.GELU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
        )
        self.refine = ConvBlock(out_channels, kernel_size=5, expansion=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.refine(x)
        x = self.pool(x)
        return x


class DilatedGatedResBlock(nn.Module):
    """
    Original gated regional block, intentionally retained.

    This lets the experiment test whether GLU is necessary throughout the
    hierarchy without simultaneously stripping gating from the regional stack.
    """

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.SyncBatchNorm(dim),
            nn.Conv1d(dim, dim * 2, kernel_size=1, bias=False),
            nn.GLU(dim=1),
            nn.Conv1d(
                dim,
                dim,
                kernel_size=5,
                dilation=dilation,
                padding=2 * dilation,
                groups=dim,
                bias=False,
            ),
            nn.SyncBatchNorm(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class RelativePositionAttention(nn.Module):
    """Self-attention with learnable relative positional bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_position: int = 1024,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout

        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.max_position = max_position
        self.rel_pos_bias = nn.Embedding(2 * max_position + 1, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        pos = torch.arange(T, device=x.device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        rel_pos = torch.clamp(rel_pos, -self.max_position, self.max_position)
        rel_pos_idx = rel_pos + self.max_position

        bias = self.rel_pos_bias(rel_pos_idx).permute(2, 0, 1).unsqueeze(0)
        scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Original full-capacity Transformer block."""

    def __init__(
        self,
        dim: int = MODEL_DIM,
        num_heads: int = 12,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = RelativePositionAttention(
            dim=dim,
            num_heads=num_heads,
            max_position=1024,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        target_len = self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (seq_len - target_len) // 2
        if trim == 0:
            return x

        return x[:, trim : trim + target_len]


class CustomSeq2Func(nn.Module):
    """Width-scaled custom model with exactly one transformer block.

    ``full`` is the original architecture. Smaller presets scale the stem,
    convolution tower, regional blocks, transformer width, and prediction
    trunk together, keeping the module order and all layer counts unchanged.
    """

    def __init__(self, model_size: str = "full"):
        super().__init__()
        if model_size not in MODEL_WIDTHS:
            raise ValueError(f"Unknown custom model size {model_size!r}; choose from {tuple(MODEL_WIDTHS)}")
        width = MODEL_WIDTHS[model_size]
        self.model_size = model_size
        self.model_dim = model_dim = MODEL_DIM * width // 8
        stem_dim = 256 * width // 8
        tower_dims = [dim * width // 8 for dim in (384, 512, 768, 1024, 1280, MODEL_DIM)]

        self.stem = MultiScaleStem(in_channels=4, out_channels=stem_dim)

        stages = []
        in_channels = stem_dim
        for out_channels in tower_dims:
            stages.append(ConvTowerStage(in_channels, out_channels))
            in_channels = out_channels
        self.conv_tower = nn.Sequential(*stages)

        # Full original regional stack restored. Tower refinement blocks remain
        # ungated; this isolates the simplification that has already held performance.
        self.dilated = nn.Sequential(
            DilatedGatedResBlock(model_dim, 1),
            DilatedGatedResBlock(model_dim, 2),
            DilatedGatedResBlock(model_dim, 4),
            DilatedGatedResBlock(model_dim, 8),
            DilatedGatedResBlock(model_dim, 16),
            DilatedGatedResBlock(model_dim, 32),
        )

        self.transformer = nn.Sequential(
            TransformerBlock(
                dim=model_dim,
                num_heads=12,
                mlp_ratio=2,
                dropout=0.1,
            ),
            nn.LayerNorm(model_dim),
        )

        self.crop = TargetLengthCrop(TARGET_LENGTH)

        # Restored full baseline prediction trunk -- the previous experiment
        # showed this capacity is important.
        self.final_pointwise = nn.Sequential(
            nn.SyncBatchNorm(model_dim),
            nn.GELU(),
            nn.Conv1d(model_dim, 2 * model_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.heads = nn.ModuleDict(
            {
                "human": nn.Sequential(
                    nn.Linear(2 * model_dim, 5313),
                    nn.Softplus(),
                ),
                "mouse": nn.Sequential(
                    nn.Linear(2 * model_dim, 1643),
                    nn.Softplus(),
                ),
            }
        )

    def forward(self, x: torch.Tensor, species: str = "human") -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError(f"Expected input shape [B, L, 4], got {tuple(x.shape)}")

        x = x.transpose(1, 2)       # [B, 4, L]
        x = self.stem(x)            # [B, stem_dim, 65536]
        x = self.conv_tower(x)      # [B, dim, 1024]
        x = self.dilated(x)         # [B, dim, 1024]

        x = x.transpose(1, 2)       # [B, 1024, dim]
        x = self.transformer(x)     # [B, 1024, dim]
        x = self.crop(x)            # [B, 896, dim]

        x = x.transpose(1, 2)       # [B, dim, 896]
        x = self.final_pointwise(x) # [B, 2*dim, 896]
        x = x.transpose(1, 2)       # [B, 896, 2*dim]

        if species not in self.heads:
            raise ValueError(
                f"Unknown species: {species}. Choose from {list(self.heads.keys())}"
            )

        return self.heads[species](x)


if __name__ == "__main__":
    model = CustomSeq2Func()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params / 1e6:.2f}M")

    for sp in ["human", "mouse"]:
        last_linear = None
        for mod in reversed(model.heads[sp]):
            if isinstance(mod, nn.Linear):
                last_linear = mod
                break
        assert last_linear is not None
        print(f"{sp} head out_features: {last_linear.out_features}")

    dummy_in = torch.randn(1, INPUT_LENGTH, 4)
    with torch.no_grad():
        out_human = model(dummy_in, species="human")
        out_mouse = model(dummy_in, species="mouse")
    assert out_human.shape == (1, TARGET_LENGTH, 5313), f"Shape mismatch: {out_human.shape}"
    assert out_mouse.shape == (1, TARGET_LENGTH, 1643), f"Shape mismatch: {out_mouse.shape}"
    assert torch.isfinite(out_human).all(), "Non-finite human output"
    assert torch.isfinite(out_mouse).all(), "Non-finite mouse output"
    print("Forward pass smoke test passed successfully.")
