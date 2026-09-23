from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from enformer_pytorch.modeling_custom import MODEL_WIDTHS

from scripts.track_selection import select_tracks, save_manifest, validate_resume

from dataloaders import MultiSpeciesEnformerDataset
from enformer_pytorch import EnformerConfig, MultiSpeciesEnformer, MultiSpeciesCustom


@dataclass(frozen=True)
class TrainingSchedule:
    """Resolved schedule after translating the four-GPU baseline to this run."""

    num_gpus: int
    target_global_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    actual_global_batch_size: int
    sample_scale: float
    max_steps: int
    warmup_steps: int
    eval_steps: int
    logging_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a human+mouse custom or Enformer model on the SPACE genomic "
            "profile targets."
        )
    )

    # Data and model configuration
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./DATASET"),
        help="Directory containing the downloaded human and mouse data.",
    )
    parser.add_argument("--human-genome", type=Path, default=None)
    parser.add_argument("--mouse-genome", type=Path, default=None)
    for species in ("human", "mouse"):
        for split in ("train", "valid", "test"):
            parser.add_argument(f"--{species}-{split}-h5", type=Path, default=None)
            parser.add_argument(f"--{species}-{split}-bed", type=Path, default=None)
    parser.add_argument("--seqlen", type=int, default=131072)
    parser.add_argument("--target-length", type=int, default=896)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of the training dataset to use. For example, 0.1 uses a "
            "deterministic random 10%% subset. Subsets are nested for a fixed seed."
        ),
    )
    parser.add_argument(
        "--model", choices=("custom", "enformer"), default="custom",
        help="Architecture to train from random weights (default: custom).",
    )
    parser.add_argument(
        "--model-size", choices=tuple(MODEL_WIDTHS), default="full",
        help="Custom width preset; full preserves the original model. All sizes use one transformer block.",
    )
    parser.add_argument(
        "--model-config", default=None,
        help="Local Enformer JSON file or directory containing config.json; defaults to built-in values.",
    )
    parser.add_argument(
        "--rope", action=argparse.BooleanOptionalAction, default=None,
        help="Override Enformer config's rotary position encoding setting.",
    )
    parser.add_argument(
        "--rope-theta", type=float, default=None,
        help="Override Enformer config's rotary frequency base.",
    )
    parser.add_argument("--human-targets", type=int, default=None)
    parser.add_argument("--mouse-targets", type=int, default=None)
    parser.add_argument(
        "--use-checkpointing",
        action="store_true",
        help="Enable Enformer activation checkpointing; unused by the custom model.",
    )
    parser.add_argument(
        "--shift-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly shift training windows by up to three bases.",
    )
    parser.add_argument(
        "--rc-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly reverse-complement training examples and targets.",
    )

    # GPU-aware scaling. The defaults reproduce the upstream Enformer reference:
    # 4 GPUs * 8 examples/GPU * 1 accumulation step = global batch 32.
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override detected training GPU/process count for schedule calculation.",
    )
    parser.add_argument("--reference-gpus", type=int, default=4)
    parser.add_argument("--reference-per-device-batch-size", type=int, default=8)
    parser.add_argument("--reference-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--target-global-batch-size", type=int, default=None)
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size; the default is memory-conscious and uses accumulation.",
    )
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--reference-max-steps", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--reference-warmup-steps", type=int, default=5_000)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--reference-eval-save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--reference-logging-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=None)

    # Hugging Face Trainer configuration
    parser.add_argument("--output-dir", type=Path, default=Path("./OUTPUTS/enformer"))
    parser.add_argument("--run-name", default="enformer-human-mouse-baseline")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--report-to",
        default="wandb",
        help="Comma-separated Trainer integrations; use 'none' to disable logging.",
    )
    parser.add_argument("--wandb-project", default="enformer-baseline")
    parser.add_argument("--wandb-dir", type=Path, default=Path("./wandb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--track-fraction", type=float, default=1.0,
                        help="Fraction of tracks supervised per species (default: all).")
    parser.add_argument("--track-seed", type=int, default=None,
                        help="Track selection seed; defaults to --seed without changing training randomness.")
    parser.add_argument("--human-track-count", type=int, default=None)
    parser.add_argument("--mouse-track-count", type=int, default=None)

    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16, matching the upstream GPU training command.",
    )
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory to resume, or omit to start from random weights.",
    )
    return parser.parse_args()


def data_path(value: Path | None, data_dir: Path, default_name: str) -> Path:
    return value if value is not None else data_dir / default_name


def resolve_data_paths(args: argparse.Namespace) -> dict[str, Path]:
    data_dir = args.data_dir
    return {
        "human_genome": args.human_genome or data_dir / "genome" / "hg38.fa",
        "mouse_genome": args.mouse_genome or data_dir / "genome" / "mm10.fa",
        "human_train_h5": data_path(args.human_train_h5, data_dir, "human_train.h5"),
        "human_valid_h5": data_path(args.human_valid_h5, data_dir, "human_valid.h5"),
        "human_test_h5": data_path(args.human_test_h5, data_dir, "human_test.h5"),
        "human_train_bed": data_path(args.human_train_bed, data_dir, "human_train.bed"),
        "human_valid_bed": data_path(args.human_valid_bed, data_dir, "human_valid.bed"),
        "human_test_bed": data_path(args.human_test_bed, data_dir, "human_test.bed"),
        "mouse_train_h5": data_path(args.mouse_train_h5, data_dir, "mouse_train.h5"),
        "mouse_valid_h5": data_path(args.mouse_valid_h5, data_dir, "mouse_valid.h5"),
        "mouse_test_h5": data_path(args.mouse_test_h5, data_dir, "mouse_test.h5"),
        "mouse_train_bed": data_path(args.mouse_train_bed, data_dir, "mouse_train.bed"),
        "mouse_valid_bed": data_path(args.mouse_valid_bed, data_dir, "mouse_valid.bed"),
        "mouse_test_bed": data_path(args.mouse_test_bed, data_dir, "mouse_test.bed"),
    }


def detect_gpu_count(requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--num-gpus must be at least 1.")
        return requested

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if requested is not None and requested != world_size:
        raise ValueError(
            f"--num-gpus={requested}, but distributed WORLD_SIZE={world_size}. "
            "Launch with torchrun/accelerate using the requested number of processes."
        )
    return world_size


def positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1; got {value}.")


def calculate_schedule(args: argparse.Namespace, num_gpus: int) -> TrainingSchedule:
    for name, value in (
        ("--reference-gpus", args.reference_gpus),
        ("--reference-per-device-batch-size", args.reference_per_device_batch_size),
        ("--reference-gradient-accumulation-steps", args.reference_gradient_accumulation_steps),
        ("--per-device-train-batch-size", args.per_device_train_batch_size),
        ("--reference-max-steps", args.reference_max_steps),
        ("--reference-warmup-steps", args.reference_warmup_steps),
        ("--reference-eval-save-steps", args.reference_eval_save_steps),
        ("--reference-logging-steps", args.reference_logging_steps),
    ):
        positive(name, value)

    reference_global_batch = (
        args.target_global_batch_size
        if args.target_global_batch_size is not None
        else args.reference_gpus
        * args.reference_per_device_batch_size
        * args.reference_gradient_accumulation_steps
    )
    positive("--target-global-batch-size", reference_global_batch)

    samples_per_accumulation = num_gpus * args.per_device_train_batch_size
    if args.gradient_accumulation_steps is None:
        gradient_accumulation_steps = math.ceil(
            reference_global_batch / samples_per_accumulation
        )
    else:
        positive("--gradient-accumulation-steps", args.gradient_accumulation_steps)
        gradient_accumulation_steps = args.gradient_accumulation_steps

    actual_global_batch = samples_per_accumulation * gradient_accumulation_steps
    sample_scale = reference_global_batch / actual_global_batch

    def scaled(reference: int, override: int | None) -> int:
        if override is not None:
            positive("schedule override", override)
            return override
        return max(1, math.ceil(reference * sample_scale))

    if args.eval_steps is not None and args.save_steps is not None:
        if args.eval_steps != args.save_steps:
            raise ValueError(
                "--eval-steps and --save-steps must match so the latest model "
                "is saved at every evaluation."
            )
    event_steps = scaled(
        args.reference_eval_save_steps,
        args.eval_steps if args.eval_steps is not None else args.save_steps,
    )

    eval_batch_size = (
        args.per_device_eval_batch_size
        if args.per_device_eval_batch_size is not None
        else args.per_device_train_batch_size
    )
    positive("--per-device-eval-batch-size", eval_batch_size)

    return TrainingSchedule(
        num_gpus=num_gpus,
        target_global_batch_size=reference_global_batch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        actual_global_batch_size=actual_global_batch,
        sample_scale=sample_scale,
        max_steps=scaled(args.reference_max_steps, args.max_steps),
        warmup_steps=scaled(args.reference_warmup_steps, args.warmup_steps),
        eval_steps=event_steps,
        logging_steps=scaled(args.reference_logging_steps, args.logging_steps),
    )


def build_model(args: argparse.Namespace) -> MultiSpeciesCustom | MultiSpeciesEnformer:
    if args.model == "custom":
        # Preserve the original initialization path and RNG consumption exactly.
        return MultiSpeciesCustom(model_size=args.model_size)

    if args.model_size != "full":
        raise ValueError("--model-size applies only to the custom model.")

    config_values = {}
    if args.model_config is not None and args.model_config.lower() not in {"local", "none"}:
        config_path = Path(args.model_config)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        # Read JSON directly: model IDs and remote config downloads are unsupported.
        with config_path.open() as config_file:
            config_values = json.load(config_file)
    config = EnformerConfig(**config_values)
    config.target_length = args.target_length
    config.use_checkpointing = args.use_checkpointing
    config.output_heads = {
        "human": args.human_targets if args.human_targets is not None else config.output_heads["human"],
        "mouse": args.mouse_targets if args.mouse_targets is not None else config.output_heads["mouse"],
    }
    for species, targets in config.output_heads.items():
        positive(f"--{species}-targets", targets)
    if args.rope is not None:
        config.rope = args.rope
    if args.rope_theta is not None:
        config.rope_theta = args.rope_theta
    if config.rope_theta <= 0 or not math.isfinite(config.rope_theta):
        raise ValueError("Enformer rope_theta must be finite and positive.")
    return MultiSpeciesEnformer(config)


def trainer_report_to(value: str) -> list[str]:
    if value.strip().lower() in {"", "none", "null"}:
        return []
    return [integration.strip() for integration in value.split(",") if integration.strip()]


def configure_wandb(args: argparse.Namespace, report_to: list[str]) -> None:
    if "wandb" not in {integration.lower() for integration in report_to}:
        return
    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_DIR"] = str(args.wandb_dir)


def build_training_args(
    args: argparse.Namespace,
    schedule: TrainingSchedule,
) -> TrainingArguments:
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16.")

    report_to = trainer_report_to(args.report_to)
    configure_wandb(args, report_to)
    training_kwargs: dict[str, Any] = dict(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        max_steps=schedule.max_steps,
        per_device_train_batch_size=schedule.per_device_train_batch_size,
        per_device_eval_batch_size=schedule.per_device_eval_batch_size,
        gradient_accumulation_steps=schedule.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=schedule.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=schedule.logging_steps,
        save_steps=schedule.eval_steps,
        eval_steps=schedule.eval_steps,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_human_pearson",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        save_safetensors=False,
        dataloader_num_workers=args.dataloader_num_workers,

        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if args.dataloader_num_workers > 0 else None,

        remove_unused_columns=False,
        report_to=report_to,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        label_names=["human_labels", "mouse_labels"],
    )
    parameters = inspect.signature(TrainingArguments).parameters
    strategy_parameter = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    training_kwargs[strategy_parameter] = "steps"
    if "batch_eval_metrics" not in parameters:
        raise RuntimeError(
            "This Transformers version does not support streaming evaluation "
            "metrics; install the project dependencies with uv sync."
        )
    training_kwargs["batch_eval_metrics"] = True
    return TrainingArguments(**training_kwargs)


def _species_pair(value: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Expected human and mouse predictions/labels.")

    def to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    return to_numpy(value[0]), to_numpy(value[1])


class StreamingPearson:
    """
    Enformer-style streaming Pearson correlation.

    For predictions/labels shaped [B, bins, tracks], compute one Pearson
    correlation per track over the pooled (examples x genomic bins) axis,
    then average the per-track correlations.

    This matches the intended Enformer evaluation semantics:
        PearsonR(reduce_axis=(0, 1))
    """

    def __init__(self, track_indices: dict[str, list[int]] | None = None) -> None:
        self.track_indices = track_indices
        self.reset()

    def reset(self) -> None:
        self.stats: dict[str, dict[str, Any]] = {}

    def _ensure_species(self, species: str, num_tracks: int) -> None:
        if species in self.stats:
            return

        self.stats[species] = {
            "n": 0,
            "sum_x": np.zeros(num_tracks, dtype=np.float64),
            "sum_y": np.zeros(num_tracks, dtype=np.float64),
            "sum_x2": np.zeros(num_tracks, dtype=np.float64),
            "sum_y2": np.zeros(num_tracks, dtype=np.float64),
            "sum_xy": np.zeros(num_tracks, dtype=np.float64),
        }

    def _update(
        self,
        species: str,
        predictions: Any,
        labels: Any,
    ) -> None:
        predictions_array = np.asarray(predictions)
        labels_array = np.asarray(labels)

        if predictions_array.ndim == 2:
            predictions_array = predictions_array[None, ...]
        if labels_array.ndim == 2:
            labels_array = labels_array[None, ...]

        if predictions_array.shape != labels_array.shape:
            raise ValueError(
                f"{species}: prediction shape {predictions_array.shape} "
                f"does not match label shape {labels_array.shape}."
            )

        if predictions_array.ndim != 3:
            raise ValueError(
                f"{species}: expected [batch, bins, tracks], "
                f"got {predictions_array.shape}."
            )

        num_tracks = predictions_array.shape[-1]
        self._ensure_species(species, num_tracks)

        # Pool examples and genomic bins together while preserving tracks.
        x = predictions_array.reshape(-1, num_tracks).astype(np.float64, copy=False)
        y = labels_array.reshape(-1, num_tracks).astype(np.float64, copy=False)

        stats = self.stats[species]
        stats["n"] += x.shape[0]
        stats["sum_x"] += x.sum(axis=0)
        stats["sum_y"] += y.sum(axis=0)
        stats["sum_x2"] += np.square(x).sum(axis=0)
        stats["sum_y2"] += np.square(y).sum(axis=0)
        stats["sum_xy"] += (x * y).sum(axis=0)

    def _result(self, species: str) -> tuple[float, np.ndarray]:
        stats = self.stats[species]
        n = float(stats["n"])

        covariance = (
            stats["sum_xy"]
            - stats["sum_x"] * stats["sum_y"] / n
        )
        variance_x = (
            stats["sum_x2"]
            - np.square(stats["sum_x"]) / n
        )
        variance_y = (
            stats["sum_y2"]
            - np.square(stats["sum_y"]) / n
        )

        variance_x = np.maximum(variance_x, 0.0)
        variance_y = np.maximum(variance_y, 0.0)

        denominator = np.sqrt(variance_x * variance_y)

        correlations = np.full(
            denominator.shape,
            np.nan,
            dtype=np.float64,
        )

        valid = (
            (denominator > 0)
            & np.isfinite(denominator)
            & np.isfinite(covariance)
        )

        correlations[valid] = covariance[valid] / denominator[valid]

        return float(np.nanmean(correlations)), correlations

    def __call__(
        self,
        evaluation: Any,
        compute_result: bool = False,
    ) -> dict[str, float]:
        predictions = _species_pair(evaluation.predictions)
        labels = _species_pair(evaluation.label_ids)

        for species, prediction, label in zip(
            ("human", "mouse"),
            predictions,
            labels,
        ):
            if self.track_indices is not None:
                label = label[..., self.track_indices[species]]
            self._update(species, prediction, label)

        if not compute_result:
            return {}

        human_mean, human_tracks = self._result("human")
        mouse_mean, mouse_tracks = self._result("mouse")

        result = {
            "human_pearson": human_mean,
            "mouse_pearson": mouse_mean,
            "mean_pearson": float(np.nanmean([human_mean, mouse_mean])),
            "human_pearson_median": float(np.nanmedian(human_tracks)),
            "mouse_pearson_median": float(np.nanmedian(mouse_tracks)),
        }

        # Important: reset here so the next evaluation starts cleanly.
        self.reset()
        return result


def save_model_metadata(directory: Path, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model_architecture.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError(f"Model architecture differs from {path}; use a new output directory.")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class CheckpointAliasesCallback(TrainerCallback):
    """Keep lightweight latest/ best aliases alongside Trainer checkpoints."""

    def __init__(self, track_manifest: dict | None = None, model_metadata: dict | None = None) -> None:
        self.track_manifest = track_manifest
        self.model_metadata = model_metadata

    @staticmethod
    def _link(output_dir: Path, alias_name: str, checkpoint_name: str) -> None:
        alias = output_dir / alias_name
        if alias.is_symlink():
            alias.unlink()
        elif alias.exists():
            if alias.is_dir():
                raise FileExistsError(
                    f"Cannot create {alias}; remove the existing directory first."
                )
            alias.unlink()
        alias.symlink_to(checkpoint_name, target_is_directory=True)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not state.is_world_process_zero:
            return
        output_dir = Path(args.output_dir)
        checkpoint = output_dir / f"checkpoint-{state.global_step}"
        if not checkpoint.is_dir():
            return
        if self.track_manifest is not None:
            save_manifest(checkpoint, self.track_manifest)
        if self.model_metadata is not None:
            save_model_metadata(checkpoint, self.model_metadata)
        self._link(output_dir, "latest", checkpoint.name)
        if state.best_model_checkpoint:
            best_checkpoint = Path(state.best_model_checkpoint)
            if best_checkpoint.is_dir():
                self._link(output_dir, "best", best_checkpoint.name)


def make_dataset(
    paths: dict[str, Path],
    *,
    split: str,
    seqlen: int,
    shift_aug: bool,
    rc_aug: bool,
    seed: int,
) -> MultiSpeciesEnformerDataset:
    return MultiSpeciesEnformerDataset(
        human_h5_path=paths[f"human_{split}_h5"],
        human_bed_path=paths[f"human_{split}_bed"],
        human_genome_path=paths["human_genome"],
        mouse_h5_path=paths[f"mouse_{split}_h5"],
        mouse_bed_path=paths[f"mouse_{split}_bed"],
        mouse_genome_path=paths["mouse_genome"],
        seqlen=seqlen,
        shift_aug=shift_aug,
        rc_aug=rc_aug,
        seed=seed,
    )


def validate_target_shapes(
    dataset: MultiSpeciesEnformerDataset,
    *,
    target_length: int,
    human_targets: int,
    mouse_targets: int,
    split: str,
) -> None:
    expected_human = (target_length, human_targets)
    expected_mouse = (target_length, mouse_targets)
    if dataset.human.target_shape != expected_human:
        raise ValueError(
            f"{split} human targets have shape {dataset.human.target_shape}; "
            f"expected {expected_human}."
        )
    if dataset.mouse.target_shape != expected_mouse:
        raise ValueError(
            f"{split} mouse targets have shape {dataset.mouse.target_shape}; "
            f"expected {expected_mouse}."
        )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError(
            f"--train-fraction must be in (0, 1]; got {args.train_fraction}."
        )

    paths = resolve_data_paths(args)
    num_gpus = detect_gpu_count(args.num_gpus)
    schedule = calculate_schedule(args, num_gpus)
    set_seed(args.seed)

    model = build_model(args)
    model_metadata = {
        "model": args.model,
        "config": model.config.to_dict(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if args.resume_from_checkpoint:
        metadata_path = Path(args.resume_from_checkpoint) / "model_architecture.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != model_metadata:
            raise ValueError(f"Model architecture does not match {metadata_path}.")
    print(f"Model: {args.model}, size={args.model_size}, trainable parameters={model_metadata['trainable_parameters']:,}")
    human_targets = model.config.output_heads["human"]
    mouse_targets = model.config.output_heads["mouse"]

    track_seed = args.seed if args.track_seed is None else args.track_seed
    track_indices = {
        species: select_tracks(total, args.track_fraction, track_seed, species, count)
        for species, total, count in (
            ("human", human_targets, args.human_track_count),
            ("mouse", mouse_targets, args.mouse_track_count),
        )
    }
    track_subset = any(len(track_indices[s]) != model.config.output_heads[s]
                       for s in track_indices)
    if track_subset and args.model != "custom":
        raise ValueError("Track subsets currently require --model custom.")
    if args.model == "custom":
        model.set_track_indices(track_indices)
    track_manifest = {
        "schema_version": 1,
        "selection_algorithm": "python-random-shuffle-species-seeded-v1",
        "index_convention": "zero-based HDF5 targets last-axis indices; sorted output order",
        "training_seed": args.seed,
        "track_seed": track_seed,
        "model": args.model,
        "fraction": args.track_fraction,
        "species": {
            species: {"total_tracks": model.config.output_heads[species],
                      "selected_count": len(indices), "indices": indices}
            for species, indices in track_indices.items()
        },
        "data_paths": {key: str(value.resolve()) for key, value in paths.items()},
        "evaluation": "validation and test metrics use the selected training tracks only",
        "heads": "full output heads; loss restricted to selected tracks",
    }
    if args.model == "custom" and args.model_size != "full":
        track_manifest["model_size"] = args.model_size
    if args.resume_from_checkpoint:
        validate_resume(Path(args.resume_from_checkpoint), track_manifest, subset=track_subset)
    manifest_path = args.output_dir / "track_selection.json"
    if manifest_path.exists():
        validate_resume(args.output_dir, track_manifest, subset=track_subset)
    elif track_subset and args.output_dir.exists() and any(args.output_dir.glob("checkpoint-*")):
        raise ValueError("Existing checkpoints have no track manifest; use a new output directory.")
    for species, indices in track_indices.items():
        print(f"Supervised {species} tracks: {len(indices)}/{model.config.output_heads[species]}, track seed={track_seed}")

    print("Resolved training schedule:")
    print(f"  GPUs/processes: {schedule.num_gpus}")
    print(f"  target global batch size: {schedule.target_global_batch_size}")
    print(f"  per-device train batch size: {schedule.per_device_train_batch_size}")
    print(f"  gradient accumulation steps: {schedule.gradient_accumulation_steps}")
    print(f"  actual global batch size: {schedule.actual_global_batch_size}")
    print(f"  schedule scale: {schedule.sample_scale:.6g}")
    print(f"  max steps: {schedule.max_steps}")
    print(f"  warmup steps: {schedule.warmup_steps}")
    print(f"  eval/save steps: {schedule.eval_steps}")
    print(f"  logging steps: {schedule.logging_steps}")

    print("Resolved data paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    train_dataset = make_dataset(
        paths,
        split="train",
        seqlen=args.seqlen,
        shift_aug=args.shift_aug,
        rc_aug=args.rc_aug,
        seed=args.seed,
    )
    valid_dataset = make_dataset(
        paths,
        split="valid",
        seqlen=args.seqlen,
        shift_aug=False,
        rc_aug=False,
        seed=args.seed,
    )
    validate_target_shapes(
        train_dataset,
        target_length=args.target_length,
        human_targets=human_targets,
        mouse_targets=mouse_targets,
        split="train",
    )
    validate_target_shapes(
        valid_dataset,
        target_length=args.target_length,
        human_targets=human_targets,
        mouse_targets=mouse_targets,
        split="valid",
    )

    # Apply the training subset only after target-shape validation because
    # torch.utils.data.Subset does not expose the wrapped dataset's custom
    # .human/.mouse attributes. Using the same seed across ablations makes
    # the subsets nested: 10% is a subset of 25%, which is a subset of 50%, etc.
    full_train_size = len(train_dataset)
    if args.train_fraction < 1.0:
        subset_size = max(1, round(full_train_size * args.train_fraction))
        generator = torch.Generator().manual_seed(args.seed)
        train_indices = torch.randperm(
            full_train_size,
            generator=generator,
        )[:subset_size].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    else:
        subset_size = full_train_size

    print(
        f"Training subset: {subset_size:,}/{full_train_size:,} examples "
        f"({100.0 * subset_size / full_train_size:.2f}%), seed={args.seed}"
    )

    test_paths = [
        paths["human_test_h5"],
        paths["human_test_bed"],
        paths["mouse_test_h5"],
        paths["mouse_test_bed"],
    ]
    test_dataset = None
    if any(path.exists() for path in test_paths):
        if not all(path.exists() for path in test_paths):
            raise FileNotFoundError(
                "Test evaluation requires all human and mouse test H5/BED files."
            )
        test_dataset = make_dataset(
            paths,
            split="test",
            seqlen=args.seqlen,
            shift_aug=False,
            rc_aug=False,
            seed=args.seed,
        )
        validate_target_shapes(
            test_dataset,
            target_length=args.target_length,
            human_targets=human_targets,
            mouse_targets=mouse_targets,
            split="test",
        )

    training_args = build_training_args(args, schedule)
    if int(os.environ.get("RANK", "0")) == 0:
        save_manifest(args.output_dir, track_manifest)
        save_model_metadata(args.output_dir, model_metadata)
    pearson = StreamingPearson(track_indices if track_subset else None)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=pearson,
        callbacks=[
            CheckpointAliasesCallback(track_manifest, model_metadata),
        ],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()

    if test_dataset is not None:
        test_metrics = trainer.evaluate(
            eval_dataset=test_dataset,
            metric_key_prefix="test",
        )
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        print(f"Test metrics: {test_metrics}")
    else:
        print("Test data not found; skipped test evaluation.")


if __name__ == "__main__":
    main()
