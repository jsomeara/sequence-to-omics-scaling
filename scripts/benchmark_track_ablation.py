"""Compare focused-track models against the full-track model on identical tracks.

Run from the repository root, inside an allocated GPU job:
    uv run python -m scripts.benchmark_track_ablation

Optional alternate data location or baseline:
    uv run python -m scripts.benchmark_track_ablation --data-dir /path/to/DATASET
    uv run python -m scripts.benchmark_track_ablation --baseline OUTPUTS/full-run

By default, test data are cached once in RAM within --ram-budget-gib (80 GiB).
Use --cache none to stream. Progress bars show caching and every inference batch.
Only inference is performed. Each species' native test set is visited exactly
once per evaluated model (no paired-dataset repetition and no augmentation).
Final run weights are the validation-selected weights saved by the trainer.
Test performance is never used to choose a checkpoint. Smaller parameter-model
ablations are excluded: comparisons require matching architecture signatures.

Hypothesis (not a result): full-track supervision will improve average selected-
track Pearson, especially for 1%/3% subsets, because the shared trunk can learn
features supported by other tracks. Focus could win where unrelated objectives
compete for capacity or dilute selected-track gradients. Repeat track seeds to
assess whether an observed difference depends on the selected subset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import h5py
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloaders import H5EnformerDataset
from enformer_pytorch.multispecies_custom import MultiSpeciesCustom

SPECIES = ("human", "mouse")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-root", type=Path, default=Path("OUTPUTS"))
    parser.add_argument("--baseline", type=Path, default=None,
                        help="100%% track run; defaults to the seed42/trackseed42 tracks100pct run.")
    parser.add_argument("--runs", type=Path, nargs="+", default=None,
                        help="Explicit partial-track run directories; otherwise discover all track ablations.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override saved test H5/BED/FASTA paths with standard filenames here.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New directory for results; defaults to a timestamped directory under OUTPUTS.")
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                        help="Same precision for both sides; bf16 matches training defaults.")
    parser.add_argument("--cache", choices=("ram", "none"), default="ram",
                        help="Cache test sequences and the union of selected labels once (default: ram).")
    parser.add_argument("--ram-budget-gib", type=float, default=80.0,
                        help="Maximum cache plus loading scratch, excluding model/OS (default: 80 GiB).")
    parser.add_argument("--hash-weights", action="store_true",
                        help="Also SHA256 weights; requires an extra full read of every checkpoint.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text())


def file_identity(path, *, digest=False):
    path = path.resolve()
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        h = hashlib.sha256()
        with path.open("rb") as source:
            with tqdm(total=stat.st_size, desc=f"Hash {path.name}", unit="B", unit_scale=True,
                      disable=stat.st_size < 8 * 1024 * 1024, dynamic_ncols=True) as bar:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    h.update(block)
                    bar.update(len(block))
        result["sha256"] = h.hexdigest()
    return result


def load_run(directory, checkpoint):
    directory = directory.resolve()
    if not (directory / ".completed").is_file():
        raise ValueError(f"Run is not marked completed: {directory}. Wait for training to finish.")
    manifest = read_json(directory / "track_selection.json")
    architecture = read_json(directory / "model_architecture.json")
    if manifest.get("model") != "custom" or architecture.get("model") != "custom":
        raise ValueError(f"Expected custom model: {directory}")
    config = architecture["config"]
    indices = {}
    for species in SPECIES:
        record = manifest["species"][species]
        selected = record["indices"]
        total = record["total_tracks"]
        if (not selected or any(type(i) is not int for i in selected)
                or selected != sorted(set(selected)) or selected[0] < 0 or selected[-1] >= total
                or len(selected) != record["selected_count"]
                or total != config["output_heads"][species]):
            raise ValueError(f"Invalid {species} selection in {directory}")
        indices[species] = selected
    weights_dir = directory if checkpoint == "final" else (directory / "best").resolve(strict=True)
    if checkpoint == "best":
        if read_json(weights_dir / "track_selection.json") != manifest:
            raise ValueError(f"Checkpoint selection differs from run selection: {weights_dir}")
    candidates = [weights_dir / "pytorch_model.bin", weights_dir / "model.safetensors"]
    weights = next((path for path in candidates if path.is_file()), None)
    if weights is None:
        raise FileNotFoundError(f"No model weights in {weights_dir}")
    return {"directory": directory, "manifest": manifest, "architecture": architecture,
            "config": config, "indices": indices, "weights": weights}


def signature(run):
    config = run["config"]
    return {key: config.get(key) for key in (
        "model_size", "model_dim", "transformer_blocks", "attention_heads",
        "sequence_length", "target_length", "output_heads")}


def is_full(run):
    return all(len(run["indices"][s]) == run["config"]["output_heads"][s] for s in SPECIES)


def test_paths(run, data_dir):
    if data_dir is not None:
        return {s: {"h5_path": data_dir / f"{s}_test.h5",
                    "bed_path": data_dir / f"{s}_test.bed",
                    "genome_path": data_dir / "genome" / ("hg38.fa" if s == "human" else "mm10.fa")}
                for s in SPECIES}
    paths = run["manifest"]["data_paths"]
    return {s: {"h5_path": Path(paths[f"{s}_test_h5"]),
                "bed_path": Path(paths[f"{s}_test_bed"]),
                "genome_path": Path(paths[f"{s}_genome"])} for s in SPECIES}


class RamTestDataset:
    """One lossless cache shared by every model, containing only needed tracks."""
    def __init__(self, source, indices, species):
        self.indices = list(indices)
        self.target_shape = (source.target_shape[0], len(indices))
        self.labels = np.empty((len(source), *self.target_shape), dtype=np.float32)
        # The existing encoder emits only 0/1; uint8 retains unknown-base zeros.
        self.sequences = np.empty((len(source), source.seqlen, 4), dtype=np.uint8)
        with h5py.File(source.h5_path, "r") as handle:
            targets = handle[source.target_key]
            rows = targets.chunks[0] if targets.chunks else 16
            with tqdm(total=len(source), desc=f"Cache {species} targets", unit="examples", dynamic_ncols=True) as bar:
                for start in range(0, len(source), rows):
                    stop = min(start + rows, len(source))
                    # Read a whole first-axis chunk slab once. Scalar example
                    # reads repeatedly inflate these multi-example chunks.
                    block = np.asarray(targets[start:stop], dtype=np.float32)
                    self.labels[start:stop] = block[..., indices]
                    del block
                    bar.update(stop - start)
        try:
            for i in tqdm(range(len(source)), desc=f"Cache {species} FASTA", unit="examples", dynamic_ncols=True):
                sequence = source._sequence_window(source.chroms[i], int(source.starts[i]), int(source.ends[i]))
                self.sequences[i] = source._one_hot(sequence)
        finally:
            source.close()

    def __len__(self):
        return len(self.labels)

    def close(self):
        pass


class SelectedTestDataset:
    """Select labels before collation/transfer; models still receive float32 DNA."""
    def __init__(self, source, indices):
        self.source = source
        self.indices = indices
        if isinstance(source, RamTestDataset):
            positions = {track: i for i, track in enumerate(source.indices)}
            self.columns = [positions[i] for i in indices]
            if self.columns == list(range(len(source.indices))):
                self.columns = None

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if isinstance(self.source, RamTestDataset):
            labels = self.source.labels[index]
            if self.columns is not None:
                labels = labels[:, self.columns]
            return {"x": torch.from_numpy(self.source.sequences[index]).float(),
                    "labels": torch.from_numpy(np.ascontiguousarray(labels))}
        item = self.source[index]
        item["labels"] = item["labels"][..., self.indices]
        return item


def cache_plan(datasets, union, budget_gib):
    resident = 0
    scratch = 0
    for species, source in datasets.items():
        resident += len(source) * (source.seqlen * 4 + source.target_shape[0] * len(union[species]) * 4)
        with h5py.File(source.h5_path, "r") as handle:
            target = handle[source.target_key]
            rows = min(len(source), target.chunks[0] if target.chunks else 16)
            # Conservative allowance for source dtype conversion and selected copy.
            scratch = max(scratch, rows * source.target_shape[0] *
                          (source.target_shape[1] * (target.dtype.itemsize + 4) + len(union[species]) * 4))
    required = resident + scratch
    print(f"RAM cache: {resident / 2**30:.2f} GiB resident + up to {scratch / 2**30:.2f} GiB loading scratch; "
          f"budget {budget_gib:.1f} GiB. Leave additional RAM for the model, batches, and OS.", flush=True)
    if required > budget_gib * 2**30:
        raise MemoryError("Cache exceeds --ram-budget-gib. Increase it only within your allocation, "
                          "or use --cache none for streaming.")
    return {"resident_bytes": resident, "loading_scratch_bytes": scratch, "budget_gib": budget_gib}


class TrackStats:
    """Float64 streaming sums, pooling examples and bins for each track."""
    def __init__(self, tracks):
        self.n = 0
        self.sums = np.zeros((6, tracks), dtype=np.float64)

    def update(self, prediction, target):
        x = prediction.reshape(-1, prediction.shape[-1]).astype(np.float64)
        y = target.reshape(-1, target.shape[-1]).astype(np.float64)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite values or mismatched prediction/target shapes.")
        if (x < 0).any() or (y < 0).any():
            raise ValueError("Poisson evaluation requires nonnegative predictions and targets.")
        self.n += x.shape[0]
        self.sums += np.stack((x.sum(0), y.sum(0), (x*x).sum(0), (y*y).sum(0),
                               (x*y).sum(0), (x-y*np.log(np.maximum(x, 1e-20))).sum(0)))

    def result(self):
        if self.n == 0:
            raise ValueError("Test set is empty.")
        sx, sy, sxx, syy, sxy, loss = self.sums
        covariance = sxy - sx*sy/self.n
        denom = np.sqrt(np.maximum(sxx-sx*sx/self.n, 0) * np.maximum(syy-sy*sy/self.n, 0))
        pearson = np.full(sx.shape, np.nan)
        np.divide(covariance, denom, out=pearson, where=denom > 0)
        return {"pearson": pearson, "poisson": loss/self.n, "pooled_bins": self.n}


def evaluate(run, indices, datasets, args, device):
    model = MultiSpeciesCustom(model_size=run["config"].get("model_size", "full"))
    if run["weights"].suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(run["weights"]), device="cpu")
    else:
        state = torch.load(run["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    results = {}
    try:
        with torch.inference_mode():
            for species in SPECIES:
                index = torch.tensor(indices[species], dtype=torch.long, device=device)
                stats = TrackStats(len(indices[species]))
                selected_dataset = SelectedTestDataset(datasets[species], indices[species])
                # Keep the large RAM cache in this process: no worker copies or IPC.
                workers = 0 if isinstance(datasets[species], RamTestDataset) else args.num_workers
                loader = DataLoader(selected_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=workers, drop_last=False,
                                    pin_memory=device.type == "cuda")
                progress = tqdm(loader, desc=f"{run['directory'].name} / {species}",
                                unit="batch", dynamic_ncols=True)
                for batch in progress:
                    context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                               if args.precision == "bf16" else nullcontext())
                    with context:
                        full_prediction = model.model(batch["x"].to(device, non_blocking=True), species=species)
                    prediction = full_prediction.index_select(-1, index).float().cpu().numpy()
                    labels = batch["labels"].numpy()
                    stats.update(prediction, labels)

                results[species] = stats.result()

    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


def mean_or_none(values):
    return finite_or_none(np.mean(values)) if len(values) else None


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run with python, not torchrun: this benchmark uses one device sequentially.")
    if not math.isfinite(args.ram_budget_gib) or args.ram_budget_gib <= 0:
        raise ValueError("--ram-budget-gib must be finite and positive.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers nonnegative.")
    device = torch.device(args.device)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported devices are cuda and cpu.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Device does not support bf16; use --precision fp32.")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    baseline_dir = args.baseline or args.outputs_root / "basic-model-trainseed42-trackseed42-tracks100pct"
    baseline = load_run(baseline_dir, args.checkpoint)
    if not is_full(baseline):
        raise ValueError("Baseline must have been supervised on every track of both species.")
    directories = args.runs or sorted(args.outputs_root.glob("basic-model-trainseed*-trackseed*-tracks*pct"))
    runs = []
    for directory in directories:
        # Exclude all-track runs before completion checks; they are not alternatives.
        manifest = read_json(directory / "track_selection.json")
        if all(len(manifest["species"][s]["indices"]) == manifest["species"][s]["total_tracks"] for s in SPECIES):
            continue
        run = load_run(directory, args.checkpoint)
        if signature(run) != signature(baseline):
            raise ValueError(f"Architecture differs from baseline: {directory}")
        if run["manifest"]["training_seed"] != baseline["manifest"]["training_seed"]:
            raise ValueError(f"Training seed differs for {directory}; choose a matching --baseline.")
        runs.append(run)
    if not runs:
        raise ValueError("No partial-track runs found.")
    if len({run['directory'] for run in runs}) != len(runs):
        raise ValueError("Duplicate run directories.")
    paths = test_paths(baseline, args.data_dir)
    if args.data_dir is None:
        for run in runs:
            if test_paths(run, None) != paths:
                raise ValueError("Saved test paths differ; specify --data-dir to use one common test set.")
    datasets = {}
    output = args.output_dir or args.outputs_root / ("track-comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    # Never overwrite earlier benchmark reports or training artifacts.
    output.mkdir(parents=True, exist_ok=False)
    try:
        for species in SPECIES:
            datasets[species] = H5EnformerDataset(**paths[species],
                seqlen=baseline["config"]["sequence_length"], shift_aug=False, rc_aug=False)
            expected = (baseline["config"]["target_length"], baseline["config"]["output_heads"][species])
            if datasets[species].target_shape != expected or len(datasets[species]) == 0:
                raise ValueError(f"Unexpected/empty {species} test dataset; expected targets {expected}.")
        union = {s: sorted({i for r in runs for i in r["indices"][s]}) for s in SPECIES}
        cache_info = cache_plan(datasets, union, args.ram_budget_gib) if args.cache == "ram" else None
        provenance = {
            "cache": args.cache, "cache_plan": cache_info,
            "hash_weights": args.hash_weights,
            "checkpoint_policy": args.checkpoint, "precision": args.precision,
            "device": str(device), "batch_size": args.batch_size,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "test_policy": "each species example once; no augmentation; all bins pooled per track",
            "pearson_policy": "paired summaries exclude tracks undefined for either model",
            "delta_policy": "focused minus full; positive Pearson / negative Poisson favors focus",
            "baseline": {"weights": file_identity(baseline["weights"], digest=args.hash_weights),
                         "manifest": baseline["manifest"], "architecture": baseline["architecture"]},
            "alternatives": [{"run": str(r["directory"]), "weights": file_identity(r["weights"], digest=args.hash_weights),
                              "manifest": r["manifest"], "architecture": r["architecture"]} for r in runs],
            "test_data": {s: {"examples": len(datasets[s]), "files": {
                key: file_identity(path, digest=key == "bed_path") for key, path in paths[s].items()}} for s in SPECIES},
            "script": file_identity(Path(__file__), digest=True),
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        # Evaluate the baseline once on the union, then slice to each alternative's
        # exact selection. This is equivalent to repeated restricted evaluations.
        if args.cache == "ram":
            for species in SPECIES:
                datasets[species] = RamTestDataset(datasets[species], union[species], species)
        baseline_scores = evaluate(baseline, union, datasets, args, device)
        positions = {s: {track: i for i, track in enumerate(union[s])} for s in SPECIES}
        rows, summaries = [], []
        for run in runs:
            focused = evaluate(run, run["indices"], datasets, args, device)
            for species in SPECIES:
                ids = run["indices"][species]
                select = [positions[species][track] for track in ids]
                fp, fl = focused[species]["pearson"], focused[species]["poisson"]
                bp = baseline_scores[species]["pearson"][select]
                bl = baseline_scores[species]["poisson"][select]
                valid = np.isfinite(fp) & np.isfinite(bp)
                difference = fp[valid] - bp[valid]
                for j, track in enumerate(ids):
                    rows.append({"run": str(run["directory"]), "species": species, "track_index": track,
                                 "focused_pearson": finite_or_none(fp[j]), "full_pearson": finite_or_none(bp[j]),
                                 "pearson_delta_focused_minus_full": finite_or_none(fp[j]-bp[j]),
                                 "focused_poisson": float(fl[j]), "full_poisson": float(bl[j]),
                                 "poisson_delta_focused_minus_full": float(fl[j]-bl[j])})
                summaries.append({"run": str(run["directory"]), "species": species,
                    "fraction": run["manifest"]["fraction"], "track_seed": run["manifest"]["track_seed"],
                    "training_seed": run["manifest"]["training_seed"], "selected_tracks": len(ids),
                    "test_examples": len(datasets[species]), "paired_valid_pearson_tracks": int(valid.sum()),
                    "focused_mean_pearson": mean_or_none(fp[valid]), "full_mean_pearson": mean_or_none(bp[valid]),
                    "mean_pearson_delta_focused_minus_full": mean_or_none(difference),
                    "focused_pearson_win_fraction": mean_or_none((difference > 0).astype(float)),
                    "focused_mean_poisson": float(fl.mean()), "full_mean_poisson": float(bl.mean()),
                    "mean_poisson_delta_focused_minus_full": float((fl-bl).mean())})
            # Preserve completed comparisons if a later model fails.
            write_csv(output / "per_track.csv", rows)
            write_csv(output / "summary.csv", summaries)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
        (output / ".completed").touch()
        print(f"Results saved to {output.resolve()}")
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
