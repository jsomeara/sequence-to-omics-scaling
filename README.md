# Sequence-to-omics scaling

Experiments on how sequence-to-function prediction changes with **training examples**, **supervised output tracks**, and **model width**. The default custom model jointly predicts human and mouse functional-genomics profiles from 131,072-base one-hot DNA windows, with 896 output bins and 5,313 human / 1,643 mouse tracks.

This is a reconstruction of the pre-conservation track-ablation benchmark state. The full custom model has **135,938,104 parameters and one transformer block**. It does not include the subsequent custom architecture, conservation inputs, or later recovery/continuation changes. Enformer remains an optional `--model enformer` alternative; custom is the default. Model configurations are local: no hosted EleutherAI configuration is fetched. The dataset downloader uses Hugging Face dataset hosting; training uses the Transformers library.

## Installation

Use Linux, Python 3.10–3.12, `uv`, and CUDA-capable GPUs with a compatible NVIDIA driver. Run commands from the repository root:

```bash
git clone https://github.com/jsomeara/sequence-to-omics-scaling.git
cd sequence-to-omics-scaling
uv sync --locked
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The preserved lockfile fixes Python package versions. Benchmark provenance records Torch 2.14.0+cu130, CUDA 13.0, NumPy 2.5.2 and an NVIDIA L40S. Matching seeds alone does not guarantee bitwise reproduction across hardware, software versions, or distributed batch layouts. If CUDA is unavailable, install the appropriate CUDA-enabled PyTorch build before training. Shell sweeps report to Weights & Biases: run `uv run wandb login`, or append `--report-to none` to disable it.

## Prepare the data

Datasets and weights are deliberately excluded. The included downloader reconstructs the split training HDF5 files and optionally downloads genome FASTAs:

```bash
uv run python -m scripts.download_dataset \
  --output-dir ./DATASET --include-test --download-hg38 --download-mm10
```

The default dataset ID is `yangyz1230/space`. Keep BED order, target order and genome assemblies unchanged. Required files are `human_{train,valid,test}.{h5,bed}`, the corresponding mouse files, and `genome/hg38.fa`, `genome/mm10.fa`. Track annotation files are downloaded too. FASTA indexes are created by the loader; keep the data directory writable.

Create the faster, uncompressed training layout used by the sweeps (requires additional disk space):

```bash
uv run python -m scripts.repack_targets DATASET/human_train.h5 DATASET/human_train_fast.h5 --workers 4
uv run python -m scripts.repack_targets DATASET/mouse_train.h5 DATASET/mouse_train_fast.h5 --workers 4
```

Repacking changes storage layout, not target values. Original HDF5 files can instead be supplied directly through the training path flags, at a potential I/O cost.

## Configure the historical sweep scripts

The original launchers are preserved, including their site-specific constants. Before running, edit the following in each launcher you use:

- `DATA_DIR`: absolute path to your prepared dataset directory.
- `NPROC_PER_NODE` **and** `--num-gpus` in `COMMON_ARGS`: set both to your GPU count.
- `OUTPUT_ROOT`: local experiment directory (default `./OUTPUTS`).

Set `CUDA_VISIBLE_DEVICES` to the same number of GPUs. Changing that environment variable alone does **not** change the launcher's four-process default. FASTA and repacked-HDF5 paths are constructed from `DATA_DIR`.

The saved track experiments used **2 GPUs**, per-device batch 2, accumulation 8, global batch 32, 10,000 optimizer steps, 5,000 warmup steps, and validation/save every 500 steps. The saved parameter experiments used **3 GPUs**, per-device batch 2, accumulation 6, global batch 36, 8,889 steps, 4,445 warmup steps, and validation/save every 445 steps. These schedules are confirmed by the captured log headers in `results/run-settings/`.

## Reproduce the experiments

### Track scaling

After setting `tracks_abal.sh` to two GPUs and your data path:

```bash
export CUDA_VISIBLE_DEVICES=0,1
bash tracks_abal.sh
```

Runs 100%, 1%, 10%, 3%, and 30% of tracks, with seed 42. The trunk and output-head sizes are unchanged; only the selected tracks contribute to loss and validation/test metrics. Each run saves `track_selection.json` with exact zero-based target indices. The full baseline is trained first.

The independent track-selection seed defaults to the training seed. To repeat only the 1% experiment with a different selection, keeping training seed 42:

```bash
TRACK_PERCENT=1 bash tracks_abal.sh --track-seed 43
```

Selections are nested within each species for a fixed track seed. For arbitrary counts, invoke `scripts/train_enformer.py` directly with `--human-track-count` and `--mouse-track-count`. Run names encode the training and track seeds. Completed sweeps are skipped using `.completed`; `FORCE=1` reruns them, so use a fresh output root when retaining previous results.

### Dataset scaling

Configure `data_abal.sh` for your paths and GPU count, then:

```bash
bash data_abal.sh
```

Runs the same five percentages of training examples, retaining full track supervision. Training subsets are deterministic and nested for a fixed seed. Validation/test sets remain unchanged; these experiments retain the configured update budget rather than scaling it down with the training fraction. No dataset-scaling result artifacts were available in the copied server output; this repository provides the reproduction code without claiming measured results for that sweep.

### Parameter scaling

Configure `parameters_abal.sh` for three GPUs to match the included results:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
bash parameters_abal.sh
# Optional single preset:
MODEL_SIZE=tiny bash parameters_abal.sh
```

Presets are `tiny`, `small`, `medium`, `large`, and `full`, with transformer widths 192, 384, 768, 1152, and 1536. All have exactly one transformer block and 12 attention heads; convolutional widths scale too. `full` is the original baseline, not a larger new model. Exact parameter counts and test metrics are preserved in each results directory.

## Reproduce the paired track benchmark

Train all five track runs first, or supply your own compatible checkpoints and manifests. Checkpoints are not distributed with this repository. The benchmark compares each focused model with the full model **on exactly the focused model's selected tracks**:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.benchmark_track_ablation \
  --outputs-root ./OUTPUTS \
  --data-dir ./DATASET \
  --checkpoint final --precision bf16 --batch-size 1 \
  --cache ram --ram-budget-gib 80 \
  --output-dir ./OUTPUTS/reproduced-track-comparison
```

The output directory should be new. Default discovery uses seed-42 run names. For other trials, pass `--baseline OUTPUTS/<full-run>` and `--runs OUTPUTS/<focused-run> ...`. Use `--cache none` if RAM is limited. The archived run planned approximately 15.6 GiB of resident cache plus 0.65 GiB of loading scratch, excluding model/process overhead. Progress bars cover caching and inference.

Each species' native test set is evaluated once per model: 1,937 human and 2,017 mouse examples, with no augmentation. Per-track Pearson pools examples and bins. Paired mean Pearson excludes tracks undefined for either model. Poisson loss omits the target-only factorial term. Positive Pearson deltas / negative Poisson deltas favor focused supervision. `final` means the run-root weights exported after validation-selected best weights were loaded, not necessarily the last optimization step. Test metrics never select checkpoints.

## Included results

`results/track-comparison-20260919T202732.863735Z/` contains the original unmodified `summary.csv`, `summary.json`, `per_track.csv`, and `provenance.json`. Provenance embeds selection manifests and original weight paths, sizes and timestamps; weight hashing was disabled. Historical absolute paths are retained as provenance and do not need to exist on your machine. `--data-dir` overrides saved input paths when rerunning.

Mean selected-track Pearson (**focused / full**):

| Tracks | Human | Mouse |
|---|---|---|
| 1% | 0.619670 / 0.618029 | 0.659757 / 0.664492 |
| 3% | 0.604273 / 0.604211 | 0.689113 / 0.681215 |
| 10% | 0.626970 / 0.625954 | 0.710858 / 0.705554 |
| 30% | 0.622344 / 0.622931 | 0.706207 / 0.705071 |

Differences are small and mixed in this single selection/training-seed trial. They do not establish that either focus or additional supervision is generally better. Repeat independent seeds before drawing a broad conclusion. Run-level track and parameter results also include architecture metadata, test summaries, and track selections under `results/basic-model-*`.

## Checkpoints, resume, and long runs

Training writes checkpoints and `latest`/`best` aliases under `OUTPUTS`. Resume requires `--resume-from-checkpoint OUTPUTS/<run>/checkpoint-<step>`; reusing an output directory alone does not resume. Keep the original schedule when resuming an interrupted experiment. Extending the step budget rebuilds the cosine schedule and can raise the learning rate; the later continuation override is intentionally absent from this historical snapshot. Historical metadata comparison can also reject JSON-normalized configurations on resume; the original implementation is retained rather than silently incorporating later fixes. Fresh runs are the supported reproduction path here.

For SSH sessions, start `tmux new -s scaling`, run the experiment inside it, then press Ctrl+B followed by D to detach. Reconnect with `tmux attach -t scaling`. This survives SSH disconnects, not application errors or machine restarts.

## Snapshot provenance and license

The server did not contain usable Git history. This snapshot was reconstructed from locally saved pre-conservation backups and server benchmark artifacts. The original custom model comes from the parameter-scaling backup; it matches `modeling_custom_old.py` except for its standalone smoke-test entry point. The benchmark script matches the executed RAM-cache version byte-for-byte. Training/data wrappers and the lockfile come from the pre-conservation backup. No server source files or running jobs were modified to create this publication.

`SNAPSHOT_SHA256SUMS` records the copied source and result contents. The original MIT license and copyright notice are retained in `LICENSE`. Dataset access and redistribution terms are separate from the source-code license.
