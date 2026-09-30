# Scaling Sequence-to-Omics Models

Code and recorded results for experiments on how sequence-to-omics prediction scales with **training data, model size, supervised tracks, and transformer depth**. The experiments use human and mouse data from the Enformer/Basenji2 dataset.

The repository includes the measurements and code needed to regenerate the figures, tables, and equations. You can reproduce the analysis on a CPU without downloading the training dataset or model checkpoints.

## Start here

| Goal | Where to start |
|---|---|
| Browse runs interactively with smoothing and visibility controls | [Run data viewer](#interactive-run-data-viewer) |
| Generate the figures, tables, and equations | [Offline analysis](#reproduce-the-figures-and-tables) |
| Inspect the analyzed training runs | [Run index](analysis/raw/wandb/README.md) and [run manifest](analysis/run_manifest.json) |
| Inspect the focused-versus-full track benchmark | [Benchmark results](results/track-comparison-20260919T202732.863735Z/) |
| Analyze individual transformer blocks | [Transformer usage notebook](Transformer_Usage_Analysis.ipynb) |
| Train new models or repeat the sweeps | [Training](#train-the-models) |

## Experiments

| Experiment | What varies | What is included |
|---|---|---|
| Dataset scaling | 1%, 3%, 10%, 30%, and 100% of training examples | Test scores and complete scalar run histories |
| Parameter scaling | Five model widths, from 4.5M to 135.9M parameters | Test scores, architecture metadata, and complete scalar histories |
| Track scaling | 1%, 3%, 10%, 30%, and 100% of supervised tracks | Exact track selections, run histories, and a matched-track test benchmark |
| Transformer depth | Zero, one, and eight blocks; original encoding and RoPE | Validation histories and comparisons at matched steps |
| Transformer usage | Each block in official Enformer, Borzoi, and AlphaGenome models | Individual layer scores across eight genomic sequences and the executed notebook |

The data and parameter experiments show diminishing gains across the tested scales. The track benchmark finds no consistent advantage for focused or full-track supervision. One transformer block reaches similar validation performance to eight blocks in the depth comparison.

These are single-seed experiments. The scaling fits describe the recorded measurements; their asymptotes are extrapolations, not established performance ceilings. Transformer input-output cosine similarity measures representation changes, not whether a layer is necessary. [Analysis methods and provenance](analysis/README.md) explain the evaluations and their limits.

## Interactive run data viewer

Download [`run-viewer.html`](run-viewer.html) and open it in a modern browser. It is a standalone file with all 20 runs embedded: no server, installation, API key, or internet connection is needed. GitHub displays HTML as source, so use **Download raw file** and open the downloaded file locally.

The viewer includes:

- Show/hide controls for individual runs and experiment groups, run search, and custom colors.
- Configurable metric charts for losses, Pearson scores, learning rates, gradients, and other logged scalars.
- EMA and moving-average smoothing, an unsmoothed mode, and an optional raw-value overlay.
- Optimizer-step, W&B-step, elapsed-time, and wall-clock axes; logarithmic axes and shared zoom.
- Exact raw/smoothed values on hover, sortable metric summaries, and run-setting comparisons.
- PNG chart downloads, full-resolution CSV exports for visible chart series, and raw JSON downloads for selected runs.

Drag on a plot to zoom, scroll to adjust the range, or double-click to reset. Run selections and chart settings persist locally in your browser. CSV exports respect the selected X range and include smoothing settings. The viewer explains its smoothing definitions and handling of missing or repeated steps.

To rebuild the HTML after refreshing the recorded run data:

```bash
python3 analysis/build_viewer.py
```

The builder uses only Python's standard library and the explicit [run manifest](analysis/run_manifest.json). Its editable source is [`analysis/viewer/template.html`](analysis/viewer/template.html). The viewer displays the training-run scalar histories; the separate paired track benchmark and official-model cosine results remain available through the offline analysis and notebook.

## Reproduce the figures and tables

Clone the repository and create a lightweight analysis environment:

```bash
git clone https://github.com/jsomeara/sequence-to-omics-scaling.git
cd sequence-to-omics-scaling

uv venv .analysis-venv --python 3.11
uv pip install --python .analysis-venv/bin/python -r analysis/requirements.txt
.analysis-venv/bin/python analysis/generate.py
```

This generates the following in `analysis/generated/`:

- Dataset and parameter scaling plots, including extrapolation, fitted equations, and fit coefficients.
- Focused-versus-full track comparisons and differences.
- Transformer-depth learning curves and tables, including RoPE at steps **6,000 and 11,000**.
- Individual layer cosine plots and tables for Enformer, Borzoi, and AlphaGenome.
- PNG/PDF/SVG figures and equation images, Markdown and rendered tables, and full-precision CSVs.

No GPU, W&B account, API key, training dataset, or checkpoint is needed. Generated files are ignored by Git. Change the output directory or extrapolation range with:

```bash
.analysis-venv/bin/python analysis/generate.py \
  --output-dir ./figures --extrapolation-factor 10
```

Both scaling axes use the same additive-offset power law, fitted by unweighted nonlinear least squares:

$$\widehat{\bar r}(x)=r_\infty-Ax^{-\alpha}.$$

For dataset scaling, $x$ is the training fraction. For parameter scaling, it is the parameter count in millions. See [the analysis README](analysis/README.md) for constraints, source files, and the optional W&B export command.

## Recorded data

Only the **20 training runs analyzed in the post** are exported: five each for dataset size, parameter size, track supervision, and transformer depth. Their unsampled numeric scalar histories contain 194,222 logged rows. No checkpoints, W&B artifacts, or training datasets are distributed.

| Location | Contents |
|---|---|
| [`analysis/raw/wandb/`](analysis/raw/wandb/) | Per-run scalar histories, numeric summaries, relevant settings, and the archived measurements used in the post |
| [`analysis/raw/transformer_usage/`](analysis/raw/transformer_usage/) | Per-layer/per-sequence cosine scores, sequence coordinates, source commits, environments, and AlphaGenome validation |
| [`analysis/raw/window_counts.json`](analysis/raw/window_counts.json) | Reconstructed training-window counts; these are not original subset manifests |
| [`results/`](results/) | Saved parameter/track test summaries, architecture metadata, track selections, benchmark results, and provenance |

The figure generator reads these committed files directly. Dataset and parameter scores are training-pipeline test summaries; transformer-depth scores are validation metrics. The focused-versus-full comparison uses each species' native test set and evaluates both models on exactly the same selected tracks.

## Transformer usage notebook

[Open the notebook in Colab](https://colab.research.google.com/github/jsomeara/sequence-to-omics-scaling/blob/main/Transformer_Usage_Analysis.ipynb), or inspect its saved outputs in [GitHub](Transformer_Usage_Analysis.ipynb).

The completed analysis measures the input-output cosine similarity of every full transformer block: **11 Enformer blocks, 8 Borzoi blocks, and 9 AlphaGenome blocks**, across eight shared hg38 chr22 genomic centers. The offline generator can recreate those plots from the saved measurements.

Rerunning inference requires a GPU and model-weight downloads. AlphaGenome requires access to Google's gated weights; enter your Hugging Face token through the notebook's masked prompt. The notebook documents its T4 numerical adaptation and checks the tiled convolutional encoder before running the full-context transformer analysis.

## Train the models

### Install the training environment

Training uses Linux, Python 3.10–3.12, `uv`, and CUDA-capable GPUs. From the repository root:

```bash
uv sync --locked
uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The default **custom** model predicts 5,313 human tracks and 1,643 mouse tracks from 131,072-base one-hot DNA windows, producing 896 output bins. The full preset has **135,938,104 parameters and one transformer block**. Enformer is available through `--model enformer`. Model configurations are local; no hosted EleutherAI configuration is fetched.

The sweep scripts report to W&B. Run `uv run wandb login`, or append `--report-to none` to a sweep command to disable reporting. Training dependencies and analysis dependencies are separate so plotting does not require installing the training stack.

### Prepare the dataset

```bash
uv run python -m scripts.download_dataset \
  --output-dir ./DATASET --include-test --download-hg38 --download-mm10
```

The downloader uses `yangyz1230/space` on Hugging Face. Required files are `human_{train,valid,test}.{h5,bed}`, the corresponding mouse files, and `genome/hg38.fa` / `genome/mm10.fa`. Preserve BED row order, target order, and genome assemblies. The data directory must be writable for FASTA indexes.

The historical sweeps use uncompressed training HDF5 files for faster reads:

```bash
uv run python -m scripts.repack_targets DATASET/human_train.h5 DATASET/human_train_fast.h5 --workers 4
uv run python -m scripts.repack_targets DATASET/mouse_train.h5 DATASET/mouse_train_fast.h5 --workers 4
```

Repacking changes storage layout without changing target values. It requires additional disk space.

### Configure and run the sweeps

The launchers preserve their historical paths and default to four processes. Before running a launcher, edit its `DATA_DIR`, `NPROC_PER_NODE`, and `--num-gpus` in `COMMON_ARGS`. The two GPU-count settings must agree. Set `OUTPUT_ROOT` if you want a different output location.

| Launcher | Sweep | GPUs used for the archived results |
|---|---|---|
| [`data_abal.sh`](data_abal.sh) | Training-example fractions | Check each run's [recorded settings](analysis/raw/wandb/runs/) |
| [`parameters_abal.sh`](parameters_abal.sh) | `tiny`, `small`, `medium`, `large`, `full` | 3 |
| [`tracks_abal.sh`](tracks_abal.sh) | Supervised-track fractions | 2 |

After configuring the launchers:

```bash
# Dataset sweep: match visibility to the GPU count you configured.
bash data_abal.sh

# Parameter sweep, configured for three GPUs.
CUDA_VISIBLE_DEVICES=0,1,2 bash parameters_abal.sh

# Track sweep, configured for two GPUs.
CUDA_VISIBLE_DEVICES=0,1 bash tracks_abal.sh
```

All parameter presets have one transformer block and 12 attention heads. Their transformer widths are 192, 384, 768, 1152, and 1536; convolutional widths scale too. `full` is the original custom baseline.

Track selection has a separate seed, defaulting to training seed 42. The selected indices are saved in `track_selection.json`, and subsets are nested for a fixed selection seed. For example:

```bash
MODEL_SIZE=tiny bash parameters_abal.sh
TRACK_PERCENT=1 bash tracks_abal.sh --track-seed 43
```

Only selected tracks contribute to track-sweep loss and evaluation; the output-head sizes remain unchanged. Dataset sweeps retain all tracks and leave validation/test sets unchanged. Completed runs are skipped through `.completed` markers. Use a new output root to preserve earlier trials; `FORCE=1` reruns completed experiments.

GPU count affects automatic gradient accumulation and schedule scaling. The archived track runs used global batch 32 and 10,000 steps; parameter runs used global batch 36 and 8,889 steps. Dataset runs used 10,000 steps, with different evaluation intervals for the full-data run. Consult [run settings](results/run-settings/) and the [W&B exports](analysis/raw/wandb/) when matching an experiment. Matching seeds alone does not guarantee bitwise reproduction across hardware or software versions.

### Rerun the matched-track benchmark

Train the track models first; their checkpoints are not included. Then compare each focused model with the full model on the focused model's selected tracks:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.benchmark_track_ablation \
  --outputs-root ./OUTPUTS \
  --data-dir ./DATASET \
  --checkpoint final --precision bf16 --batch-size 1 \
  --cache ram --ram-budget-gib 80 \
  --output-dir ./OUTPUTS/reproduced-track-comparison
```

Use a new benchmark output directory. Default discovery uses the seed-42 run names. For other trials, pass `--baseline OUTPUTS/<full-run>` and `--runs OUTPUTS/<focused-run> ...`. Use `--cache none` if RAM is limited; the archived RAM cache used approximately 15.6 GiB plus loading scratch and model overhead.

The benchmark evaluates 1,937 human and 2,017 mouse native test examples without augmentation. Pearson is calculated per track across pooled examples and bins, then averaged over tracks with defined correlations in both models. Positive Pearson differences favor focused supervision. `--checkpoint final` loads run-root weights exported after the validation-selected best checkpoint was restored.

### Long runs and resuming

Use `tmux new -s scaling` before starting a run over SSH. Detach with Ctrl+B, then D, and reconnect with `tmux attach -t scaling`.

Reusing an output directory does not automatically resume training. Resume explicitly with `--resume-from-checkpoint OUTPUTS/<run>/checkpoint-<step>`, preserving the original schedule. This historical training snapshot predates later recovery fixes; fresh runs are the supported path for reproducing the experiments.

## Repository provenance and license

The training source was reconstructed from pre-conservation backups and recorded server results. It preserves the custom architecture used for these scaling experiments and the executed RAM-cache benchmark. [`SNAPSHOT_SHA256SUMS`](SNAPSHOT_SHA256SUMS) records the original copied snapshot; later analysis and documentation additions are tracked in Git.

Source code is distributed under the [MIT license](LICENSE). Dataset and official model-weight access terms are separate.
