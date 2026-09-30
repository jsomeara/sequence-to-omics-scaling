# Reproduce the post's analyses

Run from the repository root. This analysis needs no GPUs, datasets, checkpoints,
W&B account, or API key. The committed scalar measurements are sufficient.

```bash
uv venv .analysis-venv --python 3.11
uv pip install --python .analysis-venv/bin/python -r analysis/requirements.txt
.analysis-venv/bin/python analysis/generate.py
```

Outputs go to `analysis/generated/` (ignored by Git): PNG/PDF/SVG figures,
full-precision CSV tables, Markdown and rendered image tables, equation images, and `fits.json`.
Use `--output-dir PATH` to change the directory. Both scaling plots extend to
100 times the largest measured scale; use `--extrapolation-factor 10` to shorten
that range. Solid curves cover measured scales; dashed curves are extrapolations.

The script creates:

- Dataset and parameter scaling plots, fitted equations and tables.
- Focused versus full supervision plots, track differences and matched-track table.
- Original-encoding and RoPE learning curves, plus a combined depth plot and table
  including both RoPE models at steps 6,000 and 11,000.
- Separate layer cosine plots and individual layer-by-sequence tables for Enformer,
  Borzoi and AlphaGenome, using the completed notebook's measured results.
- Pearson, species-mean Pearson, matched-track difference and cosine metric equations.

## Sources and scope

`run_manifest.json` explicitly identifies the 20 training runs relevant to the
post: five dataset fractions, five parameter sizes, five track fractions, and
five transformer-depth/encoding runs. No other training runs are included.

`raw/wandb/archived_analysis_metrics.json` preserves the W&B validation histories
and final test summaries used when writing the post. These published values
remain the canonical source even if an online run is later continued or edited.
`raw/wandb/runs/` contains the unsampled numeric scalar histories and numeric
summaries exported from those allowlisted runs. Each export records its run ID,
URL, name, group, state and export time. Config exports retain numeric settings
and a small set of method labels; text/media, filesystem paths, credentials,
artifacts and checkpoints are omitted. Missing scalar fields remain missing;
metrics are not interpolated or smoothed.

The parameter table reads the original `results/basic-model-parameters-*/`
`test_results.json` and `model_architecture.json`. The track comparison reads
`results/track-comparison-20260919T202732.863735Z/`; its summaries are checked
against `per_track.csv`, averaging only tracks with defined correlations in
both models. Original track selections and benchmark provenance are already
preserved under `results/`.

`raw/window_counts.json` contains reconstructed window counts using the archived
seed-42 sampling algorithm and training BED counts. These are not original saved
subset manifests or counts of augmented sequences actually observed.

`raw/transformer_usage/` contains the raw per-sequence/per-layer cosine scores,
coordinates, source commits, environment records and AlphaGenome validation
exported from `Transformer_Usage_Analysis.ipynb`. These are eight diagnostic
hg38 chr22 windows, not a shared held-out benchmark. The notebook documents
AlphaGenome's T4 numerical adaptation and tiled convolutional encoder validation.
High cosine does not establish that a block is unnecessary.

## Fitting method

Both scaling axes use the same three-parameter additive-offset power law:

$$\widehat{\bar r}(x)=r_\infty-Ax^{-\alpha}.$$

Dataset x is the training fraction; parameter x is parameter count in millions.
We use unweighted nonlinear least squares on the Pearson scale, with multiple
initializations and constraints `max(observed) <= r_inf <= 1`, `A >= 0`,
`0.001 <= alpha <= 5`. RMSE is in-sample. Fits reproduce the post to displayed
precision; tiny differences in solver output can depend on library versions.
The asymptote is conditional on the fitted form, not a proven performance ceiling.
There is one training seed per scale and no repeat-seed confidence interval.
Dataset/parameter test metrics, depth validation metrics and the native-test
matched-track benchmark are distinct evaluations.

## Refresh the allowlisted W&B scalar export (optional)

Offline figure generation does not require this step. Install the separate API
client and authenticate locally; never commit an API key:

```bash
uv pip install --python .analysis-venv/bin/python wandb==0.19.11
.analysis-venv/bin/wandb login
.analysis-venv/bin/python analysis/export_wandb.py
```

`WANDB_API_KEY` is also supported. The exporter uses `scan_history()` to retrieve
all logged history rows rather than sampled `history()`. It validates the explicit
allowlist and refuses ambiguous unpinned names. It does not fetch W&B artifacts,
model weights, datasets or media. To export somewhere else, use `--output PATH`.
