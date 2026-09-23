#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NPROC_PER_NODE=4
TRAIN_SCRIPT="scripts/train_enformer.py"
DATA_DIR="/scr/jsomeara/DATASET"
OUTPUT_ROOT="./OUTPUTS"
RUN_PREFIX="basic-model"

COMMON_ARGS=(
  --num-gpus 4
  --data-dir "${DATA_DIR}"
  --human-genome "${DATA_DIR}/genome/hg38.fa"
  --mouse-genome "${DATA_DIR}/genome/mm10.fa"
  --report-to wandb
  --per-device-train-batch-size 2
  --per-device-eval-batch-size 2
  --dataloader-num-workers 4
  --use-checkpointing
  --human-train-h5 "${DATA_DIR}/human_train_fast.h5"
  --mouse-train-h5 "${DATA_DIR}/mouse_train_fast.h5"
)

# Parallel arrays: fraction passed to Python, human-readable label for run/output names.
FRACTIONS=(1.00 0.01 0.10 0.03 0.30)
LABELS=(100 1 10 3 30)

for i in "${!FRACTIONS[@]}"; do
  fraction="${FRACTIONS[$i]}"
  label="${LABELS[$i]}"

  run_name="${RUN_PREFIX}-data${label}pct"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  completed_marker="${output_dir}/.completed"

  if [[ -f "${completed_marker}" && "${FORCE:-0}" != "1" ]]; then
    echo
    echo "============================================================"
    echo "Skipping ${run_name}: ${completed_marker} exists"
    echo "Set FORCE=1 to rerun completed experiments."
    echo "============================================================"
    continue
  fi

  mkdir -p "${output_dir}"

  echo
  echo "============================================================"
  echo "Starting dataset-size ablation"
  echo "  fraction:   ${fraction}"
  echo "  percent:    ${label}%"
  echo "  run name:   ${run_name}"
  echo "  output dir: ${output_dir}"
  echo "  GPUs:       ${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  uv run torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --train-fraction "${fraction}" \
    --output-dir "${output_dir}" \
    --run-name "${run_name}" \
    "$@" \
    2>&1 | tee "${output_dir}/run.log"

  touch "${completed_marker}"

  echo
  echo "Completed ${run_name}"
done

echo
echo "============================================================"
echo "All requested dataset-size ablations are complete."
echo "============================================================"
