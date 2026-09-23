#!/usr/bin/env bash
set -euo pipefail

# Five width presets, exactly one transformer block in every model.
# full is the existing custom architecture; all other presets are smaller.
# Usage: bash parameters_abal.sh
# Extra training arguments are forwarded, e.g. --seed 43 or --max-steps 50000.
# Rerun completed experiments: FORCE=1 bash parameters_abal.sh
# Run one preset only: MODEL_SIZE=tiny bash parameters_abal.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NPROC_PER_NODE=4
TRAIN_SCRIPT="scripts/train_enformer.py"
DATA_DIR="/scr/jsomeara/DATASET"
OUTPUT_ROOT="./OUTPUTS"
TRAIN_SEED=42
extra_args=("$@")
for ((j=0; j<${#extra_args[@]}; j++)); do
  arg="${extra_args[$j]}"
  case "${arg}" in
    --seed)
      if ((j + 1 >= ${#extra_args[@]})); then echo "--seed requires an integer" >&2; exit 2; fi
      ((j+=1)); TRAIN_SEED="${extra_args[$j]}" ;;
    --seed=*) TRAIN_SEED="${arg#*=}" ;;
    --model|--model=*|--model-size|--model-size=*|--output-dir|--output-dir=*|--run-name|--run-name=*)
      echo "${arg} is controlled by parameters_abal.sh; invoke train_enformer.py directly for a custom run." >&2
      exit 2 ;;
  esac
done
if [[ ! "${TRAIN_SEED}" =~ ^[0-9]+$ ]]; then echo "--seed must be a nonnegative integer" >&2; exit 2; fi
RUN_PREFIX="basic-model-parameters-seed${TRAIN_SEED}"

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

# Start with the unchanged baseline, then increase width among smaller models.
MODEL_SIZES=(full tiny small medium large)
if [[ -n "${MODEL_SIZE:-}" ]]; then
  case "${MODEL_SIZE}" in
    full|tiny|small|medium|large) ;;
    *) echo "MODEL_SIZE must be full, tiny, small, medium, or large" >&2; exit 2 ;;
  esac
fi

for model_size in "${MODEL_SIZES[@]}"; do
  if [[ -n "${MODEL_SIZE:-}" && "${MODEL_SIZE}" != "${model_size}" ]]; then continue; fi
  run_name="${RUN_PREFIX}-${model_size}"
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
  echo "Starting parameter-count ablation"
  echo "  model size: ${model_size}"
  echo "  run name:   ${run_name}"
  echo "  output dir: ${output_dir}"
  echo "  GPUs:       ${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  uv run torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --model-size "${model_size}" \
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
echo "All requested parameter-count ablations are complete."
echo "============================================================"
