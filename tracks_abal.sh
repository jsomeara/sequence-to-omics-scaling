#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NPROC_PER_NODE=4
TRAIN_SCRIPT="scripts/train_enformer.py"
DATA_DIR="/scr/jsomeara/DATASET"
OUTPUT_ROOT="./OUTPUTS"
# Positional training options are forwarded, but fraction/count options belong
# to this sweep. Resolve seeds for collision-free trial names.
TRAIN_SEED=42
TRACK_SEED=""
extra_args=("$@")
for ((j=0; j<${#extra_args[@]}; j++)); do
  arg="${extra_args[$j]}"
  case "${arg}" in
    --seed|--track-seed)
      if ((j + 1 >= ${#extra_args[@]})); then
        echo "${arg} requires an integer" >&2; exit 2
      fi
      ((j+=1))
      if [[ "${arg}" == "--seed" ]]; then TRAIN_SEED="${extra_args[$j]}"; else TRACK_SEED="${extra_args[$j]}"; fi
      ;;
    --seed=*) TRAIN_SEED="${arg#*=}" ;;
    --track-seed=*) TRACK_SEED="${arg#*=}" ;;
    --track-fraction|--track-fraction=*|--human-track-count|--human-track-count=*|--mouse-track-count|--mouse-track-count=*|--output-dir|--output-dir=*|--run-name|--run-name=*)
      echo "${arg} is controlled by tracks_abal.sh; invoke train_enformer.py directly for a single custom run." >&2
      exit 2 ;;
  esac
done
TRACK_SEED="${TRACK_SEED:-${TRAIN_SEED}}"
if [[ ! "${TRAIN_SEED}" =~ ^[0-9]+$ || ! "${TRACK_SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "--seed must be a nonnegative integer; --track-seed must be an integer" >&2; exit 2
fi
RUN_PREFIX="basic-model-trainseed${TRAIN_SEED}-trackseed${TRACK_SEED}"

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
if [[ -n "${TRACK_PERCENT:-}" ]]; then
  case "${TRACK_PERCENT}" in
    100|1|10|3|30) ;;
    *) echo "TRACK_PERCENT must be one of 100, 1, 10, 3, 30" >&2; exit 2 ;;
  esac
fi

for i in "${!FRACTIONS[@]}"; do
  fraction="${FRACTIONS[$i]}"
  label="${LABELS[$i]}"
  if [[ -n "${TRACK_PERCENT:-}" && "${TRACK_PERCENT}" != "${label}" ]]; then
    continue
  fi

  run_name="${RUN_PREFIX}-tracks${label}pct"
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
  echo "Starting track-count ablation"
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
    --track-fraction "${fraction}" \
    --track-seed "${TRACK_SEED}" \
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
echo "All requested track-count ablations are complete."
echo "============================================================"
