#!/usr/bin/env bash
set -euo pipefail

# Batch-run ViPE on the pose_demo GT videos and evaluate against the matching
# *_aligned_processed.csv files using all GT frame_idx rows.
#
# Default layout:
#   pose_estimation/
#     adapter/scripts/{prepare_gt_full_input.py,run_adapter.py,evaluate_with_gt.py}
#     pose_demo/data/*_aligned.mp4
#     pose_demo/data/telemetry_vehicle_nav_*_aligned_processed.csv
#     inputs/
#     outputs/
#
# Usage:
#   bash scripts/run_vipe_full_gt_batch.sh
#
# Useful overrides:
#   LIMIT=1 bash scripts/run_vipe_full_gt_batch.sh
#   FORCE=1 bash scripts/run_vipe_full_gt_batch.sh
#   RUN_EVAL=0 bash scripts/run_vipe_full_gt_batch.sh
#   GT_ROOT=/path/to/data OUTPUT_ROOT=/path/to/outputs bash scripts/run_vipe_full_gt_batch.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${POSE_ROOT}"

GT_ROOT="${GT_ROOT:-${POSE_ROOT}/pose_demo/data}"
INPUT_ROOT="${INPUT_ROOT:-${POSE_ROOT}/inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${POSE_ROOT}/outputs}"
PY="${PY:-${POSE_ROOT}/../envs/vipe/bin/python}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
FORCE="${FORCE:-0}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

VIPE_MODEL_ROOT="${VIPE_MODEL_ROOT:-/mlp-01/duanmengxuan/data_pipeline/vipe/models}"
export TORCH_HOME="${VIPE_MODEL_ROOT}/torch"
export HF_HOME="${VIPE_MODEL_ROOT}/huggingface"

PREPARE_SCRIPT="${POSE_ROOT}/adapter/scripts/prepare_gt_full_input.py"
RUN_ADAPTER_SCRIPT="${POSE_ROOT}/adapter/scripts/run_adapter.py"
EVAL_SCRIPT="${POSE_ROOT}/adapter/scripts/evaluate_with_gt.py"
SUMMARY_SCRIPT="${POSE_ROOT}/adapter/scripts/summarize_vipe_full_gt.py"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

if [[ ! -d "${GT_ROOT}" ]]; then
  echo "GT_ROOT not found: ${GT_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${PY}" ]]; then
  echo "PY not found or not executable: ${PY}" >&2
  exit 1
fi
for required in "${PREPARE_SCRIPT}" "${RUN_ADAPTER_SCRIPT}" "${EVAL_SCRIPT}" "${SUMMARY_SCRIPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "required script not found: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${INPUT_ROOT}" "${OUTPUT_ROOT}"

echo "POSE_ROOT=${POSE_ROOT}"
echo "GT_ROOT=${GT_ROOT}"
echo "INPUT_ROOT=${INPUT_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "PY=${PY}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "RUN_EVAL=${RUN_EVAL}"
echo "RUN_SUMMARY=${RUN_SUMMARY}"
echo "FORCE=${FORCE}"
echo "LIMIT=${LIMIT}"
echo "DRY_RUN=${DRY_RUN}"
echo

shopt -s nullglob
mp4s=("${GT_ROOT}"/*_aligned.mp4)
IFS=$'\n' mp4s=($(printf '%s\n' "${mp4s[@]}" | sort))
unset IFS

if (( ${#mp4s[@]} == 0 )); then
  echo "No *_aligned.mp4 found under ${GT_ROOT}" >&2
  exit 1
fi

processed=0
for mp4 in "${mp4s[@]}"; do
  if (( LIMIT > 0 && processed >= LIMIT )); then
    break
  fi

  base="$(basename "${mp4}")"
  stem="${base%_aligned.mp4}"
  gt_csv="${GT_ROOT}/telemetry_vehicle_nav_${stem}_aligned_processed.csv"
  clip_id="gt_${stem}_full"
  clip_in="${INPUT_ROOT}/${clip_id}"
  clip_out="${OUTPUT_ROOT}/${clip_id}"
  vipe_out="${clip_out}/vipe"

  echo "============================================================"
  echo "[clip] ${clip_id}"
  echo "mp4=${mp4}"
  echo "gt_csv=${gt_csv}"

  if [[ ! -f "${gt_csv}" ]]; then
    echo "[skip] missing aligned_processed CSV: ${gt_csv}" >&2
    continue
  fi

  if [[ "${FORCE}" == "1" || ! -f "${clip_in}/meta.json" ]]; then
    echo "[prepare] full GT frame input -> ${clip_in}"
    run_cmd "${PY}" "${PREPARE_SCRIPT}" "${mp4}" "${gt_csv}" "${clip_in}"
  else
    echo "[skip prepare] ${clip_in}/meta.json exists"
  fi

  if [[ "${FORCE}" == "1" || ! -f "${vipe_out}/poses_c2w.npy" || ! -f "${vipe_out}/intrinsics.npy" ]]; then
    echo "[run vipe] ${clip_id}"
    run_cmd "${PY}" "${RUN_ADAPTER_SCRIPT}" \
      --method vipe \
      --clip-dir "${clip_in}" \
      --out-root "${OUTPUT_ROOT}"
  else
    echo "[skip vipe] ${vipe_out} exists"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${FORCE}" == "1" || ! -f "${clip_out}/gt_eval.json" ]]; then
      echo "[evaluate] ${clip_id}"
      run_cmd "${PY}" "${EVAL_SCRIPT}" \
        --clip-out "${clip_out}" \
        --clip-in "${clip_in}" \
        --gt-csv "${gt_csv}" \
        --axis-search
    else
      echo "[skip eval] ${clip_out}/gt_eval.json exists"
    fi
  fi

  processed=$((processed + 1))
  echo
done

if [[ "${RUN_EVAL}" == "1" && "${RUN_SUMMARY}" == "1" ]]; then
  echo "[summarize] ${OUTPUT_ROOT}/vipe_full_gt_summary.{csv,md}"
  run_cmd "${PY}" "${SUMMARY_SCRIPT}" \
    --outputs-root "${OUTPUT_ROOT}" \
    --pattern "gt_*_full" \
    --csv-out "${OUTPUT_ROOT}/vipe_full_gt_summary.csv" \
    --md-out "${OUTPUT_ROOT}/vipe_full_gt_summary.md"
fi

echo "Done. processed=${processed}"
