#!/usr/bin/env bash
set -euo pipefail

# Batch-run MegaSaM on all GT videos under pose_demo/pose_demo and evaluate
# against the matching *_aligned_processed.csv using all GT frame_idx rows.
# Reuses the inputs/gt_<timestamp>_full/ directories prepared by the ViPE
# batch (prepare_gt_full_input.py) — does NOT regenerate them.
#
# Usage:
#   bash scripts/run_megasam_full_gt_batch.sh
#
# Optional:
#   GT_ROOT=/path/to/pose_demo/pose_demo bash scripts/run_megasam_full_gt_batch.sh
#   RUN_EVAL=0 bash scripts/run_megasam_full_gt_batch.sh
#   FORCE=1 bash scripts/run_megasam_full_gt_batch.sh
#   MEGASAM_GPU_ID=0 bash scripts/run_megasam_full_gt_batch.sh
#   ORDER=given bash scripts/run_megasam_full_gt_batch.sh   # disable shortest-first

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GT_ROOT="${GT_ROOT:-/data1/hexuming/pose_estimation/pose_demo/pose_demo}"
MEGASAM_PY="${MEGASAM_PY:-/data1/duanmengxuan/envs/conda_envs/mega_sam/bin/python}"
EVAL_PY="${EVAL_PY:-/home/hexuming/miniconda3/envs/vipe/bin/python}"
RUN_EVAL="${RUN_EVAL:-1}"
FORCE="${FORCE:-0}"
ORDER="${ORDER:-shortest_first}"

export MEGASAM_GPU_ID="${MEGASAM_GPU_ID:-1}"

if [[ ! -d "${GT_ROOT}" ]]; then
  echo "GT_ROOT not found: ${GT_ROOT}" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "GT_ROOT=${GT_ROOT}"
echo "MEGASAM_PY=${MEGASAM_PY}"
echo "EVAL_PY=${EVAL_PY}"
echo "MEGASAM_GPU_ID=${MEGASAM_GPU_ID}"
echo "RUN_EVAL=${RUN_EVAL}"
echo "FORCE=${FORCE}"
echo "ORDER=${ORDER}"
echo

shopt -s nullglob
mp4s=("${GT_ROOT}"/*_aligned.mp4)
if (( ${#mp4s[@]} == 0 )); then
  echo "No *_aligned.mp4 found under ${GT_ROOT}" >&2
  exit 1
fi

# Build (frames, mp4) pairs so we can sort shortest-first to fail fast.
pairs=()
for mp4 in "${mp4s[@]}"; do
  base="$(basename "${mp4}")"
  stem="${base%_aligned.mp4}"
  clip_in="inputs/gt_${stem}_full"
  meta="${clip_in}/meta.json"
  if [[ -f "${meta}" ]]; then
    n=$(${EVAL_PY} -c "import json; print(json.load(open('${meta}'))['num_raw_frames'])")
  else
    n=999999  # unprepared inputs sort last; the loop will skip-or-prepare them
  fi
  pairs+=("${n} ${mp4}")
done

if [[ "${ORDER}" == "shortest_first" ]]; then
  IFS=$'\n' sorted=($(printf '%s\n' "${pairs[@]}" | sort -n))
  unset IFS
  mp4s=()
  for line in "${sorted[@]}"; do
    mp4s+=("$(echo "${line}" | cut -d' ' -f2-)")
  done
fi

mkdir -p logs

for mp4 in "${mp4s[@]}"; do
  base="$(basename "${mp4}")"
  stem="${base%_aligned.mp4}"
  gt_csv="${GT_ROOT}/telemetry_vehicle_nav_${stem}_aligned_processed.csv"
  clip_id="gt_${stem}_full"
  clip_in="inputs/${clip_id}"
  clip_out="outputs/${clip_id}"
  megasam_out="${clip_out}/megasam"
  log_file="logs/megasam_${clip_id}.log"

  echo "============================================================"
  echo "[clip] ${clip_id}"
  echo "mp4=${mp4}"
  echo "gt_csv=${gt_csv}"
  echo "log=${log_file}"

  if [[ ! -f "${gt_csv}" ]]; then
    echo "[skip] missing aligned_processed CSV: ${gt_csv}" >&2
    continue
  fi

  if [[ "${FORCE}" == "1" || ! -f "${clip_in}/meta.json" ]]; then
    echo "[prepare] full GT frame input -> ${clip_in}"
    "${EVAL_PY}" adapter/scripts/prepare_gt_full_input.py "${mp4}" "${gt_csv}" "${clip_in}"
  else
    echo "[skip prepare] ${clip_in}/meta.json exists"
  fi

  if [[ "${FORCE}" == "1" || ! -f "${megasam_out}/poses_c2w.npy" || ! -f "${megasam_out}/intrinsics.npy" ]]; then
    echo "[run megasam] ${clip_id} (gpu=${MEGASAM_GPU_ID})"
    /usr/bin/time -f "MEGASAM_ELAPSED_S=%e" \
      "${MEGASAM_PY}" adapter/scripts/run_adapter.py \
        --method megasam \
        --clip-dir "${clip_in}" \
        --out-root outputs 2>&1 | tee "${log_file}"
  else
    echo "[skip megasam] ${megasam_out} exists"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    if [[ "${FORCE}" == "1" || ! -f "${clip_out}/gt_eval.json" || ! -f "${megasam_out}/poses_c2w.npy" ]]; then
      echo "[evaluate] ${clip_id}"
      "${EVAL_PY}" adapter/scripts/evaluate_with_gt.py \
        --clip-out "${clip_out}" \
        --clip-in "${clip_in}" \
        --gt-csv "${gt_csv}" \
        --axis-search
    else
      # Re-run eval if megasam dir exists but the JSON predates it (so megasam columns get added).
      if [[ "${megasam_out}/poses_c2w.npy" -nt "${clip_out}/gt_eval.json" ]]; then
        echo "[re-evaluate] megasam newer than gt_eval.json"
        "${EVAL_PY}" adapter/scripts/evaluate_with_gt.py \
          --clip-out "${clip_out}" \
          --clip-in "${clip_in}" \
          --gt-csv "${gt_csv}" \
          --axis-search
      else
        echo "[skip eval] ${clip_out}/gt_eval.json up to date"
      fi
    fi
  fi

  echo
done

echo "Done."
