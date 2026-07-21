#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${POSE_ROOT}/scripts/run_vipe_full_gt_batch.sh" "$@"
