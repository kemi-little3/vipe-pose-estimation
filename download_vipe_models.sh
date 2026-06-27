#!/usr/bin/env bash
# Download all models needed by ViPE (default + optional pipelines).
# Runs on the HOST — this box has no direct internet, must go through local proxy.
#
# Layout:
#   $TORCH_HOME/hub/{checkpoints,geocalib,sam,aot,droid_slam}/...   (wget / torch.hub style)
#   $HF_HOME/hub/models--<org>--<name>/...                          (HuggingFace cache style)
#
# Usage:
#   ./download.sh            # default pipeline (~4 GB)
#   ./download.sh lyra       # + MoGE ViT-L          (~1.5 GB extra)
#   ./download.sh dav3       # + Depth-Anything-V3   (~6 GB extra, may need direct hf.co)
#
# Idempotent: existing files are skipped.

set -euo pipefail

# --- destinations ---
MODEL_ROOT="${MODEL_ROOT:-/mlp-01/models/ViPE_models}"
export TORCH_HOME="${MODEL_ROOT}/torch"
export HF_HOME="${MODEL_ROOT}/huggingface"
LOCAL_HF_TOKEN_FILE="${LOCAL_HF_TOKEN_FILE:-/mlp-01/models/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -f "${LOCAL_HF_TOKEN_FILE}" ]]; then
    export HF_TOKEN="$(cat "${LOCAL_HF_TOKEN_FILE}")"
fi

# --- network ---
# hf-mirror.com used to be reliable but now returns 308 redirects back to
# huggingface.co for many repos (e.g. bert-base-uncased), which confuses
# huggingface_hub. Since our local proxy can reach huggingface.co directly,
# just talk to the real endpoint. To force the mirror, uncomment the next line.
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 30s socket timeout (default 10s is too aggressive for proxied links)
export HF_HUB_DOWNLOAD_TIMEOUT=30
# hf_transfer is too aggressive for proxied connections — disable.
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "${TORCH_HOME}"/hub/{checkpoints,geocalib,sam,aot,droid_slam}
mkdir -p "${HF_HOME}/hub"

EXTRA="${1:-}"

# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

wget_if_missing() {
    # Args: <url> <dst_path>
    local url="$1" dst="$2"
    if [ -s "$dst" ]; then
        echo "[skip] $(basename "$dst") (already present)"
    else
        echo "[wget] -> $dst"
        wget -c --show-progress "$url" -O "$dst"
    fi
}

gdrive_if_missing() {
    # Args: <google_drive_file_id> <dst_path>
    local gid="$1" dst="$2"
    if [ -s "$dst" ]; then
        echo "[skip] $(basename "$dst") (already present)"
    else
        echo "[gdown] -> $dst"
        gdown "$gid" -O "$dst"
    fi
}

hf_download() {
    # Args: <repo_id> [file1 file2 ...]
    #   If files are given, download only those. Otherwise snapshot the whole repo.
    local repo_id="$1"
    shift
    if [ $# -eq 0 ]; then
        echo "[hf]   $repo_id  (full repo)"
        hf download "$repo_id"
    else
        echo "[hf]   $repo_id  ($*)"
        hf download "$repo_id" "$@"
    fi
}

ensure_tool() {
    # Args: <cmd> <pip_pkg>
    local tool="$1" pip_pkg="$2"
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[info] $tool missing, pip install $pip_pkg"
        pip install "$pip_pkg"
    fi
}

check_file() {
    if [ -s "$1" ]; then
        echo "  OK  $1  ($(du -h "$1" | cut -f1))"
    else
        echo "  !!  $1  MISSING"
    fi
}

check_hf_repo() {
    local repo_id="$1"
    local dir="${HF_HOME}/hub/models--${repo_id//\//--}"
    if [ -d "$dir" ]; then
        echo "  OK  $repo_id  ($(du -sh "$dir" | cut -f1))"
    else
        echo "  !!  $repo_id  MISSING"
    fi
}

# ---------------------------------------------------------
# 1/3   direct URLs (torch.hub cache layout)
# ---------------------------------------------------------

echo "========== 1/3  wget =========="

wget_if_missing \
    "https://github.com/rpautrat/SuperPoint/raw/refs/heads/master/weights/superpoint_v6_from_tf.pth" \
    "${TORCH_HOME}/hub/checkpoints/superpoint_v6_from_tf.pth"

wget_if_missing \
    "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar" \
    "${TORCH_HOME}/hub/geocalib/pinhole.tar"

wget_if_missing \
    "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth" \
    "${TORCH_HOME}/hub/checkpoints/video_depth_anything_vits.pth"

wget_if_missing \
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth" \
    "${TORCH_HOME}/hub/checkpoints/groundingdino_swint_ogc.pth"

wget_if_missing \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
    "${TORCH_HOME}/hub/sam/sam_vit_b_01ec64.pth"

# ---------------------------------------------------------
# 2/3   HuggingFace repos (HF cache layout)
# ---------------------------------------------------------

echo
echo "========== 2/3  huggingface =========="

ensure_tool hf "huggingface_hub[cli]"

# GroundingDINO text encoder — transformers looks this up at runtime via
# AutoTokenizer/BertModel.from_pretrained("bert-base-uncased"). Download only
# the PyTorch/tokenizer files needed at runtime instead of the full repo, which
# also contains TF/Rust/CoreML artifacts.
hf_download bert-base-uncased \
    config.json \
    pytorch_model.bin \
    tokenizer.json \
    tokenizer_config.json \
    vocab.txt

# UniDepth V2 ViT-L — default keyframe_depth + post.depth_align_model.
hf_download lpiccinelli/unidepth-v2-vitl14

# PriorDA — only two specific files are actually loaded.
hf_download Rain729/Prior-Depth-Anything \
    depth_anything_v2_vitb.pth \
    prior_depth_anything_vitb.pth

# Pipeline-specific extras
if [[ "$EXTRA" == "lyra" ]]; then
    hf_download Ruicheng/moge-vitl
fi

if [[ "$EXTRA" == "dav3" ]]; then
    hf_download depth-anything/DA3-GIANT
    hf_download depth-anything/DA3METRIC-LARGE
fi

# ---------------------------------------------------------
# 3/3   Google Drive (gdown)
# ---------------------------------------------------------

echo
echo "========== 3/3  Google Drive =========="

ensure_tool gdown "gdown"

gdrive_if_missing "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh" \
    "${TORCH_HOME}/hub/droid_slam/droid.pth"

gdrive_if_missing "1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ" \
    "${TORCH_HOME}/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth"

# ---------------------------------------------------------
# verify
# ---------------------------------------------------------

echo
echo "========== verify =========="
echo "TORCH_HOME = ${TORCH_HOME}"
echo "HF_HOME    = ${HF_HOME}"
echo

check_file "${TORCH_HOME}/hub/checkpoints/superpoint_v6_from_tf.pth"
check_file "${TORCH_HOME}/hub/geocalib/pinhole.tar"
check_file "${TORCH_HOME}/hub/checkpoints/video_depth_anything_vits.pth"
check_file "${TORCH_HOME}/hub/checkpoints/groundingdino_swint_ogc.pth"
check_file "${TORCH_HOME}/hub/sam/sam_vit_b_01ec64.pth"
check_file "${TORCH_HOME}/hub/droid_slam/droid.pth"
check_file "${TORCH_HOME}/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth"

echo
check_hf_repo bert-base-uncased
check_hf_repo lpiccinelli/unidepth-v2-vitl14
check_hf_repo Rain729/Prior-Depth-Anything
if [[ "$EXTRA" == "lyra" ]]; then
    check_hf_repo Ruicheng/moge-vitl
fi
if [[ "$EXTRA" == "dav3" ]]; then
    check_hf_repo depth-anything/DA3-GIANT
    check_hf_repo depth-anything/DA3METRIC-LARGE
fi

echo
echo "========== done =========="
echo
echo "Every new shell that runs vipe needs:"
echo
echo "  export TORCH_HOME=${TORCH_HOME}"
echo "  export HF_HOME=${HF_HOME}"
# echo "  export HF_ENDPOINT=${HF_ENDPOINT}"
echo "  export http_proxy=${http_proxy}"
echo "  export https_proxy=${https_proxy}"
echo
echo "Add these to ~/.zshrc (or source an activate script) to make it persistent."
