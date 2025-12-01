#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/molex_ssl.conf}"
META_DIR="${META_DIR:-/path/to/meta}"
FEAT_FILE="${FEAT_FILE:-/path/to/wav.scp}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/experiments}"
EXP_IDX="${EXP_IDX:-0}"
SEED="${SEED:-1234}"
NUM_GPUS="${NUM_GPUS:-2}"


torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${REPO_DIR}/src/main.py" \
  --config "${CONFIG_PATH}" \
  --meta_dir "${META_DIR}" \
  --feat_file "${FEAT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --exp_idx "${EXP_IDX}" \
  --seed "${SEED}"

