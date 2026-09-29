#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

: "${DATA_DIR:?Set DATA_DIR to a unified Stage-III parquet file or directory}"
: "${PRETRAINED:?Set PRETRAINED to a Gestalt/LLaDA checkpoint directory}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/stage3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/training/gestalt_stage_3.yaml}"
DEEPSPEED="${DEEPSPEED:-${PROJECT_ROOT}/configs/deepspeed/zero_stage2.json}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

[[ -f "$CONFIG" ]] || { echo "Training config not found: $CONFIG" >&2; exit 2; }
[[ "$DEEPSPEED" == "none" || -f "$DEEPSPEED" ]] || {
    echo "DeepSpeed config not found: $DEEPSPEED" >&2
    exit 2
}

mkdir -p "${OUTPUT_DIR}/logs"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/training/train_stage3.py \
    --config "$CONFIG" \
    --data "$DATA_DIR" \
    --pretrained "$PRETRAINED" \
    --deepspeed "$DEEPSPEED" \
    --output "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"
