#!/bin/bash
# Launch script for AfriqueLLM continued pre-training via LLaMA-Factory.
#
# Single node (adjust NUM_GPUS to however many you have):
#   bash run_train.sh
#
# Multi-node (run on each node, set MASTER_ADDR to the IP of node 0):
#   MASTER_ADDR=<node0_ip> NNODES=4 NODE_RANK=<0,1,2,3> bash run_train.sh

set -e

# ── configuration ───────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

CONFIG="train_cpt.yaml"
# ────────────────────────────────────────────────────────────────────────────

echo "Launching with:"
echo "  GPUs per node : $NUM_GPUS"
echo "  Nodes         : $NNODES"
echo "  Node rank     : $NODE_RANK"
echo "  Master        : $MASTER_ADDR:$MASTER_PORT"
echo "  Config        : $CONFIG"

# Compute gradient_accumulation_steps to hit 4M token global batch size.
# global_batch_tokens = per_device_batch * grad_accum * num_gpus_total * seq_len
# 4_000_000 = 4 * grad_accum * (NUM_GPUS * NNODES) * 16384
# grad_accum = 4_000_000 / (4 * NUM_GPUS * NNODES * 16384)
TOTAL_GPUS=$(( NUM_GPUS * NNODES ))
GRAD_ACCUM=$(python3 -c "import math; v = 4_000_000 / (4 * $TOTAL_GPUS * 16384); print(max(1, math.ceil(v)))")
echo "  Computed grad_accumulation_steps: $GRAD_ACCUM (for 4M token global batch)"

# Patch gradient_accumulation_steps in the YAML on the fly
# (avoids editing the file manually each time you change node count)
PATCHED_CONFIG=$(mktemp /tmp/train_cpt_XXXX.yaml)
sed "s/^gradient_accumulation_steps:.*/gradient_accumulation_steps: $GRAD_ACCUM/" "$CONFIG" > "$PATCHED_CONFIG"

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NUM_GPUS" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m llamafactory.train \
    "$PATCHED_CONFIG"

rm -f "$PATCHED_CONFIG"
