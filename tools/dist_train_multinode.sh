#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr ${MASTER_ADDR} \
    --master_port=${PORT} \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
