#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
WORK_DIR=$2
GPUS=$3
PY_ARGS=${@:4}
PORT=${PORT:-29500}
SEED=${SEED:-0}

# WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --work_dir $WORK_DIR --seed $SEED --launcher pytorch ${PY_ARGS}
