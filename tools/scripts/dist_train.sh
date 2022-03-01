#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo $MASTER_PORT
poetry run python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py  --launcher pytorch ${PY_ARGS}

