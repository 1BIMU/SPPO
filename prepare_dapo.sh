#!/usr/bin/env bash
set -uxo pipefail
source uv_verl/bin/activate

export TRAIN_FILE=${TRAIN_FILE:-"data/dapo-math-17k.parquet"}
export OVERWRITE=${OVERWRITE:-1}


if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

python transformer_dapo_prmopt.py