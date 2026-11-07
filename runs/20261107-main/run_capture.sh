#!/bin/bash
# Reproduces data build + activation capture. Seeds fixed; model revision pinned by the snapshot path.
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
MODEL=/home/shumway/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/
$PY -m liedet build-data --out data --seed 0 --n-arith 160
$PY -m liedet capture --model "$MODEL" --data data --out features --dtype bfloat16 --tokens
