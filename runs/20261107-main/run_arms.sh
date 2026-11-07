#!/bin/bash
# Arms A0..A3 on DEV only (train + calibration). A3 calls `claude -p --model claude-opus-5-5`.
set -e
cd "$(dirname "$0")"
../../.venv/bin/python -m liedet wsp-run --data data --features features --out arms_followup --variant followup \
  --proposer claude --proposer-model claude-opus-5-5 --rounds 6 --per-round 5 --seed 0
