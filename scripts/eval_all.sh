#!/usr/bin/env bash
# Evaluate all traces and produce stratified leaderboard
# Usage: ./scripts/eval_all.sh --suite v3 --report score,tokens,sim_cost,safety --stratify difficulty,perception

set -euo pipefail
python scripts/eval_all.py "$@"
