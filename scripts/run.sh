#!/usr/bin/env bash
# Run oracle or nop agent against a single case
# Usage: ./scripts/run.sh -p cases/neuro/MW-NEURO-01234 --agent oracle

set -euo pipefail
python scripts/run_doctor.py "$@"
