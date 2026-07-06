#!/usr/bin/env bash
# Run a real LLM doctor (SUT) against a case
# Usage: ./scripts/run_doctor.sh --model claude-opus-4-8 --perception frame_stream -p cases/<...>

set -euo pipefail
python scripts/run_doctor.py "$@"
