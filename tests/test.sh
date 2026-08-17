#!/usr/bin/env bash
# Runs the verifier evaluate.py against the agent's submission dir.
# Usage (in the task image after an agent run): bash /workspace/tests/test.sh
set -euo pipefail
python3 /workspace/tests/evaluate.py
echo "verifier exit: $? — see /logs/verifier/reward.txt"