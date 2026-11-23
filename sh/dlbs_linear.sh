#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_folds.sh" configs/dlbs/linear.json
