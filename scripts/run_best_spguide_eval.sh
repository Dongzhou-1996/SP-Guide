#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/paper_eval_records/spguide_online_gcf_${STAMP}}"
exec "$REPO_ROOT/scripts/run_spguide_gcf_rollout_best.sh" "$@"
