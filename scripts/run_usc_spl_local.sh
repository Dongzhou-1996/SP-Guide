#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="/home/tanghx/miniconda3/envs/gdino/bin/python"
CKPT="/home/tanghx/VLN/refer_repo/USCNav/checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/logs"
LOG_PATH="$LOG_DIR/usc_gcf_spl_local_clean_${STAMP}.log"

mkdir -p "$LOG_DIR"

echo "LOG=$LOG_PATH" | tee -a "$LOG_PATH"
echo "START $(date)" | tee -a "$LOG_PATH"

for split in val_seen val_unseen test_unseen; do
  echo "=== ${split} $(date) ===" | tee -a "$LOG_PATH"
  "$PYTHON_BIN" -u scripts/eval_single_split.py \
    --split "${split}" \
    --model instr_decoder_usc_with_map \
    --checkpoint "$CKPT" \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_use_map_cache \
    --gsam_box_threshold 0.20 \
    --eval_batch_size 100 \
    --eval_max_timestep 20 \
    --eval_agent_mode gcf \
    --eval_goal_selector raw \
    --progress_stop_val 0.75 \
    --gcf_window 5 \
    --gcf_beta 8.0 \
    --gcf_progress_thr 0.80 \
    --gcf_min_step 8 \
    --gcf_arrival_dist 15.0 \
    --gcf_debug 2>&1 | tee -a "$LOG_PATH"
done

echo "DONE $(date)" | tee -a "$LOG_PATH"
