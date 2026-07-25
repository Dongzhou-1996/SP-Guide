#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/tanghx/miniconda3/envs/gdino/bin/python}"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/paper_eval_records/spguide_best_protocol_val_repro_${STAMP}}"
RAW_DIR="$OUT_DIR/raw_rollouts"
OLD_DIR="$OUT_DIR/preexisting_root_json"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_TXT="$OUT_DIR/summary.txt"
CSV_OUT="$OUT_DIR/spguide_best_protocol_metrics.csv"
JSON_OUT="$OUT_DIR/spguide_best_protocol_metrics.json"

mkdir -p "$RAW_DIR" "$OLD_DIR" "$LOG_DIR"

raw_name() {
  local split="$1"
  echo "instr_decoder_usc_with_map_mturk_50.0_0.2_${split}_0.75_progress_stop_raw.json"
}

archive_if_exists() {
  local split="$1"
  local name
  name="$(raw_name "$split")"
  if [[ -f "$REPO_ROOT/$name" ]]; then
    mv "$REPO_ROOT/$name" "$OLD_DIR/${split}_${STAMP}_${name}"
  fi
}

run_raw_split() {
  local split="$1"
  local name
  name="$(raw_name "$split")"

  echo "=== raw progress-stop rollout: ${split} ===" | tee -a "$SUMMARY_TXT"
  "$PYTHON_BIN" -u "$REPO_ROOT/scripts/eval_single_split.py" \
    --split "$split" \
    --model instr_decoder_usc_with_map \
    --checkpoint "$CKPT" \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_use_map_cache \
    --gsam_box_threshold 0.20 \
    --eval_batch_size 100 \
    --eval_max_timestep 20 \
    --eval_agent_mode progress_stop \
    --eval_goal_selector raw \
    --progress_stop_val 0.75 \
    2>&1 | tee "$LOG_DIR/${split}_raw.log"

  mv "$REPO_ROOT/$name" "$RAW_DIR/$name"
}

{
  echo "OUT_DIR=$OUT_DIR"
  echo "CKPT=$CKPT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "START $(date)"
  echo "PROTOCOL=raw progress_stop rollout + GCF terminal fusion"
  echo "RAW_ROLLOUT progress_stop_val=0.75 eval_max_timestep=20 eval_batch_size=100 altitude=50 gsam_box_threshold=0.20"
  echo "GCF window=5 beta=4.0"
} | tee "$SUMMARY_TXT"

for split in val_seen val_unseen; do
  archive_if_exists "$split"
  run_raw_split "$split"
done

TEST_RAW="$REPO_ROOT/$(raw_name test_unseen)"
if [[ ! -f "$TEST_RAW" ]]; then
  echo "[warn] missing known high-score test rollout: $TEST_RAW" | tee -a "$SUMMARY_TXT"
fi

echo "=== protocol metrics ===" | tee -a "$SUMMARY_TXT"
METRIC_ITEMS=(
  --item "raw_pred:$RAW_DIR/$(raw_name val_seen)"
  --item "gcf:$RAW_DIR/$(raw_name val_seen)"
  --item "raw_pred:$RAW_DIR/$(raw_name val_unseen)"
  --item "gcf:$RAW_DIR/$(raw_name val_unseen)"
)
if [[ -f "$TEST_RAW" ]]; then
  METRIC_ITEMS+=(
    --item "raw_pred:$TEST_RAW"
    --item "gcf:$TEST_RAW"
  )
fi

"$PYTHON_BIN" -u "$REPO_ROOT/scripts/compute_protocol_spl.py" \
  "${METRIC_ITEMS[@]}" \
  --gcf_window 5 \
  --gcf_beta 4.0 \
  --output_csv "$CSV_OUT" \
  --output_json "$JSON_OUT" \
  --altitude 50 \
  --gsam_use_segmentation_mask \
  --gsam_box_threshold 0.20 \
  2>&1 | tee "$LOG_DIR/protocol_metrics.log"

echo "DONE $(date)" | tee -a "$SUMMARY_TXT"
echo "Artifacts:" | tee -a "$SUMMARY_TXT"
echo "  $RAW_DIR" | tee -a "$SUMMARY_TXT"
echo "  $CSV_OUT" | tee -a "$SUMMARY_TXT"
echo "  $JSON_OUT" | tee -a "$SUMMARY_TXT"
