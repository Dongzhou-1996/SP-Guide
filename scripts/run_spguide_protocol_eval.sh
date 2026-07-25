#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/paper_eval_records/spguide_protocol_eval_${STAMP}}"
RAW_DIR="$OUT_DIR/raw_rollouts"
ARCHIVE_DIR="$OUT_DIR/preexisting_root_json"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_TXT="$OUT_DIR/summary.txt"
CSV_OUT="$OUT_DIR/protocol_metrics.csv"
JSON_OUT="$OUT_DIR/protocol_metrics.json"

PROGRESS_STOP_VAL="${PROGRESS_STOP_VAL:-0.75}"
PROGRESS_STOP_TAG="$("$PYTHON_BIN" - <<PY
print(float("$PROGRESS_STOP_VAL"))
PY
)"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
EVAL_MAX_TIMESTEP="${EVAL_MAX_TIMESTEP:-20}"
ALTITUDE="${ALTITUDE:-50}"
GSAM_BOX_THRESHOLD="${GSAM_BOX_THRESHOLD:-0.20}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

GCF_WINDOW="${GCF_WINDOW:-5}"
GCF_BETA="${GCF_BETA:-4.0}"
GCF_PROGRESS_THR="${GCF_PROGRESS_THR:-0.80}"
GCF_VARIANCE_THR="${GCF_VARIANCE_THR:-400.0}"
GCF_MIN_STEP="${GCF_MIN_STEP:-8}"
GCF_ARRIVAL_DIST="${GCF_ARRIVAL_DIST:-15.0}"

mkdir -p "$RAW_DIR" "$ARCHIVE_DIR" "$LOG_DIR"

if (($#)); then
  SPLITS=("$@")
else
  SPLITS=(val_seen val_unseen test_unseen)
fi

MODEL_NAME="instr_decoder_usc_with_map"
CKPT_TAG="$(basename "$(dirname "$CKPT")")"

expected_json_name() {
  local split="$1"
  echo "${MODEL_NAME}_${CKPT_TAG}_${split}_${PROGRESS_STOP_TAG}_progress_stop_raw.json"
}

archive_existing_json() {
  local expected="$1"
  local split="$2"
  if [[ -f "$REPO_ROOT/$expected" ]]; then
    mv "$REPO_ROOT/$expected" "$ARCHIVE_DIR/${split}_${STAMP}_$expected"
  fi
}

run_split() {
  local split="$1"
  local expected_json
  expected_json="$(expected_json_name "$split")"

  archive_existing_json "$expected_json" "$split"

  echo "=== protocol rollout: ${split} ===" | tee -a "$SUMMARY_TXT"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u "$REPO_ROOT/scripts/eval_single_split.py" \
    --split "$split" \
    --model "$MODEL_NAME" \
    --checkpoint "$CKPT" \
    --mode eval \
    --data_root "$DATA_ROOT" \
    --altitude "$ALTITUDE" \
    --gsam_use_segmentation_mask \
    --gsam_use_map_cache \
    --gsam_box_threshold "$GSAM_BOX_THRESHOLD" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_max_timestep "$EVAL_MAX_TIMESTEP" \
    --eval_agent_mode progress_stop \
    --eval_goal_selector raw \
    --progress_stop_val "$PROGRESS_STOP_VAL" \
    2>&1 | tee "$LOG_DIR/${split}_progress_stop.log"

  if [[ ! -f "$REPO_ROOT/$expected_json" ]]; then
    echo "Missing expected rollout JSON: $REPO_ROOT/$expected_json" >&2
    exit 1
  fi
  mv "$REPO_ROOT/$expected_json" "$RAW_DIR/$expected_json"
}

{
  echo "OUT_DIR=$OUT_DIR"
  echo "CKPT=$CKPT"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "GPU_ID=$GPU_ID"
  echo "START $(date)"
  echo "SPLITS=${SPLITS[*]}"
  echo "ROLL_OUT_MODE=progress_stop"
  echo "PROTOCOLS=raw_pred,gcf"
  echo "PARAMS progress_stop=$PROGRESS_STOP_VAL window=$GCF_WINDOW beta=$GCF_BETA progress_thr=$GCF_PROGRESS_THR variance_thr=$GCF_VARIANCE_THR min_step=$GCF_MIN_STEP arrival_dist=$GCF_ARRIVAL_DIST"
} | tee "$SUMMARY_TXT"

for split in "${SPLITS[@]}"; do
  run_split "$split"
done

"$PYTHON_BIN" -u "$REPO_ROOT/scripts/compute_protocol_spl.py" \
  $(for split in "${SPLITS[@]}"; do printf ' --item raw_pred:%s' "$RAW_DIR/$(expected_json_name "$split")"; done) \
  $(for split in "${SPLITS[@]}"; do printf ' --item gcf:%s' "$RAW_DIR/$(expected_json_name "$split")"; done) \
  --model "$MODEL_NAME" \
  --checkpoint "$CKPT" \
  --mode eval \
  --data_root "$DATA_ROOT" \
  --altitude "$ALTITUDE" \
  --gsam_use_segmentation_mask \
  --gsam_use_map_cache \
  --gsam_box_threshold "$GSAM_BOX_THRESHOLD" \
  --success_dist 20.0 \
  --progress_stop_val "$PROGRESS_STOP_VAL" \
  --gcf_window "$GCF_WINDOW" \
  --gcf_beta "$GCF_BETA" \
  --gcf_progress_thr "$GCF_PROGRESS_THR" \
  --gcf_variance_thr "$GCF_VARIANCE_THR" \
  --gcf_min_step "$GCF_MIN_STEP" \
  --output_csv "$CSV_OUT" \
  --output_json "$JSON_OUT" \
  --allow_missing

echo "DONE $(date)" | tee -a "$SUMMARY_TXT"
echo "Artifacts:" | tee -a "$SUMMARY_TXT"
echo "  $RAW_DIR" | tee -a "$SUMMARY_TXT"
echo "  $LOG_DIR" | tee -a "$SUMMARY_TXT"
echo "  $CSV_OUT" | tee -a "$SUMMARY_TXT"
echo "  $JSON_OUT" | tee -a "$SUMMARY_TXT"
