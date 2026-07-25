#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/hxtang/miniconda3/envs/uscnav/bin/python}"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/paper_eval_records/spguide_gcf_rollout_best_${STAMP}}"
RAW_DIR="$OUT_DIR/rollouts"
OLD_DIR="$OUT_DIR/preexisting_root_json"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_TXT="$OUT_DIR/summary.txt"
CSV_OUT="$OUT_DIR/spguide_online_gcf_metrics.csv"
JSON_OUT="$OUT_DIR/spguide_online_gcf_metrics.json"

PROGRESS_STOP_VAL="${PROGRESS_STOP_VAL:-0.80}"
PROGRESS_STOP_TAG="$("$PYTHON_BIN" - <<PY
print(float("$PROGRESS_STOP_VAL"))
PY
)"
GCF_WINDOW="${GCF_WINDOW:-5}"
GCF_BETA="${GCF_BETA:-4.0}"
GCF_PROGRESS_THR="${GCF_PROGRESS_THR:-0.80}"
GCF_VARIANCE_THR="${GCF_VARIANCE_THR:-400.0}"
GCF_MIN_STEP="${GCF_MIN_STEP:-8}"
GCF_ARRIVAL_DIST="${GCF_ARRIVAL_DIST:-15.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
EVAL_MAX_TIMESTEP="${EVAL_MAX_TIMESTEP:-20}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$RAW_DIR" "$OLD_DIR" "$LOG_DIR"

if (($#)); then
  SPLITS=("$@")
else
  SPLITS=(val_seen val_unseen test_unseen)
fi

raw_name() {
  local split="$1"
  echo "instr_decoder_usc_with_map_mturk_50.0_0.2_${split}_${PROGRESS_STOP_TAG}_gcf_raw.json"
}

raw_find_pattern() {
  local split="$1"
  echo "instr_decoder_usc_with_map*_mturk_50.0_0.2*_${split}_${PROGRESS_STOP_TAG}_gcf_raw.json"
}

find_latest_raw() {
  local split="$1"
  local pattern
  local produced
  pattern="$(raw_find_pattern "$split")"
  produced="$(
    find "$REPO_ROOT" -maxdepth 1 -type f -name "$pattern" -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -z "$produced" ]]; then
    echo "Expected eval output matching $REPO_ROOT/$pattern" >&2
    return 1
  fi
  echo "$produced"
}

archive_if_exists() {
  local split="$1"
  local pattern
  local path
  pattern="$(raw_find_pattern "$split")"
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    mv "$path" "$OLD_DIR/${split}_${STAMP}_$(basename "$path")"
  done < <(find "$REPO_ROOT" -maxdepth 1 -type f -name "$pattern")
}

run_split() {
  local split="$1"
  local produced
  local name

  echo "=== online GCF rollout: ${split} ===" | tee -a "$SUMMARY_TXT"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u "$REPO_ROOT/scripts/eval_single_split.py" \
    --split "$split" \
    --model instr_decoder_usc_with_map \
    --checkpoint "$CKPT" \
    --mode eval \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_use_map_cache \
    --gsam_box_threshold 0.20 \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_max_timestep "$EVAL_MAX_TIMESTEP" \
    --eval_agent_mode gcf \
    --eval_goal_selector raw \
    --progress_stop_val "$PROGRESS_STOP_VAL" \
    --gcf_window "$GCF_WINDOW" \
    --gcf_beta "$GCF_BETA" \
    --gcf_progress_thr "$GCF_PROGRESS_THR" \
    --gcf_variance_thr "$GCF_VARIANCE_THR" \
    --gcf_min_step "$GCF_MIN_STEP" \
    --gcf_arrival_dist "$GCF_ARRIVAL_DIST" \
    --gcf_debug \
    2>&1 | tee "$LOG_DIR/${split}_gcf_rollout.log"

  produced="$(find_latest_raw "$split")"
  name="$(basename "$produced")"
  mv "$produced" "$RAW_DIR/$name"
}

{
  echo "OUT_DIR=$OUT_DIR"
  echo "CKPT=$CKPT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "GPU_ID=$GPU_ID"
  echo "START $(date)"
  echo "SPLITS=${SPLITS[*]}"
  echo "PROTOCOL=online_terminal_gcf"
  echo "PARAMS progress_stop=$PROGRESS_STOP_VAL output_tag=$PROGRESS_STOP_TAG window=$GCF_WINDOW beta=$GCF_BETA progress_thr=$GCF_PROGRESS_THR variance_thr=$GCF_VARIANCE_THR min_step=$GCF_MIN_STEP arrival_dist=$GCF_ARRIVAL_DIST"
} | tee "$SUMMARY_TXT"

for split in "${SPLITS[@]}"; do
  archive_if_exists "$split"
  run_split "$split"
done

"$PYTHON_BIN" - <<PY
import csv
import json
from pathlib import Path

raw_dir = Path("$RAW_DIR")
rows = []
for path in sorted(raw_dir.glob("*.json")):
    data = json.load(open(path))
    if "_val_seen_" in path.name:
        split = "val_seen"
    elif "_val_unseen_" in path.name:
        split = "val_unseen"
    elif "_test_unseen_" in path.name:
        split = "test_unseen"
    else:
        split = "unknown"
    metrics = data["metrics"]
    debug = data.get("gcf_debug") or {}
    rows.append({
        "model": data.get("model", "instr_decoder_usc_with_map"),
        "split": split,
        "protocol": "online_terminal_gcf",
        "ne": metrics["mean_final_pos_to_goal_dist"],
        "sr": metrics["success_rate_final_pos_to_goal"],
        "osr": metrics["success_rate_oracle_pos_to_goal"],
        "spl": data.get("spl", 0.0),
        "pred_ne": metrics["mean_final_pred_to_goal_dist"],
        "pred_sr": metrics["success_rate_final_pred_to_goal"],
        "pred_osr": metrics["success_rate_oracle_pred_to_goal"],
        "lock_rate": debug.get("lock_rate"),
        "mean_lock_step": debug.get("mean_lock_step"),
    })

if rows:
    csv_path = Path("$CSV_OUT")
    json_path = Path("$JSON_OUT")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    for row in rows:
        print(
            f"{row['split']} | online_terminal_gcf | "
            f"NE={float(row['ne']):.2f} SR={float(row['sr'])*100:.2f} "
            f"OSR={float(row['osr'])*100:.2f} SPL={float(row['spl'])*100:.2f} "
            f"lock={row['lock_rate']} step={row['mean_lock_step']}"
        )
PY

echo "DONE $(date)" | tee -a "$SUMMARY_TXT"
echo "Artifacts:" | tee -a "$SUMMARY_TXT"
echo "  $RAW_DIR" | tee -a "$SUMMARY_TXT"
echo "  $LOG_DIR" | tee -a "$SUMMARY_TXT"
echo "  $CSV_OUT" | tee -a "$SUMMARY_TXT"
echo "  $JSON_OUT" | tee -a "$SUMMARY_TXT"
