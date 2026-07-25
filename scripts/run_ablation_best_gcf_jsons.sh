#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/hxtang/miniconda3/envs/uscnav/bin/python}"
GPU_ID="${GPU_ID:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/paper_eval_records/ablation_best_gcf_009_${STAMP}}"
ROLL_DIR="$OUT_DIR/rollouts"
LOG_DIR="$OUT_DIR/logs"
OLD_DIR="$OUT_DIR/preexisting_root_json"
SUMMARY_CSV="$OUT_DIR/ablation_metrics.csv"
SUMMARY_JSON="$OUT_DIR/ablation_metrics.json"
SUMMARY_TXT="$OUT_DIR/summary.txt"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-200}"
EVAL_MAX_TIMESTEP="${EVAL_MAX_TIMESTEP:-20}"
PROGRESS_STOP_VAL="${PROGRESS_STOP_VAL:-0.75}"
GCF_WINDOW="${GCF_WINDOW:-4}"
GCF_BETA="${GCF_BETA:-4.0}"
GCF_PROGRESS_THR="${GCF_PROGRESS_THR:-0.50}"
GCF_VARIANCE_THR="${GCF_VARIANCE_THR:-1000}"
GCF_MIN_STEP="${GCF_MIN_STEP:-5}"
GCF_ARRIVAL_DIST="${GCF_ARRIVAL_DIST:-15}"

IED_CKPT="${IED_CKPT:-$REPO_ROOT/checkpoints/baseline_with_map/instr_decoder_with_map/mturk_50.0_0.2/009.pth}"
SPGUIDE_CKPT="${SPGUIDE_CKPT:-$REPO_ROOT/checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2_spguide_retrain_20260613_141044/009.pth}"

mkdir -p "$ROLL_DIR" "$LOG_DIR" "$OLD_DIR"

if (($#)); then
  SPLITS=("$@")
else
  SPLITS=(val_seen val_unseen test_unseen)
fi

json_pattern() {
  local model="$1"
  local split="$2"
  local mode="$3"
  echo "${model}_*_${split}_${PROGRESS_STOP_VAL}_${mode}_raw.json"
}

archive_root_matches() {
  local model="$1"
  local split="$2"
  local mode="$3"
  local path
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    mv "$path" "$OLD_DIR/$(date +%s)_$(basename "$path")"
  done < <(find "$REPO_ROOT" -maxdepth 1 -type f -name "$(json_pattern "$model" "$split" "$mode")")
}

find_latest_json() {
  local model="$1"
  local split="$2"
  local mode="$3"
  find "$REPO_ROOT" -maxdepth 1 -type f -name "$(json_pattern "$model" "$split" "$mode")" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

run_one() {
  local variant="$1"
  local model="$2"
  local checkpoint="$3"
  local protocol="$4"
  local split="$5"
  local mode="$6"
  local log_path="$LOG_DIR/${variant}_${protocol}_${split}.log"
  local output_path="$ROLL_DIR/${variant}_${protocol}_${split}.json"
  local produced

  if [[ -s "$output_path" ]]; then
    echo "[skip] $output_path exists" | tee -a "$SUMMARY_TXT"
    return 0
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "[error] missing checkpoint: $checkpoint" | tee -a "$SUMMARY_TXT"
    return 1
  fi

  archive_root_matches "$model" "$split" "$mode"
  echo "[run] variant=$variant protocol=$protocol split=$split gpu=$GPU_ID" | tee -a "$SUMMARY_TXT"

  if [[ "$protocol" == "gcf" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u "$REPO_ROOT/scripts/eval_single_split.py" \
      --split "$split" \
      --model "$model" \
      --checkpoint "$checkpoint" \
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
      2>&1 | tee "$log_path"
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u "$REPO_ROOT/scripts/eval_single_split.py" \
      --split "$split" \
      --model "$model" \
      --checkpoint "$checkpoint" \
      --mode eval \
      --altitude 50 \
      --gsam_use_segmentation_mask \
      --gsam_use_map_cache \
      --gsam_box_threshold 0.20 \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --eval_max_timestep "$EVAL_MAX_TIMESTEP" \
      --eval_agent_mode progress_stop \
      --eval_goal_selector raw \
      --progress_stop_val "$PROGRESS_STOP_VAL" \
      2>&1 | tee "$log_path"
  fi

  produced="$(find_latest_json "$model" "$split" "$mode")"
  if [[ -z "$produced" ]]; then
    echo "[error] no JSON produced for $variant $protocol $split" | tee -a "$SUMMARY_TXT"
    return 1
  fi
  mv "$produced" "$output_path"
  echo "[json] $output_path" | tee -a "$SUMMARY_TXT"
}

{
  echo "OUT_DIR=$OUT_DIR"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "GPU_ID=$GPU_ID"
  echo "SPLITS=${SPLITS[*]}"
  echo "PROGRESS_STOP_VAL=$PROGRESS_STOP_VAL"
  echo "GCF window=$GCF_WINDOW beta=$GCF_BETA progress_thr=$GCF_PROGRESS_THR variance_thr=$GCF_VARIANCE_THR min_step=$GCF_MIN_STEP arrival_dist=$GCF_ARRIVAL_DIST"
  echo "IED_CKPT=$IED_CKPT"
  echo "SPGUIDE_CKPT=$SPGUIDE_CKPT"
  echo "START $(date)"
} >> "$SUMMARY_TXT"

for split in "${SPLITS[@]}"; do
  run_one "ied" "instr_decoder_with_map" "$IED_CKPT" "progress_stop" "$split" "progress_stop"
  run_one "ied" "instr_decoder_with_map" "$IED_CKPT" "gcf" "$split" "gcf"
  run_one "spguide" "instr_decoder_usc_with_map" "$SPGUIDE_CKPT" "progress_stop" "$split" "progress_stop"
  run_one "spguide" "instr_decoder_usc_with_map" "$SPGUIDE_CKPT" "gcf" "$split" "gcf"
done

"$PYTHON_BIN" - "$ROLL_DIR" "$SUMMARY_CSV" "$SUMMARY_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

roll_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])

rows = []
for path in sorted(roll_dir.glob("*.json")):
    stem = path.stem
    # variant_protocol_split, where split may contain underscores.
    parts = stem.split("_")
    variant = parts[0]
    protocol = parts[1]
    split = "_".join(parts[2:])
    data = json.load(open(path))
    metrics = data["metrics"]
    debug = data.get("gcf_debug") or {}
    rows.append({
        "variant": variant,
        "protocol": protocol,
        "split": split,
        "ne": float(metrics["mean_final_pos_to_goal_dist"]),
        "sr": float(metrics["success_rate_final_pos_to_goal"]) * 100.0,
        "osr": float(metrics["success_rate_oracle_pos_to_goal"]) * 100.0,
        "spl": float(data.get("spl", 0.0)) * 100.0,
        "pred_ne": float(metrics.get("mean_final_pred_to_goal_dist", 0.0)),
        "pred_sr": float(metrics.get("success_rate_final_pred_to_goal", 0.0)) * 100.0,
        "pred_osr": float(metrics.get("success_rate_oracle_pred_to_goal", 0.0)) * 100.0,
        "lock_rate": None if not debug else float(debug.get("lock_rate", 0.0)) * 100.0,
        "mean_lock_step": None if not debug else debug.get("mean_lock_step"),
        "json": str(path),
    })

fieldnames = [
    "variant", "protocol", "split", "ne", "sr", "osr", "spl",
    "pred_ne", "pred_sr", "pred_osr", "lock_rate", "mean_lock_step", "json",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)

for row in rows:
    print(
        f"{row['variant']} {row['protocol']} {row['split']}: "
        f"NE={row['ne']:.2f} SR={row['sr']:.2f} OSR={row['osr']:.2f} SPL={row['spl']:.2f}"
    )
PY

echo "DONE $(date)" | tee -a "$SUMMARY_TXT"
echo "SUMMARY_CSV=$SUMMARY_CSV" | tee -a "$SUMMARY_TXT"
echo "SUMMARY_JSON=$SUMMARY_JSON" | tee -a "$SUMMARY_TXT"
