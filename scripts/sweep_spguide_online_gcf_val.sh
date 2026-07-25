#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/tanghx/miniconda3/envs/gdino/bin/python}"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/paper_eval_records/spguide_online_gcf_sweep_${STAMP}}"
ROLL_DIR="$OUT_DIR/rollouts"
LOG_DIR="$OUT_DIR/logs"
OLD_DIR="$OUT_DIR/preexisting_root_json"
SUMMARY_TXT="$OUT_DIR/summary.txt"
GPU_ID="${GPU_ID:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
EVAL_MAX_TIMESTEP="${EVAL_MAX_TIMESTEP:-20}"
PROGRESS_STOP_VAL="${PROGRESS_STOP_VAL:-0.75}"
SPLITS=(${SPLITS:-val_seen val_unseen})

# name:window:beta:progress_thr:variance_thr:min_step:arrival_dist
if [[ -n "${CONFIGS_RAW:-}" ]]; then
  read -r -a CONFIGS <<< "$CONFIGS_RAW"
else
  CONFIGS=(
    paper_w5_b4_p80_v400_m8_a15:5:4.0:0.80:400:8:15
    early_w5_b4_p70_v600_m6_a15:5:4.0:0.70:600:6:15
    stable_w6_b4_p80_v300_m8_a15:6:4.0:0.80:300:8:15
    uniform_w5_b0_p70_v600_m6_a15:5:0.0:0.70:600:6:15
    late_w7_b4_p80_v400_m10_a20:7:4.0:0.80:400:10:20
  )
fi

mkdir -p "$ROLL_DIR" "$LOG_DIR" "$OLD_DIR"

progress_tag() {
  "$PYTHON_BIN" - <<PY
print(float("$PROGRESS_STOP_VAL"))
PY
}

raw_name() {
  local split="$1"
  local tag
  tag="$(progress_tag)"
  echo "instr_decoder_usc_with_map_mturk_50.0_0.2_${split}_${tag}_gcf_raw.json"
}

raw_find_pattern() {
  local split="$1"
  local tag
  tag="$(progress_tag)"
  echo "instr_decoder_usc_with_map*_mturk_50.0_0.2*_${split}_${tag}_gcf_raw.json"
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

run_config_split() {
  local config_name="$1"
  local window="$2"
  local beta="$3"
  local progress_thr="$4"
  local variance_thr="$5"
  local min_step="$6"
  local arrival_dist="$7"
  local split="$8"
  local produced
  local name

  echo "=== online GCF: config=${config_name} split=${split} ===" | tee -a "$SUMMARY_TXT"
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
    --gcf_window "$window" \
    --gcf_beta "$beta" \
    --gcf_progress_thr "$progress_thr" \
    --gcf_variance_thr "$variance_thr" \
    --gcf_min_step "$min_step" \
    --gcf_arrival_dist "$arrival_dist" \
    --gcf_debug \
    2>&1 | tee "$LOG_DIR/${config_name}_${split}.log"

  mkdir -p "$ROLL_DIR/$config_name"
  produced="$(find_latest_raw "$split")"
  name="$(basename "$produced")"
  mv "$produced" "$ROLL_DIR/$config_name/$name"
}

{
  echo "OUT_DIR=$OUT_DIR"
  echo "CKPT=$CKPT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "GPU_ID=$GPU_ID"
  echo "START $(date)"
  echo "SPLITS=${SPLITS[*]}"
  echo "PROGRESS_STOP_VAL=$PROGRESS_STOP_VAL"
  echo "EVAL_MAX_TIMESTEP=$EVAL_MAX_TIMESTEP EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
  printf 'CONFIGS=%s\n' "${CONFIGS[*]}"
} | tee "$SUMMARY_TXT"

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r config_name window beta progress_thr variance_thr min_step arrival_dist <<< "$cfg"
  {
    echo "CONFIG $config_name window=$window beta=$beta progress_thr=$progress_thr variance_thr=$variance_thr min_step=$min_step arrival_dist=$arrival_dist"
  } | tee -a "$SUMMARY_TXT"
  for split in "${SPLITS[@]}"; do
    archive_if_exists "$split"
    run_config_split "$config_name" "$window" "$beta" "$progress_thr" "$variance_thr" "$min_step" "$arrival_dist" "$split"
  done
done

"$PYTHON_BIN" - <<PY
import csv
import json
from pathlib import Path

out_dir = Path("$OUT_DIR")
rows = []
for cfg_dir in sorted((out_dir / "rollouts").iterdir()):
    if not cfg_dir.is_dir():
        continue
    for path in sorted(cfg_dir.glob("*.json")):
        data = json.load(open(path))
        if "_val_seen_" in path.name:
            split = "val_seen"
        elif "_val_unseen_" in path.name:
            split = "val_unseen"
        else:
            split = "unknown"
        metrics = data["metrics"]
        debug = data.get("gcf_debug") or {}
        rows.append({
            "config": cfg_dir.name,
            "split": split,
            "ne": metrics["mean_final_pos_to_goal_dist"],
            "sr": metrics["success_rate_final_pos_to_goal"],
            "osr": metrics["success_rate_oracle_pos_to_goal"],
            "spl": data.get("spl", 0.0),
            "lock_rate": debug.get("lock_rate"),
            "mean_lock_step": debug.get("mean_lock_step"),
            "median_lock_step": debug.get("median_lock_step"),
        })

summary = []
for cfg in sorted({r["config"] for r in rows}):
    vals = [r for r in rows if r["config"] == cfg and r["split"] in {"val_seen", "val_unseen"}]
    if len(vals) != 2:
        continue
    by_split = {r["split"]: r for r in vals}
    def mean(key):
        return sum(float(r[key]) for r in vals) / len(vals)
    summary.append({
        "config": cfg,
        "val_avg_ne": mean("ne"),
        "val_avg_sr": mean("sr"),
        "val_avg_osr": mean("osr"),
        "val_avg_spl": mean("spl"),
        "val_seen_ne": by_split["val_seen"]["ne"],
        "val_seen_sr": by_split["val_seen"]["sr"],
        "val_seen_osr": by_split["val_seen"]["osr"],
        "val_seen_spl": by_split["val_seen"]["spl"],
        "val_seen_lock_rate": by_split["val_seen"]["lock_rate"],
        "val_seen_mean_lock_step": by_split["val_seen"]["mean_lock_step"],
        "val_unseen_ne": by_split["val_unseen"]["ne"],
        "val_unseen_sr": by_split["val_unseen"]["sr"],
        "val_unseen_osr": by_split["val_unseen"]["osr"],
        "val_unseen_spl": by_split["val_unseen"]["spl"],
        "val_unseen_lock_rate": by_split["val_unseen"]["lock_rate"],
        "val_unseen_mean_lock_step": by_split["val_unseen"]["mean_lock_step"],
    })
summary.sort(key=lambda r: (float(r["val_avg_sr"]), float(r["val_avg_spl"]), -float(r["val_avg_ne"])), reverse=True)

for name, table in [("online_gcf_sweep_rows.csv", rows), ("online_gcf_sweep_summary.csv", summary)]:
    path = out_dir / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(f"Wrote {path}")

print("Top online GCF configs:")
for row in summary:
    print(
        f"{row['config']} | val_avg SR={float(row['val_avg_sr'])*100:.2f} "
        f"SPL={float(row['val_avg_spl'])*100:.2f} NE={float(row['val_avg_ne']):.2f} | "
        f"seen SR={float(row['val_seen_sr'])*100:.2f} lock={row['val_seen_lock_rate']} step={row['val_seen_mean_lock_step']} | "
        f"unseen SR={float(row['val_unseen_sr'])*100:.2f} lock={row['val_unseen_lock_rate']} step={row['val_unseen_mean_lock_step']}"
    )
PY

echo "DONE $(date)" | tee -a "$SUMMARY_TXT"
