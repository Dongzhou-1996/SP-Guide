#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/tanghx/miniconda3/envs/gdino/bin/python}"
CKPT="${CKPT:-$REPO_ROOT/checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/paper_eval_records/spguide_progress_stop_sweep_${STAMP}}"
ROLL_DIR="$OUT_DIR/rollouts"
LOG_DIR="$OUT_DIR/logs"
OLD_DIR="$OUT_DIR/preexisting_root_json"
SUMMARY_TXT="$OUT_DIR/summary.txt"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
EVAL_MAX_TIMESTEP="${EVAL_MAX_TIMESTEP:-20}"
GPU_ID="${GPU_ID:-0}"

THRESHOLDS=(${THRESHOLDS:-0.80 0.85 0.90 0.95})
SPLITS=(${SPLITS:-val_seen val_unseen})
GCF_CONFIGS=(${GCF_CONFIGS:-6:0 12:0 5:4})

mkdir -p "$ROLL_DIR" "$LOG_DIR" "$OLD_DIR"

float_tag() {
  "$PYTHON_BIN" - <<PY
print(float("$1"))
PY
}

raw_name() {
  local split="$1"
  local tag="$2"
  echo "instr_decoder_usc_with_map_mturk_50.0_0.2_${split}_${tag}_progress_stop_raw.json"
}

archive_if_exists() {
  local split="$1"
  local tag="$2"
  local name
  name="$(raw_name "$split" "$tag")"
  if [[ -f "$REPO_ROOT/$name" ]]; then
    mv "$REPO_ROOT/$name" "$OLD_DIR/${split}_${tag}_${STAMP}_${name}"
  fi
}

run_raw_split() {
  local split="$1"
  local threshold="$2"
  local tag="$3"
  local name
  name="$(raw_name "$split" "$tag")"

  echo "=== raw progress-stop rollout: split=${split} threshold=${threshold} tag=${tag} ===" | tee -a "$SUMMARY_TXT"
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
    --eval_agent_mode progress_stop \
    --eval_goal_selector raw \
    --progress_stop_val "$threshold" \
    2>&1 | tee "$LOG_DIR/${split}_p${tag}_raw.log"

  mkdir -p "$ROLL_DIR/p${tag}"
  mv "$REPO_ROOT/$name" "$ROLL_DIR/p${tag}/$name"
}

compute_metrics_for_threshold() {
  local tag="$1"
  local items=()
  for split in "${SPLITS[@]}"; do
    items+=(--item "raw_pred:$ROLL_DIR/p${tag}/$(raw_name "$split" "$tag")")
  done

  for cfg in "${GCF_CONFIGS[@]}"; do
    local window="${cfg%%:*}"
    local beta="${cfg##*:}"
    local cfg_tag="w${window}_b${beta}"
    local metric_items=("${items[@]}")
    for split in "${SPLITS[@]}"; do
      metric_items+=(--item "gcf:$ROLL_DIR/p${tag}/$(raw_name "$split" "$tag")")
    done
    echo "=== protocol metrics: threshold=${tag} ${cfg_tag} ===" | tee -a "$SUMMARY_TXT"
    "$PYTHON_BIN" -u "$REPO_ROOT/scripts/compute_protocol_spl.py" \
      --allow_missing \
      "${metric_items[@]}" \
      --gcf_window "$window" \
      --gcf_beta "$beta" \
      --output_csv "$OUT_DIR/metrics_p${tag}_${cfg_tag}.csv" \
      --output_json "$OUT_DIR/metrics_p${tag}_${cfg_tag}.json" \
      --altitude 50 \
      --gsam_use_segmentation_mask \
      --gsam_box_threshold 0.20 \
      2>&1 | tee "$LOG_DIR/metrics_p${tag}_${cfg_tag}.log"
  done
}

{
  echo "OUT_DIR=$OUT_DIR"
  echo "CKPT=$CKPT"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "GPU_ID=$GPU_ID"
  echo "START $(date)"
  echo "THRESHOLDS=${THRESHOLDS[*]}"
  echo "SPLITS=${SPLITS[*]}"
  echo "GCF_CONFIGS=${GCF_CONFIGS[*]}"
  echo "EVAL_MAX_TIMESTEP=$EVAL_MAX_TIMESTEP EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
} | tee "$SUMMARY_TXT"

for threshold in "${THRESHOLDS[@]}"; do
  tag="$(float_tag "$threshold")"
  for split in "${SPLITS[@]}"; do
    archive_if_exists "$split" "$tag"
    run_raw_split "$split" "$threshold" "$tag"
  done
  compute_metrics_for_threshold "$tag"
done

"$PYTHON_BIN" - <<PY
import csv
import glob
from collections import defaultdict
from pathlib import Path

out_dir = Path("$OUT_DIR")
rows = []
for path in sorted(glob.glob(str(out_dir / "metrics_p*_w*_b*.csv"))):
    stem = Path(path).stem
    _, ptag, cfg = stem.split("_", 2)
    threshold = ptag[1:]
    for row in csv.DictReader(open(path)):
        row["threshold"] = threshold
        row["gcf_config"] = cfg
        rows.append(row)

summary = []
grouped = defaultdict(list)
for row in rows:
    grouped[(row["threshold"], row["gcf_config"], row["protocol"])].append(row)

for (threshold, cfg, protocol), group in grouped.items():
    vals = [r for r in group if r["split"] in {"val_seen", "val_unseen"}]
    if len(vals) != 2:
        continue
    def mean(name):
        return sum(float(r[name]) for r in vals) / len(vals)
    by_split = {r["split"]: r for r in vals}
    summary.append({
        "threshold": threshold,
        "gcf_config": cfg,
        "protocol": protocol,
        "val_avg_ne": mean("ne"),
        "val_avg_sr": mean("sr"),
        "val_avg_osr": mean("osr"),
        "val_avg_spl": mean("spl"),
        "val_seen_ne": by_split["val_seen"]["ne"],
        "val_seen_sr": by_split["val_seen"]["sr"],
        "val_seen_osr": by_split["val_seen"]["osr"],
        "val_seen_spl": by_split["val_seen"]["spl"],
        "val_unseen_ne": by_split["val_unseen"]["ne"],
        "val_unseen_sr": by_split["val_unseen"]["sr"],
        "val_unseen_osr": by_split["val_unseen"]["osr"],
        "val_unseen_spl": by_split["val_unseen"]["spl"],
    })

summary.sort(key=lambda r: (float(r["val_avg_sr"]), float(r["val_avg_spl"]), -float(r["val_avg_ne"])), reverse=True)
summary_path = out_dir / "progress_stop_sweep_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
print(f"Wrote {summary_path}")
print("Top configs:")
for row in summary[:12]:
    print(
        f"thr={row['threshold']} {row['gcf_config']} {row['protocol']} | "
        f"val_avg SR={float(row['val_avg_sr'])*100:.2f} SPL={float(row['val_avg_spl'])*100:.2f} NE={float(row['val_avg_ne']):.2f} | "
        f"seen SR={float(row['val_seen_sr'])*100:.2f} unseen SR={float(row['val_unseen_sr'])*100:.2f}"
    )
PY

echo "DONE $(date)" | tee -a "$SUMMARY_TXT"
