#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs docs

PYTHON_BIN="${PYTHON_BIN:-conda run -n citynav_a100 python}"
GPU_ID="${GPU_ID:-0}"
PROGRESS_STOP_VAL="${PROGRESS_STOP_VAL:-0.75}"
GCF_WINDOW="${GCF_WINDOW:-5}"
GCF_BETA="${GCF_BETA:-4.0}"

run_raw_eval() {
  local model="$1"
  local checkpoint="$2"
  local max_timestep="$3"
  local marker="$4"

  if [[ ! -s "$checkpoint" ]]; then
    echo "[warn] missing checkpoint for $model: $checkpoint"
    return 0
  fi

  if [[ -f "$marker" ]]; then
    echo "[skip] raw rollout exists for $model -> $marker"
    return 0
  fi

  echo "[run] raw rollout for $model"
  CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON_BIN main_goal_predictor.py \
    --mode eval \
    --model "$model" \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_use_map_cache \
    --gsam_box_threshold 0.20 \
    --eval_batch_size 100 \
    --eval_max_timestep "$max_timestep" \
    --checkpoint "$checkpoint" \
    --eval_agent_mode progress_stop \
    --eval_goal_selector raw \
    --progress_stop_val "$PROGRESS_STOP_VAL"
}

run_raw_eval \
  seq2seq_with_map \
  checkpoints/baseline_with_map/seq2seq_with_map/mturk_50.0_0.2/official_mturk.pth \
  15 \
  "seq2seq_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json"

run_raw_eval \
  cma_with_map \
  checkpoints/baseline_with_map/cma_with_map/mturk_50.0_0.2/official_mturk.pth \
  15 \
  "cma_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json"

run_raw_eval \
  mgp \
  checkpoints/goal_predictor/mturk_50.0_0.2/official_mturk.pth \
  20 \
  "mgp_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json"

run_raw_eval \
  instr_decoder_with_map \
  checkpoints/baseline_with_map/instr_decoder_with_map/mturk_50.0_0.2/009.pth \
  20 \
  "instr_decoder_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json"

run_raw_eval \
  instr_decoder_usc_with_map \
  checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth \
  20 \
  "instr_decoder_usc_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json"

COMMON_ARGS=(
  --altitude 50
  --gsam_use_segmentation_mask
  --gsam_box_threshold 0.20
  --gcf_window "$GCF_WINDOW"
  --gcf_beta "$GCF_BETA"
)

CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON_BIN scripts/eval_goal_agents.py \
  --input_json "seq2seq_with_map_mturk_50.0_0.2_val_seen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "seq2seq_with_map_mturk_50.0_0.2_val_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "seq2seq_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "cma_with_map_mturk_50.0_0.2_val_seen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "cma_with_map_mturk_50.0_0.2_val_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "cma_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "mgp_mturk_50.0_0.2_val_seen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "mgp_mturk_50.0_0.2_val_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "mgp_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "instr_decoder_with_map_mturk_50.0_0.2_val_seen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "instr_decoder_with_map_mturk_50.0_0.2_val_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "instr_decoder_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "instr_decoder_usc_with_map_mturk_50.0_0.2_val_seen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "instr_decoder_usc_with_map_mturk_50.0_0.2_val_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --input_json "instr_decoder_usc_with_map_mturk_50.0_0.2_test_unseen_${PROGRESS_STOP_VAL}_progress_stop_raw.json" \
  --selectors gcf \
  --output_csv docs/agent_matrix_gcf_unified.csv \
  --output_json docs/agent_matrix_gcf_unified.json \
  "${COMMON_ARGS[@]}"

echo "[done] unified GCF evaluation artifacts:"
ls -lh docs/agent_matrix_gcf_unified.csv docs/agent_matrix_gcf_unified.json
