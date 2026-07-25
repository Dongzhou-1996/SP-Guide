#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs docs

run_raw_eval() {
  local model="$1"
  local checkpoint="$2"
  local max_timestep="$3"
  local marker="$4"

  if [[ -f "$marker" ]]; then
    echo "[skip] raw rollout exists for $model -> $marker"
    return 0
  fi

  echo "[run] raw rollout for $model"
  CUDA_VISIBLE_DEVICES=0 conda run -n citynav_a100 python main_goal_predictor.py \
    --mode eval \
    --model "$model" \
    --altitude 50 \
    --gsam_use_segmentation_mask \
    --gsam_box_threshold 0.20 \
    --gsam_use_map_cache \
    --eval_batch_size 100 \
    --eval_max_timestep "$max_timestep" \
    --checkpoint "$checkpoint" \
    --eval_agent_mode progress_stop \
    --eval_goal_selector raw
}

run_raw_eval \
  seq2seq_with_map \
  checkpoints/baseline_with_map/seq2seq_with_map/mturk_50.0_0.2/official_mturk.pth \
  15 \
  seq2seq_with_map_mturk_50.0_0.2_test_unseen_0.75_progress_stop_raw.json

run_raw_eval \
  cma_with_map \
  checkpoints/baseline_with_map/cma_with_map/mturk_50.0_0.2/official_mturk.pth \
  15 \
  cma_with_map_mturk_50.0_0.2_test_unseen_0.75_progress_stop_raw.json

while [[ ! -s checkpoints/goal_predictor/mturk_50.0_0.2/official_mturk.pth ]]; do
  echo "[wait] mgp checkpoint upload incomplete"
  sleep 5
done

run_raw_eval \
  mgp \
  checkpoints/goal_predictor/mturk_50.0_0.2/official_mturk.pth \
  20 \
  mgp_mturk_50.0_0.2_test_unseen_0.75_progress_stop_raw.json

COMMON_ARGS=(
  --altitude 50
  --gsam_use_segmentation_mask
  --gsam_box_threshold 0.20
  --gcf_window 5
  --gcf_beta 4.0
)

CUDA_VISIBLE_DEVICES=0 conda run -n citynav_a100 python scripts/eval_goal_agents.py \
  --input_json seq2seq_with_map_mturk_50.0_0.2_val_seen_0.75_progress_stop_raw.json \
  --input_json seq2seq_with_map_mturk_50.0_0.2_val_unseen_0.75_progress_stop_raw.json \
  --input_json seq2seq_with_map_mturk_50.0_0.2_test_unseen_0.75_progress_stop_raw.json \
  --input_json cma_with_map_mturk_50.0_0.2_val_seen_0.75_progress_stop_raw.json \
  --input_json cma_with_map_mturk_50.0_0.2_val_unseen_0.75_progress_stop_raw.json \
  --input_json cma_with_map_mturk_50.0_0.2_test_unseen_0.75_progress_stop_raw.json \
  --input_json mgp_mturk_50.0_0.2_val_seen_0.75_progress_stop_raw.json \
  --input_json mgp_mturk_50.0_0.2_val_unseen_0.75_progress_stop_raw.json \
  --input_json mgp_mturk_50.0_0.2_test_unseen_0.75_progress_stop_raw.json \
  --selectors gdino gcf \
  --output_csv docs/agent_matrix_citynav_vs_gcf.csv \
  --output_json docs/agent_matrix_citynav_vs_gcf.json \
  "${COMMON_ARGS[@]}"

CUDA_VISIBLE_DEVICES=0 conda run -n citynav_a100 python scripts/eval_goal_agents.py \
  --input_json instr_decoder_with_map_mturk_50.0_0.2_val_seen_0.75_raw.json \
  --input_json instr_decoder_with_map_mturk_50.0_0.2_val_unseen_0.75_raw.json \
  --input_json instr_decoder_with_map_mturk_50.0_0.2_test_unseen_0.75_raw.json \
  --input_json instr_decoder_usc_with_map_mturk_50.0_0.2_val_seen_0.75_progress_stop_raw.json \
  --input_json instr_decoder_usc_with_map_mturk_50.0_0.2_val_unseen_0.75_progress_stop_raw.json \
  --input_json instr_decoder_usc_with_map_mturk_50.0_0.2_test_unseen_0.75_progress_stop_raw.json \
  --selectors gcf \
  --output_csv docs/agent_matrix_ours_gcf.csv \
  --output_json docs/agent_matrix_ours_gcf.json \
  "${COMMON_ARGS[@]}"

echo "[done] evaluation matrix artifacts:"
ls -lh docs/agent_matrix_citynav_vs_gcf.csv docs/agent_matrix_citynav_vs_gcf.json \
       docs/agent_matrix_ours_gcf.csv docs/agent_matrix_ours_gcf.json
