# SP-Guide Evaluation Protocols

This document records the paper-facing evaluation setup for SP-Guide.

## What the protocol runner does

Run from the repository root:

```bash
bash scripts/run_spguide_protocol_eval.sh
```

The runner executes one shared rollout per split and then scores that same
rollout under two reporting protocols:

- `raw_pred`: use the last predicted goal on the rollout endpoint
- `gcf`: use the offline Goal Convergence Filter on the same rollout logs

The generated artifacts are stored under the ignored runtime folder
`docs/paper_eval_records/`.

To point the runner at a different dataset root, set `DATA_ROOT`:

```bash
DATA_ROOT=/path/to/data bash scripts/run_spguide_protocol_eval.sh
```

## Default settings

- model: `instr_decoder_usc_with_map`
- checkpoint:
  `checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth`
- rollout mode: `progress_stop`
- progress stop threshold: `0.75`
- GCF window: `5`
- GCF beta: `4.0`
- GCF progress threshold: `0.80`
- GCF variance threshold: `400.0`
- GCF minimum step: `8`
- GCF arrival distance: `15.0`

## Common overrides

Evaluate only `test_unseen`:

```bash
bash scripts/run_spguide_protocol_eval.sh test_unseen
```

Evaluate a different checkpoint:

```bash
CKPT=checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/009.pth \
DATA_ROOT=/path/to/data \
PROGRESS_STOP_VAL=0.80 \
GCF_WINDOW=5 GCF_BETA=4.0 \
GCF_PROGRESS_THR=0.80 GCF_VARIANCE_THR=400.0 GCF_MIN_STEP=8 GCF_ARRIVAL_DIST=15.0 \
bash scripts/run_spguide_protocol_eval.sh val_seen val_unseen test_unseen
```

## Lower-level two-step flow

If you want the rollout and the final scoring step separately, run:

```bash
python scripts/eval_single_split.py \
  --split val_seen \
  --model instr_decoder_usc_with_map \
  --checkpoint checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth \
  --data_root /path/to/data \
  --mode eval \
  --altitude 50 \
  --gsam_use_segmentation_mask \
  --gsam_use_map_cache \
  --gsam_box_threshold 0.20 \
  --eval_batch_size 100 \
  --eval_max_timestep 20 \
  --eval_agent_mode progress_stop \
  --eval_goal_selector raw \
  --progress_stop_val 0.75

python scripts/compute_protocol_spl.py \
  --item raw_pred:instr_decoder_usc_with_map_mturk_50.0_0.2_val_seen_0.75_progress_stop_raw.json \
  --item gcf:instr_decoder_usc_with_map_mturk_50.0_0.2_val_seen_0.75_progress_stop_raw.json \
  --model instr_decoder_usc_with_map \
  --checkpoint checkpoints/baseline_with_map/instr_decoder_usc_with_map/mturk_50.0_0.2/003.pth \
  --data_root /path/to/data \
  --mode eval \
  --altitude 50 \
  --gsam_use_segmentation_mask \
  --gsam_use_map_cache \
  --gsam_box_threshold 0.20 \
  --success_dist 20.0 \
  --progress_stop_val 0.75 \
  --gcf_window 5 \
  --gcf_beta 4.0 \
  --gcf_progress_thr 0.80 \
  --gcf_variance_thr 400.0 \
  --gcf_min_step 8 \
  --output_csv docs/paper_eval_records/protocol_metrics.csv \
  --output_json docs/paper_eval_records/protocol_metrics.json \
  --allow_missing
```

## Output layout

Each run writes a timestamped folder:

```text
docs/paper_eval_records/spguide_protocol_eval_YYYYMMDD_HHMMSS/
├─ summary.txt
├─ logs/
├─ raw_rollouts/
├─ protocol_metrics.csv
└─ protocol_metrics.json
```

The CSV and JSON files contain both protocol rows for each requested split.
