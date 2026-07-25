#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_MODELS = {
    "seq2seq_with_map",
    "cma_with_map",
    "instr_decoder_with_map",
    "instr_decoder_masa_with_map",
    "instr_decoder_masa_residual_with_map",
    "instr_decoder_masa_progress_with_map",
    "instr_decoder_unipe_with_map",
    "instr_decoder_cpvt_with_map",
    "instr_decoder_usc_with_map",
    "instr_decoder_instrkv_usc_with_map",
    "instr_decoder_usc_dir_with_map",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Progress-head post-training wrapper built on top of the official baseline training pipeline."
    )
    parser.add_argument("--model", default="instr_decoder_with_map", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--posttrain_tag", default="progress_posttrain")
    parser.add_argument("--train_trajectory_type", choices=["sp", "mturk", "both"], default="mturk")
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--train_episode_sample_size", type=int, default=-1)
    parser.add_argument("--potential_rank_loss_weight", type=float, default=0.0)
    parser.add_argument("--potential_rank_margin", type=float, default=0.02)
    parser.add_argument("--dagger_rollout_posttrain", action="store_true")
    parser.add_argument("--dagger_episode_sample_size", type=int, default=2000)
    parser.add_argument("--dagger_mix_ratio", type=float, default=0.5)
    parser.add_argument("--eval_goal_selector", choices=["gdino", "llava", "gcf", "raw"], default="gdino")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "main_goal_predictor.py",
        "--mode",
        "train",
        "--model",
        args.model,
        "--checkpoint",
        args.checkpoint,
        "--data_root",
        args.data_root,
        "--posttrain_tag",
        args.posttrain_tag,
        "--train_trajectory_type",
        args.train_trajectory_type,
        "--altitude",
        str(args.altitude),
        "--learning_rate",
        str(args.learning_rate),
        "--train_batch_size",
        str(args.train_batch_size),
        "--epochs",
        str(args.epochs),
        "--eval_every",
        str(args.eval_every),
        "--save_every",
        str(args.save_every),
        "--train_episode_sample_size",
        str(args.train_episode_sample_size),
        "--potential_rank_loss_weight",
        str(args.potential_rank_loss_weight),
        "--potential_rank_margin",
        str(args.potential_rank_margin),
        "--eval_goal_selector",
        args.eval_goal_selector,
        "--gsam_use_segmentation_mask",
        "--gsam_box_threshold",
        "0.20",
        "--gsam_use_map_cache",
        "--progress_head_only_tune",
    ]
    if args.dagger_rollout_posttrain:
        cmd.extend([
            "--dagger_rollout_posttrain",
            "--dagger_episode_sample_size",
            str(args.dagger_episode_sample_size),
            "--dagger_mix_ratio",
            str(args.dagger_mix_ratio),
        ])
    cmd.extend(args.extra)

    print("Launching:")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
