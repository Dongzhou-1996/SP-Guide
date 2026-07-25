#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


PRESETS: dict[str, dict[str, str | bool]] = {
    "mgp_sp": {
        "model": "mgp",
        "train_trajectory_type": "sp",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "12",
        "eval_max_timestep": "20",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
    "mgp_hd": {
        "model": "mgp",
        "train_trajectory_type": "mturk",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "12",
        "eval_max_timestep": "20",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
    "seq2seq_sp": {
        "model": "seq2seq_with_map",
        "train_trajectory_type": "sp",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "12",
        "eval_max_timestep": "15",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
    "seq2seq_hd": {
        "model": "seq2seq_with_map",
        "train_trajectory_type": "mturk",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "12",
        "eval_max_timestep": "15",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
    "cma_sp": {
        "model": "cma_with_map",
        "train_trajectory_type": "sp",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "12",
        "eval_max_timestep": "15",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
    "cma_hd": {
        "model": "cma_with_map",
        "train_trajectory_type": "mturk",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "12",
        "eval_max_timestep": "15",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
    "insdec_hd": {
        "model": "instr_decoder_with_map",
        "train_trajectory_type": "mturk",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "8",
        "eval_max_timestep": "20",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
    "insdec_usc_hd": {
        "model": "instr_decoder_usc_with_map",
        "train_trajectory_type": "mturk",
        "altitude": "50",
        "learning_rate": "0.0015",
        "train_batch_size": "8",
        "eval_max_timestep": "20",
        "gsam_box_threshold": "0.20",
        "gsam_use_segmentation_mask": True,
        "gsam_use_map_cache": True,
    },
}


PAPER_BASELINE_COVERAGE = {
    "Seq2Seq w/ SP": "seq2seq_sp",
    "Seq2Seq w/ HD": "seq2seq_hd",
    "CMA w/ SP": "cma_sp",
    "CMA w/ HD": "cma_hd",
    "MGP w/ SP": "mgp_sp",
    "MGP w/ HD": "mgp_hd",
    "AerialVLN": None,
    "AerialVLN+GSM": None,
}


def build_command(args: argparse.Namespace) -> list[str]:
    preset = PRESETS[args.preset]
    cmd = [sys.executable, "main_goal_predictor.py", "--mode", args.mode]
    cmd.extend(["--data_root", args.data_root])

    for key, value in preset.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    if args.mode == "eval":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for eval")
        cmd.extend(["--checkpoint", args.checkpoint])
        cmd.extend(["--eval_goal_selector", args.eval_goal_selector])
        if args.eval_batch_size is not None:
            cmd.extend(["--eval_batch_size", str(args.eval_batch_size)])

    if args.train_batch_size is not None:
        cmd.extend(["--train_batch_size", str(args.train_batch_size)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.train_episode_sample_size is not None:
        cmd.extend(["--train_episode_sample_size", str(args.train_episode_sample_size)])
    if args.progress_stop_val is not None:
        cmd.extend(["--progress_stop_val", str(args.progress_stop_val)])

    cmd.extend(args.extra)
    return cmd


def print_coverage() -> None:
    print("Paper baseline coverage in this standalone repo:\n")
    for paper_name, preset_name in PAPER_BASELINE_COVERAGE.items():
        if preset_name is None:
            print(f"- {paper_name}: external code required (not released in upstream CityNav repo)")
        else:
            print(f"- {paper_name}: preset `{preset_name}`")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified launcher for the standalone CityNav reproduction repo."
    )
    parser.add_argument("mode", choices=["train", "eval", "coverage"])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mgp_hd")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--eval_goal_selector", choices=["gdino", "llava", "gcf", "raw"], default="gdino")
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train_episode_sample_size", type=int)
    parser.add_argument("--progress_stop_val", type=float)
    parser.add_argument("--dry_run", action="store_true")
    args, extra = parser.parse_known_args()
    args.extra = extra

    if args.mode == "coverage":
        print_coverage()
        return

    cmd = build_command(args)
    print("Launching:")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
