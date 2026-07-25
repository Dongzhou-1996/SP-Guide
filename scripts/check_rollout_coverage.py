#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories


def infer_split(path: Path) -> str:
    for split in ("val_seen", "val_unseen", "test_unseen"):
        if f"_{split}_" in path.name:
            return split
    raise ValueError(f"Cannot infer split from {path}")


def main() -> None:
    objects = get_city_refer_objects()
    for raw in sys.argv[1:]:
        path = Path(raw)
        split = infer_split(path)
        episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(split, "all", 50.0),
        )
        with open(path) as f:
            obj = json.load(f)
        expected = {episode.id for episode in episodes}
        traj = {ast.literal_eval(key) for key in obj["trajectory_logs"]}
        pred = {ast.literal_eval(key) for key in obj["pred_goal_logs"]}
        missing_traj = sorted(expected - traj)[:5]
        missing_pred = sorted(expected - pred)[:5]
        print(
            f"{path.name} | {split} | expected={len(expected)} "
            f"traj={len(traj)} pred={len(pred)} "
            f"missing_traj={len(expected - traj)} missing_pred={len(expected - pred)}"
        )
        if missing_traj:
            print(f"  missing_traj_sample={missing_traj}")
        if missing_pred:
            print(f"  missing_pred_sample={missing_pred}")


if __name__ == "__main__":
    main()
