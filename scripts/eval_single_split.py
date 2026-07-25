from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.evaluate import eval_goal_predictor
from gsamllavanav.model_registry import get_model_spec
from gsamllavanav.parser import parse_args
from gsamllavanav.terminal_belief import terminal_belief_config_dict
from main_goal_predictor import _compute_spl, _resolve_eval_mode


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val_seen", "val_unseen", "test_unseen"], required=True)
    parser.add_argument("--limit", type=int)
    cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = parse_args()
    return cli, args


def main():
    cli, args = _parse_cli()
    model_spec = get_model_spec(args.model)

    if model_spec.pipeline == "goal_predictor":
        from gsamllavanav.evaluate import run_episodes_batch
    else:
        from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch

    selector = _resolve_eval_mode(args)

    objects = get_city_refer_objects()
    episodes = generate_episodes_from_mturk_trajectories(
        objects,
        load_mturk_trajectories(cli.split, "all", args.altitude),
    )
    if cli.limit is not None:
        episodes = episodes[: cli.limit]

    model = model_spec.factory(args.map_size).to(DEVICE)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(state["predictor_state_dict"])

    if hasattr(args, "_terminal_belief_debug_report"):
        delattr(args, "_terminal_belief_debug_report")
    trajectory_logs, pred_goal_logs, pred_progress_logs = run_episodes_batch(
        args, model, episodes, DEVICE
    )
    gcf_debug_report = getattr(args, "_terminal_belief_debug_report", None)

    if selector is not None:
        predicted_positions = selector(pred_goal_logs, pred_progress_logs)
        for eps_id, pose in predicted_positions.items():
            trajectory_logs[eps_id].append(pose)

    metrics = eval_goal_predictor(
        args,
        episodes,
        trajectory_logs,
        pred_goal_logs,
        pred_progress_logs,
    )
    spl = _compute_spl(args, episodes, trajectory_logs)

    print(
        f"{cli.split} -- {metrics.mean_final_pos_to_goal_dist: .1f}, "
        f"{metrics.success_rate_final_pos_to_goal*100: .2f}, "
        f"{metrics.success_rate_oracle_pos_to_goal*100: .2f}, "
        f"{spl*100: .2f}"
    )
    if gcf_debug_report is not None:
        print(
            f"{cli.split} gcf-lock -- "
            f"{gcf_debug_report['episodes_locked']}/{gcf_debug_report['episodes_total']} "
            f"({gcf_debug_report['lock_rate']*100:.2f}%), "
            f"mean_step={gcf_debug_report['mean_lock_step']}"
        )

    noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
    alt_env = f"_{args.alt_env}" if args.alt_env else ""
    limit_tag = f"_limit{cli.limit}" if cli.limit is not None else ""
    mode_tag = args.eval_agent_mode
    selector_tag = args.eval_goal_selector if selector is not None else "raw"
    output_path = (
        f"{args.model}_{args.checkpoint.split('/')[-2]}_{cli.split}_{args.progress_stop_val}"
        f"{noise}{alt_env}{limit_tag}_{mode_tag}_{selector_tag}.json"
    )
    with open(output_path, "w") as f:
        json.dump(
            {
                "metrics": metrics.to_dict(),
                "spl": spl,
                "eval_agent_mode": args.eval_agent_mode,
                "eval_goal_selector": selector_tag,
                "gcf_config": terminal_belief_config_dict(args) if args.eval_agent_mode == "gcf" else None,
                "gcf_debug": gcf_debug_report,
                "trajectory_logs": {
                    str(eps_id): [tuple(pose) for pose in trajectory]
                    for eps_id, trajectory in trajectory_logs.items()
                },
                "pred_goal_logs": {
                    str(eps_id): [tuple(pos) for pos in pred_goals]
                    for eps_id, pred_goals in pred_goal_logs.items()
                },
                "pred_progress_logs": {
                    str(eps_id): pred_progresses
                    for eps_id, pred_progresses in pred_progress_logs.items()
                },
            },
            f,
        )


if __name__ == "__main__":
    main()
