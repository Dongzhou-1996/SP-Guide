from __future__ import annotations

import json
import math

import torch

from gsamllavanav.parser import parse_args
from gsamllavanav.evaluate import eval_goal_predictor
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.model_registry import get_model_spec
from gsamllavanav.terminal_belief import terminal_belief_config_dict


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _planar_path_length_xy(poses) -> float:
    if len(poses) < 2:
        return 0.0
    total = 0.0
    for src, dst in zip(poses[:-1], poses[1:]):
        total += math.hypot(dst.x - src.x, dst.y - src.y)
    return total


def _shortest_path_length_xy(episode) -> float:
    return math.hypot(
        episode.target_position.x - episode.start_pose.x,
        episode.target_position.y - episode.start_pose.y,
    )


def _compute_spl(args, episodes, trajectory_logs) -> float:
    if not episodes:
        return 0.0

    values = []
    for episode in episodes:
        trajectory = trajectory_logs[episode.id]
        final_pose = trajectory[-1]
        dist_to_goal = math.hypot(
            final_pose.x - episode.target_position.x,
            final_pose.y - episode.target_position.y,
        )
        success = 1.0 if dist_to_goal <= args.success_dist else 0.0
        shortest = _shortest_path_length_xy(episode)
        path_len = _planar_path_length_xy(trajectory)
        denom = max(path_len, shortest, 1e-6)
        values.append(success * (shortest / denom))
    return sum(values) / len(values)


def _resolve_eval_mode(args):
    # Keep older commands working while giving eval a single explicit switch.
    if args.eval_agent_mode == 'selector' and args.eval_goal_selector == 'raw':
        args.eval_agent_mode = 'progress_stop'

    if args.eval_goal_selector == 'gcf' and args.eval_agent_mode != 'selector':
        args.eval_agent_mode = 'selector'

    if args.eval_agent_mode == 'gcf':
        args.gcf_filter = True
    else:
        args.gcf_filter = False

    if args.eval_agent_mode != 'selector':
        return None

    if args.eval_goal_selector == 'raw':
        raise ValueError("`eval_goal_selector=raw` cannot be used with `eval_agent_mode=selector`.")

    from gsamllavanav.goal_selection import (
        goal_selection_gcf,
        goal_selection_gdino,
        goal_selection_llava,
    )

    return {
        'gdino': lambda goal_logs, progress_logs: goal_selection_gdino(args, goal_logs),
        'llava': lambda goal_logs, progress_logs: goal_selection_llava(args, goal_logs),
        'gcf': lambda goal_logs, progress_logs: goal_selection_gcf(args, goal_logs, progress_logs),
    }[args.eval_goal_selector]


def main():
    args = parse_args()
    model_spec = get_model_spec(args.model)

    if model_spec.pipeline == 'goal_predictor':
        from gsamllavanav.train import train
        from gsamllavanav.evaluate import run_episodes_batch
    else:
        from gsamllavanav.train_baseline_with_map import train
        from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch

    if args.mode == 'train':
        train(args, DEVICE)
        return

    model_trajectory = args.checkpoint.split('/')[-2]
    selector = _resolve_eval_mode(args)

    objects = get_city_refer_objects()
    model = model_spec.factory(args.map_size).to(DEVICE)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(state['predictor_state_dict'])

    for split in ('val_seen', 'val_unseen', 'test_unseen'):
        test_episodes = generate_episodes_from_mturk_trajectories(
            objects,
            load_mturk_trajectories(split, 'all', args.altitude),
        )
        if hasattr(args, '_terminal_belief_debug_report'):
            delattr(args, '_terminal_belief_debug_report')

        trajectory_logs, pred_goal_logs, pred_progress_logs = run_episodes_batch(
            args, model, test_episodes, DEVICE
        )
        gcf_debug_report = getattr(args, '_terminal_belief_debug_report', None)

        if selector is not None:
            predicted_positions = selector(pred_goal_logs, pred_progress_logs)
            for eps_id, pose in predicted_positions.items():
                trajectory_logs[eps_id].append(pose)

        metrics = eval_goal_predictor(
            args,
            test_episodes,
            trajectory_logs,
            pred_goal_logs,
            pred_progress_logs,
        )
        spl = _compute_spl(args, test_episodes, trajectory_logs)

        print(
            f"{split} -- {metrics.mean_final_pos_to_goal_dist: .1f}, "
            f"{metrics.success_rate_final_pos_to_goal*100: .2f}, "
            f"{metrics.success_rate_oracle_pos_to_goal*100: .2f}, "
            f"{spl*100: .2f}"
        )
        if gcf_debug_report is not None:
            print(
                f"{split} gcf-lock -- "
                f"{gcf_debug_report['episodes_locked']}/{gcf_debug_report['episodes_total']} "
                f"({gcf_debug_report['lock_rate']*100:.2f}%), "
                f"mean_step={gcf_debug_report['mean_lock_step']}"
            )

        noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
        alt_env = f"_{args.alt_env}" if args.alt_env else ""
        mode_tag = args.eval_agent_mode
        selector_tag = args.eval_goal_selector if selector is not None else 'raw'
        output_path = (
            f"{args.model}_{model_trajectory}_{split}_{args.progress_stop_val}"
            f"{noise}{alt_env}_{mode_tag}_{selector_tag}.json"
        )
        with open(output_path, 'w') as f:
            json.dump(
                {
                    'metrics': metrics.to_dict(),
                    'spl': spl,
                    'eval_agent_mode': args.eval_agent_mode,
                    'eval_goal_selector': selector_tag,
                    'gcf_config': terminal_belief_config_dict(args) if args.eval_agent_mode == 'gcf' else None,
                    'gcf_debug': gcf_debug_report,
                    'trajectory_logs': {
                        str(eps_id): [tuple(pose) for pose in trajectory]
                        for eps_id, trajectory in trajectory_logs.items()
                    },
                    'pred_goal_logs': {
                        str(eps_id): [tuple(pos) for pos in pred_goals]
                        for eps_id, pred_goals in pred_goal_logs.items()
                    },
                    'pred_progress_logs': {
                        str(eps_id): pred_progresses
                        for eps_id, pred_progresses in pred_progress_logs.items()
                    },
                },
                f,
            )


if __name__ == '__main__':
    main()
