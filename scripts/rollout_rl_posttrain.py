#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsamllavanav import logger
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.episode import Episode, EpisodeID
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.defaultpaths import BASELINE_WITH_MAP_CHECKPOINT_DIR
from gsamllavanav.evaluate import GoalPredictorMetrics, eval_goal_predictor, move, unnormalize_position
from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.model_registry import BASELINE_WITH_MAP_MODELS, get_model_spec
from gsamllavanav.observation import cropclient
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.terminal_belief import fit_terminal_goal_belief_window
from gsamllavanav.train import _load_train_episodes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rollout-based RL post-training for CityNav/USCNav baseline-with-map models. "
            "The current policy samples normalized goal coordinates online, rolls out the "
            "navigation agent, scores whole trajectories, and updates the model with a "
            "self-critical REINFORCE objective plus on-policy auxiliary supervision."
        )
    )
    parser.add_argument("--model", choices=BASELINE_WITH_MAP_MODELS, default="instr_decoder_with_map")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--posttrain_tag", default="rollout_rl")
    parser.add_argument("--train_trajectory_type", choices=["sp", "mturk", "both"], default="mturk")
    parser.add_argument("--altitude", type=float, default=50.0)
    parser.add_argument("--learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_episode_sample_size", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_at_start", action="store_true")
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--resume_log_id", default="")

    parser.add_argument("--rollout_max_timestep", type=int, default=20)
    parser.add_argument("--move_iteration", type=int, default=5)
    parser.add_argument("--progress_stop_val", type=float, default=0.75)
    parser.add_argument("--progress_stop_patience", type=int, default=1)
    parser.add_argument("--min_stop_step", type=int, default=0)
    parser.add_argument("--rollout_agent_mode", choices=["progress_stop", "gcf"], default="progress_stop")
    parser.add_argument("--rl_sample_stop", action="store_true")

    parser.add_argument("--goal_concentration", type=float, default=24.0)
    parser.add_argument("--rl_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=1.0e-3)
    parser.add_argument("--goal_aux_weight", type=float, default=1.0)
    parser.add_argument("--progress_aux_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--clip_advantage", type=float, default=5.0)
    parser.add_argument("--self_critical", action="store_true", default=True)
    parser.add_argument("--no_self_critical", action="store_false", dest="self_critical")

    parser.add_argument("--distance_weight", type=float, default=1.0)
    parser.add_argument("--improvement_weight", type=float, default=1.0)
    parser.add_argument("--success_bonus", type=float, default=2.0)
    parser.add_argument("--oracle_bonus", type=float, default=0.5)
    parser.add_argument("--progress_bonus", type=float, default=0.25)
    parser.add_argument("--step_penalty", type=float, default=0.05)

    parser.add_argument("--map_size", type=int, default=240)
    parser.add_argument("--map_meters", type=float, default=410.0)
    parser.add_argument("--map_update_interval", type=int, default=5)
    parser.add_argument("--max_depth", type=float, default=200.0)
    parser.add_argument("--ablate", choices=["rgb", "depth", "tracking", "landmark", "gsam", ""], default="")
    parser.add_argument("--alt_env", choices=["", "flood", "ground_fissure"], default="")
    parser.add_argument("--gsam_rgb_shape", type=int, default=500)
    parser.add_argument("--gsam_use_segmentation_mask", action="store_true")
    parser.add_argument("--gsam_use_bbox_confidence", action="store_true")
    parser.add_argument("--gsam_use_map_cache", action="store_true")
    parser.add_argument("--gsam_box_threshold", type=float, default=0.20)
    parser.add_argument("--gsam_text_threshold", type=float, default=0.25)
    parser.add_argument("--gsam_max_box_size", type=float, default=50.0)
    parser.add_argument("--gsam_max_box_area", type=float, default=3000.0)

    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--success_dist", type=float, default=20.0)
    parser.add_argument("--success_iou", type=float, default=0.4)
    parser.add_argument("--gps_noise_scale", type=float, default=0.0)

    parser.add_argument("--gcf_window", type=int, default=5)
    parser.add_argument("--gcf_beta", type=float, default=8.0)
    parser.add_argument("--gcf_progress_thr", type=float, default=0.80)
    parser.add_argument("--gcf_variance_thr", type=float, default=400.0)
    parser.add_argument("--gcf_min_step", type=int, default=8)
    parser.add_argument("--gcf_arrival_dist", type=float, default=15.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_experiment_args(cli: argparse.Namespace) -> ExperimentArgs:
    return ExperimentArgs(
        seed=cli.seed,
        mode="train",
        model=cli.model,
        log=cli.log,
        silent=cli.silent,
        resume_log_id=cli.resume_log_id,
        map_size=cli.map_size,
        map_meters=cli.map_meters,
        map_update_interval=cli.map_update_interval,
        max_depth=cli.max_depth,
        altitude=cli.altitude,
        ablate=cli.ablate,
        alt_env=cli.alt_env,
        data_root=cli.data_root,
        gsam_rgb_shape=(cli.gsam_rgb_shape, cli.gsam_rgb_shape),
        gsam_use_segmentation_mask=cli.gsam_use_segmentation_mask,
        gsam_use_bbox_confidence=cli.gsam_use_bbox_confidence,
        gsam_use_map_cache=cli.gsam_use_map_cache,
        gsam_box_threshold=cli.gsam_box_threshold,
        gsam_text_threshold=cli.gsam_text_threshold,
        gsam_max_box_size=cli.gsam_max_box_size,
        gsam_max_box_area=cli.gsam_max_box_area,
        learning_rate=cli.learning_rate,
        train_batch_size=cli.train_batch_size,
        epochs=cli.epochs,
        checkpoint=cli.checkpoint,
        save_every=cli.save_every,
        train_trajectory_type=cli.train_trajectory_type,
        train_episode_sample_size=cli.train_episode_sample_size,
        potential_rank_loss_weight=0.0,
        potential_rank_margin=0.02,
        progress_head_only_tune=False,
        posttrain_tag=cli.posttrain_tag,
        dagger_rollout_posttrain=False,
        dagger_episode_sample_size=0,
        dagger_mix_ratio=0.0,
        eval_every=cli.eval_every,
        eval_batch_size=cli.eval_batch_size,
        eval_at_start=cli.eval_at_start,
        eval_max_timestep=cli.rollout_max_timestep,
        eval_client="crop",
        success_dist=cli.success_dist,
        success_iou=cli.success_iou,
        move_iteration=cli.move_iteration,
        progress_stop_val=cli.progress_stop_val,
        min_stop_step=cli.min_stop_step,
        progress_stop_patience=cli.progress_stop_patience,
        gcf_filter=cli.rollout_agent_mode == "gcf",
        terminal_belief_window=cli.gcf_window,
        terminal_belief_beta=cli.gcf_beta,
        terminal_belief_progress_thr=cli.gcf_progress_thr,
        terminal_belief_variance_thr=cli.gcf_variance_thr,
        terminal_belief_min_step=cli.gcf_min_step,
        terminal_belief_arrival_dist=cli.gcf_arrival_dist,
        terminal_belief_debug=False,
        eval_agent_mode=cli.rollout_agent_mode,
        eval_goal_selector="raw",
        gps_noise_scale=cli.gps_noise_scale,
        sim_ip="172.23.96.1",
        sim_port=41451,
    )


def normalize_position_xy(pos_xy, map_name: str, map_meters: float) -> tuple[float, float]:
    return (
        (pos_xy.x - MAP_BOUNDS[map_name].x_min) / map_meters,
        (MAP_BOUNDS[map_name].y_max - pos_xy.y) / map_meters,
    )


def checkpoint_dir(args: ExperimentArgs) -> Path:
    ablation = f"-{args.ablate}" if args.ablate else ""
    train_size = "" if args.train_episode_sample_size < 0 else f"_{args.train_episode_sample_size}"
    posttrain_tag = f"_{args.posttrain_tag}" if args.posttrain_tag else ""
    return (
        BASELINE_WITH_MAP_CHECKPOINT_DIR
        / args.model
        / f"{args.train_trajectory_type}_{args.altitude}_{args.gsam_box_threshold}{ablation}{train_size}{posttrain_tag}"
    )


def append_csv_row(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, args: ExperimentArgs, cli: argparse.Namespace) -> None:
    out_dir = checkpoint_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "predictor_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rollout_rl_cli": vars(cli),
            "experiment_args": asdict(args),
        },
        out_dir / f"{epoch:03d}.pth",
    )


def load_posttrain_init_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_optimizer: bool,
) -> None:
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state["predictor_state_dict"])
    if resume_optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])


def compute_reward(
    cli: argparse.Namespace,
    episodes_batch: list[Episode],
    pose_logs: dict[EpisodeID, list],
    pred_progress_logs: dict[EpisodeID, list[float]],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards: list[float] = []
    successes = 0
    oracle_successes = 0
    mean_final_dist = 0.0
    mean_oracle_dist = 0.0
    mean_steps = 0.0
    mean_progress_alignment = 0.0

    for episode in episodes_batch:
        trajectory = pose_logs[episode.id]
        progresses = pred_progress_logs[episode.id]
        final_pose = trajectory[-1]
        start_dist = max(episode.start_pose.xy.dist_to(episode.target_position.xy), 1e-6)
        final_dist = final_pose.xy.dist_to(episode.target_position.xy)
        oracle_dist = min(pose.xy.dist_to(episode.target_position.xy) for pose in trajectory)
        improvement = np.clip((start_dist - final_dist) / start_dist, -1.0, 1.0)
        final_progress_target = float(np.clip(1 - final_dist / start_dist, 0.0, 1.0))
        final_progress_pred = float(progresses[-1]) if progresses else 0.0
        progress_alignment = 1.0 - abs(final_progress_pred - final_progress_target)
        success = float(final_dist <= cli.success_dist)
        oracle_success = float(oracle_dist <= cli.success_dist)
        step_ratio = len(trajectory) / max(cli.rollout_max_timestep, 1)

        reward = 0.0
        reward += cli.success_bonus * success
        reward += cli.oracle_bonus * oracle_success
        reward += cli.improvement_weight * improvement
        reward += cli.progress_bonus * progress_alignment
        reward -= cli.distance_weight * (final_dist / cli.map_meters)
        reward -= cli.step_penalty * step_ratio

        rewards.append(float(reward))
        successes += int(success)
        oracle_successes += int(oracle_success)
        mean_final_dist += float(final_dist)
        mean_oracle_dist += float(oracle_dist)
        mean_steps += float(len(trajectory))
        mean_progress_alignment += float(progress_alignment)

    n = max(len(episodes_batch), 1)
    summary = {
        "reward": float(np.mean(rewards)) if rewards else 0.0,
        "rollout_sr": successes / n,
        "rollout_osr": oracle_successes / n,
        "rollout_ne": mean_final_dist / n,
        "rollout_oracle_ne": mean_oracle_dist / n,
        "rollout_steps": mean_steps / n,
        "progress_alignment": mean_progress_alignment / n,
    }
    return torch.tensor(rewards, device=DEVICE, dtype=torch.float32), summary


def _build_goal_beta(mean_xy: torch.Tensor, concentration: float) -> Beta:
    mean_xy = mean_xy.clamp(1e-4, 1 - 1e-4)
    concentration = max(concentration, 2.01)
    alpha = mean_xy * (concentration - 2.0) + 1.0
    beta = (1.0 - mean_xy) * (concentration - 2.0) + 1.0
    return Beta(alpha, beta)


def _prepare_step_observations(
    episodes_batch: list[Episode],
    poses,
    nav_maps,
    args: ExperimentArgs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maps = np.stack([nav_map.to_array() for nav_map in nav_maps])
    rgbs = np.stack([
        cropclient.crop_model_image(eps.map_name, pose, "rgb")
        for eps, pose in zip(episodes_batch, poses)
    ]).transpose(0, 3, 1, 2)
    normalized_depths = np.stack([
        cropclient.crop_model_image(eps.map_name, pose, "depth")
        for eps, pose in zip(episodes_batch, poses)
    ]).transpose(0, 3, 1, 2) / args.max_depth

    if args.ablate == "rgb":
        rgbs = np.zeros_like(rgbs)
    if args.ablate == "depth":
        normalized_depths = np.zeros_like(normalized_depths)
    if args.ablate == "tracking":
        maps[:, :2] = 0
    if args.ablate == "landmark":
        maps[:, 2] = 0
    if args.ablate == "gsam":
        maps[:, 3:] = 0

    return (
        torch.tensor(maps, device=DEVICE),
        torch.tensor(rgbs, device=DEVICE),
        torch.tensor(normalized_depths, device=DEVICE, dtype=torch.float32),
    )


def rollout_batch(
    cli: argparse.Namespace,
    args: ExperimentArgs,
    model: torch.nn.Module,
    tokenizer: BertTokenizerFast,
    episodes_batch: list[Episode],
    stochastic: bool,
    track_grad: bool,
):
    context = nullcontext() if track_grad else torch.no_grad()
    batch_size = len(episodes_batch)
    active_dtype = torch.float32
    progress_stop_counts = np.zeros(batch_size, dtype=np.int32)
    poses = [eps.start_pose for eps in episodes_batch]
    dones = np.zeros(batch_size, dtype=bool)
    recent_goal_xys: list[list] = [[] for _ in range(batch_size)]
    recent_progresses: list[list[float]] = [[] for _ in range(batch_size)]
    locked_goal_xys = [None for _ in range(batch_size)]
    nav_maps = [
        LandmarkNavMap(
            eps.map_name,
            args.map_shape,
            args.map_pixels_per_meter,
            eps.description_landmarks,
            eps.description_target,
            eps.description_surroundings,
            args.gsam_params,
        )
        for eps in episodes_batch
    ]
    instructions = tokenizer(
        [episode.target_description for episode in episodes_batch],
        padding=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)
    normalized_goal_targets = torch.tensor(
        [normalize_position_xy(eps.target_position.xy, eps.map_name, args.map_meters) for eps in episodes_batch],
        device=DEVICE,
        dtype=torch.float32,
    )
    start_dists = torch.tensor(
        [max(eps.start_pose.xy.dist_to(eps.target_position.xy), 1e-6) for eps in episodes_batch],
        device=DEVICE,
        dtype=torch.float32,
    )
    rnn_states = model.get_initial_recurrent_hidden_states(batch_size, DEVICE)
    log_prob_sums = torch.zeros(batch_size, device=DEVICE, dtype=torch.float32)
    entropy_sums = torch.zeros(batch_size, device=DEVICE, dtype=torch.float32)
    active_step_counts = torch.zeros(batch_size, device=DEVICE, dtype=torch.float32)
    goal_loss_sum = torch.zeros((), device=DEVICE, dtype=torch.float32)
    progress_loss_sum = torch.zeros((), device=DEVICE, dtype=torch.float32)
    loss_step_count = 0
    pose_logs: dict[EpisodeID, list] = {eps.id: [] for eps in episodes_batch}
    pred_goal_logs: dict[EpisodeID, list] = {eps.id: [] for eps in episodes_batch}
    pred_progress_logs: dict[EpisodeID, list[float]] = {eps.id: [] for eps in episodes_batch}

    with context:
        for t in range(cli.rollout_max_timestep):
            gps_noise_batch = np.random.normal(scale=args.gps_noise_scale, size=(batch_size, 2))
            noisy_poses = [
                type(pose)(pose.x + n_x, pose.y + n_y, pose.z, pose.yaw)
                for pose, (n_x, n_y) in zip(poses, gps_noise_batch)
            ]
            for idx, (eps, pose, noisy_pose, nav_map, done) in enumerate(zip(episodes_batch, poses, noisy_poses, nav_maps, dones)):
                if done:
                    continue
                gsam_rgb = cropclient.crop_image(eps.map_name, pose, args.gsam_rgb_shape, "rgb")
                nav_map.update_observations(noisy_pose, gsam_rgb, None, args.gsam_use_map_cache)
                pose_logs[eps.id].append(pose)

            maps, rgbs, normalized_depths = _prepare_step_observations(episodes_batch, poses, nav_maps, args)
            not_dones = torch.from_numpy(~dones).to(device=DEVICE)
            pred_goal_means, pred_progresses, rnn_states = model(
                instructions, normalized_depths, rgbs, maps, rnn_states, not_dones
            )
            if isinstance(rnn_states, torch.Tensor):
                rnn_states = rnn_states.detach()

            active_mask = (~dones)
            active_mask_t = torch.tensor(active_mask, device=DEVICE, dtype=active_dtype)
            if active_mask_t.sum() > 0:
                current_progress_targets = torch.tensor(
                    [
                        np.clip(1 - eps.target_position.xy.dist_to(pose.xy) / max(eps.start_pose.xy.dist_to(eps.target_position.xy), 1e-6), 0.0, 1.0)
                        for eps, pose in zip(episodes_batch, poses)
                    ],
                    device=DEVICE,
                    dtype=torch.float32,
                ).unsqueeze(-1)
                goal_loss_sum = goal_loss_sum + (
                    ((pred_goal_means - normalized_goal_targets) ** 2).mean(dim=1) * active_mask_t
                ).sum()
                progress_loss_sum = progress_loss_sum + (
                    ((pred_progresses - current_progress_targets) ** 2).squeeze(-1) * active_mask_t
                ).sum()
                loss_step_count += int(active_mask_t.sum().item())

            if stochastic:
                goal_dist = _build_goal_beta(pred_goal_means, cli.goal_concentration)
                sampled_normalized_goals = goal_dist.rsample()
                goal_log_prob = goal_dist.log_prob(sampled_normalized_goals).sum(dim=-1)
                goal_entropy = goal_dist.entropy().sum(dim=-1)
            else:
                sampled_normalized_goals = pred_goal_means
                goal_log_prob = torch.zeros(batch_size, device=DEVICE)
                goal_entropy = torch.zeros(batch_size, device=DEVICE)

            total_log_prob = goal_log_prob
            total_entropy = goal_entropy
            progress_values = pred_progresses.squeeze(-1).detach().cpu().numpy()

            sampled_stop = np.zeros(batch_size, dtype=bool)
            if cli.rl_sample_stop and t >= args.min_stop_step:
                stop_probs = pred_progresses.squeeze(-1).clamp(1e-4, 1 - 1e-4)
                if stochastic:
                    stop_dist = Bernoulli(probs=stop_probs)
                    sampled_stop_t = stop_dist.sample()
                    total_log_prob = total_log_prob + stop_dist.log_prob(sampled_stop_t)
                    total_entropy = total_entropy + stop_dist.entropy()
                    sampled_stop = sampled_stop_t.detach().bool().cpu().numpy()
                else:
                    sampled_stop = stop_probs.detach().cpu().numpy() >= args.progress_stop_val

            total_log_prob = total_log_prob * active_mask_t
            total_entropy = total_entropy * active_mask_t
            log_prob_sums = log_prob_sums + total_log_prob
            entropy_sums = entropy_sums + total_entropy
            active_step_counts = active_step_counts + active_mask_t

            pred_goal_xys = [
                unnormalize_position(xy.tolist(), eps.map_name, args.map_meters)
                for eps, xy in zip(episodes_batch, sampled_normalized_goals.detach())
            ]
            pred_progress_list = pred_progresses.squeeze(-1).detach().cpu().tolist()
            for eps, done, goal_xy, progress in zip(episodes_batch, dones, pred_goal_xys, pred_progress_list):
                if done:
                    continue
                pred_goal_logs[eps.id].append(goal_xy)
                pred_progress_logs[eps.id].append(float(progress))

            if args.gcf_filter:
                for idx, (done, xy, progress) in enumerate(zip(dones, pred_goal_xys, pred_progress_list)):
                    if done:
                        continue
                    recent_goal_xys[idx].append(xy)
                    recent_progresses[idx].append(float(progress))
                    recent_goal_xys[idx] = recent_goal_xys[idx][-args.terminal_belief_window :]
                    recent_progresses[idx] = recent_progresses[idx][-args.terminal_belief_window :]
                    if locked_goal_xys[idx] is None:
                        locked_goal_xys[idx] = fit_terminal_goal_belief_window(
                            args,
                            t,
                            recent_goal_xys[idx],
                            recent_progresses[idx],
                        )
                belief_done = np.array(
                    [
                        locked_goal is not None and pose.xy.dist_to(locked_goal) <= args.terminal_belief_arrival_dist
                        for pose, locked_goal in zip(poses, locked_goal_xys)
                    ],
                    dtype=bool,
                )
                dones = dones | belief_done
            elif cli.rl_sample_stop:
                sampled_stop = sampled_stop & (~dones)
                dones = dones | sampled_stop
            else:
                can_stop = t >= args.min_stop_step
                progress_stop_counts = np.where(
                    (~dones) & can_stop & (progress_values >= args.progress_stop_val),
                    progress_stop_counts + 1,
                    0,
                )
                dones = dones | (progress_stop_counts >= args.progress_stop_patience)

            if dones.all():
                break

            move_goal_xys = [
                locked_goal if locked_goal is not None else pred_goal
                for locked_goal, pred_goal in zip(locked_goal_xys, pred_goal_xys)
            ] if args.gcf_filter else pred_goal_xys
            poses = [
                move(pose, goal_xy, args.move_iteration, noisy_pose) if not done else pose
                for pose, noisy_pose, goal_xy, done in zip(poses, noisy_poses, move_goal_xys, dones)
            ]

    goal_aux_loss = goal_loss_sum / max(loss_step_count, 1)
    progress_aux_loss = progress_loss_sum / max(loss_step_count, 1)
    rewards, reward_summary = compute_reward(cli, episodes_batch, pose_logs, pred_progress_logs)
    return {
        "rewards": rewards,
        "reward_summary": reward_summary,
        "log_prob_sums": log_prob_sums,
        "entropy_sums": entropy_sums,
        "active_step_counts": active_step_counts,
        "goal_aux_loss": goal_aux_loss,
        "progress_aux_loss": progress_aux_loss,
        "pose_logs": pose_logs,
        "pred_goal_logs": pred_goal_logs,
        "pred_progress_logs": pred_progress_logs,
    }


def evaluate_and_log(model: torch.nn.Module, args: ExperimentArgs, val_seen_episodes: list[Episode], val_unseen_episodes: list[Episode]) -> dict[str, float]:
    metrics_out: dict[str, float] = {}
    model.eval()
    for prefix, episodes in (("val_seen", val_seen_episodes), ("val_unseen", val_unseen_episodes)):
        metrics = eval_goal_predictor(args, episodes, *run_episodes_batch(args, model, episodes, DEVICE))
        metrics_dict = metrics.to_dict()
        metrics_out.update({f"{prefix}_{k}": v for k, v in metrics_dict.items()})
    logger.log(metrics_out)
    model.train()
    return metrics_out


def main() -> None:
    cli = parse_cli()
    args = build_experiment_args(cli)
    os.environ["SP_GUIDE_DATA_ROOT"] = args.data_root
    if cli.dry_run:
        print("Dry run config:")
        print(vars(cli))
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.init(args)
    for metric in GoalPredictorMetrics.names():
        logger.define_metric(f"val_seen_{metric}", "epoch")
        logger.define_metric(f"val_unseen_{metric}", "epoch")

    cropclient.load_image_cache(alt_env=args.alt_env)
    objects = get_city_refer_objects()
    train_episodes = _load_train_episodes(objects, args)
    if args.train_episode_sample_size > 0:
        sample_size = min(args.train_episode_sample_size, len(train_episodes))
        train_episodes = random.sample(train_episodes, sample_size)
    train_loader = DataLoader(train_episodes, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    should_eval = args.eval_at_start or args.eval_every > 0
    val_seen_episodes: list[Episode] = []
    val_unseen_episodes: list[Episode] = []
    if should_eval:
        val_seen_episodes = generate_episodes_from_mturk_trajectories(
            objects, load_mturk_trajectories("val_seen", "all", args.altitude)
        )
        val_unseen_episodes = generate_episodes_from_mturk_trajectories(
            objects, load_mturk_trajectories("val_unseen", "all", args.altitude)
        )

    model_spec = get_model_spec(args.model)
    model = model_spec.factory(args.map_size).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    if args.checkpoint:
        load_posttrain_init_checkpoint(args.checkpoint, model, optimizer, cli.resume_optimizer)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)
    out_dir = checkpoint_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log_csv = out_dir / "rollout_rl_train_metrics.csv"

    if args.eval_at_start:
        evaluate_and_log(model, args, val_seen_episodes, val_unseen_episodes)

    global_step = 0
    for epoch in trange(start_epoch, args.epochs, desc="epochs", unit="epoch", colour="#448844"):
        model.train()
        progress_bar = tqdm(train_loader, desc="rollout rl", unit="batch", colour="#88dd88")
        for episodes_batch in progress_bar:
            sampled = rollout_batch(cli, args, model, tokenizer, episodes_batch, stochastic=True, track_grad=True)
            if cli.self_critical:
                greedy = rollout_batch(cli, args, model, tokenizer, episodes_batch, stochastic=False, track_grad=False)
                baseline_rewards = greedy["rewards"]
            else:
                baseline_rewards = sampled["rewards"].mean().expand_as(sampled["rewards"])

            advantages = sampled["rewards"] - baseline_rewards
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
            advantages = advantages.clamp(-cli.clip_advantage, cli.clip_advantage)
            mean_log_prob = sampled["log_prob_sums"] / sampled["active_step_counts"].clamp_min(1.0)
            mean_entropy = sampled["entropy_sums"] / sampled["active_step_counts"].clamp_min(1.0)

            policy_loss = -(advantages.detach() * mean_log_prob).mean()
            entropy_loss = -mean_entropy.mean()
            aux_loss = (
                cli.goal_aux_weight * sampled["goal_aux_loss"]
                + cli.progress_aux_weight * sampled["progress_aux_loss"]
            )
            loss = cli.rl_weight * policy_loss + aux_loss + cli.entropy_weight * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cli.grad_clip_norm)
            optimizer.step()

            global_step += 1
            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "aux_loss": float(aux_loss.item()),
                "goal_aux_loss": float(sampled["goal_aux_loss"].item()),
                "progress_aux_loss": float(sampled["progress_aux_loss"].item()),
                "entropy": float(mean_entropy.mean().item()),
                "sample_reward": float(sampled["reward_summary"]["reward"]),
                "baseline_reward": float(baseline_rewards.mean().item()),
                "advantage": float(advantages.mean().item()),
                "rollout_sr": float(sampled["reward_summary"]["rollout_sr"]),
                "rollout_osr": float(sampled["reward_summary"]["rollout_osr"]),
                "rollout_ne": float(sampled["reward_summary"]["rollout_ne"]),
                "rollout_oracle_ne": float(sampled["reward_summary"]["rollout_oracle_ne"]),
                "rollout_steps": float(sampled["reward_summary"]["rollout_steps"]),
                "progress_alignment": float(sampled["reward_summary"]["progress_alignment"]),
            }
            logger.log(metrics)
            append_csv_row(train_log_csv, metrics)
            progress_bar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                reward=f"{metrics['sample_reward']:.3f}",
                sr=f"{metrics['rollout_sr']:.2%}",
            )

        logger.log({"epoch": epoch})
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(epoch, model, optimizer, args, cli)
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            evaluate_and_log(model, args, val_seen_episodes, val_unseen_episodes)

    logger.finish()


if __name__ == "__main__":
    main()
