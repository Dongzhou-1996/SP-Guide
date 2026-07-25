from __future__ import annotations

from collections import defaultdict
import sys

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizerFast

from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.dataset.episode import Episode, EpisodeID
from gsamllavanav.observation import cropclient
from gsamllavanav.models.cma_with_map import CMAwithMap
from gsamllavanav.models.instr_decoder_with_map import InstructionQueryDecoderWithDilutedMasaMap, InstructionQueryDecoderWithMap
from gsamllavanav.models.seq2seq_with_map import Seq2SeqwithMap
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.space import Point2D, Pose4D
from gsamllavanav.evaluate import move, unnormalize_position
from gsamllavanav.terminal_belief import (
    fit_terminal_goal_belief_window,
    terminal_belief_rejection_reason,
    terminal_belief_stats,
)


@torch.no_grad()
def run_episodes_batch(
    args: ExperimentArgs,
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap | InstructionQueryDecoderWithMap | InstructionQueryDecoderWithDilutedMasaMap,
    episodes: list[Episode],
    device: str,
):
    was_training = baseline_model_with_map.training
    baseline_model_with_map.eval()
    cropclient.load_image_cache(alt_env=args.alt_env)
    dataloader = DataLoader(episodes, args.eval_batch_size, shuffle=False, collate_fn=lambda x: x)
    total_episodes = len(episodes)
    total_batches = len(dataloader)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=True)
    
    pose_logs: dict[EpisodeID, list[Pose4D]] = defaultdict(list)
    pred_goal_logs: dict[EpisodeID, list[Point2D]] = defaultdict(list)
    pred_progress_logs: dict[EpisodeID, list[float]] = defaultdict(list)
    gcf_debug = {
        'episodes_total': len(episodes),
        'episodes_locked': 0,
        'lock_first_steps': [],
        'reason_counts': defaultdict(int),
        'reason_examples': defaultdict(list),
    } if args.terminal_belief_debug else None

    episodes_batch: list[Episode]
    overall_pbar = tqdm(
        total=total_episodes,
        desc='eval progress',
        unit='episode',
        colour='#55cc88',
        position=0,
        leave=True,
        file=sys.stdout,
    )
    for batch_idx, episodes_batch in enumerate(
        tqdm(
            dataloader,
            desc='eval episodes',
            unit='batch',
            colour='#88dd88',
            position=1,
            file=sys.stdout,
        ),
        start=1,
    ):

        # init episode
        batch_size = len(episodes_batch)
        poses = [eps.start_pose for eps in episodes_batch]
        dones = np.zeros(batch_size, dtype=bool)
        progress_stop_counts = np.zeros(batch_size, dtype=np.int32)
        recent_goal_xys: list[list[Point2D]] = [[] for _ in range(batch_size)]
        recent_progresses: list[list[float]] = [[] for _ in range(batch_size)]
        locked_goal_xys: list[Point2D | None] = [None for _ in range(batch_size)]
        lock_first_steps = [-1 for _ in range(batch_size)]
        nav_maps = [
            LandmarkNavMap(
                eps.map_name, args.map_shape, args.map_pixels_per_meter,
                eps.description_landmarks, eps.description_target, eps.description_surroundings, args.gsam_params
            ) for eps in episodes_batch
        ]
        instructions : torch.Tensor = tokenizer(
            [episode.target_description for episode in episodes_batch],
            padding=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='pt',
        )['input_ids'].to(device)
        rnn_states = baseline_model_with_map.get_initial_recurrent_hidden_states(batch_size, device)

        for t in trange(
            args.eval_max_timestep,
            desc='eval timestep',
            unit='step',
            colour='#66aa66',
            position=2,
            leave=False,
            file=sys.stdout,
        ):

            gps_noise_batch = np.random.normal(scale=args.gps_noise_scale, size=(batch_size, 2))
            noisy_poses = [Pose4D(x + n_x, y + n_y, z, yaw) for (x, y, z, yaw), (n_x, n_y) in zip(poses, gps_noise_batch)]
            
            # update map
            for eps, pose, noisy_pose, nav_map, done in tqdm(
                zip(episodes_batch, poses, noisy_poses, nav_maps, dones),
                desc='updating maps',
                unit='map',
                colour='#448844',
                position=3,
                leave=False,
                file=sys.stdout,
            ):
                if not done:
                    gsam_rgb = cropclient.crop_image(eps.map_name, pose, args.gsam_rgb_shape, 'rgb')
                    nav_map.update_observations(noisy_pose, gsam_rgb, None, args.gsam_use_map_cache)
                    pose_logs[eps.id].append(pose)

            # prepare inputs
            maps = np.stack([nav_map.to_array() for nav_map in nav_maps])
            rgbs = np.stack([cropclient.crop_model_image(eps.map_name, pose, 'rgb') for pose in poses]).transpose(0, 3, 1, 2)
            normalized_depths = np.stack([cropclient.crop_model_image(eps.map_name, pose, 'depth') for pose in poses]).transpose(0, 3, 1, 2) / args.max_depth

            if args.ablate == 'rgb':
                rgbs = np.zeros_like(rgbs)
            if args.ablate == 'depth':
                normalized_depths = np.zeros_like(normalized_depths)
            if args.ablate == 'tracking':
                maps[:, :2] = 0
            if args.ablate == 'landmark':
                maps[:, 2] = 0
            if args.ablate == 'gsam':
                maps[:, 3:] = 0

            maps = torch.tensor(maps, device=device)
            rgbs = torch.tensor(rgbs, device=device)
            normalized_depths = torch.tensor(normalized_depths, device=device, dtype=torch.float32)
            not_dones = torch.from_numpy(~dones).to(device=device)

            # predict
            pred_normalized_goal_xys, pred_progresses, rnn_states = baseline_model_with_map(instructions, normalized_depths, rgbs, maps, rnn_states, not_dones)
            pred_goal_xys = [unnormalize_position(xy.tolist(), eps.map_name, args.map_meters) for eps, xy in zip(episodes_batch, pred_normalized_goal_xys)]
            for eps, done, xy, progress in zip(episodes_batch, dones, pred_goal_xys, pred_progresses.flatten().tolist()):
                if not done:
                    pred_goal_logs[eps.id].append(xy)
                    pred_progress_logs[eps.id].append(progress)

            progress_values = pred_progresses.cpu().numpy().flatten()
            if args.gcf_filter:
                for idx, (done, xy, progress) in enumerate(zip(dones, pred_goal_xys, progress_values)):
                    if done:
                        continue
                    recent_goal_xys[idx].append(xy)
                    recent_progresses[idx].append(float(progress))
                    recent_goal_xys[idx] = recent_goal_xys[idx][-args.terminal_belief_window:]
                    recent_progresses[idx] = recent_progresses[idx][-args.terminal_belief_window:]
                    stats = terminal_belief_stats(
                        args,
                        recent_goal_xys[idx],
                        recent_progresses[idx],
                    )
                    if locked_goal_xys[idx] is None:
                        if gcf_debug is not None:
                            reason = terminal_belief_rejection_reason(args, t, stats)
                            if reason is None:
                                lock_first_steps[idx] = t
                            else:
                                gcf_debug['reason_counts'][reason] += 1
                                if len(gcf_debug['reason_examples'][reason]) < 3:
                                    gcf_debug['reason_examples'][reason].append({
                                        'episode_id': str(episodes_batch[idx].id),
                                        'timestep': int(t),
                                        'weighted_progress': round(stats.weighted_progress, 4),
                                        'variance': round(stats.variance, 4),
                                        'sigma': round(stats.sigma, 4),
                                        'endpoint_dist': round(stats.endpoint_dist, 4),
                                        'window_size': int(stats.window_size),
                                    })
                        locked_goal_xys[idx] = fit_terminal_goal_belief_window(
                            args,
                            t,
                            recent_goal_xys[idx],
                            recent_progresses[idx],
                        )
                dones = dones | np.array([
                    locked_goal is not None and pose.xy.dist_to(locked_goal) <= args.terminal_belief_arrival_dist
                    for pose, locked_goal in zip(poses, locked_goal_xys)
                ], dtype=bool)
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

            # move
            if args.gcf_filter:
                move_goal_xys = []
                for locked_goal, pred_goal in zip(locked_goal_xys, pred_goal_xys):
                    if locked_goal is not None:
                        move_goal_xys.append(locked_goal)
                    else:
                        move_goal_xys.append(pred_goal)
            else:
                move_goal_xys = pred_goal_xys
            poses = [
                move(pose, xy, args.move_iteration, noisy_pose) if not done else pose
                for pose, noisy_pose, xy, done in zip(poses, noisy_poses, move_goal_xys, dones)
            ]

        if gcf_debug is not None:
            for idx, first_step in enumerate(lock_first_steps):
                if first_step >= 0:
                    gcf_debug['episodes_locked'] += 1
                    gcf_debug['lock_first_steps'].append(int(first_step))
        overall_pbar.update(batch_size)
        overall_pbar.set_postfix_str(f"batch {batch_idx}/{total_batches}")
        if not sys.stdout.isatty():
            tqdm.write(
                f"[eval] batch {batch_idx}/{total_batches} finished "
                f"({min(overall_pbar.n, total_episodes)}/{total_episodes} episodes)",
                file=sys.stdout,
            )

    overall_pbar.close()
    if was_training:
        baseline_model_with_map.train()
    if gcf_debug is not None:
        lock_steps = gcf_debug['lock_first_steps']
        gcf_debug_report = {
            'episodes_total': int(gcf_debug['episodes_total']),
            'episodes_locked': int(gcf_debug['episodes_locked']),
            'lock_rate': (
                float(gcf_debug['episodes_locked']) / max(int(gcf_debug['episodes_total']), 1)
            ),
            'mean_lock_step': float(np.mean(lock_steps)) if lock_steps else None,
            'median_lock_step': float(np.median(lock_steps)) if lock_steps else None,
            'reason_counts': dict(sorted(gcf_debug['reason_counts'].items())),
            'reason_examples': dict(gcf_debug['reason_examples']),
        }
        setattr(args, '_terminal_belief_debug_report', gcf_debug_report)
    return dict(pose_logs), dict(pred_goal_logs), dict(pred_progress_logs)
