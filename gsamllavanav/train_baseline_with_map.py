from __future__ import annotations

import random
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizerFast

from gsamllavanav.defaultpaths import BASELINE_WITH_MAP_CHECKPOINT_DIR
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.observation import cropclient
from gsamllavanav.models.cma_with_map import CMAwithMap
from gsamllavanav.models.instr_decoder_with_map import (
    InstructionQueryDecoderWithCPVTMap,
    InstructionQueryDecoderWithDilutedMasaMap,
    InstructionQueryDecoderWithDirectionalUSCMap,
    InstructionQueryDecoderWithMap,
    InstructionQueryDecoderWithProgressOnlyMasaMap,
    InstructionQueryDecoderWithResidualMasaMap,
    InstructionQueryDecoderWithUSCMap,
    InstructionQueryDecoderWithUniPEMap,
    InstructionKVUSCMap,
)
from gsamllavanav.models.seq2seq_with_map import Seq2SeqwithMap
from gsamllavanav.model_registry import BASELINE_WITH_MAP_MODELS, get_model_spec
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.evaluate import eval_goal_predictor, GoalPredictorMetrics
from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch
from gsamllavanav import logger
from gsamllavanav.train import prepare_labels, _load_train_episodes


BaselineModelwithMap = {
    name: get_model_spec(name).factory
    for name in BASELINE_WITH_MAP_MODELS
}


def train(args: ExperimentArgs, device='cuda'):

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setup logger
    logger.init(args)
    for metric in GoalPredictorMetrics.names():
        logger.define_metric('val_seen_' + metric, 'epoch')
        logger.define_metric('val_unseen_' + metric, 'epoch')

    # load data
    start_epoch = 0
    objects = get_city_refer_objects()
    train_episodes = _load_train_episodes(objects, args)
    if args.train_episode_sample_size > 0:
        train_episodes = random.sample(train_episodes, args.train_episode_sample_size)
    train_dataloader = DataLoader(train_episodes, args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    val_seen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_seen', 'all', args.altitude))
    val_unseen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_unseen', 'all', args.altitude))
    cropclient.load_image_cache()

    # init model & optim
    baseline_model_with_map : Seq2SeqwithMap | CMAwithMap | InstructionQueryDecoderWithMap | InstructionQueryDecoderWithDilutedMasaMap = BaselineModelwithMap[args.model](args.map_size).to(device)
    optimizer = AdamW(baseline_model_with_map.parameters(), args.learning_rate)
    if args.checkpoint:
        start_epoch, baseline_model_with_map, optimizer = _load_checkpoint(baseline_model_with_map, optimizer, args)
    if args.progress_head_only_tune:
        _freeze_except_progress_head(baseline_model_with_map)
        optimizer = AdamW((p for p in baseline_model_with_map.parameters() if p.requires_grad), args.learning_rate)
    if args.dagger_rollout_posttrain:
        train_episodes = _build_dagger_posttrain_episodes(args, baseline_model_with_map, train_episodes, device)
        train_dataloader = DataLoader(train_episodes, args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if args.eval_at_start:
        _eval_predictor_and_log_metrics(baseline_model_with_map, val_seen_episodes, val_unseen_episodes, args, device)

    episodes_batch: list[Episode]
    loss_window = []
    global_batch = 0
    for epoch in trange(start_epoch, args.epochs, desc='epochs', unit='epoch', colour='#448844'):
        for episodes_batch in tqdm(train_dataloader, desc='train episodes', unit='batch', colour='#88dd88'):
            
            maps, rgbs, normalized_depths, instructions = prepare_inputs(episodes_batch, tokenizer, args, device)
            normalized_goal_xys, progresses = prepare_labels(episodes_batch, args, device)
            rnn_states = baseline_model_with_map.get_initial_recurrent_hidden_states(maps.shape[0], device)
            not_done_masks = torch.ones(maps.shape[0], dtype=bool, device=device)

            pred_normalized_goal_xys, pred_progresses, rnn_states = baseline_model_with_map(instructions, normalized_depths, rgbs, maps, rnn_states, not_done_masks)
                
            goal_prediction_loss = F.mse_loss(pred_normalized_goal_xys, normalized_goal_xys)
            progress_loss = F.mse_loss(pred_progresses, progresses)
            potential_rank_loss = _potential_ranking_loss(pred_progresses, episodes_batch, args)
            if args.progress_head_only_tune:
                loss = progress_loss + args.potential_rank_loss_weight * potential_rank_loss
            else:
                loss = goal_prediction_loss + progress_loss + args.potential_rank_loss_weight * potential_rank_loss
            loss.backward()
            logger.log({
                'loss': loss.item(),
                'goal_prediction_loss': goal_prediction_loss.item(),
                'progress_loss': progress_loss.item(),
                'potential_rank_loss': potential_rank_loss.item(),
            })
            global_batch += 1
            loss_window.append((loss.item(), goal_prediction_loss.item(), progress_loss.item(), potential_rank_loss.item()))
            if len(loss_window) == 100:
                _append_loss_per_100_batches(args, epoch, global_batch, loss_window)
                loss_window.clear()

            optimizer.step()
            optimizer.zero_grad()

        logger.log({'epoch': epoch})
        if loss_window:
            _append_loss_per_100_batches(args, epoch, global_batch, loss_window)
            loss_window.clear()
        
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(epoch, baseline_model_with_map, optimizer, args)
        
        if (epoch + 1) % args.eval_every == 0:
            _eval_predictor_and_log_metrics(baseline_model_with_map, val_seen_episodes, val_unseen_episodes, args, device)


def prepare_inputs(episodes_batch: list[Episode], tokenizer: BertTokenizerFast, args: ExperimentArgs, device: str):

    maps = np.concatenate([
        LandmarkNavMap.generate_maps_for_an_episode(
            episode, args.map_shape, args.map_pixels_per_meter, args.map_update_interval, args.gsam_rgb_shape, args.gsam_params, args.gsam_use_map_cache
        )
        for episode in episodes_batch
    ])

    rgbs = np.stack([
        cropclient.crop_model_image(episode.map_name, pose, 'rgb')
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2)

    normalized_depths = np.stack([
        cropclient.crop_model_image(episode.map_name, pose, 'depth')
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2) / args.max_depth

    instructions : torch.Tensor = tokenizer(
        [
            episode.target_description 
            for episode in episodes_batch
            for _ in episode.sample_trajectory(args.map_update_interval)
        ],
        padding=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors='pt',
    )['input_ids']


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
    instructions = instructions.to(device)

    return maps, rgbs, normalized_depths, instructions


def _eval_predictor_and_log_metrics(
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap | InstructionQueryDecoderWithMap,
    val_seen_episodes: list[Episode],
    val_unseen_episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
):
    val_seen_metrics = eval_goal_predictor(args, val_seen_episodes, *run_episodes_batch(args, baseline_model_with_map, val_seen_episodes, device))
    val_unseen_metrics = eval_goal_predictor(args, val_unseen_episodes,  *run_episodes_batch(args, baseline_model_with_map, val_unseen_episodes, device))
    logger.log({'val_seen_' + k: v for k, v in val_seen_metrics.to_dict().items()})
    logger.log({'val_unseen_' + k: v for k, v in val_unseen_metrics.to_dict().items()})


def _load_checkpoint(
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap | InstructionQueryDecoderWithMap,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    checkpoint = torch.load(args.checkpoint)
    start_epoch: int = checkpoint['epoch'] + 1
    baseline_model_with_map.load_state_dict(checkpoint['predictor_state_dict'])
    if not args.progress_head_only_tune:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate

    return start_epoch, baseline_model_with_map, optimizer


def _freeze_except_progress_head(baseline_model_with_map: torch.nn.Module):
    for param in baseline_model_with_map.parameters():
        param.requires_grad = False
    for param in baseline_model_with_map.progress_prediction_head.parameters():
        param.requires_grad = True


def _save_checkpoint(
    epoch: int,
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap | InstructionQueryDecoderWithMap,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    checkpoint_dir = _checkpoint_dir(args)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            'epoch': epoch,
            'predictor_state_dict': baseline_model_with_map.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        checkpoint_dir/f"{epoch:03d}.pth"
    )


def _checkpoint_dir(args: ExperimentArgs):
    ablation = f"-{args.ablate}" if args.ablate else ''
    train_size = '' if args.train_episode_sample_size < 0 else f"_{args.train_episode_sample_size}"
    posttrain_tag = f"_{args.posttrain_tag}" if args.posttrain_tag else ''
    return BASELINE_WITH_MAP_CHECKPOINT_DIR/args.model/f"{args.train_trajectory_type}_{args.altitude}_{args.gsam_box_threshold}{ablation}{train_size}{posttrain_tag}"


def _build_dagger_posttrain_episodes(
    args: ExperimentArgs,
    baseline_model_with_map: CMAwithMap | Seq2SeqwithMap | InstructionQueryDecoderWithMap,
    train_episodes: list[Episode],
    device: str,
) -> list[Episode]:
    if not args.checkpoint:
        raise ValueError("--dagger_rollout_posttrain requires --checkpoint")

    sample_size = min(args.dagger_episode_sample_size, len(train_episodes))
    rollout_source = random.sample(train_episodes, sample_size) if sample_size > 0 else train_episodes

    was_training = baseline_model_with_map.training
    baseline_model_with_map.eval()
    pose_logs, _, _ = run_episodes_batch(args, baseline_model_with_map, rollout_source, device)
    if was_training:
        baseline_model_with_map.train()

    dagger_episodes: list[Episode] = []
    for episode in rollout_source:
        trajectory = pose_logs.get(episode.id, [])
        if len(trajectory) < 1:
            continue
        teacher_actions = [0] * max(len(trajectory) - 1, 0)
        dagger_episodes.append(Episode(
            episode.target_object,
            episode.description_id,
            trajectory,
            teacher_actions,
        ))

    if not dagger_episodes:
        raise RuntimeError("DAgger rollout produced no trainable episodes")

    expert_keep = int(round(len(train_episodes) * (1.0 - args.dagger_mix_ratio)))
    expert_keep = max(0, min(expert_keep, len(train_episodes)))
    expert_episodes = random.sample(train_episodes, expert_keep) if expert_keep > 0 else []
    mixed = expert_episodes + dagger_episodes
    random.shuffle(mixed)
    print(
        f"DAgger posttrain episodes: expert={len(expert_episodes)}, "
        f"rollout={len(dagger_episodes)}, total={len(mixed)}"
    )
    return mixed


def _potential_ranking_loss(pred_progresses: torch.Tensor, episodes_batch: list[Episode], args: ExperimentArgs):
    if args.potential_rank_loss_weight <= 0:
        return pred_progresses.new_zeros(())

    losses = []
    offset = 0
    for episode in episodes_batch:
        num_steps = len(episode.sample_trajectory(args.map_update_interval))
        if num_steps > 1:
            progress = pred_progresses[offset:offset + num_steps].squeeze(-1)
            losses.append(F.relu(args.potential_rank_margin - progress[1:] + progress[:-1]))
        offset += num_steps

    if not losses:
        return pred_progresses.new_zeros(())
    return torch.cat(losses).mean()


def _append_loss_per_100_batches(args: ExperimentArgs, epoch: int, global_batch: int, loss_window):
    checkpoint_dir = _checkpoint_dir(args)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    path = checkpoint_dir/"loss_per_100_batches.csv"
    write_header = not path.exists()
    n = len(loss_window)
    avg_loss = sum(item[0] for item in loss_window) / n
    avg_goal = sum(item[1] for item in loss_window) / n
    avg_progress = sum(item[2] for item in loss_window) / n
    avg_rank = sum(item[3] for item in loss_window) / n
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "global_batch", "num_batches", "loss", "goal_prediction_loss", "progress_loss", "potential_rank_loss"])
        writer.writerow([epoch, global_batch, n, avg_loss, avg_goal, avg_progress, avg_rank])
