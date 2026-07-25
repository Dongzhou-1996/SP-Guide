import argparse
import os
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass, asdict

from gsamllavanav.maps.gsam_map import GSamParams
from gsamllavanav.model_registry import ALL_MODEL_NAMES


@dataclass
class ExperimentArgs:

    seed: int
    mode: Literal['train', 'eval']
    
    model: str

    # logger
    log: bool
    silent: bool
    resume_log_id: str

    # observation
    map_size: int
    map_meters: float
    map_update_interval: int
    max_depth: float
    altitude: float
    ablate: Literal['rgb', 'depth', 'tracking', 'landmark', 'gsam', '']
    alt_env: Literal['flood', 'ground_fissure', '']
    data_root: str

    # gsam
    gsam_rgb_shape: tuple[int, int]
    gsam_use_segmentation_mask: bool
    gsam_use_bbox_confidence: bool
    gsam_use_map_cache: bool
    gsam_box_threshold: float
    gsam_text_threshold: float
    gsam_max_box_size: float
    gsam_max_box_area: float

    # training params
    learning_rate: float
    train_batch_size: int
    epochs: int
    checkpoint: Optional[str]
    save_every: int
    train_trajectory_type: Literal['sp', 'mturk', 'both']
    train_episode_sample_size: int
    potential_rank_loss_weight: float
    potential_rank_margin: float
    progress_head_only_tune: bool
    posttrain_tag: str
    dagger_rollout_posttrain: bool
    dagger_episode_sample_size: int
    dagger_mix_ratio: float
    
    # eval params
    eval_every: int
    eval_batch_size: int
    eval_at_start: bool
    eval_max_timestep: int
    eval_client: Literal['crop', 'airsim']
    success_dist: float
    success_iou: float
    move_iteration: int
    progress_stop_val: float
    min_stop_step: int
    progress_stop_patience: int
    gcf_filter: bool
    terminal_belief_window: int
    terminal_belief_beta: float
    terminal_belief_progress_thr: float
    terminal_belief_variance_thr: float
    terminal_belief_min_step: int
    terminal_belief_arrival_dist: float
    terminal_belief_debug: bool
    eval_agent_mode: Literal['progress_stop', 'gcf', 'selector']
    eval_goal_selector: Literal['gdino', 'llava', 'gcf', 'raw']
    gps_noise_scale: float

    # airsim
    sim_ip: str
    sim_port: int

    def to_dict(self):
        return asdict(self)
    
    @property
    def map_shape(self):
        return self.map_size, self.map_size
    
    @property
    def map_pixels_per_meter(self):
        return self.map_size / self.map_meters
    
    @property
    def gsam_params(self):
        return GSamParams(
            self.gsam_use_segmentation_mask,
            self.gsam_use_bbox_confidence,
            self.gsam_box_threshold, self.gsam_text_threshold,
            self.gsam_max_box_size, self.gsam_max_box_area
        )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')

    parser.add_argument(
        '--model',
        type=str,
        choices=ALL_MODEL_NAMES,
        default='mgp',
    )

    # logger
    parser.add_argument('--log', action='store_true', default=False, help="log results to wandb")
    parser.add_argument('--silent', action='store_true', default=False, help="disable printing log info to stdout")
    parser.add_argument('--resume_log_id', type=str, default='')

    # observation
    parser.add_argument('--map_size', type=int, default=240)
    parser.add_argument('--map_meters', type=float, default=410.)
    parser.add_argument('--map_update_interval', type=int, default=5)
    parser.add_argument('--max_depth', type=float, default=200.)
    parser.add_argument('--altitude', type=float, default=50)
    parser.add_argument('--ablate', type=str, choices=['rgb', 'depth', 'tracking', 'landmark', 'gsam', ''], default='')
    parser.add_argument('--alt_env', type=str, choices=['', 'flood', 'ground_fissure'], default='')
    parser.add_argument('--data_root', type=str, default=str(Path(__file__).resolve().parents[1] / 'data'))

    # gsam
    parser.add_argument('--gsam_rgb_shape', type=int, default=500)
    parser.add_argument('--gsam_use_segmentation_mask', action='store_true', default=False)
    parser.add_argument('--gsam_use_bbox_confidence', action='store_true', default=False)
    parser.add_argument('--gsam_use_map_cache', action='store_true', default=False)
    parser.add_argument('--gsam_box_threshold', type=float, default=0.35)
    parser.add_argument('--gsam_text_threshold', type=float, default=0.25)
    parser.add_argument('--gsam_max_box_size', type=float, default=50.)
    parser.add_argument('--gsam_max_box_area', type=float, default=3000.)
    
    # training params
    parser.add_argument('--learning_rate', type=float, default=1.0e-03)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--train_trajectory_type', type=str, choices=['sp', 'mturk', 'both'], default='sp')
    parser.add_argument('--train_episode_sample_size', type=int, default=-1)
    parser.add_argument('--potential_rank_loss_weight', type=float, default=0.)
    parser.add_argument('--potential_rank_margin', type=float, default=0.02)
    parser.add_argument('--progress_head_only_tune', action='store_true', default=False)
    parser.add_argument('--posttrain_tag', type=str, default='')
    parser.add_argument('--dagger_rollout_posttrain', action='store_true', default=False)
    parser.add_argument('--dagger_episode_sample_size', type=int, default=2000)
    parser.add_argument('--dagger_mix_ratio', type=float, default=0.5)
    
    # eval params
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--eval_at_start', action='store_true', default=False)
    parser.add_argument('--eval_max_timestep', type=int, default=20)
    parser.add_argument('--eval_client', type=str, choices=['crop', 'airsim'], default='crop')
    parser.add_argument('--success_dist', type=float, default=20.)
    parser.add_argument('--success_iou', type=float, default=0.4)
    parser.add_argument('--move_iteration', type=int, default=5)
    parser.add_argument('--progress_stop_val', type=float, default=0.75)
    parser.add_argument('--min_stop_step', type=int, default=0)
    parser.add_argument('--progress_stop_patience', type=int, default=1)
    parser.add_argument('--gcf_filter', action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('--gcf_window', dest='terminal_belief_window', type=int, default=5, metavar='GCF_WINDOW')
    parser.add_argument('--gcf_beta', dest='terminal_belief_beta', type=float, default=4.0, metavar='GCF_BETA')
    parser.add_argument('--gcf_progress_thr', dest='terminal_belief_progress_thr', type=float, default=0.80, metavar='GCF_PROGRESS_THR')
    parser.add_argument('--gcf_variance_thr', dest='terminal_belief_variance_thr', type=float, default=400.0, metavar='GCF_VARIANCE_THR')
    parser.add_argument('--gcf_min_step', dest='terminal_belief_min_step', type=int, default=8, metavar='GCF_MIN_STEP')
    parser.add_argument('--gcf_arrival_dist', dest='terminal_belief_arrival_dist', type=float, default=15.0, metavar='GCF_ARRIVAL_DIST')
    parser.add_argument('--gcf_debug', dest='terminal_belief_debug', action='store_true', default=False)
    parser.add_argument(
        '--eval_agent_mode',
        type=str,
        choices=['progress_stop', 'gcf', 'selector'],
        default='selector',
    )
    parser.add_argument('--eval_goal_selector', type=str, choices=['gdino', 'llava', 'gcf', 'raw'], default='gdino')
    parser.add_argument('--gps_noise_scale', type=float, default=0.)

    # airsim
    parser.add_argument('--sim_ip', type=str, default="172.23.96.1")
    parser.add_argument('--sim_port', type=int, default=41451)

    args = parser.parse_args()
    args.gsam_rgb_shape = args.gsam_rgb_shape, args.gsam_rgb_shape
    os.environ['SP_GUIDE_DATA_ROOT'] = args.data_root

    return ExperimentArgs(**vars(args))
