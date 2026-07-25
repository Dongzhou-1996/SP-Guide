import torch

from sp_guide.actions import DiscreteAction, GoalDirectedActionSelector
from sp_guide.gcf import GCFConfig, GoalConvergenceFilter
from sp_guide.geometry import Point2D, Pose4D
from sp_guide.model import SPGuide
from sp_guide.spce import UnifiedSpatialConstraintAttention


def test_spce_attention_shape_and_normalization():
    attn = UnifiedSpatialConstraintAttention(hidden_size=32, num_heads=4, dropout=0.0)
    query = torch.randn(2, 5, 32)
    key = torch.randn(2, 7, 32)
    distances = torch.rand(2, 5, 7)
    out = attn(query, key, distances)
    assert out.shape == query.shape


def test_spguide_forward_shape():
    model = SPGuide(hidden_size=32, num_heads=4, visual_encoder_layers=1, decoder_layers=1)
    goal, progress = model(
        instruction_tokens=torch.randn(2, 6, 32),
        map_tokens=torch.randn(2, 9, 32),
        rgb_tokens=torch.randn(2, 4, 32),
        depth_tokens=torch.randn(2, 4, 32),
        map_coords=torch.rand(2, 9, 2),
        rgb_coords=torch.rand(2, 4, 2),
        depth_coords=torch.rand(2, 4, 2),
    )
    assert goal.shape == (2, 2)
    assert progress.shape == (2, 1)
    assert torch.all((goal >= 0.0) & (goal <= 1.0))
    assert torch.all((progress >= 0.0) & (progress <= 1.0))


def test_gcf_locks_on_converged_predictions():
    gcf = GoalConvergenceFilter(GCFConfig(window=5, min_step=4, progress_threshold=0.7, variance_threshold=2.0))
    state = None
    for idx in range(8):
        state = gcf.update(Point2D(10.0 + 0.1 * (idx % 2), 20.0), 0.8)
    assert state is not None
    assert state.locked_goal is not None
    assert state.variance <= 2.0


def test_locked_goal_still_uses_discrete_action():
    selector = GoalDirectedActionSelector()
    pose = Pose4D(0.0, 0.0, 0.0, 0.0)
    action = selector.select(pose, Point2D(20.0, 20.0))
    assert isinstance(action, DiscreteAction)
    new_pose, action = selector.step(pose, Point2D(20.0, 0.0))
    assert action is DiscreteAction.MOVE_FORWARD
    assert new_pose.x > pose.x

