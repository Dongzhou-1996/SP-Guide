from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sp_guide import GoalConvergenceFilter, GoalDirectedActionSelector, SPGuide
from sp_guide.geometry import Point2D, Pose4D


def main() -> None:
    torch.manual_seed(7)
    batch_size = 2
    hidden = 128
    model = SPGuide(hidden_size=hidden, num_heads=4, visual_encoder_layers=1, decoder_layers=1)
    model.eval()

    instruction_tokens = torch.randn(batch_size, 12, hidden)
    map_tokens = torch.randn(batch_size, 64, hidden)
    rgb_tokens = torch.randn(batch_size, 16, hidden)
    depth_tokens = torch.randn(batch_size, 16, hidden)
    map_coords = torch.rand(batch_size, 64, 2)
    rgb_coords = torch.rand(batch_size, 16, 2)
    depth_coords = torch.rand(batch_size, 16, 2)

    with torch.no_grad():
        goal_xy, progress = model(
            instruction_tokens=instruction_tokens,
            map_tokens=map_tokens,
            rgb_tokens=rgb_tokens,
            depth_tokens=depth_tokens,
            map_coords=map_coords,
            rgb_coords=rgb_coords,
            depth_coords=depth_coords,
        )
    print("predicted normalized goals:", goal_xy)
    print("predicted progress:", progress.flatten())

    gcf = GoalConvergenceFilter()
    selector = GoalDirectedActionSelector()
    pose = Pose4D(0.0, 0.0, 0.0, 0.0)

    for step in range(12):
        noisy_goal = Point2D(50.0 + 0.3 * (step % 3), 10.0 - 0.2 * (step % 2))
        state = gcf.update(noisy_goal, progress=0.72 + 0.03 * step)
        target = state.locked_goal or noisy_goal
        pose, action = selector.step(pose, target)
        print(f"step={step:02d} action={action.name:12s} pose={pose} locked={state.locked_goal}")
        if action.name == "STOP":
            break


if __name__ == "__main__":
    main()
