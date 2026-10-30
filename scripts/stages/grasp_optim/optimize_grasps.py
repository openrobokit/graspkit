"""CLI for the grasp optimization pipeline (graspkit.stages.grasp_optim.optimize_pipeline).

Example usage:
  uv run scripts/stages/grasp_optim/optimize_grasps.py --experiment_name inspire_right_dex1b_v0_4_dev
  uv run scripts/stages/grasp_optim/optimize_grasps.py --experiment_name g1_inspire_right_wbc_dex1b_v0_5_mini

  # --overwrite wipes a prior run; --verbose also saves every stage for visualize_grasps.py.
  uv run scripts/stages/grasp_optim/optimize_grasps.py --experiment_name xarm7_xhand_right_mj_tabletop --overwrite --verbose
"""

from dataclasses import dataclass
from typing import Literal, Optional

import tyro

from graspkit.configs.experiment import build_experiment_config
from graspkit.stages.grasp_optim.optimize_pipeline import optimize_grasps


@dataclass
class Args:
    experiment_name: str
    robot_base: Optional[Literal["floating", "fixed"]] = None
    """Override the optimization robot with traj_builder's floating/fixed robot. None: cfg.robot_config."""
    overwrite: bool = False
    """Wipe any prior run in the output dir and re-optimize from scratch."""
    verbose: bool = False
    """Also save every stage (sample ... sim) per scene to `<output_dir>.viz/` for visualize_grasps.py."""
    rank: int = 0
    """This worker's rank. With world_size > 1 it selects a disjoint scene slice."""
    world_size: Optional[int] = None
    """Total number of ranks. When > 1, this rank processes scene_ids[rank::world_size]."""


if __name__ == "__main__":
    args = tyro.cli(Args, use_underscores=True)
    cfg = build_experiment_config(args.experiment_name)
    cfg.rank = args.rank
    cfg.world_size = args.world_size
    optimize_grasps(cfg, args.robot_base, overwrite=args.overwrite, verbose=args.verbose)
