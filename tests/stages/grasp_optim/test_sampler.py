"""Tests for `graspkit.stages.grasp_optim.sampler`."""

import pytest
import torch

from graspkit.stages.grasp_optim import (
    GraspOptimConfig,
    GraspRobot,
    RobotConfig,
    SceneDescription,
    SceneGeom,
    build_sampler,
)


@pytest.mark.hf_assets
class TestFloatingBaseSampler:
    def test_sample_returns_poses_outside_object(self, mano_config: RobotConfig, cube_scene: SceneDescription):
        """Floating sampler returns [tl(3), euler_rxyz(3), q(16)] poses outside the scene geometry (SDF > 0)."""
        cfg = GraspOptimConfig(num_iters=10, scene_geom_backend="mesh")
        robot = GraspRobot.from_config(mano_config, device="cpu")
        sampler = build_sampler(cfg, mano_config, robot, torch.device("cpu"), sampler_type="floating_base")
        scene_geom = SceneGeom([cube_scene], device="cpu")
        variables = sampler.sample(scene_geom, num_samples_per_scene=8)
        assert variables.shape == (1, 8, 3 + 3 + 16)
        assert torch.isfinite(variables).all()
        sdf, _, _ = scene_geom.query_scene_signed_distances(variables[..., :3])
        assert (sdf > 0).all()
