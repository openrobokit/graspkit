"""Tests for `graspkit.stages.grasp_optim.optimize_pipeline`."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from graspkit.configs.experiment_config import ExperimentConfig, GraspOptimConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.scene.scene_description import SceneDescription
from graspkit.stages.grasp_optim import optimize_pipeline


class TestOptimizeGrasps:
    @pytest.mark.parametrize("save_labels", [False, True])
    def test_optimization_pipeline_saves_survivors_and_optional_labels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_labels: bool
    ):
        assert GraspOptimConfig().save_labels is False
        scene_path = tmp_path / "scenes.msgpack"
        SceneDescription.save_scene_descriptions_msgpack({"scene": SceneDescription(scene_id="scene")}, str(scene_path))
        output = tmp_path / "grasps"
        cfg = ExperimentConfig(
            name="pipeline-test",
            robot_config=RobotConfig(name="robot", urdf_or_mjcf_path="unused.urdf", init_qpos=(0.0, 0.0)),
            grasp_optim=GraspOptimConfig(num_samples_per_scene=3, num_scenes_per_batch=1, save_labels=save_labels),
            grasp_output_dir=str(output),
            scene_description_path=str(scene_path),
            rank=0,
            world_size=1,
        )
        qpos = torch.arange(6).float().reshape(1, 3, 2)
        optimizer = Mock()
        optimizer.optimize.return_value = qpos
        builder = Mock()
        builder.build.side_effect = lambda batch: (
            replace(
                batch,
                pre_grasp_qpos=qpos + 1,
                post_grasp_qpos=qpos + 2,
                full_traj_qpos=qpos.unsqueeze(-2),
                pre_traj_penetration=torch.tensor([[0.001, 0.01, 0.0]]),
            ),
            torch.tensor([[True, True, False]]),
        )
        evaluator = Mock()
        evaluator.evaluate.return_value = {"success": torch.ones((1, 3), dtype=torch.bool)}
        monkeypatch.setattr(optimize_pipeline, "WarpGraspOptimizer", Mock(return_value=optimizer))
        monkeypatch.setattr(optimize_pipeline, "build_planning_traj_builder", Mock(return_value=builder))
        monkeypatch.setattr(optimize_pipeline, "build_env", Mock(return_value=evaluator))
        monkeypatch.setattr(optimize_pipeline, "cuda_timeit", lambda name: nullcontext())
        optimize_pipeline.optimize_grasps(cfg)
        optimize_pipeline.build_env.assert_called_once_with(
            cfg.grasp_optim.evaluator, cfg.robot_config, optimize_pipeline.device, enable_rendering=False
        )
        saved = GraspKitData.load(str(output / "scene.h5"))
        torch.testing.assert_close(saved.grasp_qpos, qpos[0, :1])
        torch.testing.assert_close(saved.pre_grasp_qpos, qpos[0, :1] + 1)
        torch.testing.assert_close(saved.post_grasp_qpos, qpos[0, :1] + 2)
        assert (output / ".optim_complete").is_file()
        evaluator.close.assert_called_once()
        labels_dir = tmp_path / "grasps.labels"
        assert labels_dir.exists() is save_labels
        if save_labels:
            labels = torch.load(labels_dir / "rank0_attempt0_scene.pt", weights_only=True)
            assert labels["scene_ids"] == ["scene"]
            torch.testing.assert_close(labels["grasp_qpos"], qpos)
            torch.testing.assert_close(labels["final_success"], torch.tensor([[True, False, False]]))
