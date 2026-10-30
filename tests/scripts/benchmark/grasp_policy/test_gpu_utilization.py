"""Tests for `scripts/benchmark/grasp_policy/gpu_utilization.py`."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from graspkit.configs.env_config import MsEnvConfig
from graspkit.configs.experiment_config import EnvConfig
from graspkit.scene.scene_description import SceneDescription
from scripts.benchmark.grasp_policy import gpu_utilization


class TestGpuUtilization:
    def test_gpu_benchmark_reuses_frozen_inputs_and_records_all_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        scene_path = tmp_path / "scenes.msgpack"
        scenes = {sid: SceneDescription(scene_id=sid) for sid in ("scene-b", "scene-a")}
        SceneDescription.save_scene_descriptions_msgpack(scenes, str(scene_path))
        evaluator_cfg = EnvConfig(
            ms_env_config=MsEnvConfig(reconfiguration_freq=0), num_envs=1024, num_seeds=2, enable_perturbation=False
        )
        config = SimpleNamespace(grasp_optim=SimpleNamespace(evaluator=evaluator_cfg), robot_config="robot")
        monkeypatch.setattr("graspkit.configs.experiment.build_experiment_config", lambda experiment: config)
        for name in ("set_device", "synchronize", "reset_peak_memory_stats"):
            monkeypatch.setattr(torch.cuda, name, Mock())
        for name in ("max_memory_allocated", "max_memory_reserved"):
            monkeypatch.setattr(torch.cuda, name, Mock(return_value=0))
        monkeypatch.setattr("graspkit.utils.seed_utils.seed_everything", Mock())
        monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor)
        trajectory = torch.arange(24).float().reshape(2, 3, 4, 1)
        payload = tmp_path / "payload.pt"
        saved = {
            "experiment": "fixed-experiment",
            "seed": 0,
            "scene_ids": ["scene-b", "scene-a"],
            "scene_description_path": str(scene_path),
            "protocol": asdict(evaluator_cfg),
            "full_traj_qpos": trajectory,
            "filter_success": torch.tensor([[True, True, False], [True, True, True]]),
            "planning_success": torch.tensor([[True, False, True], [True, True, True]]),
        }
        torch.save(saved, payload)
        payload_bytes = payload.read_bytes()
        success = torch.tensor([[True, True, True], [True, False, True]])
        evaluator = Mock()
        evaluator.evaluate.return_value = {"success": success, "lift_dz": trajectory[..., 0, 0]}
        factory = Mock(return_value=evaluator)
        monkeypatch.setattr("graspkit.envs.build_env", factory)
        for num_envs, options, rendering in (
            (2, {}, True),
            (8, {"enable_rendering": False}, False),
            (16, {"enable_rendering": True}, True),
        ):
            output = tmp_path / f"batch-{num_envs}.json"
            gpu_utilization.main(
                "run", payload, output, experiment="fixed-experiment", num_envs=num_envs, repeats=2, **options
            )
            actual = json.loads(output.read_text())
            assert actual["total_samples"] == 6 and actual["trajectory_shape"] == [2, 3, 4, 1]
            assert actual["scene_ids"] == saved["scene_ids"]
            assert actual["protocol"] == json.loads(json.dumps(saved["protocol"]))
            assert actual["num_envs"] == num_envs
            assert actual["enable_rendering"] is rendering
            assert [row["success_count"] for row in actual["measurements"]] == [5, 5]
            assert [row["final_success_count"] for row in actual["measurements"]] == [3, 3]
            assert all(row["samples_per_second"] > 0 for row in actual["measurements"])
            assert factory.call_args.args[0] == replace(evaluator_cfg, num_envs=num_envs)
            assert factory.call_args.kwargs["enable_rendering"] is rendering
            recorded = torch.load(output.with_suffix(".pt"), weights_only=True)
            assert recorded["scene_ids"] == saved["scene_ids"] and len(recorded["metrics"]) == 2
            for metrics in recorded["metrics"]:
                torch.testing.assert_close(metrics["success"], success)
        for call in evaluator.evaluate.call_args_list:
            assert call.args[0].scene_ids == saved["scene_ids"]
            assert call.args[0].scene_descriptions == [scenes[sid] for sid in saved["scene_ids"]]
            torch.testing.assert_close(call.args[0].full_traj_qpos, trajectory)
        assert evaluator.close.call_count == 3 and payload.read_bytes() == payload_bytes
        factory.reset_mock()
        saved["protocol"]["num_seeds"] = 3
        torch.save(saved, payload)
        with pytest.raises(AssertionError, match="Evaluation protocol changed"):
            gpu_utilization.main("run", payload, tmp_path / "invalid.json", experiment="fixed-experiment")
        factory.assert_not_called()
