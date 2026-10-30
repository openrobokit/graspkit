"""Robot assets (FK, collision spheres, file contact points) match the checked-in fingerprints.

A robokit pin bump or a config change that moves any link, sphere or contact point fails here. When the change is
intended, regenerate with `uv run python tests/configs/robot/test_robot_assets_golden.py` and review the diff.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from graspkit.configs.robot import _ROBOT_MODULES, build_robot_config
from graspkit.robot.grasp_robot import GraspRobot

GOLDEN = Path(__file__).parents[2] / "fixtures" / "robot_assets_golden.json"
NAMES = [n for n in _ROBOT_MODULES if n != "shadow_right"] + [  # shadow_right: its URDF does not parse
    "+yam_yam_gripper",
    "+franka_panda_allegro_right",
    "+xarm7_sharpa_right",
    "+ur5e_robotiq_2f85",
]


def fingerprint(name: str) -> dict[str, float]:
    """Index-weighted sums, so a moved, renamed or reordered entry changes them (spheres are sorted: unordered)."""
    from graspkit.utils.assembly_utils import assembled_robot_config  # imports robot assets

    cfg = assembled_robot_config(name[1:]) if name.startswith("+") else build_robot_config(name)
    robot = GraspRobot.from_config(cfg, prune=False, load_collision_spheres=cfg.collision_spheres_path is not None)
    limits = robot.actuated_joint_limits_torch
    q = torch.stack([torch.zeros(len(limits)).clamp(limits[:, 0], limits[:, 1]), limits.mean(-1)])
    if cfg.base_type == "floating":
        q = torch.cat([torch.tensor([[0.0] * 6, [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]]), q], -1)
    poses = robot.forward_kinematics_torch(q)
    out = {"fk": float((poses[..., :3, :].numpy() * np.arange(1, len(robot.link_names) + 1)[:, None, None]).sum())}
    if cfg.collision_spheres_path is not None:
        spheres = robot.get_collision_spheres(poses).numpy()
        spheres = np.stack([s[np.lexsort(s.T[::-1])] for s in spheres])
        out["spheres"] = float((spheres * np.arange(1, spheres.shape[1] + 1)[:, None]).sum())
    if cfg.contact_points_path is not None:  # sampled candidates are random; only authored points are fixed
        points, links = GraspRobot.get_contact_candidates(cfg)
        out["contacts"] = float((points.numpy() * (links.numpy() + 1)[:, None]).sum())
    return out


@pytest.mark.hf_assets
class TestRobotAssetsGolden:
    @pytest.mark.parametrize("name", NAMES)
    def test_robot_assets_match_golden(self, name: str):
        expected = json.loads(GOLDEN.read_text())[name]
        actual = fingerprint(name)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert actual[key] == pytest.approx(value, rel=1e-4, abs=1e-3), key


if __name__ == "__main__":
    GOLDEN.write_text(json.dumps({name: fingerprint(name) for name in NAMES}, indent=1) + "\n")
