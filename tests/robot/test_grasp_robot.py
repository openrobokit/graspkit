"""Self-collision residuals, gradients and representation-specific ignores of `graspkit.robot.grasp_robot`."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
import yaml
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import RobotState
from robokit.robo.robot_spec import RobotSpec
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.utils.warp_utils import wp_device_type
from robot_descriptions.loaders.yourdfpy import load_robot_description

from graspkit.configs.experiment_config import GraspOptimConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.stages.grasp_filters import compute_self_collision_depth


@pytest.fixture
def panda_robot() -> GraspRobot:
    return GraspRobot(RobotSpec.parse(load_robot_description("panda_description"), load_meshes=True))


def _bent_state(robot: GraspRobot, device: wp_device_type) -> RobotState:
    np.random.seed(7)
    q_np = (np.random.randn(1, robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
    q = wp.from_numpy(q_np, dtype=wp.float32, device=device)
    return robot.state(q=q)


def _make_task(robot: GraspRobot, weight: float = 2.0, margin: float = 0.5) -> SelfCollisionTask:
    # max_active_pairs > total link pairs so the top-K keep never truncates (FD stability).
    return SelfCollisionTask(robot=robot, representation="capsule", weight=weight, margin=margin, max_active_pairs=64)


class TestCapsuleSelfCollision:
    def test_residual_nonzero_with_margin(self, panda_robot):
        device = wp.get_device("cpu")
        task = _make_task(panda_robot)
        state = _bent_state(panda_robot, device)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        assert residual.shape == (1, task.residual_dim)
        assert (residual >= 0.0).all()
        assert residual.max() > 1e-3, f"expected active capsule pairs at margin=0.5; residual={residual}"

    def test_analytic_gradient_matches_finite_differences(self, panda_robot):
        """grad(0.5*||r||^2) = J^T r with the analytic Jacobian matches central finite differences."""
        device = wp.get_device("cpu")
        task = _make_task(panda_robot, weight=1.0)
        state = _bent_state(panda_robot, device)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        analytic = np.einsum("bri,br->bi", jacobian, residual)
        assert np.max(np.abs(analytic)) > 1e-5, "expected non-zero gradient"

        q_np = state.q.numpy().copy()
        eps = 1e-4
        np.random.seed(0)
        base_dofs = 6 if state.has_floating_base else 0
        sample_dofs = np.random.choice(q_np.shape[-1], size=4, replace=False)
        max_err = 0.0
        for d in sample_dofs:
            costs = []
            for sign in (1.0, -1.0):
                qs = q_np.copy()
                qs[0, d] += sign * eps
                s = panda_robot.state(q=wp.from_numpy(qs, dtype=wp.float32, device=device))
                r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: s})).numpy()
                costs.append(0.5 * (r**2).sum(axis=-1)[0])
            fd = (costs[0] - costs[1]) / (2.0 * eps)
            err = abs(fd - float(analytic[0, base_dofs + d]))
            max_err = max(max_err, err)
        assert max_err < 5e-3, f"finite-diff mismatch: max abs err {max_err:.3e}"

    def test_sqrt_abs_gradient_matches_finite_differences(self, panda_robot):
        """GD wiring (sqrt_abs residual, J^T r via analytic Jacobian, cache reused) matches finite differences."""
        device = wp.get_device("cpu")
        task = SelfCollisionTask(
            robot=panda_robot,
            representation="capsule",
            weight=np.sqrt(2.0),
            margin=0.5,
            residual_mode="sqrt_abs",
            residual_eps=1e-6,
            max_active_pairs=64,
        )
        state = _bent_state(panda_robot, device)
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        analytic = np.einsum("bri,br->bi", jacobian, residual)
        assert np.max(np.abs(analytic)) > 1e-5, "expected non-zero gradient"

        q_np = state.q.numpy().copy()
        eps = 1e-4
        np.random.seed(1)
        base_dofs = 6 if state.has_floating_base else 0
        sample_dofs = np.random.choice(q_np.shape[-1], size=4, replace=False)
        max_err = 0.0
        for d in sample_dofs:
            costs = []
            for sign in (1.0, -1.0):
                qs = q_np.copy()
                qs[0, d] += sign * eps
                s = panda_robot.state(q=wp.from_numpy(qs, dtype=wp.float32, device=device))
                r = task.compute_weighted_residual(VarValues({VarKey.ROBOT: s})).numpy()
                costs.append(0.5 * (r**2).sum(axis=-1)[0])
            fd = (costs[0] - costs[1]) / (2.0 * eps)
            err = abs(fd - float(analytic[0, base_dofs + d]))
            max_err = max(max_err, err)
        assert max_err < 1e-2, f"finite-diff mismatch: max abs err {max_err:.3e}"

    @pytest.mark.hf_assets
    @pytest.mark.parametrize("representation", ["sphere", "capsule"])
    def test_from_config_loads_spec_ignore_pairs(self, representation: str):
        """Both representations load the matching robokit asset against the pruned link names."""
        from graspkit.configs.robot.inspire import inspire_right

        robot = GraspRobot.from_config(inspire_right, self_collision_representation=representation)
        pairs = robot.spec.self_collision_ignored_pairs
        num_links = len(robot.spec.link_names)
        assert len(pairs) > 0, "inspire config sets self_collision_ignore_path"
        assert all(0 <= i < num_links and 0 <= j < num_links for i, j in pairs)


class TestSelfCollisionIgnores:
    def test_representation_selection_and_sphere_filter(self, collision_config: RobotConfig):
        robots = {
            rep: GraspRobot.from_config(
                collision_config, load_meshes=True, load_collision_spheres=True, self_collision_representation=rep
            )
            for rep in ("sphere", "capsule")
        }
        assert GraspOptimConfig().spen_representation == "sphere"
        assert robots["sphere"] is not robots["capsule"]
        for rep, ignored_link in (("sphere", "b"), ("capsule", "c")):
            robot = robots[rep]
            assert (
                GraspRobot.from_config(
                    collision_config, load_meshes=True, load_collision_spheres=True, self_collision_representation=rep
                )
                is robot
            )
            expected = [[robot.link_names.index("a"), robot.link_names.index(ignored_link)]]
            np.testing.assert_array_equal(robot.spec.self_collision_ignored_pairs, expected)
            task = SelfCollisionTask(robot=robot, representation=rep)
            link_ids = robot.spec.collision_spheres_link_indices if rep == "sphere" else np.arange(robot.spec.num_links)
            active = {frozenset(robot.link_names[link_ids[i]] for i in pair) for pair in task._pair_indices_np}
            assert frozenset(("a", ignored_link)) not in active
            assert frozenset(("b", "c")) in active
            # A capsule-loaded robot must still use sphere ignores for the sphere overlap metric.
            torch.testing.assert_close(compute_self_collision_depth(torch.zeros(1, 1, 3), robot), torch.zeros(1, 1))
            np.testing.assert_array_equal(robot.spec.self_collision_ignored_pairs, expected)
        unfiltered = GraspRobot.from_config(replace(collision_config, self_collision_ignore_path=None))
        assert compute_self_collision_depth(torch.zeros(1, 1, 3), unfiltered).item() == pytest.approx(0.25)
        pruned = GraspRobot.from_config(collision_config, ee_link_names=("a", "c"))
        assert pruned.spec.self_collision_ignored_pairs.shape == (0, 2)

    @pytest.mark.parametrize("representation", ["sphere", "capsule"])
    def test_invalid_supplied_ignore_raises(self, collision_config: RobotConfig, representation: str):
        path = Path(collision_config.self_collision_ignore_path)
        path.unlink()
        with pytest.raises(FileNotFoundError):
            GraspRobot.from_config(collision_config, self_collision_representation=representation)
        path.write_text("self_collision_ignore: [\n")
        with pytest.raises(yaml.YAMLError):
            GraspRobot.from_config(collision_config)
