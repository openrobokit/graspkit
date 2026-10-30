"""FingertipDistanceTask: per fingertip link, `weight * min` over its spheres of `|sdf - r|` on a box SDF, CPU-only."""

from typing import Callable

import numpy as np
import pytest
import warp as wp
from robokit.geom.scene import WarpScene
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo import RobotState
from robokit.utils.warp_utils import wp_device_type

from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.stages.grasp_optim.fingertip_task import FingertipDistanceTask

_HALF_EXTENT = 0.05


def _setup(
    cfg: RobotConfig, scene: WarpScene, weight: float
) -> tuple[GraspRobot, list[FingertipDistanceTask], np.ndarray, wp_device_type]:
    device = wp.get_device("cpu")
    robot = GraspRobot.from_config(cfg, load_collision_spheres=True, device="cpu")
    scene_indices = wp.zeros(1, dtype=wp.int32, device=device)
    tasks = [FingertipDistanceTask(robot, scene, scene_indices, link, weight) for link in cfg.fingertip_link_names]
    np.random.seed(42)
    q = (np.random.randn(robot.num_actuated_joints) * 0.1).astype(np.float32)
    return robot, tasks, q, device


def _state(robot: GraspRobot, q: np.ndarray, device: wp_device_type) -> RobotState:
    state = robot.forward_kinematics(robot.state(q=wp.from_numpy(q, dtype=wp.float32, device=device)))
    return robot.compute_motion_subspace(state)


def _cost(task: FingertipDistanceTask, robot: GraspRobot, q: np.ndarray, device: wp_device_type) -> float:
    residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: _state(robot, q, device)})).numpy()
    return 0.5 * float((residual**2).sum())


@pytest.mark.hf_assets
class TestFingertipDistanceTask:
    def test_residual_is_the_closest_sphere_gap(
        self, mano_config: RobotConfig, box_scene: Callable[[float], WarpScene]
    ):
        """residual == weight * min_i |sdf(center_i) - radius_i| over the link's spheres, closed-form box SDF."""
        robot, tasks, q, device = _setup(mano_config, box_scene(_HALF_EXTENT), weight=1.5)
        state = _state(robot, q, device)
        centers = state.collision_sphere_centers_world.numpy()[0]
        radii = np.asarray(robot.spec.collision_sphere_radii)
        links = np.asarray(robot.spec.collision_spheres_link_indices)
        for task in tasks:
            residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
            assert residual.shape == (1, 1)
            spheres = np.flatnonzero(links == robot.spec.link_names.index(task.link_name))
            outward = np.abs(centers[spheres]) - _HALF_EXTENT
            assert not (outward < 0).all(axis=-1).any(), "test assumes the fingertip spheres lie outside the box"
            sdf = np.linalg.norm(np.maximum(outward, 0.0), axis=-1)
            np.testing.assert_allclose(residual[0, 0], 1.5 * np.abs(sdf - radii[spheres]).min(), rtol=1e-5, atol=1e-6)

    def test_jacobian_matches_finite_differences(
        self, mano_config: RobotConfig, box_scene: Callable[[float], WarpScene]
    ):
        """grad(0.5 r²) = Jᵀr against central finite differences on a few joints, for every fingertip."""
        robot, tasks, q, device = _setup(mano_config, box_scene(_HALF_EXTENT), weight=1.0)
        np.random.seed(0)
        dofs = np.random.choice(q.shape[0], size=4, replace=False)
        eps = 1e-4
        for task in tasks:
            state = _state(robot, q, device)
            residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
            jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
            analytic = np.einsum("bri,br->bi", jacobian, residual)[0]
            assert np.abs(analytic).max() > 1e-6, f"{task.link_name}: expected a non-zero gradient"
            for d in dofs:
                qp, qm = q.copy(), q.copy()
                qp[d] += eps
                qm[d] -= eps
                fd = (_cost(task, robot, qp, device) - _cost(task, robot, qm, device)) / (2 * eps)
                assert abs(fd - analytic[d]) < 5e-3, (
                    f"{task.link_name} dof {d}: fd {fd:.3e} vs analytic {analytic[d]:.3e}"
                )
