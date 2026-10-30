"""Tests for `graspkit.stages.grasp_optim.warp_optimizer`: the end-to-end solver and the robokit terms of `grasp_terms`."""

import dataclasses
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
import trimesh
import warp as wp
from robokit.geom.scene import WarpScene
from robokit.opt.var_values import VarKey, VarValues
from robokit.robo.robot import Robot
from robokit.terms.autodiff import autodiff_weighted_jacobian
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.utils.warp_utils import wp_device_type, wp_vec7
from robot_descriptions.loaders.yourdfpy import load_robot_description

from graspkit.configs.robot import build_robot_config
from graspkit.stages.grasp_optim import (
    GraspOptimConfig,
    GraspRobot,
    MeshDescription,
    RobotConfig,
    SceneDescription,
    WarpGraspOptimizer,
)
from graspkit.stages.grasp_optim.warp_optimizer import grasp_terms
from graspkit.utils.seed_utils import seed_everything


def _setup_contact_points(robot: Robot, device: wp_device_type) -> tuple[wp.array, wp.array, int]:
    num_contacts = 3
    link_names = ["panda_link4", "panda_link6", "panda_hand"]
    link_indices = np.array([robot.link_names.index(n) for n in link_names], dtype=np.int32)
    points_np = np.array(
        [[0.01, 0.02, 0.03], [0.02, -0.01, 0.01], [-0.01, 0.01, 0.02]],
        dtype=np.float32,
    )
    robot.spec.local_contact_point_centers = points_np
    robot.spec.contact_points_link_indices = link_indices
    local_contact_points = wp.from_numpy(np.array([points_np], dtype=np.float32), dtype=wp.vec3, device=device)
    contact_link_indices = wp.from_numpy(np.array([link_indices], dtype=np.int32), dtype=wp.int32, device=device)
    return local_contact_points, contact_link_indices, num_contacts


def _setup_task(robot: Robot, task_cls: type, scene: WarpScene, weight: float = 1.5) -> tuple:
    device = wp.get_device("cpu")
    lcp, cli, n = _setup_contact_points(robot, device)
    task = task_cls(
        robot=robot,
        num_contact_points=n,
        warp_meshes=scene,
        weight=weight,
        local_contact_points=lcp,
        contact_points_link_indices=cli,
    )
    np.random.seed(42)
    q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
    state = robot.state(q=wp.from_numpy(q_np, dtype=wp.float32, device=device))
    state = robot.forward_kinematics(state)
    state = robot.compute_motion_subspace(state)
    cost_buf = wp.zeros((1,), dtype=wp.float32, device=device)
    grad_buf = wp.zeros((1, state.tangent_dim), dtype=wp.float32, device=device)
    return state, task, cost_buf, grad_buf, device


@pytest.fixture
def panda_robot_mutable() -> Robot:
    return Robot.load(load_robot_description("panda_description"))


@pytest.mark.parametrize("task_cls,half_extent", [(SceneDistanceTask, 0.05), (ForceClosureTask, 2.0)])
class TestWarpTaskCostAndGradient:
    """Validate compute_weighted_cost_and_gradient (Gauss-Newton form: cost = 0.5*||r||^2, grad = J^T r)."""

    def test_cost_matches_residual_form(self, panda_robot_mutable, box_scene, task_cls, half_extent):
        state, task, cost_buf, grad_buf, _ = _setup_task(
            panda_robot_mutable, task_cls, box_scene(half_extent), weight=1.5
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        expected = 0.5 * (residual.numpy() ** 2).sum(axis=-1)
        np.testing.assert_allclose(cost_buf.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_gradient_matches_jacobian_times_residual(self, panda_robot_mutable, box_scene, task_cls, half_extent):
        state, task, cost_buf, grad_buf, _ = _setup_task(
            panda_robot_mutable, task_cls, box_scene(half_extent), weight=1.5
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        jacobian = task.compute_weighted_jacobian_analytic(VarValues({VarKey.ROBOT: state})).numpy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        expected = np.einsum("bri,br->bi", jacobian, residual)
        np.testing.assert_allclose(grad_buf.numpy(), expected, rtol=1e-4, atol=1e-5)

    def test_gradient_matches_finite_differences(self, panda_robot_mutable, box_scene, task_cls, half_extent):
        state, task, cost_buf, grad_buf, device = _setup_task(
            panda_robot_mutable, task_cls, box_scene(half_extent), weight=1.0
        )
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        analytic = grad_buf.numpy()

        q_np = state.q.numpy().copy()
        eps = 1e-4
        np.random.seed(0)
        D = q_np.shape[-1]
        sample_dofs = np.random.choice(D, size=4, replace=False)
        max_err = 0.0
        for d in sample_dofs:
            qp = q_np.copy()
            qp[0, d] += eps
            sp = panda_robot_mutable.state(q=wp.from_numpy(qp, dtype=wp.float32, device=device))
            r_p = task.compute_weighted_residual(VarValues({VarKey.ROBOT: sp})).numpy()
            cp = 0.5 * (r_p**2).sum(axis=-1)[0]

            qm = q_np.copy()
            qm[0, d] -= eps
            sm = panda_robot_mutable.state(q=wp.from_numpy(qm, dtype=wp.float32, device=device))
            r_m = task.compute_weighted_residual(VarValues({VarKey.ROBOT: sm})).numpy()
            cm = 0.5 * (r_m**2).sum(axis=-1)[0]

            fd = (cp - cm) / (2.0 * eps)
            err = abs(fd - float(analytic[0, d]))
            max_err = max(max_err, err)
        assert max_err < 5e-3, f"finite-diff mismatch: max abs err {max_err:.3e}"

    def test_accumulation_into_shared_buffers(self, panda_robot_mutable, box_scene, task_cls, half_extent):
        state, task, cost_buf, grad_buf, _ = _setup_task(
            panda_robot_mutable, task_cls, box_scene(half_extent), weight=1.0
        )
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        single_cost = cost_buf.numpy().copy()
        single_grad = grad_buf.numpy().copy()
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        np.testing.assert_allclose(cost_buf.numpy(), 2.0 * single_cost, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(grad_buf.numpy(), 2.0 * single_grad, rtol=1e-5, atol=1e-6)


class TestWarpForceClosureTask:
    """Test ForceClosureTask residual and Jacobian."""

    def test_residual_shape_and_dim(self, panda_robot_mutable, box_scene):
        robot = panda_robot_mutable
        device = wp.get_device("cpu")
        local_contact_points, contact_link_indices, num_contacts = _setup_contact_points(robot, device)
        wm = box_scene(2.0)

        task = ForceClosureTask(robot=robot, num_contact_points=num_contacts, warp_meshes=wm, weight=1.0)
        assert task.residual_dim == 6

        q = wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32, device=device)
        state = robot.state(q=q)

        residual = task.compute_weighted_residual(
            VarValues({VarKey.ROBOT: state}),
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_link_indices,
        )
        assert residual.shape == (1, 6)
        assert np.all(np.isfinite(residual.numpy()))

    def test_residual_finite(self, panda_robot_mutable, box_scene):
        """Verify residual produces finite values with SDF normals."""
        robot = panda_robot_mutable
        device = wp.get_device("cpu")
        local_contact_points, contact_link_indices, num_contacts = _setup_contact_points(robot, device)
        wm = box_scene(2.0)

        task = ForceClosureTask(robot=robot, num_contact_points=num_contacts, warp_meshes=wm, weight=2.0)

        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        state = robot.state(q=q)

        residual = task.compute_weighted_residual(
            VarValues({VarKey.ROBOT: state}),
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_link_indices,
        )
        assert residual.shape == (1, 6)
        assert np.all(np.isfinite(residual.numpy()))

    def test_jacobian_analytic_vs_autodiff(self, panda_robot_mutable, box_scene):
        """Standard analytic vs autodiff Jacobian consistency check."""
        robot = panda_robot_mutable
        device = wp.get_device("cpu")
        local_contact_points, contact_link_indices, num_contacts = _setup_contact_points(robot, device)
        wm = box_scene(2.0)

        task = ForceClosureTask(robot=robot, num_contact_points=num_contacts, warp_meshes=wm, weight=1.0)

        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        state = robot.state(q=q)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        analytic_jac = task.compute_weighted_jacobian_analytic(
            VarValues({VarKey.ROBOT: state}),
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_link_indices,
        ).numpy()

        autodiff_jac = autodiff_weighted_jacobian(
            task,
            VarValues({VarKey.ROBOT: state}),
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_link_indices,
        ).numpy()

        assert analytic_jac.shape == autodiff_jac.shape
        assert analytic_jac.shape == (1, 6, robot.num_actuated_joints)
        # Torque rows (3:6) should match autodiff
        np.testing.assert_allclose(analytic_jac[:, 3:, :], autodiff_jac[:, 3:, :], atol=1e-3, rtol=1e-3)

    def test_jacobian_analytic_vs_autodiff_floating_base(self, panda_robot_mutable, box_scene):
        """Jacobian test with floating base."""
        robot = panda_robot_mutable
        device = wp.get_device("cpu")
        local_contact_points, contact_link_indices, num_contacts = _setup_contact_points(robot, device)
        wm = box_scene(2.0)

        task = ForceClosureTask(robot=robot, num_contact_points=num_contacts, warp_meshes=wm, weight=1.5)

        np.random.seed(42)
        q_np = (np.random.randn(robot.num_actuated_joints) * 0.1 + robot.spec.midrange_q).astype(np.float32)
        q = wp.from_numpy(q_np, dtype=wp.float32, device=device)
        T_base_np = np.array([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        T_base = wp.from_numpy(T_base_np, dtype=wp_vec7, device=device)
        state = robot.state(q=q, T_world_base=T_base)
        state = robot.forward_kinematics(state)
        state = robot.compute_motion_subspace(state)

        analytic_jac = task.compute_weighted_jacobian_analytic(
            VarValues({VarKey.ROBOT: state}),
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_link_indices,
        ).numpy()

        autodiff_jac = autodiff_weighted_jacobian(
            task,
            VarValues({VarKey.ROBOT: state}),
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_link_indices,
        ).numpy()

        total_dofs = 6 + robot.num_actuated_joints
        assert analytic_jac.shape == (1, 6, total_dofs)
        assert autodiff_jac.shape == (1, 6, total_dofs)
        np.testing.assert_allclose(analytic_jac[:, 3:, :], autodiff_jac[:, 3:, :], atol=1e-3, rtol=1e-3)

    def test_stored_contact_points_constructor(self, panda_robot_mutable, box_scene):
        """Test using local_contact_points set in constructor."""
        robot = panda_robot_mutable
        device = wp.get_device("cpu")
        local_contact_points, contact_link_indices, num_contacts = _setup_contact_points(robot, device)
        wm = box_scene(2.0)

        task = ForceClosureTask(
            robot=robot,
            num_contact_points=num_contacts,
            warp_meshes=wm,
            weight=1.0,
            local_contact_points=local_contact_points,
            contact_points_link_indices=contact_link_indices,
        )

        q = wp.from_numpy(robot.spec.midrange_q, dtype=wp.float32, device=device)
        state = robot.state(q=q)

        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
        assert residual.shape == (1, 6)
        assert np.all(np.isfinite(residual.numpy()))

    def test_force_closure_gradient_matches_autodiff(self, panda_robot_mutable, box_scene):
        """5th ForceClosureTask check: grad seeded from J^T r via autodiff torque rows (force rows zero, normals fixed)."""
        state, task, cost_buf, grad_buf, _ = _setup_task(
            panda_robot_mutable, ForceClosureTask, box_scene(2.0), weight=1.5
        )
        residual = task.compute_weighted_residual(VarValues({VarKey.ROBOT: state})).numpy()
        autodiff_jac = autodiff_weighted_jacobian(task, VarValues({VarKey.ROBOT: state})).numpy()
        expected = np.einsum("bri,br->bi", autodiff_jac[:, 3:, :], residual[:, 3:])
        task.compute_weighted_cost_and_gradient(
            VarValues({VarKey.ROBOT: state}), out_cost=cost_buf, out_gradient=grad_buf
        )
        np.testing.assert_allclose(grad_buf.numpy(), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.hf_assets
@pytest.mark.cuda
class TestWarpGraspOptimizer:
    @pytest.mark.parametrize("backend", ["warp_gd", "warp_lm"])
    @pytest.mark.parametrize("geometry", ["mesh", "sdf_volume"])
    def test_reuse_matches_fresh_solver(
        self, tmp_path: Path, mano_config: RobotConfig, cube_scene: SceneDescription, backend: str, geometry: str
    ):
        """Changing seeds and geometry must preserve buffers/graphs and reproduce a fresh solve."""
        cfg = GraspOptimConfig(num_iters=100, scene_geom_backend=geometry, optimizer_backend=backend)
        scenes = [cube_scene]
        other_path = str(tmp_path / "other.obj")
        trimesh.creation.box((0.08, 0.05, 0.1)).export(other_path)
        original = scenes[0].meshes["object"]
        changed = dataclasses.replace(original, collision_mesh_path=other_path, position=(0.3, 0.0, 0.0))
        optimizer = WarpGraspOptimizer(cfg, mano_config, torch.device("cuda:0"))
        buffers, graphs, cached_solver = None, None, None
        for seed, mesh in ((0, original), (7, changed), (0, original)):
            scenes[0].meshes["object"] = mesh  # mutate the same list/description, not just its identity
            seed_everything(seed)
            actual = optimizer.optimize(scenes, 4, record=backend == "warp_gd")
            _, var, _, solver = next(iter(optimizer._problems.values()))
            inner = solver._optimizers[0] if backend == "warp_lm" else solver
            current = (var.get(VarKey.ROBOT).q.ptr, var.get(VarKey.ROBOT).T_world_base.ptr, inner.costs.ptr)
            if cached_solver is None:
                cached_solver, buffers, graphs = solver, current, dict(inner._graphs)
            assert solver is cached_solver and current == buffers and inner._graphs == graphs
            seed_everything(seed)
            fresh = WarpGraspOptimizer(cfg, mano_config, torch.device("cuda:0"))
            expected = fresh.optimize(scenes, 4)
            torch.testing.assert_close(actual, expected)
            del fresh
        seed_everything(0)
        optimizer.optimize(scenes, 2)
        seed_everything(0)
        actual = optimizer.optimize(scenes, 4)
        assert next(iter(optimizer._problems.values()))[3] is cached_solver
        torch.testing.assert_close(actual, expected)
        assert optimizer.last_snapshots is None

    @pytest.mark.parametrize("representation", ["sphere", "capsule"])
    def test_warp_gd_floating_runs(self, mano_config: RobotConfig, cube_scene: SceneDescription, representation: str):
        cfg = GraspOptimConfig(num_iters=200, scene_geom_backend="mesh", spen_representation=representation)
        optimizer = WarpGraspOptimizer(cfg, mano_config, torch.device("cuda:0"))
        grasp_qpos = optimizer.optimize([cube_scene], num_samples_per_scene=8)
        assert grasp_qpos.shape == (1, 8, 3 + 3 + 16)  # tl + euler_rxyz + q
        assert torch.isfinite(grasp_qpos).all()

    @pytest.mark.parametrize("representation", ["sphere", "capsule"])
    def test_warp_lm_floating_runs(self, mano_config: RobotConfig, cube_scene: SceneDescription, representation: str):
        cfg = GraspOptimConfig(
            scene_geom_backend="mesh", optimizer_backend="warp_lm", spen_representation=representation
        )
        optimizer = WarpGraspOptimizer(cfg, mano_config, torch.device("cuda:0"))
        grasp_qpos = optimizer.optimize([cube_scene], num_samples_per_scene=8)
        assert grasp_qpos.shape == (1, 8, 3 + 3 + 16)
        assert torch.isfinite(grasp_qpos).all()

    def test_warp_gd_fingertip_term_runs(self, mano_config: RobotConfig, cube_scene: SceneDescription):
        """warp_gd with the fingertip-distance term enabled runs end-to-end (autodiff path)."""
        cfg = GraspOptimConfig(num_iters=200, w_fingertip_dis=10.0, scene_geom_backend="mesh")
        optimizer = WarpGraspOptimizer(cfg, mano_config, torch.device("cuda:0"))
        grasp_qpos = optimizer.optimize([cube_scene], num_samples_per_scene=8)
        assert grasp_qpos.shape == (1, 8, 3 + 3 + 16)
        assert torch.isfinite(grasp_qpos).all()

    def test_warp_gd_fingertip_requires_fingertip_links(self, mano_config: RobotConfig):
        """w_fingertip_dis > 0 on a robot without fingertip links raises at construction."""
        cfg = GraspOptimConfig(num_iters=10, w_fingertip_dis=10.0, scene_geom_backend="mesh")
        no_ft = dataclasses.replace(mano_config, fingertip_link_names=None)
        with pytest.raises(ValueError, match="fingertip links"):
            WarpGraspOptimizer(cfg, no_ft, torch.device("cuda:0"))

    def test_two_scenes_keep_their_own_geometry(self, mano_config: RobotConfig, cube_scene: SceneDescription):
        """Scene-major batches: every grasp of scene k is scored against scene k, for GD and LM alike."""
        far = dataclasses.replace(
            cube_scene,
            scene_id="s1",
            meshes={
                "object": MeshDescription(
                    "object", cube_scene.meshes["object"].collision_mesh_path, (1.0, 1.0, 1.0), position=(0.4, 0.0, 0.0)
                )
            },
        )
        for backend in ("warp_gd", "warp_lm"):
            cfg = GraspOptimConfig(num_iters=100, scene_geom_backend="mesh", optimizer_backend=backend)
            grasp_qpos = WarpGraspOptimizer(cfg, mano_config, torch.device("cuda:0")).optimize([cube_scene, far], 4)
            assert grasp_qpos.shape == (2, 4, 3 + 3 + 16)
            assert torch.isfinite(grasp_qpos).all()
            assert (
                grasp_qpos[1, :, 0] - grasp_qpos[0, :, 0]
            ).mean() > 0.2  # the second scene's hands sit near its object

    def test_record_keeps_snapshots_of_the_annealing(self, mano_config: RobotConfig, cube_scene: SceneDescription):
        cfg = GraspOptimConfig(num_iters=100, scene_geom_backend="mesh")
        optimizer = WarpGraspOptimizer(cfg, mano_config, torch.device("cuda:0"))
        grasp_qpos = optimizer.optimize([cube_scene], num_samples_per_scene=4, record=True)
        snapshots = optimizer.last_snapshots
        assert snapshots is not None and snapshots.shape[:2] == (1, 4) and snapshots.shape[-1] == grasp_qpos.shape[-1]
        assert 20 <= snapshots.shape[2] <= 27
        torch.testing.assert_close(snapshots[:, :, -1], grasp_qpos.cpu())
        optimizer_ref = weakref.ref(optimizer)
        del optimizer
        assert optimizer_ref() is None  # recorded callbacks must not keep the cached solver/optimizer alive

    def test_warp_gd_fixed_base_runs(self, cube_scene: SceneDescription):
        """A fixed-base arm and hand: the sampler solves IK for the init, the annealing moves only the joints."""
        cfg = GraspOptimConfig(num_iters=50, scene_geom_backend="mesh", sampler_type="fixed_base")
        optimizer = WarpGraspOptimizer(cfg, build_robot_config("xarm7_xhand_right"), torch.device("cuda:0"))
        scene = dataclasses.replace(
            cube_scene, meshes={"object": dataclasses.replace(cube_scene.meshes["object"], position=(0.4, 0.0, 0.03))}
        )
        grasp_qpos = optimizer.optimize([scene], num_samples_per_scene=4)
        assert grasp_qpos.shape == (1, 4, optimizer.robot_helper.num_actuated_joints)
        assert torch.isfinite(grasp_qpos).all()

    def test_warp_lm_ik_scoring_keeps_reachable_candidates(self, cube_scene: SceneDescription):
        cfg = GraspOptimConfig(scene_geom_backend="mesh", optimizer_backend="warp_lm", enable_ik_scoring=True)
        optimizer = WarpGraspOptimizer(
            cfg,
            build_robot_config("xhand_right"),
            torch.device("cuda:0"),
            fixed_base_robot_config=build_robot_config("xarm7_xhand_right"),
        )
        scene = dataclasses.replace(
            cube_scene, meshes={"object": dataclasses.replace(cube_scene.meshes["object"], position=(0.4, 0.0, 0.03))}
        )
        grasp_qpos = optimizer.optimize([scene], num_samples_per_scene=4)
        assert grasp_qpos.shape == (1, 4, 3 + 3 + 12)
        assert torch.isfinite(grasp_qpos).all()


class TestGraspTerms:
    @pytest.mark.parametrize("representation", ["sphere", "capsule"])
    @pytest.mark.parametrize("ignore_kind", ["present", "empty", "missing", "other_only"])
    @pytest.mark.parametrize("enabled", [True, False])
    def test_grasp_term_warning(
        self,
        collision_config: RobotConfig,
        box_scene: Callable[[float], WarpScene],
        caplog: pytest.LogCaptureFixture,
        representation: str,
        ignore_kind: str,
        enabled: bool,
    ):
        path = Path(collision_config.self_collision_ignore_path).with_name(f"ignore.{representation}.yml")
        if ignore_kind == "missing":
            collision_config = replace(collision_config, self_collision_ignore_path=None)
        elif ignore_kind == "empty":
            path.write_text("self_collision_ignore: {}\n")
        elif ignore_kind == "other_only":
            other = "capsule" if representation == "sphere" else "sphere"
            collision_config = replace(
                collision_config, self_collision_ignore_path=str(path.with_name(f"ignore.{other}.yml"))
            )
            path.unlink()
        robot = GraspRobot.from_config(
            collision_config,
            load_meshes=True,
            load_collision_spheres=True,
            self_collision_representation=representation,
        )
        assert "self-collision ignore path" not in caplog.text
        scene = box_scene()
        terms = grasp_terms(
            GraspOptimConfig(w_spen=float(enabled), spen_representation=representation, w_fingertip_dis=0.0),
            robot,
            scene,
            scene,
            lm=False,
        )
        assert ("self_collision" in terms) == enabled
        warned = enabled and ignore_kind in ("missing", "other_only")
        messages = [r.message for r in caplog.records if "self-collision ignore path" in r.message]
        assert len(messages) == int(warned)
        if warned:
            assert collision_config.name in messages[0] and representation in messages[0]
        if enabled:
            caplog.clear()
            state = robot.state(q=wp.zeros((1, 3), dtype=wp.float32, device="cpu"))
            for _ in range(2):
                terms["self_collision"].compute_weighted_residual(VarValues({VarKey.ROBOT: state}))
            assert "self-collision ignore path" not in caplog.text
