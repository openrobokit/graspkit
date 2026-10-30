# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""WarpGraspOptimizer: "warp_gd" anneals pose and contacts; "warp_lm" keeps the best of 24 LM seeds per grasp."""

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import torch
import warp as wp
from robokit.geom.scene import WarpScene
from robokit.helpers.grasp import ContactSamplerConfig, ContactVar
from robokit.lie.se3 import se3_to_matrix
from robokit.opt.gd_optimizer import AnnealedGDConfig, AnnealedGDOptimizer
from robokit.opt.lm_optimizer import LMOptimizerConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.sampler import roberts_samples
from robokit.opt.var_values import VarKey, VarValues
from robokit.terms.dense.force_closure_task import ForceClosureTask
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.rotation_task import AxisLimitTask
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.dense.scene_distance_task import SceneDistanceTask
from robokit.terms.dense.self_collision_task import SelfCollisionTask
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec7
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion
from torch.types import Device

from graspkit.configs.experiment_config import GraspOptimConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec
from graspkit.scene.scene_description import SceneDescription
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_optim.fingertip_task import FingertipDistanceTask
from graspkit.stages.grasp_optim.ik_scorer import FloatingBaseIKScorer
from graspkit.stages.grasp_optim.sampler import build_sampler

LM_NUM_CANDIDATES, LM_ITERS, LM_RESAMPLE_INTERVAL = 24, 64, 8


def grasp_terms(
    cfg: GraspOptimConfig,
    robot: GraspRobot,
    target_scene: WarpScene,
    collision_scene: WarpScene,
    lm: bool,
    hand_up_link_index: Optional[int] = None,
    hand_up_dir: tuple = (0.0, 0.0, 1.0),
) -> Dict[str, ResidualTask]:
    """Grasp energy on `contacts`; GD reads `w` as `0.5 * w * r²`, LM as a raw residual scale with `lm_*` overrides."""
    scale = float if lm else (lambda w: float(np.sqrt(2.0 * w)))
    mode = "abs" if lm else "sqrt_abs"
    eps = cfg.lm_residual_eps if lm else 1e-6
    device = target_scene.device
    scene_indices = wp.from_numpy(np.arange(target_scene.num_scenes, dtype=np.int32), dtype=wp.int32, device=device)
    distance = SceneDistanceTask(
        robot=robot,
        warp_meshes=target_scene,
        num_contact_points=cfg.num_contact_points,
        points_var_key=VarKey.CONTACTS,
        weight=scale(cfg.w_dis),
        residual_mode=mode,
        residual_eps=eps,
    )
    terms: Dict[str, ResidualTask] = {"distance": distance}
    terms["force_closure"] = ForceClosureTask(
        robot=robot,
        warp_meshes=target_scene,
        num_contact_points=cfg.num_contact_points,
        points_var_key=VarKey.CONTACTS,
        weight=scale(cfg.w_fc),
        distance_task=distance,  # one contact query serves both terms
    )
    terms["joint_limit"] = PositionLimit(
        robot=robot, weight=scale(cfg.lm_w_joints if lm else cfg.w_joints), residual_mode=mode, residual_eps=eps
    )
    terms["scene_collision"] = SceneCollisionTask(
        robot=robot,
        scene=collision_scene,
        scene_indices=scene_indices,
        weight=scale(cfg.lm_w_pen if lm else cfg.w_pen),
        # GD penalizes penetration only (DexGraspNet's E_pen); LM's margin keeps contact-free fingertips off the surface
        margin=cfg.lm_collision_margin if lm else 0.0,
        residual_mode=mode,
        residual_eps=eps,
        use_link_bounding_sphere_filter=cfg.use_link_bounding_sphere_filter,
    )
    if cfg.w_spen > 0:
        robot.config.get_self_collision_ignore_path(cfg.spen_representation, warn=True)
        terms["self_collision"] = SelfCollisionTask(
            robot=robot,
            representation=cfg.spen_representation,
            weight=scale(cfg.w_spen),
            margin=cfg.spen_threshold,
            residual_mode=mode,
            residual_eps=eps,
        )
    if cfg.w_angle_z > 0.0:  # a hand axis driven towards world +z
        assert hand_up_link_index is not None, "w_angle_z > 0 requires hand_up_link_index"
        terms["hand_up"] = AxisLimitTask(
            robot=robot,
            frame_index=hand_up_link_index,
            local_axis=tuple(hand_up_dir),
            world_axis=(0.0, 0.0, 1.0),
            min_angle=cfg.angle_z_min,
            weight=cfg.w_angle_z,
        )
    w_fingertip = cfg.lm_w_fingertip_dis if lm else cfg.w_fingertip_dis
    if w_fingertip > 0.0:  # every fingertip link's closest sphere drawn onto the object; one term per link
        for link in robot.config.fingertip_link_names:
            terms[f"fingertip/{link}"] = FingertipDistanceTask(
                robot, target_scene, scene_indices, link, scale(w_fingertip), residual_mode=mode, residual_eps=eps
            )
    return terms


class WarpGraspOptimizer:
    """Batched Warp grasp optimizer: samples init grasps per scene, then runs `GraspOptimConfig.optimizer_backend`."""

    def __init__(
        self,
        cfg: GraspOptimConfig,
        robot_config: RobotConfig,
        device: Device,
        fixed_base_robot_config: Optional[RobotConfig] = None,
        ik_oversample_ratio: int = 16,
    ):
        self.cfg, self.device, self.robot_config = cfg, device, robot_config
        self.lm = cfg.optimizer_backend == "warp_lm"
        self.floating = robot_config.base_type == "floating"
        if self.lm and not self.floating:
            raise ValueError("warp_lm is floating-base only; use cfg.enable_ik_scoring=True for a fixed-base arm")
        if self.lm and cfg.enable_ik_scoring and fixed_base_robot_config is None:
            raise ValueError("cfg.enable_ik_scoring=True requires fixed_base_robot_config")
        wp.init()
        self.robot_helper = GraspRobot.from_config(
            robot_config,
            load_meshes=True,
            load_collision_spheres=True,
            device=device,
            self_collision_representation=cfg.spen_representation,
        )
        # a floating hand samples floating no matter what cfg.sampler_type says
        self.sampler = build_sampler(
            cfg, robot_config, self.robot_helper, device, sampler_type="floating_base" if self.floating else None
        )
        self._fixed_base_robot_config, self._ik_oversample_ratio = fixed_base_robot_config, ik_oversample_ratio
        if (
            cfg.lm_w_fingertip_dis if self.lm else cfg.w_fingertip_dis
        ) > 0 and robot_config.fingertip_link_names is None:
            raise ValueError(f"robot config '{robot_config.name}' has no fingertip links set")
        self.last_snapshots: Optional[torch.Tensor] = None
        self._problems: Dict[tuple, tuple] = {}  # one solver and graph-bound var per scene layout and batch width

    def _compose_vars(self, q: torch.Tensor, base_mat: torch.Tensor, n_scene: int) -> torch.Tensor:
        """Grasp variables `(n_scene, num_samples, dof)`: a floating hand prepends its base, `[xyz, euler_rxyz, q]`."""
        if self.floating:
            q = torch.cat([base_mat[:, :3, 3], matrix_to_euler_angles(base_mat[:, :3, :3], "rxyz"), q], dim=-1)
        return q.reshape(n_scene, -1, q.shape[-1])

    def _problem(
        self,
        scene_geom: SceneGeom,
        variables: torch.Tensor,
        switch_probability: float,
        seed: int,
        problem: Optional[tuple[VarValues, Dict[str, ResidualTask]]] = None,
    ) -> tuple[VarValues, Dict[str, ResidualTask]]:
        """The grasp energy over `variables` rows (`[xyz, euler_rxyz, q]` floating, `q` fixed) on `scene_geom`."""
        cfg, robot, lm, device = self.cfg, self.robot_helper, self.lm, self.device
        if self.floating:
            base = torch.cat(
                [variables[:, :3], matrix_to_quaternion(euler_angles_to_matrix(variables[:, 3:6], axes="rxyz"))], -1
            )
            q = variables[:, 6:]
        else:  # the config's base pose, broadcast; the variables are the joints
            base = torch.tensor(
                self.robot_config.init_position + self.robot_config.init_quaternion, dtype=torch.float32, device=device
            ).expand(variables.shape[0], -1)
            q = variables
        if problem is not None:
            var, terms = problem
            state = var.get(VarKey.ROBOT)
            wp.copy(state.q, wp.from_torch(q.contiguous(), dtype=wp.float32))
            wp.copy(state.T_world_base, wp.from_torch(base.contiguous(), dtype=wp_vec7))
            var.invalidate()
            var.get(VarKey.CONTACTS).reset(seed)
            return var, terms
        state = robot.state(
            q=wp.from_torch(q.contiguous(), dtype=wp.float32),
            T_world_base=wp.from_torch(base.contiguous(), dtype=wp_vec7),
        )
        state.has_floating_base = self.floating
        state.clamp_q_on_integrate = True  # every step lands inside the joint limits
        if lm:
            state.base_step_scale = cfg.lm_base_step_scale
        target_scene, collision_scene = scene_geom.solver_scenes(
            backend=cfg.scene_geom_backend,
            sdf_bake_padding=cfg.sdf_bake_padding,
            sdf_resolution=cfg.sdf_resolution,
            sdf_voxel_size=cfg.sdf_voxel_size,
        )
        hand_up_link_index, hand_up_dir = None, (0.0, 0.0, 1.0)
        if cfg.w_angle_z > 0.0:  # the hand-up term keeps a hand axis pointing toward world +z
            if robot.config.ee_coord_spec is None:
                raise ValueError(f"w_angle_z > 0 requires ee_coord_spec for robot {robot.config.name!r}")
            hand_up_link_index = robot.spec.link_names.index(
                robot.config.hand_base_link_name or robot.spec.link_names[0]
            )
            hand_up_dir = tuple(float(x) for x in hand_coord_spec(robot.config).four_fingers_up_vec)
        # grasp contacts; annealed GD redraws them with `switch_possibility` per step, LM leaves that to the sampler
        candidates, candidate_links = (t.cpu().numpy() for t in GraspRobot.get_contact_candidates(robot.config, device))
        contacts = ContactVar(
            candidates,
            candidate_links,
            state.batch_size,
            cfg.num_contact_points,
            switch_probability=switch_probability,
            seed=seed,
            device=state.q.device,
        )
        terms = grasp_terms(cfg, robot, target_scene, collision_scene, lm, hand_up_link_index, hand_up_dir)
        return VarValues({VarKey.ROBOT: state, VarKey.CONTACTS: contacts}), terms

    def optimize(
        self, scene_descriptions: List[SceneDescription], num_samples_per_scene: int, record: bool = False
    ) -> torch.Tensor:
        """Optimize grasps for each scene; returns `(n_scene, num_samples_per_scene, dof)` grasp variables."""
        cfg, robot, lm, device = self.cfg, self.robot_helper, self.lm, self.device
        n_scene = len(scene_descriptions)
        key = (num_samples_per_scene, tuple((len(s.meshes), len(s.boxes)) for s in scene_descriptions))
        cached = self._problems.get(key)
        scene_geom = SceneGeom(scene_descriptions, device=device) if cached is None else cached[0]
        scene_geom.update(scene_descriptions)
        ik_scorer = None
        if lm and cfg.enable_ik_scoring:
            ik_scorer = FloatingBaseIKScorer(
                self.robot_config, self._fixed_base_robot_config, scene_descriptions, num_samples_per_scene, device
            )
            init_vars = ik_scorer.filter_reachable(
                self.sampler.sample(scene_geom, num_samples_per_scene * self._ik_oversample_ratio)
            )
        else:
            init_vars = self.sampler.sample(scene_geom, num_samples_per_scene)
        flat = init_vars.reshape(-1, init_vars.shape[-1])
        num_candidates = LM_NUM_CANDIDATES if lm else 1
        if lm:  # graspkit's LM seeds: Roberts offsets of `lm_init_sample_range_q` joint ranges and `_base` meters
            limits = robot.actuated_joint_limits_torch
            n = flat.shape[0]
            flat = flat.repeat_interleave(num_candidates, 0)
            offsets_q = torch.as_tensor(
                roberts_samples(num_candidates, flat.shape[1] - 6) - 0.5, dtype=torch.float32, device=device
            )
            offsets_base = torch.as_tensor(roberts_samples(num_candidates, 3) - 0.5, dtype=torch.float32, device=device)
            q = flat[:, 6:] + offsets_q.repeat(n, 1) * cfg.lm_init_sample_range_q * (limits[:, 1] - limits[:, 0])
            q = torch.max(torch.min(q, limits[:, 1]), limits[:, 0])
            xyz = flat[:, :3] + offsets_base.repeat(n, 1) * cfg.lm_init_sample_range_base
            flat = torch.cat([xyz, flat[:, 3:6], q], -1)
        seed = int(torch.initial_seed() % 2147483647)
        var, term_by_name = self._problem(
            scene_geom,
            flat,
            0.0 if lm else cfg.switch_possibility,
            seed,
            None if cached is None else (cached[1], cached[2]),
        )
        state = var.get(VarKey.ROBOT)
        if cached is None:
            active_dof_mask = None
            ignore = robot.ignore_joint_indices.cpu().numpy()
            if ignore.size > 0:  # actuated joints that are neither arm nor hand stay where the sampler put them
                assert not self.floating, "ignore_joint_indices not supported for floating-base"
                mask = np.ones(state.tangent_dim, np.float32)
                mask[ignore] = 0.0
                active_dof_mask = wp.from_numpy(mask, dtype=wp.float32, device=state.q.device)
            terms = list(term_by_name.values())
            if lm:  # every 8 iterations each candidate is joined by a copy with fresh contacts; the better one stays
                population = PopulationSolverConfig(
                    num_iters=LM_ITERS,
                    optimizer=LMOptimizerConfig(
                        max_iter=LM_ITERS, lm_lambda=1.0, cuda_graph_mode="iter", early_stopping_interval=0
                    ),
                    samplers=[ContactSamplerConfig(seed=seed)],
                    sample_at=range(0, LM_ITERS, LM_RESAMPLE_INTERVAL),
                    num_candidates=[num_candidates],
                    cuda_graph_mode="iter",
                    use_early_stopping=False,
                )
                solver = PopulationSolver([terms], population, state.q.device, active_dof_mask=active_dof_mask)
            else:
                gd = AnnealedGDConfig(  # no early stopping: annealing never converges; the check syncs each iteration
                    max_iter=cfg.num_iters,
                    cuda_graph_mode="iter",
                    early_stopping_interval=0,
                    gradient_mode="autodiff",
                    learning_rate=cfg.step_size,
                    mu=cfg.mu,
                    stepsize_period=cfg.stepsize_period,
                    starting_temperature=cfg.starting_temperature,
                    temperature_decay=cfg.temperature_decay,
                    annealing_period=cfg.annealing_period,
                    seed=seed,
                )
                solver = AnnealedGDOptimizer(terms, gd, state.q.device, active_dof_mask)
            self._problems[key] = (scene_geom, var, term_by_name, solver)
        else:
            solver = cached[3]
            if lm:
                solver.samplers[0].seed = seed
            else:
                solver.config = replace(solver.config, seed=seed)
        # --record keeps ~25 snapshots of the annealed search for the grasp-opt replay
        stride, snaps = max(1, cfg.num_iters // 25), []

        def snapshot(iteration: int, population, costs: wp.array) -> None:
            if iteration % stride == 0 or iteration == cfg.num_iters:
                hand = population.get(VarKey.ROBOT)
                snaps.append(
                    self._compose_vars(
                        wp.to_torch(hand.q).clone(), wp.to_torch(se3_to_matrix(hand.T_world_base)).clone(), n_scene
                    )
                )

        values, _ = solver.solve(var, snapshot if record and not lm else None)
        solver._iteration_callback = None  # the recording closure captures this optimizer; release it after the solve
        hand = values.get(VarKey.ROBOT)
        q, base_mat = wp.to_torch(hand.q).clone(), wp.to_torch(se3_to_matrix(hand.T_world_base)).clone()
        if ik_scorer is not None:  # rank every seed by energy plus arm reachability, keep num_samples per scene
            population = solver.final_var.get(VarKey.ROBOT)
            base_mat, q = ik_scorer.select_reachable_seeds(
                wp.to_torch(population.q).clone(),
                wp.to_torch(se3_to_matrix(population.T_world_base)).clone(),
                wp.to_torch(solver.candidate_scores).clone(),
            )
            q, base_mat = q.reshape(-1, q.shape[-1]), base_mat.reshape(-1, 4, 4)
        self.last_snapshots = torch.stack(snaps, dim=2).cpu() if snaps else None
        torch.cuda.synchronize()
        wp.synchronize()
        return self._compose_vars(q, base_mat, n_scene)

    def compute_costs(
        self, scene_descriptions: List[SceneDescription], grasps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Per-term `0.5·Σr²` cost of `(n_scene, n, dof)` grasps, with contacts re-picked nearest the object."""
        robot, device, n_scene = self.robot_helper, self.device, len(scene_descriptions)
        grasps = grasps.to(device)
        scene_geom = SceneGeom(scene_descriptions, device=device)
        var, terms = self._problem(scene_geom, grasps.reshape(-1, grasps.shape[-1]), 0.0, 0)
        contacts = var.get(VarKey.CONTACTS)
        slots = torch.arange(contacts.num_hand_points, device=device).expand(*grasps.shape[:2], -1)
        points = robot.get_contact_points(robot.forward_kinematics_torch(grasps), slots)
        sdf = scene_geom.query_target_object_signed_distances(points.reshape(n_scene, -1, 3))[0]
        nearest = sdf.abs().reshape(-1, contacts.num_hand_points).topk(self.cfg.num_contact_points, largest=False)
        wp.copy(contacts.indices, wp.from_torch(nearest.indices.int().contiguous()))
        contacts.lookup()
        out = {}
        for term in terms.values():
            buf = wp.zeros((var.batch_size, term.residual_dim), dtype=wp.float32, device=var.device)
            term.precompute(var, need_gradient=False)
            term.compute_weighted_residual(var, out_residual=buf, row_offset=0)
            out[term.cost_name] = 0.5 * (wp.to_torch(buf) ** 2).sum(-1).reshape(n_scene, -1).cpu()
        return out


__all__ = ["WarpGraspOptimizer", "grasp_terms"]
