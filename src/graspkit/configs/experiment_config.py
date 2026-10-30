import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from graspkit.configs.env_config import CameraSpec, IsaacEnvConfig, MjEvalConfig, MjNativeEnvConfig, MsEnvConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot_config import HandAxis, RobotConfig


# ---------------------------------------------------------------------------
# Stage sub-configs: pure algorithm + runtime params, hand-independent defaults
# ---------------------------------------------------------------------------
@dataclass
class GraspOptimConfig:
    """Grasp optimizer algorithm and runtime parameters, with hand-independent defaults.

    The config of the `optimize_grasps` stage (`ExperimentConfig.grasp_optim`); every field is a CLI flag, e.g.
    `--grasp_optim.num_iters 2000`. The annealing fields drive `warp_gd` only and the `lm_*` fields `warp_lm` only.
    The init-sampler fields feed `stages/grasp_optim/sampler.py`; the runtime fields feed `optimize_pipeline.py`,
    not a bare `optimize()` call.
    """

    # algorithm
    num_iters: int = 6000
    """Annealing iterations (`warp_gd`). Each replays one CUDA graph; fewer trade grasp quality for speed, and 2000
    suits smoke tests."""
    num_contact_points: int = 4
    """Active contact points per grasp, drawn from the robot's contact candidates; they parameterize the distance and
    force-closure terms. 4-5 fits most five-finger hands."""
    switch_possibility: float = 0.5
    """Per-grasp probability, each iteration, of re-drawing the whole contact set; the Metropolis test accepts or
    rejects it together with the pose step."""
    mu: float = 0.98
    """RMSprop weight on the current squared gradient, `ema = mu * g^2 + (1 - mu) * ema`. Near 1 the step approaches
    the gradient's sign times `step_size`."""
    step_size: float = 0.005
    """Base RMSprop step; the effective step is `step_size * temperature_decay^(t // stepsize_period)`."""
    stepsize_period: int = 50
    """Iterations between step-size decays."""
    starting_temperature: float = 18.0
    """Initial Metropolis temperature `T`: a proposal that worsens the energy by `d` is still accepted with
    probability `exp(-d / T)`."""
    annealing_period: int = 30
    """Iterations per temperature decay, `T = starting_temperature * temperature_decay^(t // annealing_period)`."""
    temperature_decay: float = 0.95
    """Decay factor shared by the temperature and step-size schedules; closer to 1 cools slower and explores more."""
    spen_threshold: float = 0.01
    """Self-collision margin (m) between sphere/capsule pairs; 2 cm pinned the middle finger of most hands off the object."""
    spen_representation: Literal["capsule", "sphere"] = "sphere"
    """Self-collision geometry for `warp_gd`: per-link capsules or the collision spheres. `warp_lm` always uses
    capsules."""
    w_fc: float = 1.0
    """Force-closure weight: the active contact normals must balance external wrenches. A zero weight disables any
    term."""
    w_dis: float = 100.0
    """Weight pulling each active contact point onto the target surface."""
    w_fingertip_dis: float = 0.0
    """Weight of the fingertip term (warp_gd only; requires fingertip links set): each fingertip link's closest
    collision sphere is drawn onto the object. 0 disables; a hand should touch without it, this is a last resort."""
    w_pen: float = 100.0
    """`warp_gd` scene-penetration weight: hand collision spheres against the target and table, with a fixed 5 mm
    margin. `warp_lm` uses `lm_w_pen`."""
    w_spen: float = 10.0
    """Self-penetration weight between robot links; zero skips building the term."""
    w_joints: float = 1.0
    """`warp_gd` joint-limit weight; q is also clamped to its limits every iteration. `warp_lm` uses `lm_w_joints`."""
    w_angle_z: float = 0.0
    """Weight of the hand-orientation prior (`HandUpAngleTask`, both backends)."""
    angle_z_min: float = math.pi / 2
    """Minimum angle between the hand-up axis and world +z; tabletop configs use it to forbid underhand grasps."""
    hand_axis: Optional[HandAxis] = None
    """Hand-axis cone filter on init samples: which `hand_coord_spec` axis to check, after IK for fixed-base
    samplers. None disables it. The cone on optimized grasps lives on `GraspTrajBuilderConfig`."""
    hand_axis_target: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Direction `hand_axis` should point, in the robot base frame (world for floating hands)."""
    filter_hand_axis_max_angle_deg: float = 30.0
    """Hand-axis cone half-angle in degrees."""
    optimizer_backend: Literal["warp_gd", "warp_lm"] = "warp_gd"
    """`warp_gd`: simulated-annealing gradient descent for floating or fixed-base robots. `warp_lm`: multi-seed
    Levenberg-Marquardt, floating base only, typically 2-3x faster and needs fingertip links; to target an arm, set
    `enable_ik_scoring` and let `GraspTrajBuilder` IK-convert."""
    scene_geom_backend: Literal["mesh", "sdf_volume"] = "sdf_volume"
    """`sdf_volume`: a cached SDF of the target mesh plus analytic table boxes; fast, accuracy bounded by voxel size,
    and the target must be the scene's only mesh. `mesh`: per-scene BVHs, exact but slower."""
    sdf_bake_padding: float = 0.25
    """SDF bake padding, in mesh units (robokit bakes the unscaled mesh and applies `scales` at query time)."""
    sdf_resolution: int = 64
    """Voxel grid resolution along the longest axis when sdf_voxel_size is None."""
    sdf_voxel_size: Optional[float] = None
    """Voxel size in mesh units (before `scales`). None = derive voxel size from sdf_resolution."""
    use_link_bounding_sphere_filter: bool = False
    """Sphere-level broad-phase skip on the warp scene-collision task. Only for scene_geom_backend="sdf_volume"."""

    # warp_lm solver internals (consumed only by stages/grasp_optim/warp_optimizer.py)
    lm_w_pen: float = 150.0
    """warp_lm counterpart of w_pen: scene-collision term weight (LM raw-residual scale)."""
    lm_w_joints: float = 10.0
    """warp_lm counterpart of w_joints: position-limit term weight (LM raw-residual scale)."""
    lm_w_fingertip_dis: float = 10.0
    """warp_lm counterpart of w_fingertip_dis: fingertip-distance term weight (LM raw-residual scale)."""
    lm_collision_margin: float = 0.005
    """warp_lm scene-collision margin (meters)."""
    lm_residual_eps: float = 1e-6
    """warp_lm residual epsilon for distance / position-limit / self-penetration terms."""
    lm_base_step_scale: float = 1e-3
    """warp_lm floating-base step relative to the joint step; keeps base updates from overshooting."""
    lm_init_sample_range_q: float = 0.2
    """warp_lm per-seed joint half-range: the 24 seeds are the init qpos plus Roberts offsets in `[-range, range]`,
    clamped to limits."""
    lm_init_sample_range_base: float = 0.05
    """warp_lm per-seed base-translation sampling half-range (floating base only)."""
    enable_ik_scoring: bool = False
    """warp_lm floating base only: oversample inits 16x, keep those an arm IK can reach, and rank solutions by energy
    plus reachability. Use when `GraspTrajBuilder` will IK-convert onto an arm."""
    evaluator: Optional["EnvConfig"] = None
    """In-loop simulation filter: each optimized batch is rolled out (e.g. with perturbation or a heavy object) and
    only survivors are saved. None gates on motion-plan success only."""

    # init sampler (consumed by the samplers in stages/grasp_optim/sampler.py)
    sampler_type: Literal["fixed_base", "wbc", "floating_base"] = "fixed_base"
    """`fixed_base`: look-at poses solved to collision-free arm IK. `wbc`: world-rooted arms, any azimuth, never from
    below the table. `floating_base`: no IK, emits `[xyz, euler_rxyz, hand_qpos]`; floating optimizers force it."""
    distance_lower: float = 0.02
    """Lower bound (m) of the hand-to-surface standoff, sampled uniformly. Tight bounds are the usual cause of
    starved init sampling."""
    distance_upper: float = 0.2
    """Upper bound (m) of the hand-to-surface standoff."""
    front_theta_lower: float = -math.pi / 2
    """Lower bound of the roll jitter about the look-at approach axis."""
    front_theta_upper: float = 0.0
    """Upper bound of the roll jitter about the look-at approach axis."""
    side_theta_lower: float = -math.pi / 18
    """Lower bound of the tilt jitter about the two axes perpendicular to the approach."""
    side_theta_upper: float = math.pi / 18
    """Upper bound of the tilt jitter about the two axes perpendicular to the approach."""
    hand_qpos_jitter_strength: float = 0.05
    """Std of the per-joint Gaussian jitter added to `hand_qpos_mu`."""
    hand_qpos_mu: Optional[Tuple[float, ...]] = None
    """Mean hand qpos for init sampling, typically half-open; None uses the robot config's init qpos."""
    init_ik_position_tolerance: float = 0.05
    """Position tolerance (m) for the init-time arm IK solve (FixedBaseSampler / WBCSampler)."""
    init_ik_orientation_tolerance: float = 0.3
    """Orientation tolerance (rad) for that solve."""
    init_ik_prefer_rest: bool = True
    """That solve keeps the solution nearest the rest pose; False keeps its best-scoring random seed."""
    init_rest_pose_max_dist: float = 3.0
    """Reject init IK solutions whose arm-qpos L2 distance from the rest pose exceeds this."""
    init_num_ik_candidates_per_sample: int = 50
    """IK candidates per requested sample before collision/rest filtering (this × num_samples_per_scene)."""
    init_num_points_per_sample: int = 1000
    """Empty-space points sampled per requested sample when seeding approach directions."""
    init_ik_batch_size: int = 512
    """Chunk size of the IK + collision sweep over candidates (memory only)."""

    # runtime
    num_samples_per_scene: int = 64
    """Grasps optimized per scene per attempt."""
    num_scenes_per_batch: int = 16
    """Scenes batched into one solve: `num_scenes_per_batch * num_samples_per_scene` grasps on the GPU."""
    min_grasps_threshold: int = 16
    """A scene is done once its output holds this many grasps; scenes below it are re-attempted."""
    num_attempts: int = 1
    """Top-up passes over unfinished scenes; attempt `i` runs with `seed + i`."""
    max_scenes: Optional[int] = None
    """If set, caps scenes processed per attempt (after sort/shuffle/rank slicing). Useful for quick testing."""
    save_labels: bool = False
    """Save all candidates and their filter/plan/simulation verdicts to `<grasp_output_dir>.labels/`."""


@dataclass
class EnvConfig:
    """The tabletop grasp task; select Isaac or MuJoCo with their optional configuration, otherwise ManiSkill."""

    ms_env_config: MsEnvConfig = field(default_factory=MsEnvConfig)
    """ManiSkill-only settings"""
    mj_env_config: Optional[MjEvalConfig] = None
    """MuJoCo evaluator config; set this to select the MuJoCo backend."""
    isaac_env_config: Optional[IsaacEnvConfig] = None
    """Isaac backend config; mutually exclusive with mj_env_config."""
    mj_native_env_config: Optional[MjNativeEnvConfig] = None
    """Closed-loop MuJoCo backend config; at most one of the backend configs may be set."""
    load_scenes: bool = True
    """Whether to load scenes (actors and tables)."""
    control_freq: int = 50
    """Control frequency."""
    target_lift_dist: float = 0.2
    """Target lift distance."""
    object_mass: float = 0.01
    """Mass of the object."""
    floating_object: bool = False
    """Whether to use a floating object."""
    camera_width: int = 960
    """Width of the camera."""
    camera_height: int = 540
    """Height of the camera."""
    camera_intrinsic: Tuple[Tuple[float, float, float], ...] = field(
        default_factory=lambda: ((677, 0, 492), (0, 677, 268), (0, 0, 1))
    )
    """Intrinsic matrix of the camera."""
    camera_near: float = 0.01
    """Near clipping plane of the camera."""
    cameras: Tuple[CameraSpec, ...] = ()
    """Cameras rendered instead of ManiSkill's `base_camera` (with `ms_env_config.robot_uid` None); Isaac and
    MuJoCo render only these."""
    contact_offset: float = 0.001
    """PhysX contact_offset (m). Larger = compliant/sticky-feeling fingertip pads."""
    num_envs: int = 2048
    """Number of environments to evaluate on"""
    num_seeds: int = 1
    """Number of seeds"""
    enable_perturbation: bool = True
    """Whether to enable perturbation"""
    perturbation_step_range: Tuple[int, int] = (35, 80)
    """Perturbation step range"""
    perturbation_magnitude: float = 150.0
    """Perturbation magnitude"""
    num_perturbations: int = 8
    """Number of perturbations"""
    perturbation_duration: int = 1
    """Number of consecutive steps to apply each perturbation force (1 = single impulse)"""


@dataclass
class MotionPlannerConfig:
    """Trajectory optimization params — hand-independent defaults."""

    backend: Literal["torch", "warp"] = "warp"
    """Motion planner backend. 'warp' (default; robokit MotionPlanner, fixed-base only) is faster than 'torch'."""

    # algorithm
    pre_grasp_arm_grad_move: float = 0.001
    """Pre-grasp arm gradient move"""
    pre_grasp_hand_grad_move: float = 0.0
    """Pre-grasp hand gradient move"""
    post_grasp_hand_grad_move: float = 1.0
    """Post-grasp hand gradient move"""
    lifting_dist: float = 0.3
    """Lifting distance"""
    padding_dist: float = 0.025
    """Padding distance"""
    opt_steps: int = 300
    """Optimization steps"""
    pen_weight: float = 1.0
    """Penetration weight"""
    arm_smooth_weight: float = 100.0
    """Arm smooth weight"""
    hand_smooth_weight: float = 10.0
    """Hand smooth weight"""
    vel_smooth_weight: float = 0.1
    """Velocity smooth weight"""
    num_steps_pre_grasp: int = 150
    """Number of steps in pre-grasp"""
    num_steps_pre_to_grasp: int = 25
    """Number of steps in pre-to-grasp"""
    num_steps_holding: int = 75
    """Number of steps in holding"""
    close_during_holding: bool = False
    """Close gradually during holding instead of settling at the closed pose."""
    num_steps_post_grasp: int = 50
    """Number of steps in post-grasp"""
    control_dt: float = 0.02
    """Seconds per trajectory step, the simulator's control period."""
    enable_joint_optimization: bool = False
    """Fixed base, warp: one robokit plan through start, pre-grasp, grasp, hold and lift, timed by robokit except the
    hold (num_steps_holding); at least num_steps long. False optimizes start → pre-grasp alone and fills the other
    phases at their step counts."""

    # robokit L-BFGS trajectory-opt overrides (warp only); raise defaults if a low approach leaves the path unconverged.
    traj_opt_max_iter: int = 100
    traj_opt_smoothness_weight: float = 200.0
    traj_opt_accel_smoothness_weight: float = 30.0
    traj_opt_self_collision_weight: float = 0.0
    """Self-collision weight (warp only). 0 disables the term; the robokit q_goal preset ships it off."""
    traj_opt_self_collision_margin: float = 0.02
    """Self-collision margin in metres; only read when the weight is > 0."""

    start_from_grasp_dof_indices: Optional[Tuple[int, ...]] = None
    """DOF indices whose start values come from grasp_qpos, not init pose; also locked during post-grasp lift IK."""

    # runtime
    num_samples_per_batch: int = 2048
    """Number of samples per batch"""
    num_grasps_per_scene: int = 32
    """Number of grasps to plan per scene (pad/truncate to this)"""

    @property
    def num_steps(self) -> int:
        return (
            self.num_steps_pre_grasp + self.num_steps_pre_to_grasp + self.num_steps_holding + self.num_steps_post_grasp
        )


@dataclass
class GraspTrajBuilderConfig:
    """Grasp traj builder: floating-base grasps -> deployable trajectories (stages/traj_builder)."""

    fixed_base_robot_config: Optional[RobotConfig] = None
    """Fixed-base arm+hand robot trajectories are planned for; also the camera-mount robot. None if floating-only."""
    floating_base_robot_config: Optional[RobotConfig] = None
    """Floating-base hand-only robot the input grasps come from."""
    convert_to_fixed: bool = False
    """True: filter + IK-convert floating grasps, build trajectories for the fixed robot. False: plan grasps as-is."""
    motion_planner: MotionPlannerConfig = field(default_factory=MotionPlannerConfig)
    """Motion planner config for the trajectory from the init pose to the grasp pose."""
    hand_axis: Optional[HandAxis] = None
    """Hand-axis cone filter on optimized grasps: which `hand_coord_spec` axis to check. None disables it."""
    hand_axis_target: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Direction `hand_axis` should point, in the robot base frame (world for floating hands)."""
    filter_hand_axis_max_angle_deg: float = 30.0
    """Hand-axis cone half-angle in degrees."""
    contact_avoid_filter: bool = False
    """Reject grasps where the contact-avoid point is closer to the target than the contact point on that link."""
    pen_filter: bool = False
    """Reject grasps whose hand spheres overlap the target by more than pen_filter_threshold (m); fingertip ok."""
    pen_filter_threshold: float = 0.005
    rest_pose_filter_max_dist: Optional[float] = None
    """Reject grasps whose arm qpos (L2 over arm joints) is further than this from robot rest pose; None disables."""
    """Max allowed hand-sphere penetration depth into the target object, meters."""
    arm_range_fraction: float = 0.30
    """Half-width (fraction of each arm joint's range around its midpoint) the IK solution must lie in; 1.0 disables."""
    ori_threshold_deg: float = 45.0
    """Ori-gate half-angle (deg) for the floating->fixed IK conversion; 45 = default, 90 effectively disables."""
    plan_table_only: bool = False
    """Plan against a table-only scene (object removed), mirroring deployment. Pre/post qpos use the real scene."""


@dataclass
class RendererConfig:
    """Point cloud renderer params — hand-independent defaults."""

    env_config: EnvConfig = field(default_factory=EnvConfig)
    """Environment the renderer simulates; its num_envs and evaluation settings are unused"""
    num_envs: int = 64
    """Number of environments"""
    seed: int = 0
    """Seed"""
    obs_mode: str = "pointcloud"
    """Observation mode"""
    points_crop_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.2, 0.2),
        (-0.2, 0.2),
        (-0.1, 0.8),
    )
    """((x_min, x_max), (y_min, y_max), (z_min, z_max))"""
    points_crop_bounds_camera: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.2, 0.2),
        (-0.2, 0.2),
        (0.1, 1.0),
    )
    """((x_min, x_max), (y_min, y_max), (z_min, z_max)) in camera frame"""
    num_points: int = 2048
    """Number of points to sample from the point cloud"""
    remove_segments: list[str] = field(default_factory=list)
    """Segments to remove from the point cloud, e.g., ['table', 'hand']"""
    extract_segments: list[str] = field(default_factory=lambda: ["object"])
    """Segments to extract from the point cloud, e.g., ['object']. If not empty, `remove_segments` is ignored."""
    # ---- camera ----
    camera_poses_path: Optional[str] = None
    """Path to camera extrinsic poses"""
    camera_intrinsic_path: Optional[str] = None
    """Path to camera intrinsic parameters"""
    render_downscale: float = 1.0
    """Divide render resolution (intrinsics + width/height) by this. 1.0 = native; 2.0 = half-res, 4x fewer pixels."""
    camera_frame: Literal["world", "robot"] = "robot"
    """Anchor frame of the extrinsic: "robot" composes with the robot base pose, "world" is already world-frame."""
    camera_mode: Literal["fixed", "sampled"] = "fixed"
    """fixed: one view at the calibrated pose. sampled: camera_sample_num_views jittered views/scene, keep valid."""
    camera_sample_num_views: int = 24
    """sampled mode: number of jittered views rendered per scene before the valid-view filter."""
    camera_sample_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """sampled mode: per-axis max eye-translation jitter in meters (x, y, z)."""
    camera_sample_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """sampled mode: per-axis max orientation jitter in degrees (rxyz), applied on top of the chosen orientation."""
    look_at_object: bool = False
    """Override camera orientation to aim at the target-object center (works in both modes)."""
    point_cloud_frame: Literal["world", "camera"] = "world"
    """Frame the rendered point cloud is returned in."""

    # object point-cloud sampling (sample_point_cloud stage, not the Renderer)
    sample_num_points: int = 2048
    """Number of points for point cloud sampling"""
    sample_enable_fps: bool = False
    """Whether to enable FPS sampling"""

    # runtime
    num_scenes_per_batch: int = 128
    """Number of scenes per rendering batch"""


@dataclass
class TrainerConfig:
    """Training-loop parameters for `Trainer`.

    AdamW's learning rate drops 10x at two thirds of `total_iters`; that schedule has no field.
    """

    seed: int = 0
    """Seeds model initialization, dataset workers and sampling."""
    evaluator: Optional[EnvConfig] = None
    """Simulation for periodic evaluation, built with `build_env`; None disables evaluation."""
    eval_only: bool = False
    """Skip training, load `ExperimentConfig.ckpt_path` and evaluate one trajectory per scene."""
    eval_on_train: bool = False
    """Evaluate on the train split instead of test, for small scene sets without test scenes."""
    track: bool = False
    """Log to Weights & Biases, which also syncs the TensorBoard scalars."""
    wandb_project_name: str = "GraspKit"
    """W&B project used when `track` is set."""
    batch_size: int = 32
    """Global batch size in scenes; each DDP rank gets `batch_size // world_size`."""
    eval_num_envs: int = 32
    """Evaluator parallelism; capped at `batch_size` during training."""
    max_eval_batches: int = 100
    """Stop evaluation after this many data-loader batches."""
    eval_pen_threshold: float = 0.005
    """Penetration depth (m) below which an eval grasp counts toward pass-rate (report-only)."""
    total_iters: int = 100_000
    """Final training iteration; the loop runs iterations 0 through this value."""
    num_dataloader_workers: int = 12
    """Worker processes for the training and evaluation data loaders."""
    lr: float = 1e-4
    """AdamW learning rate."""
    weight_decay: float = 1e-4
    """AdamW weight decay."""
    log_freq: int = 100
    """Iterations between loss logs, and TensorBoard scalars when tracking."""
    eval_freq: int = 10000
    """Iterations between simulation evaluations; iteration 0 is skipped."""
    save_freq: int = 10000
    """Iterations between `iter_<N>.pt` checkpoints, iteration 0 included."""
    cuda_graph: bool = False
    """Capture training on prepared HDF5 views as a CUDA graph; single GPU without evaluation."""
    fast_step: bool = False
    """Use TF32 and fused AdamW for ordinary training steps."""
    grad_clip_norm: float = 0.0
    """Clip the gradient norm before an optimizer step; zero disables clipping."""


@dataclass
class DatasetConfig:
    """Dataset loading parameters for `GraspDataset`.

    One index is one scene, not one grasp; raw views resample points and actions on every access.
    """

    num_actions: int = 64
    """Successful action trajectories sampled per scene, with replacement when a scene has fewer."""
    num_points: int = 2048
    """Point-cloud points per scene, with replacement when the cloud is smaller."""
    return_actions: bool = True
    """Return the `(num_actions, 3, D)` supervision tensor; training needs it and evaluation turns it off."""
    return_graspness: bool = False
    """Return per-point graspness and the best conditioning-point indices, for local-feature SpConv policies; needs
    data processed with `enable_contact_extraction`."""
    return_object_pointcloud_only: bool = False
    """Sample only target-object points instead of the full cropped scene cloud."""
    cache_in_memory: bool = False
    """Cache observations and actions in RAM; sampling stays unchanged."""
    return_rgb: bool = True
    """Include RGB and camera poses when available."""
    fuse_views: int = 0
    """Object-only multi-view data: fuse view 0 (the head) with up to this many other views, a random count per
    sample, into one cloud. 0: one random view."""
    view_noise: float = 0.0
    """With fuse_views: std of each view's rigid misalignment, meters (translation) and x10 radians (rotation)."""
    point_noise: float = 0.0
    """With fuse_views: std of per-point Gaussian noise (m)."""
    aug_prob: float = 0.0
    """With fuse_views: probability of the perception augmentation (`datasets/perception_aug.py`) per sample; needs
    renders that keep the neighbours (extract_segments=[]) and camera poses in the processed h5."""
    aug_keep_min: float = 0.1
    """Perception aug: fused-cloud keep fraction, log-uniform in [aug_keep_min, 1]."""
    aug_voxel_max: float = 0.0
    """Perception aug: voxel dedup size (m), applied with p 0.5 (the old gate pipeline's 3 mm thinning)."""
    aug_drop_wrist_prob: float = 0.0
    """Perception aug: probability of the head view alone (no wrist view)."""
    aug_rgb_prob: float = 0.0
    """Perception aug: probability of the RGB / MoGe depth-error model (per sample)."""


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Unified experiment config. Sub-configs hold hand-independent params; paths/robots/pipeline control live here."""

    name: str
    """Experiment name"""

    # ---- robot configs (direct objects, not strings) ----
    robot_config: Optional[RobotConfig] = None
    """The robot that grasp optimization and policy training target (the floating hand in the standard pipeline)."""
    traj_builder: GraspTrajBuilderConfig = field(default_factory=GraspTrajBuilderConfig)
    """Grasp traj builder config: floating/fixed robot pair, convert decision, motion planner + evaluator."""

    # ---- stage sub-configs ----
    grasp_optim: GraspOptimConfig = field(default_factory=GraspOptimConfig)
    """Grasp optimizer config."""

    renderer: RendererConfig = field(default_factory=RendererConfig)
    """Renderer config"""
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Trainer config"""
    train_single_gpu: bool = False
    """Force training onto a single GPU even when more are available. Useful for tiny datasets, DDP sync dominates."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset config"""
    enable_contact_extraction: bool = False
    """Whether to enable contact extraction during data processing"""
    agent: Optional[GraspPolicyConfig] = None
    """Open-loop policy config"""

    # ---- all paths (experiment-specific) ----
    scene_description_path: str = ""
    """Path to scene descriptions (used by rendering, training, evaluation)"""

    grasp_output_dir: Optional[str] = None
    """Output directory for optimized grasps (pre/grasp/post qpos of robot_config)."""

    rendered_data_path: str = ""
    """Output path for rendered data (h5 file)"""
    processed_data_path: str = ""
    """Output path for processed sim data (h5 file)"""
    object_pointcloud_path: str = ""
    """Path to the object point cloud h5 file (for floating-base sample_point_cloud step)"""
    ckpt_path: Optional[str] = None
    """Path to a checkpoint to evaluate"""
    real_env_config_name: Optional[str] = None
    """Default REAL_ENV_CONFIGS adapter for scripts/deploy/deploy.py --env real; CLI overrides."""

    # ---- pipeline control ----
    steps: Tuple[str, ...] = (
        "render_data",
        "optimize_grasps",
        "process_sim_data",
        "train",
    )
    """Pipeline steps to run"""
    skip_existing: bool = True
    """Skip existing scenes"""
    shuffle_order: bool = True
    """Shuffle the order of scenes"""
    seed: int = 42
    """Random seed for the entire run."""
    visualize: bool = False
    """Whether to visualize"""

    gpus: Optional[str] = None
    """Comma-separated GPU IDs (e.g. '0,1,2,3'). If None, auto-detect free GPUs."""
    num_gpus: int = 4
    """Number of GPUs to use when auto-detecting."""
    gpu_memory_threshold_mb: int = 2048
    """GPUs with less than this much memory used (MB) are considered free."""

    rank: Optional[int] = None
    """Process rank"""
    world_size: Optional[int] = None
    """Number of processes"""

    restart_from: Optional[str] = None
    """Step name to restart from (e.g. 'optimize_grasps'). Deletes markers + outputs for this step and all following."""
    append: bool = False
    """When used with --restart-from, keep the restart step's output dir (only delete its marker)."""
