from dataclasses import dataclass
from typing import Literal


@dataclass
class GraspPolicyConfig:
    """Grasp policy architecture, losses and sampling.

    Reusable presets live in `graspkit/configs/grasp_policy.py`.
    """

    # Action spec
    num_queries: int = 3
    """Configurations decoded per trajectory; the trainer and evaluator assume three: pre-grasp, grasp and
    post-grasp."""
    convert_rxyz_to_rot6d: bool = False
    """Floating base: predict a continuous 6D rotation instead of Euler `rxyz`, converted back on output."""
    analytic_euler: bool = False
    """Use exact, non-iterative Euler conversion after decoding rotations."""
    coupled_floating_base: bool = False
    """Share wrist rotation across all three poses and wrist translation during closing."""
    rot_begin_idx: int = 3
    """Start index of the floating-base rotation in each action."""
    enable_kinematic_loss: bool = False
    """Add the MSE between predicted and target link positions from forward kinematics."""

    # Vision encoder
    vision_encoder: Literal["pointnet", "pointnet_state", "spconv", "rgb", "rgb_resnet", "rgb_resnet_views"] = "spconv"
    """Observation backbone."""
    vision_encoder_feature_dim: int = 512
    """Condition feature size passed to the generator."""
    vision_encoder_normalization: Literal["layernorm", "groupnorm"] | None = None
    """Normalization for the SpConv encoder."""
    vision_encoder_global_feature: bool = True
    """Return one scene feature; False keeps per-point features for local grasp-point conditioning."""
    center_pointcloud: bool = False
    """Predict floating-base translations relative to the observed point-cloud centroid."""
    pointcloud_scale: float = 1.0
    """Fixed coordinate scale applied only to vision-encoder inputs."""
    train_yaw_augmentation: bool = False
    """Rotate training observations and target poses together around the vertical axis."""
    canonical_yaw: bool = False
    """Use a deterministic observed-point-cloud yaw frame for training and inference."""
    enable_objectness_prediction: bool = True
    """With local features, train a target-segmentation head and mask graspness at inference."""
    num_points: int = 2048
    """Points `GraspPolicy.preprocess` samples from an observed point cloud."""
    object_points_only: bool = True
    """`GraspPolicy.preprocess` keeps only the target's points, initial and current frames together (hands occlude)."""

    # Generator (persisted cvae_* names also configure flow width and sampling)
    generative_model: Literal["cvae", "flow", "flow_transformer"] = "cvae"
    """`cvae` (`models/cvae.py`) or flow matching (`models/flow_matching.py`)."""
    cvae_latent_dim: int = 256
    """Latent dimension of the CVAE."""
    cvae_hidden_dim: int = 256
    """Width of the generator networks and optional CVAE learned prior."""
    cvae_learned_prior: bool = False
    """Jointly learn a diagonal-Gaussian latent prior conditioned on the observation."""
    normalize_cvae_actions: bool = False
    """Encode standardized actions and decode physical actions using saved mean/std buffers."""
    reconstruction_loss: Literal["native", "phase_std_mse"] = "native"
    """Native reconstruction loss or MSE scaled by each phase's training action standard deviation."""

    # Loss weights
    qpos_weight: float = 1.0
    """Joint-position reconstruction weight: L1 for CVAE, MSE for flow."""
    rot_weight: float = 1.0
    """Floating-base rotation L1 weight."""
    tl_weight: float = 1.0
    """Floating-base translation L1 weight."""
    kinematic_weight: float = 1e2
    """Weight of the kinematic loss."""
    kl_weight: float = 0.01
    """Weight of the KL term to the unit Gaussian."""
    objectness_weight: float = 0.02
    """Weight of the local-feature objectness BCE."""
    graspness_weight: float = 0.02
    """Weight of the local-feature per-point graspness KL."""
    enable_penetration_loss: bool = False
    """if True, penalize hand collision-sphere penetration into the target object (needs scene geom at train time)"""
    penetration_weight: float = 0.1
    """the weight of the penetration loss"""
    penetration_threshold: float = 0.005
    """only penetration deeper than this (meters) is penalized (avoids punishing legitimate contact)"""
    penetration_every: int = 1
    """compute the (expensive) penetration loss every N steps to amortize per-batch scene-geometry build"""
    penetration_warmup_iters: int = 0
    """delay the penetration loss until this iteration so the policy learns to grasp first (replaces fine-tuning)"""
    prior_fingertip_weight: float = 0.0
    """Attract prior-sampled fingertip collision spheres to the target surface during training."""
    # Sampling
    use_mean: bool = True
    """Decode the prior mean for CVAE or integrate from zero noise for flow."""
    cvae_temperature: float = 1.0
    """Scale of the sampled generator noise; ignored when use_mean is True."""
    grasp_evaluator_guidance_steps: int = 0
    """Latent refinement steps using a trained grasp evaluator; zero disables guidance."""
    grasp_evaluator_guidance_radius: float = 2.0
    """Maximum L2 displacement from each initial latent draw."""
    grasp_evaluator_local_features: bool = False
    """Let the grasp evaluator attend to frozen local visual tokens; the actor stays unchanged."""
    grasp_evaluator_anchor_state: bool = False
    """Condition the local success critic on the current fixed-base robot configuration."""
    grasp_evaluator_hand_joint_features: bool = False
    """Encode hand joints instead of all actuated joints in the grasp evaluator."""
    grasp_evaluator_mask_constant_features: bool = True
    """Zero constant replay features after scorer normalization."""
