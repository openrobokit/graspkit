from collections.abc import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from robokit.xform.utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_nearest_euler_angles
from torch import nn
from torch.nn import functional as F
from torch.types import Device

from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.models.cvae import CVAE
from graspkit.models.flow_matching import FlowMatching
from graspkit.models.grasp_evaluator import GraspEvaluator
from graspkit.models.vision_encoder.pointnet import PointNetEncoder
from graspkit.models.vision_encoder.rgb import build_rgb_encoder, preprocess_rgb
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.utils.tui_utils import log_once
from graspkit.utils.xform_utils import matrix_to_continuous_euler_angles


class GraspPolicy(nn.Module):
    """Predicts pre-grasp, grasp and post-grasp configurations from an observation; see `build_grasp_policy`."""

    def __init__(self, cfg: GraspPolicyConfig, action_dim: int, robot_config: RobotConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        assert cfg.pointcloud_scale > 0
        assert cfg.cvae_hidden_dim > 0
        assert cfg.reconstruction_loss != "phase_std_mse" or (
            robot_config.base_type == "fixed" and (cfg.generative_model != "cvae" or cfg.normalize_cvae_actions)
        )
        assert not cfg.grasp_evaluator_local_features or (
            cfg.grasp_evaluator_guidance_steps > 0
            and cfg.vision_encoder_feature_dim == 512
            and cfg.vision_encoder_global_feature
        )
        assert not cfg.grasp_evaluator_anchor_state or (
            cfg.grasp_evaluator_local_features and robot_config.base_type == "fixed"
        )
        assert not cfg.center_pointcloud or robot_config.base_type == "floating"
        assert not cfg.train_yaw_augmentation or (
            robot_config.base_type == "floating" and not cfg.enable_penetration_loss
        )
        assert not cfg.canonical_yaw or (
            cfg.center_pointcloud
            and cfg.convert_rxyz_to_rot6d
            and cfg.rot_begin_idx == 3
            and not cfg.train_yaw_augmentation
            and not cfg.enable_penetration_loss
        )
        assert cfg.prior_fingertip_weight >= 0
        assert not cfg.prior_fingertip_weight or (
            cfg.enable_penetration_loss and robot_config.fingertip_link_names is not None
        )
        assert not cfg.coupled_floating_base or (
            robot_config.base_type == "floating" and cfg.num_queries == 3 and cfg.rot_begin_idx == 3
        )

        # build robot helper lazily
        self.robot_config = robot_config
        self.robot_helper: GraspRobot | None = None
        self._grasp_link_indices: torch.Tensor | None = None
        self._scene_geom = None  # set per-batch by the trainer when enable_penetration_loss

        # build vision encoder
        if self.cfg.vision_encoder.startswith("rgb"):
            assert cfg.vision_encoder_global_feature
            assert not (cfg.center_pointcloud or cfg.train_yaw_augmentation or cfg.canonical_yaw)
            self.vision_encoder = build_rgb_encoder(cfg.vision_encoder, cfg.vision_encoder_feature_dim)
        elif self.cfg.vision_encoder in ("pointnet", "pointnet_state"):
            self.vision_encoder = PointNetEncoder(
                feature_dim=self.cfg.vision_encoder_feature_dim,
                input_dim=3,
                global_feature=self.cfg.vision_encoder_global_feature,
            )
        elif self.cfg.vision_encoder == "spconv":
            from graspkit.models.vision_encoder.spconv_unet import SpUNetBase

            self.vision_encoder = SpUNetBase(
                in_channels=3,
                num_classes=self.cfg.vision_encoder_feature_dim,
                base_channels=32,
                grid_size=0.005,
                channels=(32, 64, 128, 256, 192, 192, 192, 192),
                layers=(2, 2, 2, 2, 2, 2, 2, 2),
                cls_mode=self.cfg.vision_encoder_global_feature,
                normalization=self.cfg.vision_encoder_normalization,
            )

        # objectness decoder
        if not self.cfg.vision_encoder_global_feature:
            if self.cfg.enable_objectness_prediction:
                self.objectness_decoder = nn.Sequential(
                    nn.Linear(self.cfg.vision_encoder_feature_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
            self.graspness_decoder = nn.Sequential(
                nn.Linear(self.cfg.vision_encoder_feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        generator_args = {
            "x_dim": self.action_dim * self.cfg.num_queries,
            "cond_dim": self.cfg.vision_encoder_feature_dim,
            "hidden_dims": [self.cfg.cvae_hidden_dim] * 3,
        }
        if self.cfg.generative_model == "cvae":
            generator = CVAE(
                **generator_args,
                latent_dim=self.cfg.cvae_latent_dim,
                learned_prior=self.cfg.cvae_learned_prior,
                normalize_actions=self.cfg.normalize_cvae_actions,
            )
        else:
            assert not self.cfg.cvae_learned_prior
            generator = FlowMatching(
                **generator_args, model=self.cfg.generative_model, num_queries=self.cfg.num_queries
            )
        # Preserve existing checkpoint keys; expose both generator types through generator.
        self.add_module("cvae", generator)
        if self.cfg.grasp_evaluator_guidance_steps > 0:
            assert self.cfg.grasp_evaluator_guidance_radius > 0
            self.grasp_evaluator = GraspEvaluator(
                self.cfg.vision_encoder_feature_dim,
                robot_config,
                self.cfg.num_queries,
                self.cfg.grasp_evaluator_local_features,
                self.cfg.grasp_evaluator_anchor_state,
                self.cfg.grasp_evaluator_hand_joint_features,
                self.cfg.grasp_evaluator_mask_constant_features,
            )
        if self.cfg.vision_encoder in ("pointnet_state", "rgb_resnet_views"):
            if self.cfg.vision_encoder == "pointnet_state":
                assert cfg.vision_encoder_global_feature and robot_config.base_type == "fixed"
            feature_dim = self.cfg.vision_encoder_feature_dim
            views = 4 if self.cfg.vision_encoder == "rgb_resnet_views" else 1
            self.rgb_conditioning = nn.Linear(views * feature_dim + 2 * self.action_dim + 28, feature_dim)
            with torch.no_grad():
                self.rgb_conditioning.weight.zero_()
                self.rgb_conditioning.weight[:, : views * feature_dim].copy_(
                    torch.eye(feature_dim).repeat(1, views) / views
                )
                self.rgb_conditioning.bias.zero_()

    @property
    def generator(self) -> CVAE | FlowMatching:
        return self.cvae

    def _couple_base(self, actions: torch.Tensor) -> torch.Tensor:
        rotation_end = 9 if self.cfg.convert_rxyz_to_rot6d else 6
        grasp_base = actions[..., 1:2, :rotation_end]
        translations = torch.cat([actions[..., :1, :3], grasp_base[..., :3], grasp_base[..., :3]], dim=-2)
        rotations = grasp_base[..., 3:].expand(*actions.shape[:-2], 3, rotation_end - 3)
        return torch.cat([translations, rotations, actions[..., rotation_end:]], dim=-1)

    def _canonical_yaw_rotation(self, points: torch.Tensor) -> torch.Tensor:
        xy = points[..., :2] - points[..., :2].mean(dim=1, keepdim=True)
        covariance = xy.transpose(1, 2) @ xy
        angle = 0.5 * torch.atan2(2 * covariance[:, 0, 1], covariance[:, 0, 0] - covariance[:, 1, 1])
        projection = xy[..., 0] * angle.cos()[:, None] + xy[..., 1] * angle.sin()[:, None]
        # The third moment resolves the principal axis's 180-degree ambiguity.
        angle = angle + torch.pi * (projection.pow(3).sum(dim=1) < 0)
        euler = points.new_zeros(len(points), 3)
        euler[:, 2] = -angle
        return euler_angles_to_matrix(euler, "rxyz")

    # NOTE: do not rename this method as `forward` hooks are required for DDP
    def forward(self, obs: dict, actions: Float[torch.Tensor, "b n t d"]) -> dict[str, torch.Tensor]:
        loss_dict = {}

        # vision encoder
        scene_pts = obs["rgb"] if self.cfg.vision_encoder.startswith("rgb") else obs["pointcloud"]
        if self.cfg.train_yaw_augmentation or self.cfg.canonical_yaw:
            center = scene_pts.mean(dim=1, keepdim=True)
            if self.cfg.canonical_yaw:
                rotation = self._canonical_yaw_rotation(scene_pts)
            else:
                yaw = scene_pts.new_zeros(len(scene_pts), 3)
                yaw[:, 2] = torch.rand(len(scene_pts), device=scene_pts.device) * (2 * torch.pi)
                rotation = euler_angles_to_matrix(yaw, "rxyz")
            scene_pts = (rotation[:, None] @ (scene_pts - center).unsqueeze(-1)).squeeze(-1) + center
            translation = (rotation[:, None, None] @ (actions[..., :3] - center[:, None]).unsqueeze(-1)).squeeze(
                -1
            ) + center[:, None]
            action_rotation = rotation[:, None, None] @ euler_angles_to_matrix(actions[..., 3:6], "rxyz")
            actions = torch.cat(
                [translation, matrix_to_nearest_euler_angles(action_rotation), actions[..., 6:]], dim=-1
            )
        if self.cfg.center_pointcloud:
            center = scene_pts.mean(dim=1, keepdim=True)
            scene_pts = scene_pts - center
        if self.cfg.vision_encoder.startswith("rgb"):
            scene_pts = preprocess_rgb(scene_pts, self.cfg.vision_encoder)
        else:
            scene_pts = scene_pts * self.cfg.pointcloud_scale
        scene_feats = self.vision_encoder(scene_pts)  # (B, D) or (B, N, D)
        if self.cfg.vision_encoder == "rgb_resnet_views":
            scene_feats = rearrange(scene_feats, "(b v) d -> b (v d)", b=len(actions))
        if self.cfg.vision_encoder in ("pointnet_state", "rgb_resnet_views"):
            scene_feats = self.rgb_conditioning(
                torch.cat((scene_feats, obs["qpos"].flatten(1), obs["camera_poses"].flatten(1)), dim=-1)
            )

        if self.cfg.vision_encoder_global_feature:
            # global feature
            scene_feats = repeat(scene_feats, "b d -> (b n) d", n=actions.shape[1])
        else:
            if self.cfg.enable_objectness_prediction:
                gt_objectness = obs["objectness"]  # (B, N)
                # objectness loss
                seg_prob = self.objectness_decoder(scene_feats).squeeze(-1)  # (B, N)
                # upweight positive points by the negative/positive ratio to balance classes
                pos_weight = ((1.0 - gt_objectness).sum() / (gt_objectness.sum() + 1e-6)).clamp(1e-4, 1e4)
                weight = 1.0 + (pos_weight - 1.0) * gt_objectness
                seg_loss = F.binary_cross_entropy(seg_prob.clamp(1e-7, 1.0 - 1e-7), gt_objectness, weight=weight)
                loss_dict["objectness"] = seg_loss * self.cfg.objectness_weight
                with torch.no_grad():
                    predictions = (seg_prob > 0.5).float()
                    tp = (predictions * gt_objectness).sum()
                    fp = (predictions * (1 - gt_objectness)).sum()
                    fn = ((1 - predictions) * gt_objectness).sum()
                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                    loss_dict["objectness_f1"] = f1

            # graspness
            graspness = self.graspness_decoder(scene_feats).squeeze(-1)
            gt_graspness = obs["graspness"]
            gt_graspness = gt_graspness / (gt_graspness.sum(-1, keepdim=True) + 1e-6)
            graspness_loss = F.kl_div(F.log_softmax(graspness, dim=-1), gt_graspness, reduction="batchmean")
            loss_dict["graspness"] = graspness_loss * self.cfg.graspness_weight

            # gather local features
            best_point_indices = obs["best_point_indices"]
            scene_feats = torch.gather(
                scene_feats, dim=-2, index=best_point_indices.unsqueeze(-1).expand(-1, -1, scene_feats.shape[-1])
            )
            scene_feats = rearrange(scene_feats, "b n d -> (b n) d")

        # Action generation
        if (self.cfg.enable_kinematic_loss or self.cfg.enable_penetration_loss) and (
            self.robot_helper is None or self.robot_helper.device != actions.device
        ):
            self.robot_helper = GraspRobot.from_config(self.robot_config, device=actions.device)
            if self.cfg.enable_kinematic_loss:
                self._grasp_link_indices = torch.tensor(
                    [
                        self.robot_helper.spec.link_names.index(name)
                        for name in [self.robot_config.palm_link_name, *self.robot_config.fingertip_link_names]
                    ],
                    device=actions.device,
                )
        if self.cfg.enable_kinematic_loss:
            grasp_links = self._grasp_link_indices
            with torch.no_grad():
                gt_link_pos = self.robot_helper.forward_kinematics_torch(actions)[..., grasp_links, :3, 3].clone()
        if self.cfg.convert_rxyz_to_rot6d:
            rot6d = matrix_to_rotation_6d(
                euler_angles_to_matrix(actions[..., self.cfg.rot_begin_idx : self.cfg.rot_begin_idx + 3], "rxyz")
            )
            actions = torch.cat(
                [actions[..., : self.cfg.rot_begin_idx], rot6d, actions[..., self.cfg.rot_begin_idx + 3 :]], dim=-1
            )

        encoded_actions = actions
        if self.cfg.center_pointcloud:
            encoded_actions = torch.cat([actions[..., :3] - center[:, None], actions[..., 3:]], dim=-1)
        recon_x, means, log_var, geometry_x = self.generator(
            rearrange(encoded_actions, "b n t d -> (b n) (t d)"), scene_feats
        )

        recon_x = rearrange(
            recon_x, "(b n) (t d) -> b n t d", b=actions.shape[0], n=actions.shape[1], t=actions.shape[2]
        )
        if self.cfg.center_pointcloud:
            recon_x = torch.cat([recon_x[..., :3] + center[:, None], recon_x[..., 3:]], dim=-1)
        if self.cfg.coupled_floating_base:
            recon_x = self._couple_base(recon_x)

        # compute loss
        klds = -0.5 * (1 + log_var - means.pow(2) - log_var.exp())
        if self.cfg.cvae_learned_prior:
            prior_means, prior_log_var = self.generator.prior(scene_feats)
            klds = 0.5 * (
                prior_log_var - log_var + (log_var.exp() + (means - prior_means).square()) * (-prior_log_var).exp() - 1
            )
        kl_loss = klds.sum(-1).mean()  # sum over channels, mean over samples

        # qpos loss
        if self.robot_config.base_type == "floating":
            rot_dim = 6 if self.cfg.convert_rxyz_to_rot6d else 3
            gt_qpos = actions[..., self.cfg.rot_begin_idx + rot_dim :]
            pred_qpos = recon_x[..., self.cfg.rot_begin_idx + rot_dim :]
        else:
            gt_qpos, pred_qpos = actions, recon_x
        if self.cfg.reconstruction_loss == "phase_std_mse":
            assert self.robot_config.base_type == "fixed"
            scale = self.generator.action_scale.reshape(self.cfg.num_queries, -1)
            gt_qpos, pred_qpos = gt_qpos / scale, pred_qpos / scale
        elif self.cfg.generative_model == "cvae" and self.cfg.enable_kinematic_loss:
            scale = self.robot_helper.actuated_joint_limits_torch.diff(dim=-1).squeeze(-1).clamp_min(1e-3)
            gt_qpos, pred_qpos = gt_qpos / scale, pred_qpos / scale
        elif self.cfg.generative_model != "cvae":
            scale = self.generator.action_scale.reshape(self.cfg.num_queries, -1)
            scale = scale[..., -gt_qpos.shape[-1] :].amax(dim=0)
            gt_qpos, pred_qpos = gt_qpos / scale, pred_qpos / scale
        reconstruction_loss = (
            F.l1_loss
            if self.cfg.generative_model == "cvae" and self.cfg.reconstruction_loss == "native"
            else F.mse_loss
        )
        qpos_loss = reconstruction_loss(pred_qpos, gt_qpos, reduction="mean")
        loss_dict["qpos"] = qpos_loss * self.cfg.qpos_weight

        # translation and rotation loss
        if self.robot_config.base_type == "floating":
            tl_loss = reconstruction_loss(recon_x[..., :3], actions[..., :3], reduction="mean")
            loss_dict["tl"] = tl_loss * self.cfg.tl_weight
            rot_loss = reconstruction_loss(
                recon_x[..., self.cfg.rot_begin_idx : self.cfg.rot_begin_idx + rot_dim],
                actions[..., self.cfg.rot_begin_idx : self.cfg.rot_begin_idx + rot_dim],
                reduction="mean",
            )
            loss_dict["rot"] = rot_loss * self.cfg.rot_weight

        if self.cfg.enable_kinematic_loss:
            geometry_x = rearrange(
                geometry_x, "(b n) (t d) -> b n t d", b=actions.shape[0], n=actions.shape[1], t=actions.shape[2]
            )
            if self.cfg.center_pointcloud:
                geometry_x = torch.cat([geometry_x[..., :3] + center[:, None], geometry_x[..., 3:]], dim=-1)
            if self.cfg.coupled_floating_base:
                geometry_x = self._couple_base(geometry_x)
            recon_link_pos = self.robot_helper.forward_kinematics_torch(geometry_x)[..., grasp_links, :3, 3]
            loss_dict["kinematic"] = (
                F.mse_loss(recon_link_pos, gt_link_pos, reduction="mean") * self.cfg.kinematic_weight
            )
            # FK clamps invalid joints; this term supplies the missing recovery gradient.
            joints = geometry_x[..., -self.robot_helper.num_actuated_joints :]
            limits = self.robot_helper.actuated_joint_limits_torch
            violation = (joints - joints.clamp(limits[:, 0], limits[:, 1])) / limits.diff(dim=-1).squeeze(-1).clamp_min(
                1e-3
            )
            loss_dict["joint_limits"] = violation.square().mean() * self.cfg.kinematic_weight
        if self.cfg.enable_penetration_loss and self._scene_geom is not None:
            # Penalize penetration on PRIOR-decoded grasps (matches get_action); GT reconstruction hides it.
            x_prior = self.generator.sample(scene_feats, use_mean=False, temperature=self.cfg.cvae_temperature)
            x_prior = rearrange(
                x_prior, "(b n) (t d) -> b n t d", b=actions.shape[0], n=actions.shape[1], t=actions.shape[2]
            )
            if self.cfg.center_pointcloud:
                x_prior = torch.cat([x_prior[..., :3] + center[:, None], x_prior[..., 3:]], dim=-1)
            if self.cfg.coupled_floating_base:
                x_prior = self._couple_base(x_prior)
            grasp_qpos = x_prior[..., 1, :]  # grasp phase: (b, n, dof)
            spheres = self.robot_helper.get_collision_spheres(
                link_poses=self.robot_helper.forward_kinematics_torch(grasp_qpos)
            )
            b, n, n_sphere = spheres.shape[0], spheres.shape[1], spheres.shape[-2]
            sdf, _, _ = self._scene_geom.query_target_object_signed_distances(
                spheres[..., :3].reshape(b, n * n_sphere, 3)
            )
            surface_distance = sdf.reshape(b, n, n_sphere) - spheres[..., 3]
            pen = (-surface_distance).clamp(min=0)
            excess = (pen - self.cfg.penetration_threshold).clamp(min=0)
            loss_dict["penetration"] = excess.max(dim=-1).values.mean() * self.cfg.penetration_weight
            if self.cfg.prior_fingertip_weight:
                tip_links = torch.tensor(
                    [self.robot_helper.link_names.index(name) for name in self.robot_config.fingertip_link_names],
                    device=grasp_qpos.device,
                )
                tip_mask = torch.isin(self.robot_helper.canonical_collision_spheres[1], tip_links)
                assert tip_mask.any()
                loss_dict["prior_fingertip"] = (
                    surface_distance[..., tip_mask].abs().mean() * self.cfg.prior_fingertip_weight
                )
        loss_dict["kl"] = kl_loss * self.cfg.kl_weight
        loss_dict["loss"] = sum(v for k, v in loss_dict.items() if k != "objectness_f1")
        return loss_dict

    def preprocess(self, initial: dict, current: dict, calibrated: bool) -> dict[str, torch.Tensor]:
        """One scene's input from `env.observe()` slices plus "qpos" (1, 1, D); calibrated views skip the initial."""
        device = next(self.parameters()).device
        if self.cfg.vision_encoder.startswith("rgb"):
            frames = (current,) if calibrated else (initial, current)
            obs = {"rgb": torch.cat([frame["rgb"].permute(0, 1, 4, 2, 3).flatten(1, 2) for frame in frames], 1)}
        else:
            frames = (current,) if calibrated or not self.cfg.object_points_only else (initial, current)
            points = torch.cat([frame["xyz"] for frame in frames], dim=1)
            segments = torch.cat([frame["seg"] for frame in frames], dim=1)
            point_indices = np.arange(points.shape[1])
            if self.cfg.object_points_only:
                point_indices = np.flatnonzero(segments.numpy().reshape(-1) == int(current["seg_id"][0]))
            replace = len(point_indices) <= self.cfg.num_points
            obs = {"pointcloud": points[:, np.random.choice(point_indices, self.cfg.num_points, replace=replace)]}
        if self.cfg.vision_encoder in ("pointnet_state", "rgb_resnet_views") or self.cfg.grasp_evaluator_anchor_state:
            assert not calibrated
            obs["qpos"] = torch.cat((initial["qpos"], current["qpos"]), dim=1)
            obs["camera_poses"] = torch.cat([frame["camera_poses"][:, :, None] for frame in (initial, current)], dim=1)
        return {key: value.to(device) for key, value in obs.items()}

    def get_action(
        self, obs: dict, num_samples_per_scene: int = 1, noise: torch.Tensor | None = None
    ) -> Float[torch.Tensor, "b n t d"]:
        """Sample actions; guided sampling accepts raw noise shaped (batch * samples, latent_dim)."""
        assert noise is None or self.cfg.grasp_evaluator_guidance_steps > 0
        # vision encoder
        scene_pts = obs["rgb"] if self.cfg.vision_encoder.startswith("rgb") else obs["pointcloud"]
        scene_pts = scene_pts.float()
        if self.cfg.canonical_yaw:
            yaw_center = scene_pts.mean(dim=1, keepdim=True)
            yaw_rotation = self._canonical_yaw_rotation(scene_pts)
            scene_pts = (yaw_rotation[:, None] @ (scene_pts - yaw_center).unsqueeze(-1)).squeeze(-1) + yaw_center
        if self.cfg.center_pointcloud:
            center = scene_pts.mean(dim=1, keepdim=True)
            scene_pts = scene_pts - center
        if self.cfg.vision_encoder.startswith("rgb"):
            scene_pts = preprocess_rgb(scene_pts, self.cfg.vision_encoder)
        else:
            scene_pts = scene_pts * self.cfg.pointcloud_scale
        if self.cfg.grasp_evaluator_local_features:
            assert self.cfg.vision_encoder_global_feature
            assert self.cfg.vision_encoder in ("pointnet", "pointnet_state", "rgb_resnet_views")
            captured = []
            module = (
                self.vision_encoder.avgpool
                if self.cfg.vision_encoder == "rgb_resnet_views"
                else self.vision_encoder.conv3
            )
            hook = module.register_forward_hook(lambda module, arguments, output: captured.append(output))
        scene_feats = self.vision_encoder(scene_pts)  # (B, D) or (B, N, D)
        if self.cfg.grasp_evaluator_local_features:
            hook.remove()
            assert len(captured) == 1
            if self.cfg.vision_encoder == "rgb_resnet_views":
                tokens = rearrange(captured[0], "(b v) d h w -> b (v h w) d", b=len(obs["rgb"]))
                assert tokens.shape[1] == 64
                position = torch.arange(64, device=tokens.device, dtype=tokens.dtype)[:, None]
                frequency = torch.exp(
                    -torch.log(tokens.new_tensor(10000.0))
                    * torch.arange(0, tokens.shape[-1], 2, device=tokens.device)
                    / tokens.shape[-1]
                )
                tokens = tokens.clone()
                tokens[..., 0::2] += torch.sin(position * frequency)
                tokens[..., 1::2] += torch.cos(position * frequency)
            else:
                points = captured[0].transpose(1, 2)
                indices = torch.linspace(0, points.shape[1] - 1, 64, device=points.device).long()
                tokens = self.vision_encoder.feat_linear(points[:, indices])
            obs["critic_tokens"] = tokens
        if self.cfg.vision_encoder == "rgb_resnet_views":
            scene_feats = rearrange(scene_feats, "(b v) d -> b (v d)", b=len(obs["rgb"]))
        if self.cfg.vision_encoder in ("pointnet_state", "rgb_resnet_views"):
            scene_feats = self.rgb_conditioning(
                torch.cat((scene_feats, obs["qpos"].flatten(1), obs["camera_poses"].flatten(1)), dim=-1)
            )

        if not self.cfg.vision_encoder_global_feature:  # local feature
            if "conditioned_point_indices" in obs:  # conditioned on point indices
                scene_feats = torch.gather(
                    scene_feats,
                    dim=-2,
                    index=repeat(obs["conditioned_point_indices"], "b n -> b n d", d=scene_feats.shape[-1]),
                )
            else:
                # graspness
                graspness = self.graspness_decoder(scene_feats).squeeze(-1)

                if self.cfg.enable_objectness_prediction:
                    # segmentation / objectness
                    seg_prob = self.objectness_decoder(scene_feats).squeeze(-1)
                    graspness = graspness.masked_fill(seg_prob <= 0.5, -torch.inf)

                # gather local features
                max_indices = torch.topk(graspness, k=num_samples_per_scene, dim=-1).indices
                scene_feats = torch.gather(
                    scene_feats, dim=-2, index=max_indices.unsqueeze(-1).expand(-1, -1, scene_feats.shape[-1])
                ).squeeze(-2)

        # Action sampling
        if scene_feats.ndim == 2:
            scene_feats = repeat(scene_feats, "b d -> (b n) d", n=num_samples_per_scene)
        else:
            scene_feats = rearrange(scene_feats, "b n d -> (b n) d")
        obs["scene_features"] = scene_feats  # what `select` scores the actions on
        if self.cfg.use_mean and num_samples_per_scene > 1:
            log_once(f"Sampling {num_samples_per_scene} actions from the generator mean")
        if self.cfg.grasp_evaluator_guidance_steps > 0:
            anchor_state = None
            if self.cfg.grasp_evaluator_anchor_state:
                anchor_state = self.grasp_evaluator.encode_action(
                    obs["qpos"][:, -1:].expand(-1, self.cfg.num_queries, -1).flatten(1)
                ).unflatten(-1, (self.cfg.num_queries, -1))[:, :1]
            latent = scene_feats.new_zeros(scene_feats.shape[0], self.generator.latent_dim)
            if noise is not None:
                assert noise.shape == latent.shape and noise.dtype == latent.dtype and noise.device == latent.device
            if self.cfg.cvae_learned_prior:
                mean, log_var = self.generator.prior(scene_feats)
                latent = (
                    mean
                    if self.cfg.use_mean
                    else mean
                    + (torch.randn_like(mean) if noise is None else noise)
                    * (0.5 * log_var).exp()
                    * self.cfg.cvae_temperature
                )
            elif not self.cfg.use_mean:
                latent = (torch.randn_like(latent) if noise is None else noise) * self.cfg.cvae_temperature
            actions = self.grasp_evaluator.refine(
                self.generator,
                latent,
                scene_feats,
                self.cfg.grasp_evaluator_guidance_steps,
                self.cfg.grasp_evaluator_guidance_radius,
                obs["critic_tokens"] if self.cfg.grasp_evaluator_local_features else None,
                anchor_state=anchor_state,
            )
        else:
            actions = self.generator.sample(
                scene_feats, use_mean=self.cfg.use_mean, temperature=self.cfg.cvae_temperature
            )

        actions = rearrange(actions, "(b n) (t d) -> b n t d", n=num_samples_per_scene, t=self.cfg.num_queries)
        if self.cfg.center_pointcloud:
            actions = torch.cat([actions[..., :3] + center[:, None], actions[..., 3:]], dim=-1)
        if self.cfg.coupled_floating_base:
            actions = self._couple_base(actions)
        if self.cfg.convert_rxyz_to_rot6d:
            matrices = rotation_6d_to_matrix(actions[..., self.cfg.rot_begin_idx : self.cfg.rot_begin_idx + 6])
            if self.cfg.canonical_yaw:
                inverse = yaw_rotation.transpose(-1, -2)[:, None, None]
                translation = (inverse @ (actions[..., :3] - yaw_center[:, None]).unsqueeze(-1)).squeeze(-1)
                actions = torch.cat([translation + yaw_center[:, None], actions[..., 3:]], dim=-1)
                matrices = inverse @ matrices
            euler_angles = (
                matrix_to_nearest_euler_angles(matrices)
                if self.cfg.analytic_euler
                else matrix_to_continuous_euler_angles(matrices, "rxyz")
            )
            actions = torch.cat(
                [actions[..., : self.cfg.rot_begin_idx], euler_angles, actions[..., self.cfg.rot_begin_idx + 6 :]],
                dim=-1,
            )
        return actions

    def select(self, obs: dict, actions: Float[torch.Tensor, "b n t d"]) -> torch.Tensor:
        """Each scene's best proposal index by the grasp evaluator, on the features `get_action` left in `obs`."""
        anchor_state = None
        if self.cfg.grasp_evaluator_anchor_state:
            anchor_state = self.grasp_evaluator.encode_action(
                obs["qpos"][:, -1:].expand(-1, self.cfg.num_queries, -1).flatten(1)
            ).unflatten(-1, (self.cfg.num_queries, -1))[:, :1]
        scores = self.grasp_evaluator(
            obs["scene_features"].unflatten(0, actions.shape[:2]),
            actions.flatten(2),
            obs["critic_tokens"] if self.cfg.grasp_evaluator_local_features else None,
            anchor_state=anchor_state,
        )
        return scores.flatten(1).argmax(1)


def build_grasp_policy(
    cfg: GraspPolicyConfig,
    robot_config: RobotConfig,
    device: Device = "cpu",
    policy_cls: Callable[..., GraspPolicy] = GraspPolicy,
) -> GraspPolicy:
    """Build `policy_cls` (a script may pass a subclass); `action_dim` is the robot DOF plus an SE3 head if floating."""
    action_dim = GraspRobot.from_config(robot_config).num_actuated_joints
    if robot_config.base_type == "floating":
        action_dim += 3 + (6 if cfg.convert_rxyz_to_rot6d else 3)
    return policy_cls(cfg, action_dim, robot_config).to(device)
