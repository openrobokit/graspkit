"""Condition-driven ARX X5 grasp generator: GraspPolicy's CVAE trained on hindsight conditions, parts dropped at random.

Condition v0 (the point cloud's frame, per grasp), COND_DIM numbers, see `hindsight`:
  pad_a (3), pad_b (3)      where the two jaw pads should touch (a contact segment); order-free (the jaw is symmetric)
  approach (3), cone (1)    the direction the hand travels toward the object and the cone's half-angle (rad)
  grasp_type                one-hot, TYPES (from the approach's tilt)
  masks                     has_contact, has_approach, has_type: absent parts are zeroed and masked
v2 (`encode_point`): a single grasp point + optional approach.
"""

import math
from dataclasses import replace
from functools import partial

import torch
from einops import rearrange, repeat
from robokit.xform.utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_nearest_euler_angles
from torch import nn
from torch.types import Device

from graspkit.configs.experiment.arx_x5_gripper import arx_x5_diverse
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.models.cvae import CVAE
from graspkit.models.grasp_policy import GraspPolicy, build_grasp_policy
from graspkit.models.mlp import MLP

TCP = 0.145
TYPES = ("top", "tilted", "side", "rim", "axis")  # v0 layout: pads 0:6, approach 6:9, cone 9, types 10:15, masks 15:18
T0, M0, COND_DIM = 10, 15, 18
COND_FEAT = 64
DROP = 0.3  # per-part dropout in training

# v2: one grasp point (a VLM's click lifted to 3D): point (3), approach (3), cone (1), masks (has_point, has_approach).
V2_DIM = 9


def encode_point(point=None, approach=None, cone_deg: float = 30.0, device="cpu") -> torch.Tensor:
    """One v2 condition: grasp here (`point`, the cloud's frame), optionally travelling along `approach`."""
    c = torch.zeros(V2_DIM, device=device)
    if point is not None:
        c[0:3], c[7] = torch.as_tensor(point, dtype=torch.float32), 1.0
    if approach is not None:
        a = torch.as_tensor(approach, dtype=torch.float32)
        c[3:6], c[6], c[8] = a / a.norm(), math.radians(cone_deg), 1.0
    return c


def grasp_centre(grasp: torch.Tensor) -> torch.Tensor:
    """(..., >=7) floating qpos -> (..., 3) the pads' midpoint (the TCP)."""
    R = euler_angles_to_matrix(grasp[..., 3:6], "rxyz")
    return grasp[..., :3] + TCP * R[..., :, 0]


def hindsight_point(grasp: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """(B, n, >=7) grasps, (B, N, 3) cloud -> (B, n, V2_DIM): a contact-patch point, noised like a click 1-2 cm off."""
    R = euler_angles_to_matrix(grasp[..., 3:6], "rxyz")
    x = R[..., :, 0]
    mid = grasp[..., :3] + TCP * x
    d = torch.cdist(mid, pts)  # (B, n, N)
    r = d.min(-1, keepdim=True).values + 0.02 * torch.rand(*d.shape[:-1], 1, device=d.device)
    pick = (torch.rand_like(d) * (d <= r)).argmax(-1)  # a uniform point within r of the grasp centre
    c = torch.gather(pts, 1, pick[..., None].expand(-1, -1, 3))
    c = c + (torch.randn_like(c) * 0.005).clamp(-0.012, 0.012)
    cone = torch.deg2rad(15 + 30 * torch.rand(*grasp.shape[:-1], 1, device=grasp.device))
    return torch.cat([c, x, cone, torch.ones(*grasp.shape[:-1], 2, device=grasp.device)], -1)


def _transform2(cond, rot, center, scale):
    c, ap = cond[..., 0:3], cond[..., 3:6]
    if rot is not None:
        c = ((c - center) @ rot.transpose(1, 2)) + center
        ap = ap @ rot.transpose(1, 2)
    return torch.cat([(c - center) * scale * cond[..., 7:8], ap, cond[..., 6:]], -1)


def _drop2(cond):
    keep = (torch.rand(*cond.shape[:-1], 2, device=cond.device) > DROP).float()
    out = cond.clone()
    out[..., 0:3] *= keep[..., 0:1]
    out[..., 3:7] *= keep[..., 1:2]
    out[..., 7:9] = keep
    return out


def hindsight(grasp: torch.Tensor) -> torch.Tensor:
    """(..., >=7) qpos [xyz, euler rxyz, joint7] -> (..., COND_DIM): pads TCP -+ y * joint7, random 15-45 deg cone."""
    R = euler_angles_to_matrix(grasp[..., 3:6], "rxyz")
    x, y = R[..., :, 0], R[..., :, 1]
    tcp = grasp[..., :3] + TCP * x
    q = grasp[..., 6:7].clamp(0.002, 0.044)
    tilt = torch.rad2deg(torch.arccos((-x[..., 2]).clamp(-1, 1)))
    t = torch.where(tilt < 25, 0, torch.where(tilt < 60, 1, 2))
    cone = torch.deg2rad(15 + 30 * torch.rand(grasp.shape[:-1], device=grasp.device))
    masks = torch.ones(*grasp.shape[:-1], COND_DIM - M0, device=grasp.device)
    return torch.cat(
        [tcp - y * q, tcp + y * q, x, cone[..., None], nn.functional.one_hot(t, len(TYPES)).float(), masks], -1
    )


def _transform(cond: torch.Tensor, rot: torch.Tensor | None, center: torch.Tensor, scale: float) -> torch.Tensor:
    """Condition (B, n, C) world -> policy frame: rotated about `center`, centred, scaled; masked parts stay zero."""
    pa, pb, ap = cond[..., 0:3], cond[..., 3:6], cond[..., 6:9]
    if rot is not None:
        pa = ((pa - center) @ rot.transpose(1, 2)) + center
        pb = ((pb - center) @ rot.transpose(1, 2)) + center
        ap = ap @ rot.transpose(1, 2)
    m = cond[..., M0 : M0 + 1]
    pa, pb = (pa - center) * scale * m, (pb - center) * scale * m
    return torch.cat([pa, pb, ap, cond[..., 9:]], -1)


def _drop(cond: torch.Tensor) -> torch.Tensor:
    keep = (torch.rand(*cond.shape[:-1], COND_DIM - M0, device=cond.device) > DROP).float() * cond[..., M0:]
    out = cond.clone()
    out[..., 0:6] *= keep[..., 0:1]
    out[..., 6:10] *= keep[..., 1:2]
    out[..., T0:M0] *= keep[..., 2:3]
    out[..., M0:] = keep
    swap = torch.rand(*cond.shape[:-1], 1, device=cond.device) < 0.5  # pads are unordered
    out[..., 0:6] = torch.where(swap, torch.cat([out[..., 3:6], out[..., 0:3]], -1), out[..., 0:6])
    out[..., 0:6] = out[..., 0:6] + torch.randn_like(out[..., 0:6]) * 0.004 * keep[..., 0:1]  # coarse contact marks
    return out


LOCAL_FEAT = 128
LOCAL_SCALE = 2.0  # the policy frame is x10 (1 = 10 cm); local patches x2 more, so 5 cm -> 1


class LocalPatchEncoder(nn.Module):
    """v2 local pathway: a small PointNet on the K points nearest c, the few cm the jaw closes on; zero without c."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.point_mlp = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 256))
        self.out = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, LOCAL_FEAT))

    def forward(self, pts: torch.Tensor, c: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
        """pts (B, N, 3), c (B, n, 3) policy frame, has (B, n) -> (B * n, LOCAL_FEAT)."""
        idx = torch.cdist(c, pts).topk(self.k, dim=-1, largest=False).indices  # (B, n, k)
        n = c.shape[1]
        nb = torch.gather(pts[:, None].expand(-1, n, -1, -1), 2, idx[..., None].expand(-1, -1, -1, 3))
        f = self.point_mlp((nb - c[:, :, None]) * LOCAL_SCALE).max(2).values  # (B, n, 256)
        return rearrange(self.out(f) * has[..., None], "b n d -> (b n) d")


class CondGraspPolicy(GraspPolicy):
    def __init__(
        self, cfg: GraspPolicyConfig, action_dim: int, robot_config: RobotConfig, version: int, local_k: int = 0
    ):
        """`version`: 0 = pads/approach/cone/type, 2 = grasp point + optional approach; `local_k`: v2 patch size."""
        super().__init__(cfg, action_dim, robot_config)
        assert cfg.vision_encoder_global_feature and cfg.center_pointcloud and not cfg.canonical_yaw
        self.v = version
        self.cond_encoder = MLP(
            input_dim=V2_DIM if self.v == 2 else COND_DIM, output_dim=COND_FEAT, hidden_dims=[128, 128]
        )
        self.local_k = local_k if self.v == 2 else 0
        if self.local_k:
            self.local_encoder = LocalPatchEncoder(self.local_k)
        self.cvae = CVAE(
            x_dim=self.action_dim * cfg.num_queries,
            cond_dim=cfg.vision_encoder_feature_dim + COND_FEAT + (LOCAL_FEAT if self.local_k else 0),
            latent_dim=cfg.cvae_latent_dim,
            hidden_dims=[cfg.cvae_hidden_dim] * 3,
            learned_prior=cfg.cvae_learned_prior,
        )

    def forward(self, obs: dict, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """GraspPolicy.forward for the floating, global-feature, centred config, with the hindsight condition."""
        cfg = self.cfg
        assert not cfg.enable_penetration_loss and not cfg.cvae_learned_prior
        if self.v == 2:  # the grasp point from the target's own (augmented) cloud
            cond = _drop2(hindsight_point(actions[..., 1, :], obs["pointcloud"]))
        else:  # (B, n, C) world frame, from the ground-truth grasp
            cond = _drop(hindsight(actions[..., 1, :]))
        pts = obs["pointcloud"]
        center = pts.mean(dim=1, keepdim=True)
        rot = None
        if cfg.train_yaw_augmentation:
            yaw = pts.new_zeros(len(pts), 3)
            yaw[:, 2] = torch.rand(len(pts), device=pts.device) * (2 * torch.pi)
            rot = euler_angles_to_matrix(yaw, "rxyz")
            pts = (rot[:, None] @ (pts - center).unsqueeze(-1)).squeeze(-1) + center
            tl = (rot[:, None, None] @ (actions[..., :3] - center[:, None]).unsqueeze(-1)).squeeze(-1) + center[:, None]
            R = rot[:, None, None] @ euler_angles_to_matrix(actions[..., 3:6], "rxyz")
            actions = torch.cat([tl, matrix_to_nearest_euler_angles(R), actions[..., 6:]], dim=-1)
        scene_feats = self.vision_encoder((pts - center) * cfg.pointcloud_scale)
        feats = self._cond_feats(repeat(scene_feats, "b d -> (b n) d", n=actions.shape[1]),
                                 self._tf(cond, rot, center), (pts - center) * cfg.pointcloud_scale)  # fmt: skip
        loss = {}
        if cfg.enable_kinematic_loss:
            if self.robot_helper is None or self.robot_helper.device != actions.device:
                from graspkit.robot.grasp_robot import GraspRobot

                self.robot_helper = GraspRobot.from_config(self.robot_config, device=actions.device)
            with torch.no_grad():
                gt_link = self.robot_helper.forward_kinematics_torch(actions)[..., :3, 3].clone()
        if cfg.convert_rxyz_to_rot6d:
            r6 = matrix_to_rotation_6d(
                euler_angles_to_matrix(actions[..., cfg.rot_begin_idx : cfg.rot_begin_idx + 3], "rxyz")
            )
            actions = torch.cat([actions[..., : cfg.rot_begin_idx], r6, actions[..., cfg.rot_begin_idx + 3 :]], dim=-1)
        enc = torch.cat([actions[..., :3] - center[:, None], actions[..., 3:]], dim=-1)
        recon, mu, logv, _ = self.cvae(rearrange(enc, "b n t d -> (b n) (t d)"), feats)
        recon = rearrange(recon, "(b n) (t d) -> b n t d", b=actions.shape[0], n=actions.shape[1], t=actions.shape[2])
        recon = torch.cat([recon[..., :3] + center[:, None], recon[..., 3:]], dim=-1)
        if cfg.coupled_floating_base:
            recon = self._couple_base(recon)
        kl = (-0.5 * (1 + logv - mu.pow(2) - logv.exp())).sum(-1).mean()
        rd = 6 if cfg.convert_rxyz_to_rot6d else 3
        F_ = nn.functional
        loss["qpos"] = (
            F_.l1_loss(recon[..., cfg.rot_begin_idx + rd :], actions[..., cfg.rot_begin_idx + rd :]) * cfg.qpos_weight
        )
        loss["tl"] = F_.l1_loss(recon[..., :3], actions[..., :3]) * cfg.tl_weight
        loss["rot"] = F_.l1_loss(recon[..., cfg.rot_begin_idx : cfg.rot_begin_idx + rd],
                                 actions[..., cfg.rot_begin_idx : cfg.rot_begin_idx + rd]) * cfg.rot_weight  # fmt: skip
        if cfg.enable_kinematic_loss:
            loss["kinematic"] = (
                F_.mse_loss(self.robot_helper.forward_kinematics_torch(recon)[..., :3, 3], gt_link)
                * cfg.kinematic_weight
            )
        loss["kl"] = kl * cfg.kl_weight
        loss["loss"] = sum(loss.values())
        return loss

    def _tf(self, cond, rot, center):
        if self.v == 2:
            return _transform2(cond, rot, center, self.cfg.pointcloud_scale)
        return _transform(cond, rot, center, self.cfg.pointcloud_scale)

    def _cond_feats(
        self, scene_feats: torch.Tensor, cond_policy_frame: torch.Tensor, pts_policy_frame=None
    ) -> torch.Tensor:
        out = [scene_feats, self.cond_encoder(rearrange(cond_policy_frame, "b n c -> (b n) c"))]
        if self.local_k:  # v2 layout: c at 0:3, has_point at 7
            out.append(self.local_encoder(pts_policy_frame, cond_policy_frame[..., 0:3], cond_policy_frame[..., 7]))
        return torch.cat(out, -1)

    @torch.no_grad()
    def get_action(self, obs: dict, num_samples_per_scene: int = 1) -> torch.Tensor:
        """obs["pointcloud"] (B, N, 3); obs["condition"] (B, C) or (B, n, C) world frame (default: none)."""
        pts = obs["pointcloud"].float()
        B = len(pts)
        cond = obs.get("condition")
        if cond is None:
            cond = pts.new_zeros(B, V2_DIM if self.v == 2 else COND_DIM)
        if cond.ndim == 2:
            cond = repeat(cond, "b c -> b n c", n=num_samples_per_scene)
        center = pts.mean(dim=1, keepdim=True)
        scene_feats = self.vision_encoder((pts - center) * self.cfg.pointcloud_scale)
        feats = self._cond_feats(repeat(scene_feats, "b d -> (b n) d", n=num_samples_per_scene),
                                 self._tf(cond.float(), None, center), (pts - center) * self.cfg.pointcloud_scale)  # fmt: skip
        actions = self.cvae.sample(feats, use_mean=self.cfg.use_mean, temperature=self.cfg.cvae_temperature)
        actions = rearrange(actions, "(b n) (t d) -> b n t d", n=num_samples_per_scene, t=self.cfg.num_queries)
        actions = torch.cat([actions[..., :3] + center[:, None], actions[..., 3:]], dim=-1)
        if self.cfg.coupled_floating_base:
            actions = self._couple_base(actions)
        if self.cfg.convert_rxyz_to_rot6d:
            m = rotation_6d_to_matrix(actions[..., self.cfg.rot_begin_idx : self.cfg.rot_begin_idx + 6])
            actions = torch.cat([actions[..., : self.cfg.rot_begin_idx], matrix_to_nearest_euler_angles(m),
                                 actions[..., self.cfg.rot_begin_idx + 6 :]], -1)  # fmt: skip
        return actions


# Conditional generators (lineage in README.md); point_local starts as point_v2_20k via cond_init --pad (zero-padded).
cond = replace(arx_x5_diverse, name="arx_x5_cond", processed_data_path="runs/arx_diverse/sim_data_mix2.h5")
point_local = replace(
    cond,
    name="arx_x5_point_local",
    trainer=replace(cond.trainer, total_iters=12000, save_freq=4000),
    processed_data_path="runs/arx_diverse/sim_data_mix3.h5",
    scene_description_path="runs/arx_diverse/scenes/scenes_mix3.msgpack",  # + the part scenes' duplicates (label_diverse mix3)
)
EXPERIMENTS = {  # name -> (config, condition version, local patch points)
    "arx_x5_cond": (cond, 0, 0),
    "arx_x5_point": (replace(cond, name="arx_x5_point"), 2, 0),
    "arx_x5_point_local": (point_local, 2, 64),
}


def build_cond_policy(name: str, device: Device = "cpu") -> tuple[CondGraspPolicy, ExperimentConfig]:
    cfg, version, local_k = EXPERIMENTS[name]
    policy_cls = partial(CondGraspPolicy, version=version, local_k=local_k)
    return build_grasp_policy(cfg.agent, cfg.robot_config, device, policy_cls), cfg
