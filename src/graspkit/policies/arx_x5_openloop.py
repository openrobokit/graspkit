"""Open-loop ARX X5 grasp policy inference (`arx_x5_gripper_dex1b_mini_tabletop`); all I/O in one z-up world frame.

Usage:
    from graspkit.policies.arx_x5_openloop import plan_openloop_grasps
    grasps = plan_openloop_grasps(object_points, table_z, obstacle_points, "arx_x5_openloop.pt")
    T_link6, q, T_pre = grasps[0]
"""

import os
import zlib
from functools import lru_cache

import numpy as np
import pytorch_kinematics as pk
import torch
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_axis_angle
from robokit.assets.robots.assembly import arx_x5
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix
from torch import nn

from graspkit.configs.experiment import build_experiment_config
from graspkit.models.grasp_policy import GraspPolicy, build_grasp_policy
from graspkit.policies.cond_policy import EXPERIMENTS, build_cond_policy, encode_point
from graspkit.robot.grasp_robot import GraspRobot

EXPERIMENT = "arx_x5_gripper_dex1b_mini_tabletop"
OPEN_Q = 0.044
# conditional checkpoints by their condition encoder's input width (cond_policy: v2 9, v0 18)
COND_EXPERIMENTS = {9: "arx_x5_point", 18: "arx_x5_cond"}
NUM_POINTS = 512  # ranker input points
BASE_XY = {"left": (-0.30, -0.45), "right": (0.30, -0.45)}  # RoboDojo arm bases (yaw +90 deg) on the table top
PRE_DIST, LIFT = 0.08, 0.10


class GraspRanker(nn.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, width), nn.ReLU()
        )
        self.head = nn.Sequential(nn.Linear(width + 4, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, local: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """local (B, N, 3) object points in link6 frames (x10); table (B, 4) -> success logits (B,)."""
        return self.head(torch.cat([self.point(local).amax(1), table], -1))[:, 0]


def featurize(pts: torch.Tensor, T: torch.Tensor, table_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pts (B, N, 3) world, T (B, 4, 4) link6 poses, table_z (B,) -> the ranker's (local, table) inputs."""
    R, t = T[:, :3, :3], T[:, :3, 3]
    local = (pts - t[:, None]) @ R * 10.0
    up = R[:, 2, :]  # world up in link6 coordinates: R^T e_z = row 2 of R
    return local, torch.cat([up, (t[:, 2:3] - table_z[:, None]) * 10.0], -1)


@lru_cache(maxsize=2)
def load_ranker(path: str, device: str = "cuda") -> GraspRanker:
    model = GraspRanker()
    model.load_state_dict(torch.load(path, map_location=device)["ranker"])
    return model.to(device).eval()


@torch.no_grad()
def score_grasps(path: str, pts: torch.Tensor, T: torch.Tensor, table_z: float) -> torch.Tensor:
    """(N, 3) world object points, (n, 4, 4) link6 poses -> (n,) success logits (higher = better)."""
    model = load_ranker(path, str(pts.device))
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randint(len(pts), (NUM_POINTS,), generator=g).to(pts.device)
    P = pts[idx][None].expand(len(T), -1, -1)
    local, table = featurize(P, T, torch.full((len(T),), float(table_z), device=pts.device))
    return model(local, table)


@lru_cache(maxsize=2)
def _chain(device: str):
    return pk.build_serial_chain_from_urdf(arx_x5.URDF_PATH.read_bytes(), "link6").to(device=device)


def _ik(chain, target: torch.Tensor, q: torch.Tensor, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    lim = torch.tensor([[-10.0, 10.0]] * 5 + [[-3.14, 3.14]], device=q.device)
    eye = torch.eye(6, device=q.device)
    for _ in range(iters):
        cur = chain.forward_kinematics(q).get_matrix()
        e = torch.cat([target[:, :3, 3] - cur[:, :3, 3],
                       matrix_to_axis_angle(target[:, :3, :3] @ cur[:, :3, :3].transpose(1, 2))], 1)  # fmt: skip
        J = chain.jacobian(q)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + 2.5e-3 * eye, e[..., None]))[..., 0]
        q = torch.minimum(torch.maximum(q + dq, lim[:, 0]), lim[:, 1])
    cur = chain.forward_kinematics(q).get_matrix()
    ep = (target[:, :3, 3] - cur[:, :3, 3]).norm(dim=1)
    er = matrix_to_axis_angle(target[:, :3, :3] @ cur[:, :3, :3].transpose(1, 2)).norm(dim=1)
    return q, (ep < 0.002) & (er < np.radians(1.5))


@torch.no_grad()
def reachable(T: torch.Tensor, table_z: float) -> torch.Tensor:
    """(n, 4, 4) world link6 grasp poses -> (n,) bool: either RoboDojo arm reaches pre-grasp, grasp and lift."""
    dev = T.device
    chain = _chain(str(dev))
    n = len(T)
    ok_any = torch.zeros(n, dtype=torch.bool, device=dev)
    seeds = torch.cat([torch.zeros(1, 6, device=dev),
                       (torch.rand(7, 6, device=dev, generator=torch.Generator(dev).manual_seed(0)) * 2 - 1) * np.pi])  # fmt: skip
    for bx, by in BASE_XY.values():
        Wb = torch.eye(4, device=dev)
        Wb[:3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], device=dev)
        Wb[:3, 3] = torch.tensor([bx, by, table_z], device=dev)
        bT = torch.linalg.inv(Wb) @ T
        pre = bT.clone()
        pre[:, :3, 3] -= PRE_DIST * bT[:, :3, 0]
        q, ok = _ik(chain, pre.repeat_interleave(8, 0), seeds.repeat(n, 1), 60)
        ok = ok.view(n, 8)
        q = q.view(n, 8, 6)[torch.arange(n), ok.float().argmax(1)]
        ok = ok.any(1)
        q, c1 = _ik(chain, bT, q, 20)
        lift = bT.clone()
        lift[:, 2, 3] += LIFT  # base z is world z
        _, c2 = _ik(chain, lift, q, 20)
        ok_any |= ok & c1 & c2
    return ok_any


def experiment_of(state: dict) -> str:
    """The experiment config a checkpoint's weights belong to (env GRASP_POLICY_EXP overrides)."""
    if os.environ.get("GRASP_POLICY_EXP"):
        return os.environ["GRASP_POLICY_EXP"]
    if any(k.startswith("local_encoder.") for k in state):
        return "arx_x5_point_local"
    w = [v for k, v in state.items() if k.startswith("cond_encoder.") and k.endswith("weight") and v.ndim == 2]
    return COND_EXPERIMENTS[int(w[0].shape[1])] if w else EXPERIMENT


@lru_cache(maxsize=2)
def load_policy(ckpt_path: str, device: str = "cuda") -> tuple[GraspPolicy, GraspRobot]:
    state = torch.load(ckpt_path, map_location=device)["agent"]
    name = experiment_of(state)
    if name in EXPERIMENTS:
        policy, cfg = build_cond_policy(name, device)
    else:
        cfg = build_experiment_config(name)
        policy = build_grasp_policy(cfg.agent, cfg.robot_config, device)
    policy.load_state_dict(state)
    robot = GraspRobot.from_config(cfg.robot_config, load_collision_spheres=True, device=device)
    return policy.eval(), robot


@torch.no_grad()
def plan_openloop_grasps(
    object_points: np.ndarray,
    table_z: float,
    obstacle_points: np.ndarray | None,
    ckpt_path: str,
    num_samples: int = 64,
    num_points: int = 1024,
    pre_dist: float = 0.08,
    margin: float = 0.003,
    seed: int = 0,
    device: str = "cuda",
    rank: str | None = None,
    condition: dict | None = None,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """World target points, table top z, other points -> `num_samples` (T_link6, q, T_pre) grasps ranked best first."""
    policy, robot = load_policy(ckpt_path, device)
    seed = (seed * 1000003 + zlib.crc32(np.round(np.asarray(object_points, np.float64), 4).tobytes())) % 2**31
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.choice(len(object_points), num_points, replace=len(object_points) < num_points)
    pts = torch.as_tensor(object_points[idx], dtype=torch.float32, device=device)
    obs = {"pointcloud": pts[None]}
    c = None
    if condition is not None:
        if getattr(policy, "v", None) != 2:
            raise ValueError(f"{ckpt_path} takes no point condition (not an arx_x5_point checkpoint)")
        c = (
            None
            if condition.get("point") is None
            else torch.as_tensor(condition["point"], dtype=torch.float32, device=device)
        )
        obs["condition"] = encode_point(c, condition.get("approach"), float(condition.get("cone_deg") or 30.0),
                                           device=device)[None]  # fmt: skip
    if os.environ.get("GRASP_POLICY_TEMP"):  # CVAE prior temperature (default 1; the parts diag used 0.6)
        policy.cfg.cvae_temperature = float(os.environ["GRASP_POLICY_TEMP"])
    actions = policy.get_action(obs, num_samples_per_scene=num_samples)[0]  # (n, 3, 7)
    grasp = actions[:, 1]  # pre (0) is the same pose with the jaws open, post (2) closed
    T = torch.eye(4, device=device).repeat(num_samples, 1, 1)
    T[:, :3, :3] = euler_angles_to_matrix(grasp[:, 3:6], "rxyz")
    T[:, :3, 3] = grasp[:, :3]
    T_pre = T.clone()
    T_pre[:, :3, 3] -= pre_dist * T[:, :3, 0]

    # open-jaw spheres at the grasp and at the pre-grasp: (n, 2, S, 4)
    poses = torch.stack([T, T_pre], 1)
    qpos = torch.cat([poses[..., :3, 3], grasp[:, None, 3:6].expand(-1, 2, -1)], -1)
    qpos = torch.cat([qpos, torch.full_like(qpos[..., :1], OPEN_Q)], -1)
    spheres = robot.get_collision_spheres(robot.forward_kinematics_torch(qpos))
    depth = (table_z - (spheres[..., 2] - spheres[..., 3])).clamp(min=0).amax((-1, -2))
    others = [pts] + ([] if obstacle_points is None or len(obstacle_points) == 0 else [
        torch.as_tensor(obstacle_points, dtype=torch.float32, device=device)])  # fmt: skip
    sph = spheres[..., :3].reshape(num_samples, -1, 3)
    for (
        cloud
    ) in others:  # nearest cloud point per sphere, in chunks (a big obstacle cloud x 256 samples ran out of memory)
        step = max(1, int(2e8 // max(1, sph.shape[1] * len(cloud))))
        d = torch.cat([torch.cdist(sph[i : i + step], cloud).amin(-1) for i in range(0, num_samples, step)])
        pen = (spheres[..., 3].reshape(num_samples, -1) - d).clamp(min=0).amax(-1)
        depth = torch.maximum(depth, pen)
    local = (pts[None] - T[:, None, :3, 3]) @ T[:, :3, :3]  # object points in each link6 frame
    inside = (local[..., 0] > 0.125) & (local[..., 0] < 0.16) & (local[..., 2].abs() < 0.012)
    inside &= local[..., 1].abs() < OPEN_Q
    order = torch.argsort((depth > margin) * num_points * 2 - inside.sum(-1), stable=True)
    rank = rank or os.environ.get("GRASP_POLICY_RANK", "helper")
    near = 0.0  # the point condition's distance term (logits): 1 per cm from the pads' midpoint to c
    if c is not None:
        dist = (T[:, :3, 3] + 0.145 * T[:, :3, 0] - c).norm(dim=1)
        near = dist / 0.01 * float(os.environ.get("GRASP_POINT_DIST_W", "1.0"))
        if os.environ.get("GRASP_POINT_RADIUS"):  # grasps within this many cm of c first, then the ranker
            near = near - 100.0 * (dist <= float(os.environ["GRASP_POINT_RADIUS"]) / 100)
    if rank == "none":
        order = torch.arange(num_samples, device=device)
    if rank == "ranker":
        order = torch.argsort(-(score_grasps(os.environ["GRASP_POLICY_RANKER"], pts, T, table_z) - near), stable=True)
    if rank == "ranker_reach":  # reachable by either RoboDojo arm first (`reachable`), then the ranker's score
        score = score_grasps(os.environ["GRASP_POLICY_RANKER"], pts, T, table_z)
        order = torch.argsort(-(score - near + 1e3 * reachable(T, table_z).float()), stable=True)
    if os.environ.get("GRASP_POLICY_LOG"):  # diagnostics: every call's inputs and ranked outputs, one npz per call
        log_dir = os.environ["GRASP_POLICY_LOG"]
        os.makedirs(log_dir, exist_ok=True)
        k = len([f for f in os.listdir(log_dir) if f.startswith(f"{os.getpid()}_")])
        np.savez_compressed(
            os.path.join(log_dir, f"{os.getpid()}_{k:03d}.npz"),
            object_points=np.asarray(object_points, np.float32),
            obstacle_points=np.zeros((0, 3), np.float32) if obstacle_points is None else np.asarray(obstacle_points, np.float32),
            table_z=table_z, seed=seed, rank=rank, T=T[order].cpu().numpy(), T_pre=T_pre[order].cpu().numpy(),
            q=grasp[order, 6].cpu().numpy(), order=order.cpu().numpy(), depth=depth[order].cpu().numpy(),
            inside=inside.sum(-1)[order].cpu().numpy(),
        )  # fmt: skip
    T, T_pre, q = T[order].cpu().numpy(), T_pre[order].cpu().numpy(), grasp[order, 6].clamp(0, OPEN_Q).cpu().numpy()
    return [(T[i], float(q[i]), T_pre[i]) for i in range(num_samples)]
