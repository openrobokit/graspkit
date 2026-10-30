"""`MjEnv`: the ported MuJoCo evaluator scores `GraspKitData` grasps on the planning robot, mapped to native joints."""

import glob
import multiprocessing as mp
import os
import signal
from typing import Optional

import numpy as np
import torch
import trimesh
from robokit.xform.numpy import quaternion_apply, quaternion_invert, quaternion_raw_multiply
from robokit.xform.numpy import quaternion_to_matrix as quaternion_to_matrix_np
from robokit.xform.warp.torch_wrappers import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

from graspkit.configs.env_config import HandSpec, MjEvalConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.constants import GRASPKIT_HOME
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs.base import BaseEnv, Rollout
from graspkit.mujoco import MjEvaluator
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription

# one MjEvaluator per worker (MuJoCo is CPU-only, not thread-safe); DexGraspBench fans out the same way
_WORKER_EVAL: MjEvaluator = None


def _worker_init(cfg: MjEvalConfig, hand: HandSpec):
    # workers ignore SIGINT so only the main process handles Ctrl-C; else they race the parent and hang
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_EVAL
    _WORKER_EVAL = MjEvaluator(cfg, hand)


def _worker_eval(args: tuple) -> dict:
    grasp_data, record = args
    return _WORKER_EVAL.evaluate(grasp_data, record=record)


class MjEnv(BaseEnv):
    def __init__(
        self,
        cfg: MjEvalConfig,
        robot_config: RobotConfig,
        hand: HandSpec = None,
        num_workers: int = 48,
        base_pos: list[float] | None = None,
        base_quat: list[float] | None = None,
    ):
        # fixed-base arm roots at base_pos/base_quat vs MjDexGraspEnv's mjcf origin; floating hands leave these None
        super().__init__()
        self.robot_helper = GraspRobot.from_config(robot_config, device="cpu")
        self._full_helper = GraspRobot.from_config(robot_config, device="cpu", prune=False)
        self._chain_to_full = [
            self._full_helper.actuated_joint_names.index(name) for name in self.robot_helper.actuated_joint_names
        ]
        self._mj_joint_indices = [
            self._full_helper.spec.joint_names.index(name) for name in self._full_helper.spec.nonfixed_joint_names
        ]
        self._mj_to_chain = [
            self._full_helper.spec.nonfixed_joint_names.index(name) for name in self.robot_helper.actuated_joint_names
        ]
        self.base_pos = np.asarray(base_pos, dtype=float) if base_pos is not None else None
        self.base_quat = np.asarray(base_quat, dtype=float) if base_quat is not None else None
        self.evaluator = MjEvaluator(cfg, hand)
        # spawn, not fork: forking a CUDA/warp-initialized, multithreaded parent deadlocks; workers are pure-CPU
        self.pool = mp.get_context("spawn").Pool(
            min(num_workers, mp.cpu_count()),
            initializer=_worker_init,
            initargs=(self.evaluator.cfg, self.evaluator.hand),
        )
        # mp.Pool teardown deadlocks under Ctrl-C (finalizer vs. respawn race), so SIGKILL the whole job group at once
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    def _to_variables(self, q_mj: np.ndarray) -> torch.Tensor:
        """Gather chain joints and convert the floating root back to Euler angles."""
        q = torch.as_tensor(q_mj, dtype=torch.float32)
        num_free = q.shape[-1] - len(self._mj_joint_indices)
        joints = q[..., num_free:][..., self._mj_to_chain]
        if num_free:
            euler = matrix_to_euler_angles(quaternion_to_matrix(q[..., 3:7]), "rxyz")
            return torch.cat((q[..., :3], euler, joints), dim=-1)
        return joints

    def _to_grasp_data(self, batch: GraspKitData, i: int, k: int) -> dict:
        scene = batch.scene_descriptions[i]
        mesh = scene.meshes[scene.target_actor_id]
        obj_scale = float(mesh.scales[0])
        mesh_path = os.path.join(GRASPKIT_HOME, mesh.collision_mesh_path)
        # obj_path is a single .glb (convex pieces) or a folder of convex_piece_*.obj files; sum volumes for density
        if mesh_path.endswith(".glb"):
            obj_path = mesh_path
            parts = trimesh.load(mesh_path, process=False).dump(concatenate=False)
        else:
            obj_path = os.path.dirname(mesh_path)
            parts = [trimesh.load(f, process=False) for f in glob.glob(os.path.join(obj_path, "*convex_piece_*.obj"))]
        volume = sum(float(p.volume) * obj_scale**3 for p in parts)
        q = (
            torch.stack((batch.pre_grasp_qpos[i, k], batch.grasp_qpos[i, k], batch.post_grasp_qpos[i, k]))
            .detach()
            .cpu()
        )
        num_free = q.shape[-1] - self.robot_helper.num_actuated_joints
        full_q = self._full_helper.init_qpos.to(q).expand(*q.shape[:-1], -1).clone()
        full_q[..., self._chain_to_full] = q[..., num_free:]
        mj_q = self._full_helper.map_to_full_joint_values_torch(full_q)[..., self._mj_joint_indices]
        if num_free:
            quat = matrix_to_quaternion(euler_angles_to_matrix(q[..., 3:6], "rxyz"))
            mj_q = torch.cat((q[..., :3], quat, mj_q), dim=-1)
        pre, grasp, squeeze = mj_q.numpy()
        obj_pose = np.array([*mesh.position, *mesh.quaternion])
        r_inv = quaternion_invert(self.base_quat) if self.base_pos is not None else None
        if self.base_pos is not None:  # world -> robot-base frame (fixed-base arm rooted at mjcf origin)
            obj_pose = np.concatenate(
                [quaternion_apply(r_inv, obj_pose[:3] - self.base_pos), quaternion_raw_multiply(r_inv, obj_pose[3:])]
            )
        table_box = None
        if self.evaluator.cfg.setting == "tabletop":
            box = scene.boxes["table"]
            box_pos, box_quat = np.asarray(box.position, dtype=float), np.asarray(box.quaternion, dtype=float)
            if self.base_pos is not None:
                box_pos = quaternion_apply(r_inv, box_pos - self.base_pos)
                box_quat = quaternion_raw_multiply(r_inv, box_quat)
            table_box = (box_pos, np.asarray(box.size) / 2.0, box_quat)
        return {
            "obj_path": obj_path,
            "obj_scale": obj_scale,
            "obj_pose": obj_pose,
            "obj_density": self.evaluator.cfg.obj_mass / volume,
            "table_box": table_box,
            "pregrasp_qpos": pre,
            "grasp_qpos": grasp,
            "squeeze_qpos": squeeze,
        }

    def evaluate(self, batch: GraspKitData, record: bool = False) -> dict[str, torch.Tensor]:
        self.last_rollouts = []
        batch = batch.to_padded()
        num_scenes, num_samples = batch.pre_grasp_qpos.shape[:2]
        grasp_datas = [self._to_grasp_data(batch, i, k) for i in range(num_scenes) for k in range(num_samples)]
        results = self.pool.map(_worker_eval, [(gd, record) for gd in grasp_datas])
        keys = ("success", "delta_pos", "delta_angle", "ho_pene", "contact_num", "sim_steps")
        infos = {key: torch.zeros(num_scenes, num_samples) for key in keys}
        for idx, r in enumerate(results):
            i, k = divmod(idx, num_samples)
            for key in keys:
                infos[key][i, k] = float(r["succ_flag" if key == "success" else key])
        infos["success"] = infos["success"].bool()
        if record:
            # grasps that bail on initial penetration have an empty log (skip); lift object pose to world for replay
            rot = quaternion_to_matrix_np(self.base_quat) if self.base_pos is not None else None
            self.last_rollouts = [
                Rollout(
                    qpos=self._to_variables(r["replay_hand_qpos"]).numpy(),
                    obj_pose=r["replay_obj_pose"].astype(np.float32)
                    if self.base_pos is None
                    else np.concatenate(
                        [
                            r["replay_obj_pose"][:, :3] @ rot.T + self.base_pos,
                            quaternion_raw_multiply(self.base_quat, r["replay_obj_pose"][:, 3:]),
                        ],
                        axis=1,
                    ).astype(np.float32),
                    scene_id=batch.scene_ids[idx // num_samples],
                    success=bool(r["succ_flag"]),
                    grasp_index=idx % num_samples,
                )
                for idx, r in enumerate(results)
                if r["replay_hand_qpos"].shape[0] > 0
            ]
        return infos

    def close(self):
        self.pool.terminate()

    def reset(
        self,
        scene: Optional[SceneDescription] = None,
        init_qpos: Optional[torch.Tensor] = None,
        root_pose: Optional[torch.Tensor] = None,
        camera_pose: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError("MjEnv is offline-eval only; use evaluate().")

    def get_qpos(self) -> torch.Tensor:
        raise NotImplementedError("MjEnv is offline-eval only; use evaluate().")

    def step_qpos(self, q: torch.Tensor):
        raise NotImplementedError("MjEnv is offline-eval only; use evaluate().")

    def get_info(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("MjEnv is offline-eval only; use evaluate().")

    def get_rgb(self) -> np.ndarray:
        raise NotImplementedError("MjEnv is offline-eval only; use evaluate().")

    def get_pointcloud(self, mask: np.ndarray) -> torch.Tensor:
        raise NotImplementedError("MjEnv is offline-eval only; use evaluate().")
