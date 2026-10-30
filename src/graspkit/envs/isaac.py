"""BaseEnv adapter for an isolated Isaac runtime (`graspkit/isaac/worker.py`), using native chain-order actions."""

import itertools
import socket
import subprocess
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from torch.types import Device

from graspkit.configs.experiment_config import EnvConfig, RendererConfig
from graspkit.configs.robot_config import RobotConfig
from graspkit.constants import LOG_ROOT
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs.base import BaseEnv, Rollout
from graspkit.isaac.job import prepare_robot, write_job
from graspkit.isaac.worker import pack_array, unpack_array
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.point_cloud_utils import crop_and_resample_point_cloud


class IsaacEnv(BaseEnv):
    """Fixed-base tabletop env per scene on one Isaac worker, with robot cameras; task settings as in ManiSkill."""

    def __init__(self, cfg: EnvConfig, robot_config: RobotConfig, device: Device, renderer_cfg: RendererConfig):
        super().__init__()
        self.cfg, self.renderer_cfg, self.device = cfg, renderer_cfg, torch.device(device)
        self.isaac = cfg.isaac_env_config
        parent = LOG_ROOT / "isaac"
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(mkdtemp(prefix="env-", dir=parent))
        from graspkit.assets import isaac_appearance  # imported here: the HF fetch happens on first use only

        self.robot = prepare_robot(robot_config, cfg.cameras, self.directory, isaac_appearance.PATH)
        self.attrs = self.robot["attrs"] | {
            "control_freq": cfg.control_freq,
            "substeps": self.isaac.substeps,
            "lift_dist": cfg.target_lift_dist,
            "object_mass": cfg.object_mass,
            "point_bounds": np.asarray(renderer_cfg.points_crop_bounds, np.float32),
        }
        self._scenes = []
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        path = str(self.directory / "worker.sock")
        server.bind(path)
        server.listen(1)
        server.settimeout(self.isaac.timeout)
        self._log = (self.directory / "worker.log").open("a")
        self.process = subprocess.Popen(
            [
                *self.isaac.worker_command,
                str(Path(__file__).resolve().parents[1] / "isaac/worker.py"),
                "--socket",
                path,
                "--usd-cache",
                str(parent / "usd"),
                "--threads",
                str(self.isaac.runtime_threads),
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        channel, _ = server.accept()
        channel.settimeout(None)
        self.connection = Connection(channel.detach())
        server.close()

    def close(self):
        self._request("close")
        self.connection.close()
        self.process.wait(timeout=self.isaac.timeout)
        assert self.process.returncode == 0
        self._log.close()
        (self.directory / "worker.sock").unlink()

    def _request(self, operation: str, **arguments):
        arguments = {
            key: pack_array(value) if isinstance(value, np.ndarray) else value for key, value in arguments.items()
        }
        self.connection.send(dict(operation=operation, **arguments))
        assert self.connection.poll(self.isaac.timeout), f"Isaac worker timed out; see {self.directory / 'worker.log'}"
        result = self.connection.recv()
        if result is None:
            return None
        if isinstance(result, dict):
            return {key: unpack_array(value) for key, value in result.items()}
        return unpack_array(result)

    def _full_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Pruned chain qpos (..., D) -> every Isaac joint, expanding mimics and parked branches."""
        return qpos @ self.robot["coefficients"].T + self.robot["bias"]

    def reset(
        self,
        scene: SceneDescription | list[SceneDescription] | None = None,
        init_qpos: torch.Tensor | None = None,
        root_pose: torch.Tensor | None = None,
        camera_pose: torch.Tensor | None = None,
    ):
        """One env per scene in `scene` (a scene or a list); `None` reuses the previous scenes."""
        assert camera_pose is None, "Isaac observes through the robot's own cameras."
        if scene is not None:
            self._scenes = scene if isinstance(scene, list) else [scene]
        n = len(self._scenes)
        qpos = self.robot["init_qpos"] if init_qpos is None else init_qpos.detach().cpu().numpy()
        root = self.robot["root_pose"] if root_pose is None else root_pose.detach().cpu().numpy()
        job = self.directory / "reset.h5"
        qpos = np.broadcast_to(qpos.reshape(-1, 1, qpos.shape[-1]), (n, 1, qpos.shape[-1]))
        write_job(job, self._scenes, self._full_qpos(qpos), root.reshape(-1, 7)[0], self.attrs)
        self._request("reset", job=str(job), indices=list(range(n)))

    def get_qpos(self) -> torch.Tensor:
        return torch.as_tensor(self._request("get_qpos"), device=self.device)

    def step_qpos(self, q: torch.Tensor):
        self._request("step_qpos", qpos=self._full_qpos(q.detach().cpu().numpy().reshape(len(self._scenes), -1)))

    def get_info(self) -> dict[str, torch.Tensor]:
        return {key: torch.as_tensor(value, device=self.device) for key, value in self._request("get_info").items()}

    def get_object_pose(self) -> torch.Tensor:
        return torch.as_tensor(self._request("get_object_pose"), device=self.device)

    def set_object_pose(self, pose: torch.Tensor):
        self._request("set_object_pose", pose=pose.detach().cpu().numpy().astype(np.float32))

    def observe(self) -> dict[str, torch.Tensor]:
        """Camera frames from the worker; points are cropped and resampled per view as the ManiSkill renderer does."""
        frames = self._request("observe")
        n, v = frames["counts"].shape
        cloud = torch.zeros(n * v, max(1, int(frames["counts"].max())), 4)
        target = torch.zeros(*cloud.shape[:2], 1)
        for view, (begin, end) in enumerate(itertools.pairwise([0, *frames["counts"].reshape(-1).cumsum()])):
            cloud[view, : end - begin, :3] = torch.from_numpy(frames["points"][begin:end])
            cloud[view, : end - begin, 3] = 1
            target[view, : end - begin, 0] = torch.from_numpy(frames["target"][begin:end])
        tables = torch.tensor([scene.boxes["table"].position for scene in self._scenes]).repeat_interleave(v, 0)
        bounds = torch.tensor(self.renderer_cfg.points_crop_bounds) + tables[:, :, None]
        points = crop_and_resample_point_cloud(
            {"xyzw": cloud, "rgb": torch.zeros_like(cloud[..., :3]), "segmentation": target},
            bounds,
            self.renderer_cfg.num_points,
        )
        return {
            "rgb": torch.from_numpy(frames["rgb"]),
            "xyz": points["xyz"].reshape(n, -1, 3),
            "seg": points["segmentation"].reshape(n, -1, 1),
            "seg_id": torch.ones(n),
            "camera_poses": torch.from_numpy(frames["cameras"]),
        }

    def get_rgb(self) -> np.ndarray:
        return self.observe()["rgb"][0, 0].numpy()

    def get_pointcloud(self, mask: np.ndarray) -> torch.Tensor:
        raise NotImplementedError("Isaac removes the robot by segmentation; use observe().")

    def evaluate(self, batch: GraspKitData, record: bool = False) -> dict[str, torch.Tensor]:
        assert batch.full_traj_qpos is not None and batch.scene_descriptions is not None
        trajectories = batch.full_traj_qpos.detach().cpu().numpy()
        if trajectories.ndim == 3:
            trajectories = trajectories[:, None]
        b, k, t, d = trajectories.shape
        assert b == len(batch.scene_descriptions) and self.cfg.num_seeds > 0
        scenes = [scene for scene in batch.scene_descriptions for _ in range(k)]
        job = self.directory / "evaluate.h5"
        write_job(job, scenes, self._full_qpos(trajectories.reshape(b * k, t, d)), self.robot["root_pose"], self.attrs)
        self.last_rollouts = []
        chunk = min(self.cfg.num_envs, self.isaac.max_envs)
        successes, final = [], None
        for seed in range(self.cfg.num_seeds):
            results = []
            for start in range(0, b * k, chunk):
                stop = min(start + chunk, b * k)
                forces = None
                if self.cfg.enable_perturbation:
                    forces = np.zeros((t, stop - start, 3), np.float32)
                    rng = np.random.default_rng(seed)
                    for _ in range(self.cfg.num_perturbations):
                        step = int(rng.integers(*self.cfg.perturbation_step_range))
                        assert step < t, "Perturbation step range exceeds the trajectory."
                        force = rng.normal(size=(stop - start, 3)).astype(np.float32)
                        force *= self.cfg.perturbation_magnitude / np.linalg.norm(force, axis=-1, keepdims=True)
                        forces[step : step + self.cfg.perturbation_duration] += force
                result = self._request(
                    "evaluate", job=str(job), indices=list(range(start, stop)), record=record, forces=forces
                )
                if record:
                    measured, poses = result.pop("qpos"), result.pop("obj_pose")
                    for offset, index in enumerate(range(start, stop)):
                        self.last_rollouts.append(
                            Rollout(
                                qpos=measured[offset],
                                obj_pose=poses[offset],
                                scene_id=batch.scene_ids[index // k],
                                grasp_index=index % k,
                                success=bool(result["success"][offset]),
                            )
                        )
                results.append(result)
            final = {key: np.concatenate([result[key] for result in results]) for key in results[0]}
            successes.append(final["success"])
        final["success"] = np.logical_and.reduce(successes)
        return {
            key: torch.as_tensor(value, device=self.device).reshape(b, k, *value.shape[1:])
            for key, value in final.items()
        }
