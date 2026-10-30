import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Link, Pose, SceneConfig, SimConfig
from torch import Tensor

from graspkit.configs.experiment_config import EnvConfig
from graspkit.mani_skill.utils.building.actor_builder import ActorBuilder
from graspkit.mani_skill.utils.scene_builder.table import TableSceneBuilder
from graspkit.scene.scene_description import SceneDescription, SceneDescriptionsHelper

logger = logging.getLogger("graspkit")


@register_env("DexGrasp", max_episode_steps=500)
class MsDexGraspEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        "shadow_right_mj_floating_base",
        "shadow_right_floating_base",
        "inspire_right_floating_base",
        "ability_right_floating_base",
        "ability_left_floating_base",
        "barrett_right_floating_base",
        "schunk_right_floating_base",
        "mano_right_floating_base",
        "leap_right_floating_base",
        "xhand_right_floating_base",
        "xhand_right",
        "sharpa_right_floating_base",
        "sharpa_left_floating_base",
        "xarm7_ability_right",
        "xarm7_xhand_right",
        "xarm7_mano_right",
        "franka_gripper",
        "g1_inspire_right",
        "g1_mano_right",
        "g1_inspire_right_wbc",
        "vega_sharpa_right",
    ]
    SUPPORTED_OBS_MODES = ("state", "state_dict", "rgb", "rgb+depth+segmentation", "pointcloud")

    def __init__(self, config: EnvConfig, *args, **kwargs):
        self.config = config
        scene_config = SceneConfig(contact_offset=config.contact_offset, solver_position_iterations=30)
        sim_config = SimConfig(sim_freq=100, control_freq=config.control_freq, scene_config=scene_config)

        self.actors: Dict[str, Actor] = {}
        self.target_actor: Optional[Actor] = None
        self.initial_target_z: Optional[Tensor] = None
        self.scene_descriptions_helper: Optional[SceneDescriptionsHelper] = None

        super().__init__(
            *args,
            **kwargs,
            sim_config=sim_config,
            reconfiguration_freq=config.ms_env_config.reconfiguration_freq,
        )

    def _load_scene(self, options: dict):
        """Called by `self._reconfigure`."""
        scene_descriptions: Optional[List[SceneDescription]] = options.get("scene_descriptions", None)

        if self.config.load_scenes and scene_descriptions is not None:
            self.actors = {}
            scene_desc_helper = SceneDescriptionsHelper(scene_descriptions)
            self.scene_descriptions_helper = scene_desc_helper
            actor_id_to_entities, actor_id_to_scene_idxs = defaultdict(list), defaultdict(list)
            # PhysX contact outcomes depend on actor insertion order.
            for coll_mesh_path, mesh_scales in sorted(set(scene_desc_helper.mesh_keys)):
                visual_mesh_path = coll_mesh_path
                scene_idxs = scene_desc_helper.get_scene_indices_by_mesh_key(coll_mesh_path, mesh_scales)
                actor_ids = scene_desc_helper.get_actor_ids_by_mesh_key(coll_mesh_path, mesh_scales)
                first_pos = scene_desc_helper.get_positions_by_actor_id(actor_ids[0])[0]
                first_quat = scene_desc_helper.get_quaternions_by_actor_id(actor_ids[0])[0]

                builder = ActorBuilder().set_scene(self.scene)
                object_physx_material = sapien.physx.PhysxMaterial(
                    static_friction=2.0, dynamic_friction=2.0, restitution=0.0
                )
                builder.add_multiple_convex_collisions_from_file(
                    coll_mesh_path, scale=mesh_scales, material=object_physx_material
                )
                builder.add_visual_from_file(visual_mesh_path, scale=mesh_scales)
                builder.set_mass(self.config.object_mass)
                builder.set_scene_idxs(scene_idxs)
                builder.set_initial_pose(sapien.Pose(p=first_pos, q=first_quat))
                built_actor = builder.build(name=f"mesh_{coll_mesh_path}_{mesh_scales}")
                for actor_id, entity, scene_idx in zip(actor_ids, built_actor._objs, scene_idxs):
                    actor_id_to_entities[actor_id].append(entity)
                    actor_id_to_scene_idxs[actor_id].append(scene_idx)
            for actor_id, entities in actor_id_to_entities.items():
                scene_idxs = actor_id_to_scene_idxs[actor_id]
                scene_idxs = torch.as_tensor(scene_idxs)
                reindex = torch.argsort(scene_idxs)
                self.actors[actor_id] = Actor.create_from_entities(
                    [entities[i] for i in reindex],
                    self.scene,
                    scene_idxs=scene_idxs[reindex],
                    shared_name=actor_id,
                )
                self.actors[actor_id].merged = True

            target_actor_entites = []
            for scene_idx, scene_desc in enumerate(scene_descriptions):
                target_actor_id = scene_desc.target_actor_id
                actor_entities = actor_id_to_entities[target_actor_id]
                actor_scene_idxs = actor_id_to_scene_idxs[target_actor_id]
                target_actor_entites.append(actor_entities[actor_scene_idxs.index(scene_idx)])
            self.target_actor = Actor.create_from_entities(
                target_actor_entites,
                self.scene,
                torch.arange(len(target_actor_entites)),
                shared_name="target",
            )
            if self.config.floating_object:
                self.target_actor.disable_gravity = True

            if not self.config.floating_object:
                self.table_scene = TableSceneBuilder(env=self)
                self.table_scene.build(box_sizes=scene_desc_helper.table_sizes)

        # load camera
        cam_mount_builder = self.scene.create_actor_builder()
        cam_mount_builder.set_initial_pose(sapien.Pose(p=(0.56, 0.07, 0.37), q=(0.02, 0.21, 0.0, -0.98)))
        self.camera_mount = cam_mount_builder.build_kinematic("camera_mount")

    def _initialize_episode(self, env_idx: Tensor, options: dict):
        """Called by `self.reset`."""

        with torch.device(self.device):
            if self.config.load_scenes and self.scene_descriptions_helper is not None:
                for actor_id in self.actors.keys():
                    ps = self.scene_descriptions_helper.get_positions_by_actor_id(actor_id)
                    qs = self.scene_descriptions_helper.get_quaternions_by_actor_id(actor_id)
                    self.actors[actor_id].set_pose(
                        Pose.create(torch.cat([torch.as_tensor(ps), torch.as_tensor(qs)], dim=-1))
                    )
                if not self.config.floating_object:
                    ps = self.scene_descriptions_helper.get_positions_by_actor_id("table")
                    qs = self.scene_descriptions_helper.get_quaternions_by_actor_id("table")
                    box_size = self.scene_descriptions_helper.table_sizes[0]
                    ps, box_size = torch.as_tensor(ps), torch.as_tensor(box_size)
                    # Table is not centered on the origin (see `TableSceneBuilder`).
                    ps[:, 2] -= box_size[2] / 2
                    self.table_scene.initialize(
                        env_idx,
                        table_pose=Pose.create(torch.cat([ps, torch.as_tensor(qs)], dim=-1)),
                    )
            if options.get("save_hand_key_points", False):
                self.save_hand_key_points = True

            init_qpos = options.get("qpos", None)
            if init_qpos is not None:
                self.agent.robot.set_qpos(init_qpos)
            root_poses = options.get("root_poses", None)
            if root_poses is not None:
                self.agent.robot.set_pose(Pose.create(root_poses))
            camera_poses = options.get("camera_poses", None)
            if camera_poses is not None:
                self.camera_mount.set_pose(Pose.create(camera_poses))
            object_poses = options.get("object_poses", None)
            if object_poses is not None:
                self.target_actor.set_pose(Pose.create(object_poses))

            # Snapshot post-init object z so is_lifted measures relative lift, since table height varies per scene.
            if self.target_actor is not None:
                self.initial_target_z = self.target_actor.pose.p[..., 2].clone()

    def _get_obs_extra(self, info: dict):
        """Get task-relevant extra observations."""
        obs_extra = {}
        if self.target_actor is not None:
            obs_extra["actor_pose"] = self.target_actor.pose.raw_pose.clone()
        return obs_extra

    def evaluate(self, **kwargs) -> Dict:
        if self.target_actor is not None:
            is_strict_grasped = self.agent.is_grasping(self.target_actor)

            if not self.config.floating_object:
                lift_dist = torch.clip(self.target_actor.pose.p[..., 2] - self.initial_target_z, 0.0, 0.4)
                is_lifted = lift_dist > self.config.target_lift_dist

                success = torch.logical_and(is_strict_grasped, is_lifted)
                return {
                    "success": success,
                    "is_lifted": is_lifted,
                    "is_strict_grasped": is_strict_grasped,
                }
            else:
                success = is_strict_grasped
                return {
                    "success": success,
                    "is_strict_grasped": is_strict_grasped,
                }
        else:
            return {}

    def compute_normalized_dense_reward(self, obs, action, info):
        return 0.0

    @property
    def _default_sensor_configs(self):
        if self.config.cameras:
            return []
        return [
            CameraConfig(
                "base_camera",
                pose=Pose.create_from_pq([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
                width=self.config.camera_width,
                height=self.config.camera_height,
                intrinsic=np.asarray(self.config.camera_intrinsic),
                near=self.config.camera_near,
                far=5,
                mount=self.camera_mount,
            )
        ]

    @property
    def target_actor_segmentation_id(self):
        return torch.tensor([obj.per_scene_id for obj in self.target_actor._objs], device=self.device)

    @property
    def table_segmentation_id(self):
        return torch.tensor([obj.per_scene_id for obj in self.table_scene.table._objs], device=self.device)

    @property
    def robot_segmentation_id(self):
        return torch.tensor(
            [obj_id for obj_id, obj in self.segmentation_id_map.items() if isinstance(obj, Link)],
            device=self.device,
        )
