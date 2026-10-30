from typing import List, Optional, Union

import sapien
from mani_skill import format_path
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.mjcf_loader import ParsedMJCFData
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.pose import Pose

from graspkit.mani_skill.agents.grasp_agent import GraspAgent
from graspkit.mani_skill.utils.sapien_mjcf_loader import MJCFLoader as SAPIENMJCFLoader


# This is a direct copy-paste of the original MJCFLoader class, but inherits from a custom SAPIENMJCFLoader
class MJCFLoader(SAPIENMJCFLoader):
    scene: ManiSkillScene
    name: str | None = None
    disable_self_collisions: bool = True

    def parse(self, mjcf_file, package_dir=None) -> ParsedMJCFData:
        articulation_builders, actor_builders, cameras = super().parse(mjcf_file, package_dir)
        for i, a in enumerate(articulation_builders):
            if len(articulation_builders) > 1:
                a.set_name(f"{self.name}-articulation-{i}")
            else:
                a.set_name(f"{self.name}")
            if self.disable_self_collisions:
                for link in a.link_builders:
                    # NOTE (stao): Currently this may not be working as intended
                    link.collision_groups[2] |= 1 << 29
        for i, b in enumerate(actor_builders):
            b.set_name(f"{self.name}-actor-{i}")
        return dict(
            articulation_builders=articulation_builders,
            actor_builders=actor_builders,
            cameras=cameras,
        )

    def load(
        self,
        mjcf_file: str,
        package_dir=None,
        name=None,
        scene_idxs=None,
    ) -> Articulation:
        """
        Args:
            urdf_file: filename for URDL file
            srdf_file: SRDF for urdf_file; defaults to the ".srdf" file with the same name as the urdf file
            package_dir: base directory to resolve asset files in the URDF file; strips leading "package://" from paths
            name (str): name of the created articulation
            scene_idxs (list[int]): the ids of the scenes to build the objects in
        Returns:
            returns a single Articulation loaded from the URDF file. It throws an error if multiple objects exists
        """
        if name is not None:
            self.name = name
        _parsed_mjcf_data = self.parse(mjcf_file, package_dir)
        articulation_builders = _parsed_mjcf_data["articulation_builders"]
        _parsed_mjcf_data["actor_builders"]
        cameras = _parsed_mjcf_data["cameras"]

        articulations: List[Articulation] = []
        for b in articulation_builders[:1]:
            b.set_scene_idxs(scene_idxs)
            b.disable_self_collisions = self.disable_self_collisions
            articulations.append(b.build())

        actors: List[Actor] = []

        if len(cameras) > 0:
            name2entity = dict()
            for a in articulations:
                for sapien_articulation in a._objs:
                    for link in sapien_articulation.links:
                        name2entity[link.name] = link.entity

            for a in actors:
                name2entity[a.name] = a

            # TODO (stao): support extracting sensors (mjcf camera components) — unimplemented.

        return articulations[0]


@register_agent()
class ShadowRightMjFloatingBase(GraspAgent):
    uid = "shadow_right_mj_floating_base"

    @property
    def mjcf_path(self) -> str:
        from robokit.assets.robots.end_effectors import shadow_hand as shadow_hand_assets

        return str(shadow_hand_assets.MJCF_FLOATING_BASE_SAPIEN_PATH_RIGHT)

    urdf_config = dict(
        _materials=dict(tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)),
        link={
            "robot0:ffdistal": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "robot0:mfdistal": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "robot0:rfdistal": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "robot0:lfdistal": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            "robot0:thdistal": dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        },
    )
    disable_self_collisions: bool = False

    def __init__(self, *args, **kwargs):
        self.contact_link_names = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.palm_link_name = "robot0:palm"
        kwargs["initial_pose"] = sapien.Pose(p=(-0.2, -0.2, 0.0))
        super().__init__(*args, **kwargs)

    def _load_articulation(self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None):
        """Load the robot articulation."""
        if self.urdf_path is not None:
            loader = self.scene.create_urdf_loader()
            asset_path = format_path(str(self.urdf_path))
        elif self.mjcf_path is not None:
            loader = MJCFLoader()
            loader.set_scene(self.scene)
            asset_path = format_path(str(self.mjcf_path))

        loader.name = self.uid
        if self._agent_idx is not None:
            loader.name = f"{self.uid}-agent-{self._agent_idx}"
        loader.fix_root_link = self.fix_root_link
        loader.load_multiple_collisions_from_file = self.load_multiple_collisions
        loader.disable_self_collisions = self.disable_self_collisions

        if self.urdf_config is not None:
            urdf_config = sapien_utils.parse_urdf_config(self.urdf_config)
            sapien_utils.check_urdf_config(urdf_config)
            sapien_utils.apply_urdf_config(loader, urdf_config)

        builder = loader.parse(asset_path)["articulation_builders"][0]
        builder.initial_pose = initial_pose
        self.robot = builder.build()
        assert self.robot is not None, f"Fail to load URDF/MJCF from {asset_path}"

        # Cache robot link ids
        self.robot_link_ids = [link.name for link in self.robot.get_links()]

    @property
    def _controller_configs(self):
        floating_base = self.floating_base_controller()
        pd_joint_pos = PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints[6:]],
            # stiffness=500,
            stiffness=1000,
            damping=50,
            normalize_action=False,
            friction=0.05,
            lower=None,
            upper=None,
            use_delta=False,
        )
        return {
            "pd_joint_pos": {"floating_base": floating_base, "pd_joint_pos": pd_joint_pos},
        }
