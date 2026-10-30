from typing import List, Optional

import torch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.utils.structs import Actor


class GraspAgent(BaseAgent):
    """Base class for grasp hands/grippers; grasping = contact-force on fingertip/contact links exceeds min_force."""

    contact_link_names: List[str] = []
    """Fingertip / contact link names used for grasp detection."""
    palm_link_name: Optional[str] = None
    """Palm link, counted as one more contact surface (None to omit)."""

    def floating_base_controller(self) -> PDJointPosControllerConfig:
        """Shared 6-DoF floating-base controller that rigidly drives the virtual base joints (`active_joints[:6]`)."""
        return PDJointPosControllerConfig(
            [x.name for x in self.robot.active_joints[:6]],
            stiffness=1e5,
            damping=1e4,
            force_limit=1000,
            normalize_action=False,
            lower=None,
            upper=None,
            use_delta=False,
        )

    def _grasp_link_names(self) -> List[str]:
        links = list(self.contact_link_names)
        if self.palm_link_name is not None:
            links.append(self.palm_link_name)
        return links

    def is_grasping(self, object: Actor, min_force: float = 0.5) -> torch.Tensor:
        link_contact_forces = []
        for name in self._grasp_link_names():
            contact_forces = self.scene.get_pairwise_contact_forces(self.robot.links_map[name], object)
            link_contact_forces.append(torch.linalg.norm(contact_forces, axis=1))
        link_contact_forces = torch.stack(link_contact_forces, dim=-1)
        # at least two contact links (tips and/or palm) are loaded
        return torch.sum(link_contact_forces >= min_force, dim=-1) >= 2
