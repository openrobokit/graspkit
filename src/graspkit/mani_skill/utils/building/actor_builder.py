"""Support manually setting mass but automatically setting inertia and center of mass."""

from mani_skill.utils.building.actor_builder import ActorBuilder as ManiSkillActorBuilder


class ActorBuilder(ManiSkillActorBuilder):
    def __init__(self):
        super().__init__()
        self._manual_set_mass: bool = False

    def set_mass(self, mass: float):
        self._mass = mass
        self._manual_set_mass = True
        return self

    def build_physx_component(self, link_parent=None):
        component = super().build_physx_component(link_parent)
        if self._manual_set_mass:
            component.mass = self._mass
        return component
