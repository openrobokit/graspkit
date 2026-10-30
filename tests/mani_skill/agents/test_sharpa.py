"""Tests for `graspkit.mani_skill.agents.sharpa`."""

import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("mani_skill")  # the maniskill extra

from graspkit.mani_skill.envs.dex_grasp import MsDexGraspEnv


@pytest.mark.hf_assets
class TestSharpaAgents:
    def test_sharpa_left_agent_assets(self):
        from robokit.assets.robots.end_effectors import sharpa_hand

        from graspkit.mani_skill.agents.sharpa import SharpaLeftFloatingBase, SharpaRightFloatingBase

        assert SharpaLeftFloatingBase.uid in MsDexGraspEnv.SUPPORTED_ROBOTS
        root = ET.parse(sharpa_hand.URDF_FLOATING_BASE_PATH_LEFT).getroot()
        links = {link.attrib["name"] for link in root.findall("link")}
        assert SharpaLeftFloatingBase.palm_link_name in links
        assert set(SharpaLeftFloatingBase.contact_link_names) <= links
        assert set(SharpaLeftFloatingBase.urdf_config["link"]) == set(SharpaLeftFloatingBase.contact_link_names)
        assert all(name.startswith("left_") for name in SharpaLeftFloatingBase.contact_link_names)
        assert all(name.startswith("right_") for name in SharpaRightFloatingBase.contact_link_names)
        joints = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]
        hand = ET.parse(sharpa_hand.URDF_PATH_LEFT).getroot()
        hand_joints = [joint.attrib["name"] for joint in hand.findall("joint") if joint.attrib["type"] != "fixed"]
        assert len(hand_joints) == 22
        assert joints[:6] == ["WRJTx", "WRJTy", "WRJTz", "WRJRx", "WRJRy", "WRJRz"]
        assert joints[6:] == hand_joints
