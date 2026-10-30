"""MuJoCo evaluator `HandSpec` instances."""

from graspkit.configs.env_config import HandSpec
from graspkit.configs.robot.shadow_mj import SHADOW_MJ_XML
from graspkit.configs.robot.vega_sharpa_mj import VEGA_SHARPA_MJ_XML
from graspkit.configs.robot.xarm7_shadow import XARM_SHADOW_XML
from graspkit.configs.robot.xarm7_xhand_mj import XARM_XHAND_MJ_XML
from graspkit.configs.robot.xhand_mj import XHAND_MJ_XML

SHADOW_HAND = HandSpec(
    xml_path=SHADOW_MJ_XML,
    mocap=True,
    finger_prefix=["rh_ff", "rh_mf", "rh_rf", "rh_lf", "rh_th"],
    valid_body_name=[
        "rh_palm",
        "rh_ffproximal",
        "rh_ffmiddle",
        "rh_ffdistal",
        "rh_mfproximal",
        "rh_mfmiddle",
        "rh_mfdistal",
        "rh_rfproximal",
        "rh_rfmiddle",
        "rh_rfdistal",
        "rh_lfmetacarpal",
        "rh_lfproximal",
        "rh_lfmiddle",
        "rh_lfdistal",
        "rh_thproximal",
        "rh_thmiddle",
        "rh_thdistal",
    ],
    tip_body_name=["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"],
    exclude_table_contact=None,
)

# Fixed-base bimanual Vega + right Sharpa; no mocap, arm+hand actuated, tabletop lift raises the `Lift` base joint 10cm.
_VS_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
VEGA_SHARPA = HandSpec(
    xml_path=VEGA_SHARPA_MJ_XML,
    mocap=False,
    finger_prefix=[f"rh_right_{f}" for f in _VS_FINGERS],
    valid_body_name=["rh_right_hand_C_MC"] + [f"rh_right_{f}_{seg}" for f in _VS_FINGERS for seg in ("PP", "MP", "DP")],
    tip_body_name=[f"rh_right_{f}_DP" for f in _VS_FINGERS],
    exclude_table_contact=None,
    lift_arm_body="rh_right_hand_C_MC",  # lock the base; the tabletop lift IKs the right arm to raise the palm 10cm
    arm_qpos_indices=tuple(
        range(34, 41)
    ),  # R_arm_j1..7 in the 63-dof mjcf order (Lift,torso,head,L_arm,L_hand,R_arm,..)
)

# Left-hand mirror on the SAME bimanual mjcf: lh_left_* contact bodies, lift via L_arm_j1..7.
VEGA_SHARPA_LEFT = HandSpec(
    xml_path=VEGA_SHARPA_MJ_XML,
    mocap=False,
    finger_prefix=[f"lh_left_{f}" for f in _VS_FINGERS],
    valid_body_name=["lh_left_hand_C_MC"] + [f"lh_left_{f}_{seg}" for f in _VS_FINGERS for seg in ("PP", "MP", "DP")],
    tip_body_name=[f"lh_left_{f}_DP" for f in _VS_FINGERS],
    exclude_table_contact=None,
    lift_arm_body="lh_left_hand_C_MC",
    arm_qpos_indices=tuple(range(5, 12)),  # L_arm_j1..7 in the 63-dof mjcf order
)


# xArm7 (fixed base) with the Shadow hand bolted to link7; mocap=False, hand bodies carry the "_r" attach suffix.
XARM_SHADOW = HandSpec(
    xml_path=XARM_SHADOW_XML,
    mocap=False,
    finger_prefix=["rh_ff", "rh_mf", "rh_rf", "rh_lf", "rh_th"],
    valid_body_name=["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
    + [f"{b}_r" for b in SHADOW_HAND.valid_body_name],
    tip_body_name=[f"{b}_r" for b in SHADOW_HAND.tip_body_name],
    exclude_table_contact=None,
    lift_arm_body="rh_palm_r",  # fixed base; the tabletop lift IKs the 7 arm joints to raise the palm 10cm
    arm_qpos_indices=(0, 1, 2, 3, 4, 5, 6),
)


# xArm7 + xHand (fully actuated, no coupling tendons -> optimizer FK == MuJoCo); fixed base, tabletop lift IKs the arm.
_XH = ["thumb_rota", "index_rota", "mid", "ring", "pinky"]
XARM_XHAND = HandSpec(
    xml_path=XARM_XHAND_MJ_XML,
    mocap=False,
    finger_prefix=[f"right_hand_{f.split('_')[0]}" for f in _XH],
    valid_body_name=["link1", "link2", "link3", "link4", "link5", "link6", "link7", "right_hand_link"]
    + [f"right_hand_{f}_link{s}" for f in _XH for s in (1, 2)],
    tip_body_name=[f"right_hand_{f}_link2" for f in _XH],
    exclude_table_contact=None,
    lift_arm_body="right_hand_link",
    arm_qpos_indices=(0, 1, 2, 3, 4, 5, 6),
)


# Floating (mocap) xHand -- same hand subtree as XARM_XHAND with the arm dropped; mocap=True, 12 dof, like SHADOW_HAND.
XHAND = HandSpec(
    xml_path=XHAND_MJ_XML,
    mocap=True,
    finger_prefix=[f"right_hand_{f.split('_')[0]}" for f in _XH],
    valid_body_name=["right_hand_link"] + [f"right_hand_{f}_link{s}" for f in _XH for s in (1, 2)],
    tip_body_name=[f"right_hand_{f}_link2" for f in _XH],
    exclude_table_contact=None,
)
