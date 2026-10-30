"""MuJoCo grasp evaluator (`fc` setting), from DexGraspBench; excludes force-closure (q1/qp/tdg/dfc) and `tabletop`."""

import json
import os
from copy import deepcopy

import mujoco
import numpy as np

from graspkit.configs.env_config import HandSpec, MjEvalConfig
from graspkit.configs.mj_hand import SHADOW_HAND
from graspkit.mujoco.dex_grasp import MjDexGraspEnv
from graspkit.mujoco.rot_np import compute_delta_qpos

# The 6 axis-aligned unit force directions used by the force-closure test.
FC_FORCE_DIRECTIONS = np.array(
    [
        [-1.0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ]
)

# Downward wrench standing in for gravity in the tabletop lift test (gravity stays disabled in MjDexGraspEnv).
TABLETOP_GRAVITY = np.array([0.0, 0, -1, 0, 0, 0])
LIFT_DIST = 0.2  # meters the hand root rises during lift (2x DexGraspBench's 0.1: visible, robust follow check)


class MjEvaluator:
    """MuJoCo grasp evaluator (`fc` setting): holds cfg/hand for callers; exposes build_scene/simulate_fc for replay."""

    def __init__(self, cfg: MjEvalConfig = None, hand: HandSpec = None):
        self.cfg = cfg if cfg is not None else MjEvalConfig()
        self.hand = hand if hand is not None else SHADOW_HAND

    def build_scene(self, grasp_data: dict, record: bool = False) -> MjDexGraspEnv:
        """Build the `MjDexGraspEnv` scene, with object mass normalized to cfg.obj_mass.

        - graspkit: passes `obj_path` (convex-piece folder) + a precomputed `obj_density`.
        - DexGraspBench: passes an example_object id; `info/simplified.json` gives mass/density to rescale to obj_mass.
        """
        cfg = self.cfg
        if "obj_density" in grasp_data:
            parts_dir = grasp_data["obj_path"]
            obj_density = grasp_data["obj_density"]
        else:
            obj_dir = os.path.join(cfg.obj_root, os.path.basename(grasp_data["obj_path"]))
            parts_dir = os.path.join(obj_dir, "urdf/meshes")
            with open(os.path.join(obj_dir, "info/simplified.json")) as f:
                obj_info = json.load(f)
            obj_coef = obj_info["mass"] / (obj_info["density"] * (obj_info["scale"] ** 3))
            obj_density = cfg.obj_mass / (obj_coef * (grasp_data["obj_scale"] ** 3))
        return MjDexGraspEnv(
            obj_path=parts_dir,
            obj_scale=grasp_data["obj_scale"],
            obj_density=obj_density,
            hand_xml_path=self.hand.xml_path,
            hand_mocap=self.hand.mocap,
            exclude_table_contact=self.hand.exclude_table_contact,
            friction_coef=cfg.miu_coef,
            # Object rests on the scene's table box (object-only collision) so the grip presses it against a surface.
            table_box=grasp_data["table_box"] if cfg.setting == "tabletop" else None,
            record=record,
            timestep=cfg.timestep,
            contact_solref_timeconst=cfg.solref_timeconst,
        )

    def simulate_fc(self, mj_ho: MjDexGraspEnv, grasp_data: dict) -> tuple:
        """Force-closure test (pregrasp->grasp->squeeze, hold under 6 forces); fails on excess motion or penetration."""
        cfg = self.cfg
        ho_c, hh_c = mj_ho.get_contact_info(grasp_data["pregrasp_qpos"], grasp_data["obj_pose"])
        ho_dist = min([c["contact_dist"] for c in ho_c]) if ho_c else 0
        hh_dist = min([c["contact_dist"] for c in hh_c]) if hh_c else 0
        if ho_dist < -cfg.max_pene or hh_dist < -cfg.max_pene:
            return False, 100.0, 100.0  # severe initial penetration -> failure sentinel

        pre_obj_qpos = deepcopy(mj_ho.get_obj_pose())
        succ_flag = True
        for direction in FC_FORCE_DIRECTIONS:
            mj_ho.reset_pose_qpos(grasp_data["pregrasp_qpos"], grasp_data["obj_pose"])
            mj_ho.control_hand_with_interp(grasp_data["pregrasp_qpos"], grasp_data["grasp_qpos"])
            mj_ho.control_hand_with_interp(grasp_data["grasp_qpos"], grasp_data["squeeze_qpos"])
            mj_ho.set_ext_force_on_obj(cfg.fc_force_scale * direction * cfg.obj_mass)
            for _ in range(10):
                mj_ho.control_hand_step(step_inner=50)
                delta_pos, delta_angle = compute_delta_qpos(pre_obj_qpos, mj_ho.get_obj_pose())
                succ_flag = (delta_pos < cfg.trans_thre) & (delta_angle < cfg.angle_thre)
                if not succ_flag:
                    break
            if not succ_flag:
                break

        delta_pos, delta_angle = compute_delta_qpos(pre_obj_qpos, mj_ho.get_obj_pose())
        succ_flag = (delta_pos < cfg.trans_thre) & (delta_angle < cfg.angle_thre)
        return bool(succ_flag), delta_pos, delta_angle

    def _arm_lift_qpos(self, mj_ho: MjDexGraspEnv, hand_qpos: np.ndarray) -> np.ndarray:
        """Damped-least-squares arm IK: raises `lift_arm_body` LIFT_DIST in z; kinematic-only, rollout untouched."""
        model, data, hand = mj_ho.model, mj_ho.data, self.hand
        ai = np.asarray(hand.arm_qpos_indices)
        bid = model.body(f"{mj_ho.hand_prefix}{hand.lift_arm_body}").id
        saved = data.qpos.copy()
        q = deepcopy(hand_qpos)
        data.qpos[: len(q)] = q
        mujoco.mj_forward(model, data)
        # Incremental resolved-rate IK: small damped-pseudo-inverse +z steps, never a full-error IK (flings the arm).
        z_goal = float(data.xpos[bid][2]) + LIFT_DIST  # drive the palm to the FULL LIFT_DIST (no undershoot)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(30):
            dz = z_goal - float(data.xpos[bid][2])  # remaining rise (closed-loop, so it reaches the goal)
            if dz < 5e-4:
                break
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)
            jac = np.vstack([jacp[:, ai], jacr[:, ai]])  # 6 x n arm Jacobian (pos + rot)
            vel = np.array([0.0, 0.0, min(dz, 0.01), 0.0, 0.0, 0.0])  # small +z steps, zero commanded rot
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), vel)
            q[ai] = np.clip(q[ai] + dq, model.jnt_range[ai, 0], model.jnt_range[ai, 1])
            data.qpos[ai] = q[ai]
            mujoco.mj_forward(model, data)
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        return q

    def simulate_tabletop(self, mj_ho: MjDexGraspEnv, grasp_data: dict) -> tuple:
        """Tabletop lift test; force applied after the grip (not approach), like ManiSkill's is_grasped+is_lifted."""
        cfg = self.cfg
        ho_c, hh_c = mj_ho.get_contact_info(grasp_data["pregrasp_qpos"], grasp_data["obj_pose"])
        ho_dist = min([c["contact_dist"] for c in ho_c]) if ho_c else 0
        hh_dist = min([c["contact_dist"] for c in hh_c]) if hh_c else 0
        if ho_dist < -cfg.max_pene or hh_dist < -cfg.max_pene:
            return False, 100.0, 100.0  # severe initial penetration -> failure sentinel

        # get_contact_info leaves the sim reset at pregrasp; success tracks the palm's ACTUAL rise, not a fixed target.
        init_obj = deepcopy(mj_ho.get_obj_pose())
        mj_ho.control_hand_with_interp(grasp_data["pregrasp_qpos"], grasp_data["grasp_qpos"], cfg.grasp_interp_steps)
        mj_ho.control_hand_with_interp(grasp_data["grasp_qpos"], grasp_data["squeeze_qpos"], cfg.grasp_interp_steps)
        mj_ho.set_ext_force_on_obj(cfg.fc_force_scale * TABLETOP_GRAVITY * cfg.obj_mass)  # perturb the held object
        lift_bid = (
            mj_ho.model.body(f"{mj_ho.hand_prefix}{self.hand.lift_arm_body}").id if self.hand.lift_arm_body else None
        )
        palm_z0 = float(mj_ho.data.xpos[lift_bid][2]) if lift_bid is not None else None
        if self.hand.lift_arm_body is not None:
            lift_qpos = self._arm_lift_qpos(mj_ho, grasp_data["squeeze_qpos"])  # fixed base: arm lifts the palm
        else:
            lift_qpos = deepcopy(grasp_data["squeeze_qpos"])
            lift_qpos[self.hand.lift_qpos_index] += LIFT_DIST
        # Raise gradually: a fast lift yanks the object out of the grip, a slow quasi-static lift lets it follow.
        mj_ho.control_hand_with_interp(grasp_data["squeeze_qpos"], lift_qpos, step_outer=cfg.lift_interp_steps)

        # Target = the resting pose raised by the palm's actual rise (LIFT_DIST for mocap, else measured arm-IK rise).
        palm_rise = (float(mj_ho.data.xpos[lift_bid][2]) - palm_z0) if lift_bid is not None else LIFT_DIST
        target_obj = deepcopy(init_obj)
        target_obj[2] += palm_rise
        final_obj = mj_ho.get_obj_pose()
        delta_pos, delta_angle = compute_delta_qpos(target_obj, final_obj)
        # Also require post-lift hand contact, from the LIVE sim state (a kinematic reset falsely reports no contact).
        ho_final, _ = mj_ho.scan_contacts()
        held = any(
            c["contact_dist"] < cfg.contact_threshold and c["body1_name"] in self.hand.valid_body_name for c in ho_final
        )
        succ_flag = (
            (delta_pos < cfg.lift_slip_thre)
            & (delta_angle < cfg.angle_thre)
            & (final_obj[2] - init_obj[2] >= cfg.min_lift_dist)
            & held
        )
        return bool(succ_flag), delta_pos, delta_angle

    def evaluate(self, grasp_data: dict, record: bool = False) -> dict:
        """Penetration/contact at the grasp pose + force-closure success; `record=True` adds the replay trajectory."""
        cfg, hand = self.cfg, self.hand
        mj_ho = self.build_scene(grasp_data, record=record)

        ho_contact, hh_contact = mj_ho.get_contact_info(
            grasp_data["grasp_qpos"], grasp_data["obj_pose"], obj_margin=cfg.contact_margin
        )
        contact_dist_dict = {name: cfg.contact_margin for name in hand.finger_prefix}
        contact_link_set = set()
        for c in ho_contact:
            for finger_prefix in contact_dist_dict:
                if c["body1_name"].startswith(finger_prefix):
                    contact_dist_dict[finger_prefix] = min(contact_dist_dict[finger_prefix], c["contact_dist"])
                    break
            if np.abs(c["contact_dist"]) < cfg.contact_threshold and c["body1_name"] in hand.valid_body_name:
                contact_link_set.add(c["body1_name"])
        contact_dist_lst = list(contact_dist_dict.values())

        if cfg.setting == "tabletop":
            succ_flag, delta_pos, delta_angle = self.simulate_tabletop(mj_ho, grasp_data)
        else:
            succ_flag, delta_pos, delta_angle = self.simulate_fc(mj_ho, grasp_data)
        result = {
            "ho_pene": max(-min([c["contact_dist"] for c in ho_contact]) if ho_contact else 0, 0),
            "self_pene": max(-min([c["contact_dist"] for c in hh_contact]) if hh_contact else 0, 0),
            "contact_num": len(contact_link_set),
            "contact_dist": np.mean([max(i, 0.0) for i in contact_dist_lst]),
            "contact_consis": np.max(contact_dist_lst) - np.min(contact_dist_lst),
            "succ_flag": succ_flag,
            "delta_pos": delta_pos,
            "delta_angle": delta_angle,
            "sim_steps": mj_ho.sim_steps,
        }
        if record:
            result["replay_hand_qpos"] = np.array(mj_ho.hand_qpos_log)  # (T, nq_hand)
            result["replay_obj_pose"] = np.array(mj_ho.obj_pose_log)  # (T, 7) [xyz, quat_wxyz]
        return result
