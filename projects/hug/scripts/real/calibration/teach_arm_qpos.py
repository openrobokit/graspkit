"""Drag-and-drop teaching for G1 arm calibration qpos.

Take both robot hands off, as for ``g1_calibration.py``; the rope takes the weight, feet resting on the ground.
Arm kp goes to zero (``--kp 5`` = soft hold) with RNEA gravity comp of the bare arms (clipped at ±10 Nm), so
they float; hold them through the 2 s damping-only warmup. An unmodeled load drifts an arm, so hold or rest
it while pressing Enter: the snapshot is the encoder reading at that instant.
Enter snapshots the arm qpos, ``s`` saves and exits, ``q`` quits. Output is (N, 14) in ``ARM_JOINT_INDICES``
order (left arm, then right): the ``--qpos-file`` for ``g1_calibration.py``, for a mount the shipped poses miss.

Usage:
    uv run --extra real-cu128 python scripts/real/calibration/teach_arm_qpos.py --network-iface enp4s0
"""

import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import pinocchio as pin
import tyro

from hug import paths
from hug.control.config import GRAVITY_COMP_CLIP_NM
from hug.real.arm_presets import ARM_JOINT_LIMITS_HIGH, ARM_JOINT_LIMITS_LOW
from hug.real.g1_motion import ARM_JOINT_INDICES
from hug.real.g1_robot import G1_NUM_MOTOR, G1Robot, channel_factory_init
from hug.wbc.pose_estimator import G1_29DOF_HEAD_JOINT_NAMES, motor_q_29_to_32

# arm damping: enough to suppress oscillation after release, light enough to drag freely
TEACH_ARM_KD = 0.5
# Pure damping on non-arm motors (legs, waist).
NONARM_KD = 2.0


def _build_arm_gravity_helper() -> Callable[[np.ndarray], np.ndarray]:
    """Closure ``arm_gravity_tau(motor_q_29) -> (14,)``: RNEA gravity torques for the arm joints on the head URDF."""
    urdf = ET.parse(paths.G1_29DOF_HEAD_URDF).getroot()
    # bare arms, as calibration runs: no hands, and Unitree's wrist yaw (the URDF adds 0.5 kg, tuned with hands on)
    bare_kg = {"left_rubber_hand": 0.0, "right_rubber_hand": 0.0, "R_hand_base_link": 0.0}
    bare_kg |= {f"{side}_wrist_yaw_link": 0.08457647 for side in ("left", "right")}
    for link, kg in bare_kg.items():
        for mass in urdf.iterfind(f"link[@name='{link}']/inertial/mass"):
            mass.set("value", str(kg))
    model = pin.buildModelFromXML(ET.tostring(urdf, encoding="unicode"))
    data = model.createData()

    # Names of qpos-bearing joints in the pinocchio model order.
    model_qpos_joint_names = [model.names[jid] for jid in range(1, model.njoints) if model.joints[jid].nq > 0]
    if set(model_qpos_joint_names) != set(G1_29DOF_HEAD_JOINT_NAMES):
        missing = sorted(set(model_qpos_joint_names) - set(G1_29DOF_HEAD_JOINT_NAMES))
        extra = sorted(set(G1_29DOF_HEAD_JOINT_NAMES) - set(model_qpos_joint_names))
        raise RuntimeError(
            f"head URDF joints differ from G1_29DOF_HEAD_JOINT_NAMES; "
            f"missing in name list={missing}, extra in name list={extra}"
        )
    # Remap input (32-vec in G1_29DOF_HEAD_JOINT_NAMES order) -> model qpos order
    indices = np.array([G1_29DOF_HEAD_JOINT_NAMES.index(n) for n in model_qpos_joint_names], dtype=int)

    # idx_v of each arm joint, to pull the arm components out of rnea's (nv,) tau
    arm_joint_names = G1_29DOF_HEAD_JOINT_NAMES[18:32]  # last 14 = arms
    assert len(arm_joint_names) == 14
    arm_idx_v: List[int] = []
    for name in arm_joint_names:
        found = False
        for jid in range(1, model.njoints):
            if model.names[jid] == name:
                arm_idx_v.append(int(model.joints[jid].idx_v))
                found = True
                break
        if not found:
            raise RuntimeError(f"joint {name!r} not found in head URDF")
    arm_idx_v_arr = np.array(arm_idx_v, dtype=int)

    qvel_zero = np.zeros(model.nv)
    qacc_zero = np.zeros(model.nv)

    def arm_gravity_tau(motor_q_29: np.ndarray) -> np.ndarray:
        # Splice HEAD_QPOS into the head slot to get a 32-vec the head URDF expects.
        qpos_32 = motor_q_29_to_32(motor_q_29)
        qpos_model = qpos_32[indices]
        tau_model = pin.rnea(model, data, qpos_model, qvel_zero, qacc_zero)
        return tau_model[arm_idx_v_arr]

    return arm_gravity_tau


def _damping_shutdown(g1: G1Robot, hold_s: float = 0.5) -> None:
    """Defensive shutdown: hand off to pure damping for ``hold_s`` then Close."""
    zeros = np.zeros(G1_NUM_MOTOR)
    kd_full = np.full(G1_NUM_MOTOR, NONARM_KD)
    n_steps = max(int(hold_s / g1._control_dt), 1)
    for _ in range(n_steps):
        try:
            g1.SetJointCmd(q=zeros, dq=zeros, tau=zeros, kp=zeros, kd=kd_full)
            g1.Step()
        except Exception:
            break
    try:
        g1.Close()
    except Exception:
        pass


def _start_input_listener(
    record_event: threading.Event,
    save_event: threading.Event,
    quit_event: threading.Event,
) -> threading.Thread:
    def listen() -> None:
        print("[teach] arms floating. Enter snapshots a pose; s Enter saves and exits; q Enter quits.")
        while not quit_event.is_set():
            try:
                cmd = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                quit_event.set()
                return
            if cmd == "s":
                save_event.set()
                quit_event.set()
            elif cmd == "q":
                quit_event.set()
            elif cmd == "":
                record_event.set()
            else:
                print(f"[teach] unknown '{cmd}': Enter snapshots, s saves and exits, q quits")

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    return t


def teach(iface: str, output_path: Path, arm_kp: float, control_dt: float) -> None:
    channel_factory_init(iface)
    arm_gravity_tau = _build_arm_gravity_helper()

    g1 = G1Robot(control_dt=control_dt, read_only=False, topic_prefix="rt")

    kp = np.zeros(G1_NUM_MOTOR)
    kp[ARM_JOINT_INDICES] = arm_kp
    kd = np.full(G1_NUM_MOTOR, NONARM_KD)
    kd[ARM_JOINT_INDICES] = TEACH_ARM_KD
    zeros = np.zeros(G1_NUM_MOTOR)

    # print the first gravity-comp tau to eyeball: upper arm 1-5 Nm, wrist < 1 Nm; > 10 Nm at rest means trouble
    initial_q = g1.GetMotorQ()
    initial_tau = arm_gravity_tau(initial_q)
    print(f"[teach] gravity comp at rest: up to {np.abs(initial_tau).max():.1f} N·m (over 10 means a model mismatch)")
    print(f"[teach] hold both arms: damping only for 2 s, then they float (kp {arm_kp:g}, kd {TEACH_ARM_KD:g})")
    warmup_steps = int(2.0 / control_dt)
    for _ in range(warmup_steps):
        motor_q = g1.GetMotorQ()
        g1.SetJointCmd(q=motor_q, dq=zeros, tau=zeros, kp=zeros, kd=np.full(G1_NUM_MOTOR, NONARM_KD))
        g1.Step()

    record_event = threading.Event()
    save_event = threading.Event()
    quit_event = threading.Event()
    _start_input_listener(record_event, save_event, quit_event)

    recorded: List[np.ndarray] = []
    try:
        while not quit_event.is_set():
            motor_q = g1.GetMotorQ()
            tau_ff_arm14 = arm_gravity_tau(motor_q)
            tau_ff = np.zeros(G1_NUM_MOTOR)
            tau_ff[ARM_JOINT_INDICES] = np.clip(tau_ff_arm14, -GRAVITY_COMP_CLIP_NM, GRAVITY_COMP_CLIP_NM)
            # q_des = current_q so the (small or zero) kp has no positional target drift.
            g1.SetJointCmd(q=motor_q, dq=zeros, tau=tau_ff, kp=kp, kd=kd)
            g1.Step()

            if record_event.is_set():
                record_event.clear()
                arm_q = motor_q[ARM_JOINT_INDICES].copy()
                outside = np.where((arm_q < ARM_JOINT_LIMITS_LOW) | (arm_q > ARM_JOINT_LIMITS_HIGH))[0]
                if outside.size:
                    # refuse: the calibration script's PD would fight the firmware clamp and trip its settle check
                    print(
                        f"[teach] rejected: arm joints {outside.tolist()} past their limit; move back in, Enter again"
                    )
                else:
                    recorded.append(arm_q)
                    print(f"[teach] pose {len(recorded)} recorded")
    except KeyboardInterrupt:
        pass
    finally:
        _damping_shutdown(g1)

    poses = f"{len(recorded)} pose" + "s" * (len(recorded) != 1)
    if save_event.is_set() and recorded:
        out = np.array(recorded, dtype=np.float64)
        assert out.shape == (len(recorded), 14)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, out)
        print(f"[teach] saved {poses} to {output_path}; calibrate with them:")
        print(
            "  uv run --extra real-cu128 python scripts/real/calibration/g1_calibration.py"
            f" --network-iface {iface} --qpos-file {output_path}"
        )
    else:
        print(f"[teach] exited without saving ({poses} discarded)")


@dataclass
class Args:
    network_iface: str = "enp4s0"  # DDS network interface
    output: Path = paths.CALIBRATION_DIR / "teach_arm_qpos.npy"  # (N, 14) arm qpos; --qpos-file for g1_calibration.py
    kp: float = 0.0  # arm stiffness while teaching: 0 = pure gravity comp, 5-10 = soft hold
    rate: float = 300.0  # control rate (Hz)


if __name__ == "__main__":
    args = tyro.cli(Args)
    teach(args.network_iface, args.output, args.kp, 1.0 / args.rate)
