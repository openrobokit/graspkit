"""Real-robot WBC teleop: drag a target in viser, the right hand tracks it.

Start with the feet on the ground and the rope taking the weight; the robot stands up to balance. The smallest
thing that proves the stack ``real/grasp.py`` depends on, with no policy, cloud or SAM: drag /w0/target_right_hand
(W0 frame) → clamp to the workspace + slew limit → IK + pelvis comp + gravity comp → the 50 Hz interface (DDS +
Inspire serial). Right hand only; the left arm holds the IK's default. Arms stay at the interface's default kp
throughout: at kp=0 they drift under gravity even with gravity comp, so there is no engage gate.

Usage:
    uv run --extra real-cu128 python scripts/real/teleop.py --network-iface enp4s0
"""

import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import tyro

from hug.control.config import HZ, IDLE_HAND, RealControlConfig, RealDeployment
from hug.real.g1_inspire_interface import G1InspireConfig, G1InspireInterface
from hug.real.lifecycle import graceful_shutdown, smooth_init_to_ready, step_to_target
from hug.teleop import TeleopConfig
from hug.utils import Rate
from hug.viewer import RealRobotView, handle_pose7, make_server, set_pose

VIZ_EVERY = 5  # 50 Hz control / 5 = 10 Hz viser pushes


@dataclass
class Args:
    network_iface: str = "enp4s0"  # DDS network interface (real robot is domain 0, prefix 'rt')
    viser_port: int = 8080  # viser server port
    warmup_s: float = 3.0  # Quintic ramp from wherever the interface left the arms into READY


def main(args: Args) -> None:
    # Do NOT call ChannelFactoryInitialize here: the interface subprocess opens its own DDS participant
    interface = G1InspireInterface(
        G1InspireConfig(
            network_interface=args.network_iface,
            enable_walking_policy=True,
        )
    )

    deployment = RealDeployment(control=RealControlConfig(), track_hands="right")
    print("[init] loading the whole-body controller ...")
    wbc = deployment.controller()
    # Open at the real idle hand, not the sim stand pose TeleopConfig defaults to
    teleop = TeleopConfig(hand_open=IDLE_HAND.astype(np.float64))

    # ---- viser: /robot is re-anchored at W0_T_pelvis every viz tick, the frame the operator drags in
    server = make_server(args.viser_port)
    server.scene.add_grid("/ground", width=2.0, height=2.0)
    robot = RealRobotView(server)

    server.gui.add_text("Mode", initial_value="REAL", disabled=True)
    grip_slider = server.gui.add_slider("grip (0 open → 1 closed)", min=0.0, max=1.0, step=0.05, initial_value=0.0)
    with server.gui.add_folder("Status"):
        err_pos_lbl = server.gui.add_text("EE pos err (mm)", initial_value="—", disabled=True)
        slew_lag_lbl = server.gui.add_text("Slew lag (mm)", initial_value="—", disabled=True)
        height_lbl = server.gui.add_text("pelvis height (m)", initial_value="—", disabled=True)
        arm_lbl = server.gui.add_text("right arm q (rad)", initial_value="—", disabled=True)

    print(f"[viser] open http://localhost:{args.viser_port} — drag the wrist frame. Ctrl+C exits.")

    # ---- warmup: ramp current → READY at full kp, then seed the handle from the achieved FK so tick 0 is not a step
    smooth_init_to_ready(interface, wbc, wbc.home_q21(), duration_s=args.warmup_s)
    motor_q = interface.get_motor_qpos()
    cmd_pose7 = wbc.hand_pose7_W0("right", motor_q)

    # Created only now: assigning .position on a live handle does not reliably propagate before the first readback
    target_handle = server.scene.add_transform_controls("/target_right_hand", scale=0.15, line_width=3.0, opacity=0.9)
    set_pose(target_handle, cmd_pose7)
    achieved_frame = server.scene.add_frame("/achieved_right_hand", axes_length=0.05, axes_radius=0.002)
    set_pose(achieved_frame, cmd_pose7)
    robot.update(motor_q, wbc.pelvis_pose7_W0(motor_q), interface.get_hand_qpos()[6:])

    # ---- 50 Hz tracking loop
    dt = 1.0 / HZ
    tick = 0
    rate = Rate(HZ)
    print("[ready] tracking live.\n")
    try:
        while True:
            requested = handle_pose7(target_handle)
            cmd_pose7 = teleop.track(cmd_pose7, requested, dt)

            q21, _ = step_to_target(interface, wbc, cmd_pose7, teleop.grip(float(grip_slider.value)))

            if tick % VIZ_EVERY == 0:
                # FK from the live motor_q is ground truth; the IK's own pose pins the torso at 0.75 m
                motor_q = interface.get_motor_qpos()
                achieved = wbc.hand_pose7_W0("right", motor_q)
                pelvis = wbc.pelvis_pose7_W0(motor_q)
                set_pose(achieved_frame, achieved)
                robot.update(motor_q, pelvis, interface.get_hand_qpos()[6:])
                err_pos_lbl.value = f"{np.linalg.norm(achieved[:3] - cmd_pose7[:3]) * 1e3:5.1f}"
                slew_lag_lbl.value = f"{np.linalg.norm(cmd_pose7[:3] - requested[:3]) * 1e3:5.1f}"
                height_lbl.value = f"{pelvis[2]:.3f}"
                arm_lbl.value = "  ".join(f"{x:+.2f}" for x in q21[14:21])

            tick += 1
            rate.sleep()
    except KeyboardInterrupt:
        print("\n[exit] interrupted — compliant shutdown.")
    finally:
        graceful_shutdown(interface, wbc)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # the interface subprocess opens its own DDS participant
    main(tyro.cli(Args))
