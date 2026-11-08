"""Task-frame whole-body teleop — SAPIEN physics + viser, single process.

Open http://localhost:8080. Drag the two wrist frames (W0 frame); the upper-body IK +
pelvis compensation track them and the GR00T policy balances the free base. Tick
"Engage walking" to drive vx / vy / vyaw. The base is welded through the warmup, then released.
The green ghost is the command the controller tracks, the red line its error; untick "Pelvis
compensation" or "Whole-body gravity comp" to see what each fixes.

Usage:
    uv run --extra sim python scripts/sim/teleop.py
"""

from dataclasses import dataclass

import numpy as np
import tyro

from hug.control.config import HZ, SimDeployment
from hug.control.step import deliver_to_sim, solve_body_command
from hug.teleop import TeleopConfig
from hug.utils import Rate
from hug.viewer import RobotView, TrackingGhost, handle_pose7, make_server, set_pose, world_T_W0_pose7
from hug.wbc.lower_body_policy import LowerBodyPolicy

VIZ_EVERY = 3


@dataclass
class Args:
    viser_port: int = 8080
    warmup_s: float = 2.0  # Seconds to hold the stand pose (balance-policy warmup) before the base is released


def main(args: Args) -> None:
    print("[init] loading SAPIEN, the whole-body controller and the lower-body policy ...")
    deployment = SimDeployment(track_hands="both")
    teleop = TeleopConfig()
    backend = deployment.backend()
    wbc = deployment.controller()
    wbc.reset(SimDeployment.stand_q21())  # seed the IK where the sim starts, so tick 0 is not a step
    lower = LowerBodyPolicy()

    # ---- viser: /w0 carries world_T_W0 so the robot and the handles (W0-local) move against the ground
    server = make_server(args.viser_port)
    w0 = server.scene.add_frame("/w0", show_axes=False)
    robot = RobotView(server, backend)
    server.scene.add_frame("/w0/estimator_center", show_axes=True, axes_length=0.2, axes_radius=0.006)
    handles = {}
    for side in ("right", "left"):
        pose = wbc.hand_pose7_W0(side, backend.read_state().motor_q)
        handles[side] = server.scene.add_transform_controls(f"/w0/target_{side}_hand", scale=0.15, line_width=3.0)
        set_pose(handles[side], pose)
    ghost = TrackingGhost(server, ("right", "left"))

    with server.gui.add_folder("Walk command"):
        walk_cb = server.gui.add_checkbox("Engage walking", initial_value=False)
        vx_s = server.gui.add_slider("vx (m/s)", min=-0.5, max=0.5, step=0.05, initial_value=0.0)
        vy_s = server.gui.add_slider("vy (m/s)", min=-0.4, max=0.4, step=0.05, initial_value=0.0)
        vyaw_s = server.gui.add_slider("vyaw (rad/s)", min=-0.6, max=0.6, step=0.05, initial_value=0.0)
    with server.gui.add_folder("Grip (0 open → 1 closed)"):
        grip = {
            "left": server.gui.add_slider("left", min=0.0, max=1.0, step=0.05, initial_value=0.0),
            "right": server.gui.add_slider("right", min=0.0, max=1.0, step=0.05, initial_value=0.0),
        }
    with server.gui.add_folder("Control"):
        pelvis_cb = server.gui.add_checkbox("Pelvis compensation", initial_value=True)
        gravity_cb = server.gui.add_checkbox("Whole-body gravity comp", initial_value=wbc.whole_body_gravity)
        ghost_cb = server.gui.add_checkbox("Show command ghost", initial_value=True)
    gravity_cb.on_update(lambda _evt: setattr(wbc, "whole_body_gravity", gravity_cb.value))
    ghost_cb.on_update(lambda _evt: ghost.set_visible(ghost_cb.value))
    with server.gui.add_folder("Status"):
        err_lbl = {
            s: server.gui.add_text(f"{s[0].upper()} EE err (mm)", initial_value="—", disabled=True)
            for s in ("right", "left")
        }
        sag_lbl = server.gui.add_text("waist pitch sag (deg)", initial_value="—", disabled=True)
        base_lbl = server.gui.add_text("base height (m)", initial_value="—", disabled=True)
    reset_btn = server.gui.add_button("Reset")

    pending = {"reset": False}
    reset_btn.on_click(lambda _evt: pending.__setitem__("reset", True))
    print(
        f"[viser] open http://localhost:{args.viser_port} — drag the wrist frames; tick 'Engage walking'. Ctrl+C exits."
    )

    # ---- loop
    dt = 1.0 / HZ
    n_warm = int(args.warmup_s * HZ)
    cmd = {side: handle_pose7(h) for side, h in handles.items()}
    tick = 0
    rate = Rate(HZ)
    while True:
        if pending["reset"]:
            pending["reset"] = False
            backend.reset()
            wbc.reset(SimDeployment.stand_q21())
            lower.reset_warmup()
            for side, h in handles.items():
                cmd[side] = wbc.hand_pose7_W0(side, backend.read_state().motor_q)
                set_pose(h, cmd[side])
            tick = 0

        # ramp the compensator over ~1 s, so toggling it drifts the hands instead of jumping them by up to ~15 cm
        wbc.compensator.weight = float(np.clip(wbc.compensator.weight + (dt if pelvis_cb.value else -dt), 0.0, 1.0))
        motor_q = backend.read_state().motor_q
        for side, h in handles.items():
            cmd[side] = teleop.track(cmd[side], handle_pose7(h), dt)
        walking = walk_cb.value and tick >= n_warm  # opt-in, and only after warmup
        body = solve_body_command(
            wbc,
            motor_q,
            right_pose7=cmd["right"],
            left_pose7=cmd["left"],
            right_hand=teleop.grip(grip["right"].value),
            left_hand=teleop.grip(grip["left"].value),
        )
        deliver_to_sim(
            backend,
            lower,
            body,
            vx=float(vx_s.value) if walking else 0.0,
            vy=float(vy_s.value) if walking else 0.0,
            yaw=float(vyaw_s.value) if walking else 0.0,
        )
        if tick == n_warm and backend.base_welded:
            backend.release_base()
            print("[ready] warmup complete — base released.", flush=True)

        if tick % VIZ_EVERY == 0:
            mq = backend.read_state().motor_q
            pelvis = wbc.pelvis_pose7_W0(mq)
            set_pose(w0, world_T_W0_pose7(pelvis, backend.base_pose7()))
            robot.update(mq, pelvis)
            achieved = {side: wbc.hand_pose7_W0(side, mq) for side in handles}
            for side in handles:
                err_lbl[side].value = f"{np.linalg.norm(achieved[side][:3] - cmd[side][:3]) * 1e3:5.1f}"
            if ghost_cb.value:
                ghost.update(cmd, achieved, right_grip=teleop.grip(grip["right"].value))
            sag_lbl.value = f"{np.degrees(mq[14] - body.waist_yrp[2]):+.2f}"
            base_lbl.value = f"{float(backend.base_pose7()[2]):.3f}"

        tick += 1
        rate.sleep()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        print("\n[exit] interrupted.")
