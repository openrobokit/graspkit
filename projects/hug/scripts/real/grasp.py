"""Real-robot grasp: chest cam + SAM2 + grasp policy + whole-body controller, one viser panel.

Feet on the ground, the rope taking the weight; it stands to balance. Run ``real/preflight.py`` on a new calibration.
    1 · Capture → 2 · Segment → 3 · Run policy   as in preflight: RGB-D + joints → object cloud in W0 → stages
    4 · Execute      approach → [gate] → grasp → [gate] → lift → place → return; enabled only when the stages
                     sit in the workspace box in a sane order — your eyes on the coloured stage hands do the rest
    Continue         passes a gate; the button says what it will do (close the hand, then lift)
    Reset / Abort    drop capture + plan / glide the arm back to READY (Ctrl+C mid-trial aborts, a second one quits)

Usage:
    uv run --extra real-cu128 python scripts/real/grasp.py --network-iface enp4s0 \\
        --right-hand-serial-port /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0
"""

import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import tyro

from hug.control.config import (
    HZ,
    IDLE_HAND,
    PLACE_ARM_QPOS,
    READY_ARM_QPOS,
    RealControlConfig,
    RealDeployment,
)
from hug.control.step import BodyCommand, body_command_from_q21, glide_q21_commands
from hug.grasp.planner import GraspTrajectory, RealGraspTiming, plan_grasp_trajectory
from hug.real.g1_inspire_interface import G1InspireConfig, G1InspireInterface
from hug.real.lifecycle import (
    deliver_to_real,
    graceful_shutdown,
    smooth_init_to_ready,
    step_to_target,
    tick_compensator,
)
from hug.real.perception import PerceptionArgs, PerceptionFrontEnd, State

RETREAT_S = 4.0  # abort glide back to READY
# The two operator gates: (trajectory segment the loop freezes after, name shown, what Continue does).
GATES = (("settle_pre_grasp", "pre-grasp", "close the hand"), ("settle_post_grasp", "grasp", "lift"))


@dataclass
class Args(PerceptionArgs):
    network_iface: str = "enp4s0"  # DDS network interface (the real robot is domain 0, prefix 'rt')
    right_hand_serial_port: str = G1InspireConfig.right_hand_serial_port  # CH340 USB-RS485; empty = hand not driven
    warmup_s: float = 3.0  # Ramp from wherever the interface left the arms into READY
    preroll_s: float = 1.5  # Settle at READY before each trajectory starts


class RealGraspDemo(PerceptionFrontEnd):
    def __init__(self, args: Args) -> None:
        # network_interface goes through G1InspireConfig: the subprocess opens its own DDS participant
        self.interface = G1InspireInterface(
            G1InspireConfig(
                network_interface=args.network_iface,
                enable_walking_policy=True,
                right_hand_serial_port=args.right_hand_serial_port,
            )
        )
        print("[init] loading the whole-body controller ...")
        self.wbc = RealDeployment(control=RealControlConfig(), track_hands="right").controller()
        self.timing = RealGraspTiming()
        super().__init__(
            args,
            motor_q=self.interface.get_motor_qpos,
            estimator=self.wbc.estimator,
            right_hand_q=lambda: self.interface.get_hand_qpos()[6:],
        )

        self.warmup_s = args.warmup_s
        self.preroll_ticks = max(int(args.preroll_s * HZ), 1)
        self.idle_pose = self.wbc.fk_hand_pose7("right", READY_ARM_QPOS).astype(np.float32)
        self.place_pose = self.wbc.fk_hand_pose7("right", PLACE_ARM_QPOS).astype(np.float32)
        # The stand between trials bypasses the IK: its rest tasks at a static target pull the torso down (bent legs)
        self.hold_q21 = self.wbc.home_q21()
        self.hold_cmd = body_command_from_q21(self.wbc, self.hold_q21, right_hand=IDLE_HAND.astype(np.float64))
        self.cmd_pose7 = self.idle_pose

        self.trial = 0
        self.traj: Optional[GraspTrajectory] = None
        self.exec_i = 0
        self.gates: list[tuple[int, str, str]] = []  # (frame, name, action)
        self.preroll_left = 0
        self.segment_label = ""
        # Glide: a command iterator; the IK re-seeds at hold_q21 when it lands.
        self.glide: Optional[Iterator[BodyCommand]] = None
        self.glide_label = ""

    # ------------------------------------------------------------------ GUI

    def _build_gui(self) -> None:
        super()._build_gui()
        gui = self.server.gui
        with self.grasp_folder:
            self.execute_btn = gui.add_button("4 · Execute", order=40, hint="Enabled once the plan passes the checks")
            self.continue_btn = gui.add_button("Continue", order=50, hint="Passes the gate the robot is paused at")
            self.abort_btn = gui.add_button("Abort", order=90, color="red", hint="Glide the arm back to READY")
        with self.status_folder:
            self.trial_lbl = gui.add_text("trial", initial_value="0", disabled=True, order=15)
            self.hand_err_lbl = gui.add_text("hand err (mm)", initial_value="—", disabled=True, order=50)
        for tag, btn in (("execute", self.execute_btn), ("continue", self.continue_btn), ("abort", self.abort_btn)):
            btn.on_click(self.click(tag))

    def _update_buttons(self) -> None:
        super()._update_buttons()
        s = self.state
        self.execute_btn.disabled = not (s == State.PLANNED and self.safety is not None and self.safety.overall)
        self.continue_btn.disabled = s != State.GATE
        self.abort_btn.disabled = s not in (State.EXECUTING, State.GATE)

    def _state_text(self) -> str:
        s = self.state
        if s == State.PLANNED and self.safety is not None and self.safety.overall:
            return "PLANNED · SAFE — click 4 · Execute"
        if s == State.EXECUTING:
            return "RETURNING to READY" if self.glide is not None else f"EXECUTING · {self.segment_label or 'PREROLL'}"
        if s == State.GATE:
            _, name, action = self.gates[0]
            return f"PAUSED at {name} — Continue will {action}"
        if s == State.RETREAT:
            return "ABORTING — gliding back to READY"
        return super()._state_text()

    def _on_robot_viz(self, motor_q: np.ndarray) -> None:
        achieved = self.wbc.hand_pose7_W0("right", motor_q)
        self.hand_err_lbl.value = f"{np.linalg.norm(achieved[:3] - self.cmd_pose7[:3]) * 1e3:5.1f}"

    # ------------------------------------------------------------ actions

    def drain_pending(self) -> set[str]:
        for tag in super().drain_pending():
            if tag == "execute" and self.state == State.PLANNED and self.safety is not None and self.safety.overall:
                self._start_trial()
            elif tag == "continue" and self.state == State.GATE:
                self._pass_gate()
            elif tag == "abort" and self.state in (State.EXECUTING, State.GATE):
                self._start_retreat()
        return set()

    def reset(self) -> None:
        self.traj = None
        self.glide = None
        super().reset()

    def _start_trial(self) -> None:
        assert self.stages is not None
        poses, hands = self.stages
        self.trial += 1
        self.trial_lbl.value = str(self.trial)
        self.wbc.ik.reset(self.hold_q21)  # the first IK tick continues from the held pose, not from setup
        # The trajectory ends at THROW; the return to READY is the joint-space glide, not a segment.
        self.traj = plan_grasp_trajectory(
            idle_pose=self.idle_pose,
            idle_hand=IDLE_HAND,
            stage_poses=[p.astype(np.float32) for p in poses],
            stage_hands=hands,
            place_pose=self.place_pose,
            timing=self.timing,
            hz=HZ,
        )
        print(f"[execute] trial {self.trial}: {self.traj.num_frames} frames ({self.traj.end('throw') / HZ:.1f} s)")
        self.exec_i = 0
        self.preroll_left = self.preroll_ticks
        self.gates = [(self.traj.end(segment), name, action) for segment, name, action in GATES]
        self.segment_label = ""
        self.set_state(State.EXECUTING)

    def _pass_gate(self) -> None:
        self.gates.pop(0)
        self.continue_btn.label = "Continue"
        self.segment_label = ""
        self.set_state(State.EXECUTING)

    def _start_glide(self, duration_s: float, label: str) -> None:
        """Joint-space glide from the measured arms to READY, for both the return home and the abort.

        Bypasses the IK (a grasp cannot be undone in EE space) and lands exactly on READY, so the stand sees no step."""
        q21_from = np.zeros(21, dtype=np.float64)
        q21_from[0] = float(self.wbc.ik.qpos[0])
        q21_from[7:21] = self.interface.get_motor_qpos()[15:29]
        self.glide = glide_q21_commands(
            self.wbc,
            q21_from,
            self.hold_q21.astype(np.float64),
            max(int(duration_s * HZ), 1),
            right_hand=IDLE_HAND.astype(np.float64),
            interp_order=self.timing.interp_order,
        )
        self.glide_label = label

    def _start_retreat(self) -> None:
        print(f"[abort] retreating to READY over {RETREAT_S:.0f} s ...")
        self.traj = None
        self._start_glide(RETREAT_S, "retreat")
        self.set_state(State.RETREAT)

    # ----------------------------------------------------------- tick loop

    def _track(self, pose7: np.ndarray, hand_6: np.ndarray) -> None:
        step_to_target(self.interface, self.wbc, pose7, hand_6)
        self.cmd_pose7 = pose7

    def _command_tick(self) -> None:
        """Emit exactly one command for this tick, per state."""
        s = self.state
        if not s.busy:
            deliver_to_real(self.interface, self.hold_cmd)
            tick_compensator(self.wbc, self.interface.get_motor_qpos())
            self.cmd_pose7 = self.idle_pose
            return
        if s == State.GATE:  # hold the frozen boundary frame until the operator clicks through
            assert self.traj is not None
            self._track(*self.traj.get_frame(self.gates[0][0] - 1))
            return
        if self.glide is not None:  # RETREAT, or EXECUTING's return home
            self._glide_tick()
            return
        assert self.traj is not None
        if self.preroll_left > 0:
            self.preroll_left -= 1
            self._track(self.idle_pose, IDLE_HAND)
            return
        if self.gates and self.exec_i == self.gates[0][0]:
            _, name, action = self.gates[0]
            print(f"[execute] paused at {name} — click Continue to {action}")
            self.continue_btn.label = f"Continue · {action}"
            self.set_state(State.GATE)
            self._command_tick()
            return
        if self.exec_i >= self.traj.end("throw"):
            print(f"[execute] returning to READY over {self.timing.glide_home:.1f} s ...")
            self._start_glide(self.timing.glide_home, "home")
            self.refresh_state_text()
            self._glide_tick()
            return
        pose7, hand_6 = self.traj.get_frame(self.exec_i)
        self.exec_i += 1
        self._track(pose7, hand_6)
        segment = self.traj.segment_at(min(self.exec_i, self.traj.num_frames - 1)).upper().replace("_", " ")
        if segment != self.segment_label:
            self.segment_label = segment
            self.refresh_state_text()

    def _glide_tick(self) -> None:
        """One glide command; on landing, re-seed the IK, drop the plan and stand at READY."""
        assert self.glide is not None
        cmd = next(self.glide, None)
        if cmd is None:
            self.wbc.ik.reset(self.hold_q21)
            label = self.glide_label
            self.reset()  # the object has been placed: the next trial starts from a new capture
            print(f"[{'execute' if label == 'home' else 'abort'}] back at READY.")
            cmd = self.hold_cmd
        deliver_to_real(self.interface, cmd)
        tick_compensator(self.wbc, self.interface.get_motor_qpos())

    def run(self) -> None:
        # Quintic ramp from wherever the interface left the arms into READY; a bare direct command would snap them.
        smooth_init_to_ready(self.interface, self.wbc, self.hold_q21, duration_s=self.warmup_s)
        self.open_tabs()
        print("  ready — click 1 · Capture. Ctrl+C exits.\n")
        self.rate.resync()
        while True:
            try:
                self.drain_pending()
                self._command_tick()
                if not self.state.busy or self.state == State.GATE:  # motion segments give every ms to the IK
                    self.viz_tick()
                self.rate.sleep()
            except KeyboardInterrupt:
                if self.state == State.RETREAT:
                    print("\n[exit] interrupted during retreat — quitting.")
                    break
                if self.state.busy:
                    print("\n[abort] interrupted mid-trial.")
                    self._start_retreat()
                    continue
                print("\n[exit] interrupted.")
                break

    def shutdown(self) -> None:
        self.close()
        graceful_shutdown(self.interface, self.wbc)


def main(args: Args) -> None:
    mp.set_start_method("spawn", force=True)
    demo = RealGraspDemo(args)
    try:
        demo.run()
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args))
