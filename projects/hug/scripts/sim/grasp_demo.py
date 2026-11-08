"""Whole-body CVAE grasp demo — SAPIEN/PhysX physics + viser, single process.

Open http://localhost:8080, click [1 · Capture + Infer], then [2 · Execute grasp]. The policy
decodes at the latent mean (same cloud → same grasp); tick "Sample latent" for a fresh random
grasp per click. Scene controls: the object dropdown (the arrow buttons step through it), a
table-top slider and Randomize. Each invalidates a planned grasp and greys out while
a run plays. Two seconds after the robot lands home the object re-seats itself, so
[2 · Execute grasp] simply runs again.

Usage:
    uv run --extra sim python scripts/sim/grasp_demo.py [--scene tomato_soup_can]
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser

from hug import paths
from hug.control.config import HZ, PLACE_ARM_QPOS, READY_ARM_QPOS, SimDeployment
from hug.control.step import body_command_from_q21, deliver_to_sim, glide_q21_commands, solve_body_command
from hug.grasp.planner import plan_grasp_trajectory
from hug.grasp.policy import POLICY_IDLE_HAND, GraspPolicy
from hug.perception.object_cloud import head_camera_W0_T_cam, pc_cam_to_w0
from hug.sim.scene import available_objects, object_scale, object_visual_mesh_path
from hug.utils import Rate, slew_pose7
from hug.viewer import (
    GhostHands,
    RobotView,
    make_server,
    set_pose,
    texture_material,
    world_T_W0_pose7,
)
from hug.wbc.lower_body_policy import LowerBodyPolicy

DEFAULT_SCENE = "mustard_bottle"  # curated: alphabetical would open on aluminum_foil, the flakiest entry
OBJECTS = tuple(available_objects())

# The slider goes below the supported 0.15 m floor on purpose, so the degradation is visible.
TABLE_TOP_MIN, TABLE_TOP_MAX = 0.10, 0.90
SLEW_LIN, SLEW_ANG = 0.40, 1.50  # m/s, rad/s: rate caps when moving between the stand and idle poses
VIZ_EVERY = 3  # control ticks per render update (~17 Hz)
AUTO_RESET_S = 2.0  # idle time after a run before the object is re-seated for the next one


@dataclass
class Args:
    scene: str = DEFAULT_SCENE  # Object name: a folder of paths.OBJECTS_DIR
    policy_path: Path = paths.DEFAULT_GRASP_POLICY  # CVAE grasp policy (.onnx)
    viser_port: int = 8080
    warmup_s: float = 2.0  # Seconds to hold the stand pose (balance-policy warmup) before the base is released


def _object_hint(name: str) -> str:
    scale = object_scale(name)
    size = "true size" if scale == 1.0 else f"shown at x{scale:g} source size"
    return f"{OBJECTS.index(name) + 1} of {len(OBJECTS)} — {size}"


def _object_position(name: str) -> str:
    return f"{OBJECTS.index(name) + 1} / {len(OBJECTS)}"


def _add_object_node(server: viser.ViserServer, name: str):
    """Textured OBJ where the object ships one, else flat-shaded geometry; scaled like the physics."""
    mesh = trimesh.load(str(object_visual_mesh_path(name)), force="mesh")
    scale = object_scale(name)
    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        mesh.visual.material = texture_material(mesh.visual.material)
        return server.scene.add_mesh_trimesh("/object", mesh, scale=scale)
    return server.scene.add_mesh_simple(
        "/object",
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        color=(230, 200, 50),
        scale=scale,
    )


class GraspDemo:
    def __init__(self, args: Args) -> None:
        print(f"[init] loading SAPIEN (scene={args.scene!r}), the whole-body controller and the policies ...")
        self.deployment = SimDeployment()
        self.backend = self.deployment.backend(scene_object=args.scene)
        self.wbc = self.deployment.controller()
        self.wbc.reset(SimDeployment.stand_q21())  # seed the IK where the sim starts, so tick 0 is not a step
        self.lower = LowerBodyPolicy()
        self.policy = GraspPolicy(args.policy_path)

        self.stand_pose = self.wbc.fk_hand_pose7("right", np.zeros(7)).astype(np.float32)
        self.idle_pose = self.wbc.fk_hand_pose7("right", READY_ARM_QPOS).astype(np.float32)
        self.place_pose = self.wbc.fk_hand_pose7("right", PLACE_ARM_QPOS).astype(np.float32)
        self.q21_home = self.wbc.home_q21().astype(np.float64)
        self.hand_idle = POLICY_IDLE_HAND.astype(np.float64)

        # ---- viser
        self.server = make_server(args.viser_port)
        self.w0 = self.server.scene.add_frame("/w0", show_axes=False)
        self.robot = RobotView(self.server, self.backend)
        self.ghosts = GhostHands(self.server)
        self.table_node = self.server.scene.add_box(
            "/table",
            dimensions=tuple(2 * s for s in self.backend.table_half_size),
            position=tuple(self.backend.table_pos),
            color=(153, 102, 51),
        )
        self.object_node = _add_object_node(self.server, args.scene)
        self.plan_nodes: list = []
        self.last_stages: tuple | None = None
        self.pending: set[str] = set()  # button clicks arrive on viser's thread; the loop drains them
        self._build_gui(args.scene)

        # ---- loop state
        self.n_warm = int(args.warmup_s * HZ)
        self.n_glide = max(int(self.deployment.timing.glide_home * HZ), 2)
        self.tick = 0
        self.current_object = args.scene
        self.ghosts_shown = self.ghost_cb.value
        self.cmd_pose = self.stand_pose.astype(np.float64)
        self.q21 = SimDeployment.stand_q21()
        self.traj = None
        self.exec_i = -1  # -1 = not executing
        self.glide = None  # command iterator while returning home
        self.glide_done = 0
        self.locked = False  # scene controls greyed out while a run plays
        self.auto_reset_at: int | None = None  # tick at which an untouched scene re-seats its object
        self.rt_marks: list[tuple[float, int]] = []
        self.rate = Rate(HZ)

    # ------------------------------------------------------------------ GUI
    def _build_gui(self, scene: str) -> None:
        gui = self.server.gui
        with gui.add_folder("Scene"):
            self.object_dd = gui.add_dropdown("Object", options=OBJECTS, initial_value=scene, hint=_object_hint(scene))
            self.object_step = gui.add_button_group(
                _object_position(scene), ("◀", "▶"), hint="Step through the object list"
            )
            self.object_step.on_click(self._step_object)
            self.table_sl = gui.add_slider(
                "Table top (m)",
                min=TABLE_TOP_MIN,
                max=TABLE_TOP_MAX,
                step=0.005,
                initial_value=round(self.backend.table_top_z, 3),
            )
        with gui.add_folder("Grasp"):
            capture_btn = gui.add_button("1 · Capture + Infer")
            self.execute_btn = gui.add_button("2 · Execute grasp", disabled=True)
            self.sample_cb = gui.add_checkbox("Sample latent (random grasp)", initial_value=False)
            self.ghost_cb = gui.add_checkbox("Ghost grasp hands", initial_value=True)
            randomize_btn = gui.add_button("Randomize object")
            reset_btn = gui.add_button("Reset object")
        with gui.add_folder("Status"):
            self.phase_lbl = gui.add_text("phase", initial_value="WARMUP", disabled=True)
            self.pc_lbl = gui.add_text("point cloud", initial_value="—", disabled=True)
            self.rt_lbl = gui.add_text("RT factor", initial_value="—", disabled=True)

        for name, btn in (
            ("capture", capture_btn),
            ("execute", self.execute_btn),
            ("randomize", randomize_btn),
            ("reset", reset_btn),
        ):
            btn.on_click(lambda _evt, name=name: self.pending.add(name))

        # Everything that edits the scene greys out while a run plays: the grasp is solved against one object pose.
        self.run_locks = (self.object_dd, self.table_sl, capture_btn, randomize_btn, reset_btn)

    # ------------------------------------------------------------ scene edits
    def _step_object(self, evt: viser.GuiEvent) -> None:
        """◀ ▶ walk the object list; _poll_gui picks the new value up and swaps it in."""
        if self.object_dd.disabled:  # viser button groups cannot grey out, so honour the dropdown's state
            return
        step = -1 if evt.target.value == "◀" else 1
        self.object_dd.value = OBJECTS[(OBJECTS.index(self.object_dd.value) + step) % len(OBJECTS)]

    def _clear_markers(self) -> None:
        """Drop what the plan drew: point cloud, stage frames, ghost hands."""
        self.ghosts.clear()
        for n in self.plan_nodes:
            n.remove()
        self.plan_nodes.clear()
        self.last_stages = None
        self.pc_lbl.value = "—"

    def _clear_plan(self) -> None:
        """Forget a planned grasp (the object it was solved against has moved)."""
        self._clear_markers()
        self.traj, self.exec_i = None, -1
        self.execute_btn.disabled = True

    @property
    def busy(self) -> bool:
        """A grasp is playing back, or the robot is gliding home from one."""
        return self.exec_i >= 0 or self.glide is not None

    def _sync_locks(self) -> None:
        if self.busy == self.locked:
            return
        self.locked = self.busy
        for handle in self.run_locks:
            handle.disabled = self.locked

    def _poll_gui(self) -> None:
        """Apply scene edits from the control thread: the physics scene must not change under a stepping loop."""
        self._sync_locks()
        if self.pending:
            self.auto_reset_at = None  # the operator got there first
        if self.busy:
            self.pending -= {"reset", "randomize"}  # a click that raced the greying
        if "reset" in self.pending:
            self.pending.discard("reset")
            self.backend.reset_object()
        if "randomize" in self.pending:
            self.pending.discard("randomize")
            x, y, yaw = self.backend.randomize_object()
            print(f"[scene] object → x={x:.3f} y={y:.3f} yaw={np.degrees(yaw):+.0f}°")
            self._clear_plan()
        if self.object_dd.value != self.current_object:
            self.current_object = self.object_dd.value
            self.backend.swap_object(self.current_object)  # scene-only: the robot keeps balancing
            self.object_node.remove()
            self.object_node = _add_object_node(self.server, self.current_object)
            self.object_dd.hint = _object_hint(self.current_object)
            self.object_step.label = _object_position(self.current_object)
            self.auto_reset_at = None
            self._clear_plan()
            print(f"[scene] {self.current_object} (table top {self.backend.table_top_z:.3f} m).")
        if abs(float(self.table_sl.value) - self.backend.table_top_z) > 1e-6:
            self.backend.set_table_height(float(self.table_sl.value))
            self.table_node.position = tuple(self.backend.table_pos)
            self.auto_reset_at = None
            self._clear_plan()
        if self.ghost_cb.value != self.ghosts_shown:
            self.ghosts_shown = self.ghost_cb.value
            if self.ghosts_shown and self.last_stages is not None:
                self.ghosts.draw(*self.last_stages)
            else:
                self.ghosts.clear()
        if self.auto_reset_at is not None and self.tick >= self.auto_reset_at:
            self.auto_reset_at = None
            self.backend.reset_object()  # back at its spawn pose, so the same plan can simply run again
            print("[scene] object re-seated — Execute runs the same grasp, or capture a new one.")

    # ---------------------------------------------------------------- grasp
    def _capture_and_plan(self, motor_q: np.ndarray) -> None:
        pc_cam = self.backend.capture_object_pc()
        if pc_cam.shape[0] == 0:
            self.pc_lbl.value = "EMPTY (object not visible?)"
            return
        self._clear_plan()
        pc_world = pc_cam_to_w0(pc_cam, head_camera_W0_T_cam(self.wbc.estimator, motor_q))
        frac = self.backend.last_cloud_z_fraction
        warn = "  CLIPPED — grasp may be poor" if frac < self.backend.scene_cfg.min_cloud_z_fraction else ""
        self.pc_lbl.value = f"{pc_world.shape[0]} pts, {frac * 100:.0f}% of object{warn}"
        c = pc_world.mean(axis=0)
        print(
            f"[capture] cloud {pc_world.shape[0]} pts ({frac * 100:.0f}% of object height), "
            f"centroid ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) m"
        )

        stage_poses, stage_hands = self.policy.predict(pc_world, sample=self.sample_cb.value)
        for name, p in zip(("pregrasp", "grasp", "postgrasp"), stage_poses):
            print(f"  {name:9s} ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) m")
        self.traj = plan_grasp_trajectory(
            idle_pose=self.idle_pose,
            idle_hand=POLICY_IDLE_HAND,
            stage_poses=stage_poses,
            stage_hands=stage_hands,
            place_pose=self.place_pose,
            timing=self.deployment.timing,
            hz=HZ,
        )
        self.execute_btn.disabled = False
        self.plan_nodes.append(
            self.server.scene.add_point_cloud("/w0/pc", points=pc_world, colors=(255, 150, 0), point_size=0.003)
        )
        for i, pose in enumerate(stage_poses):
            frame = self.server.scene.add_frame(f"/w0/stage_{i}", axes_length=0.06, axes_radius=0.003)
            set_pose(frame, pose)
            self.plan_nodes.append(frame)
        self.last_stages = (stage_poses, stage_hands)
        if self.ghost_cb.value:
            self.ghosts.draw(stage_poses, stage_hands)
        print(f"[policy] trajectory {self.traj.num_frames / HZ:.1f} s — click 'Execute grasp'.")

    def _next_target(self) -> tuple[np.ndarray, np.ndarray]:
        """This tick's right-hand ``(pose7, hand_6)``: replay the trajectory, or slew toward the stand/idle pose."""
        if self.exec_i >= 0 and self.traj is not None:
            pose7, hand_6 = self.traj.get_frame(self.exec_i)
            self.cmd_pose = pose7.astype(np.float64)
            self.exec_i += 1
            if self.exec_i >= self.traj.num_frames:
                print("[execute] trajectory done — returning home.")
                self.exec_i, self.glide_done = -1, 0
                self.glide = glide_q21_commands(
                    self.wbc,
                    self.q21.astype(np.float64),
                    self.q21_home,
                    self.n_glide,
                    right_hand=self.hand_idle,
                    left_hand=self.hand_idle,
                )
            return self.cmd_pose, hand_6.astype(np.float64)
        target = self.stand_pose if self.tick < self.n_warm else self.idle_pose
        self.cmd_pose = slew_pose7(self.cmd_pose, target.astype(np.float64), 1.0 / HZ, SLEW_LIN, SLEW_ANG)
        return self.cmd_pose, self.hand_idle

    def _glide_command(self, motor_q: np.ndarray):
        """Joint-space glide home, bypassing the IK; re-seeds the IK at home when it lands."""
        cmd = next(self.glide, None)
        self.wbc.compensator.compute_delta(motor_q)  # keep the rate limiter warm while the IK is out of the loop
        if cmd is None:
            self.glide = None
            self.wbc.reset(self.q21_home.astype(np.float32))
            self.cmd_pose = self.idle_pose.astype(np.float64)
            self.execute_btn.disabled = self.traj is None
            self.auto_reset_at = self.tick + int(AUTO_RESET_S * HZ)
            self._clear_markers()  # the grasp has been run; its cloud and ghosts are just clutter now
            print("[execute] back at the init pose.")
            cmd = body_command_from_q21(
                self.wbc, self.q21_home.astype(np.float32), right_hand=self.hand_idle, left_hand=self.hand_idle
            )
        self.glide_done += 1
        return cmd

    # ----------------------------------------------------------------- loop
    def _control_tick(self) -> None:
        motor_q = self.backend.read_state().motor_q
        if "capture" in self.pending and not self.busy and self.tick >= self.n_warm:
            self.pending.discard("capture")
            self._capture_and_plan(motor_q)
            self.rate.resync()  # inference is not missed ticks
        if "execute" in self.pending and self.traj is not None and not self.busy:
            self.pending.discard("execute")
            self.exec_i = 0
            self.execute_btn.disabled = True

        pose7, right_hand = self._next_target()
        if self.glide is not None:
            cmd = self._glide_command(motor_q)
        else:
            cmd = solve_body_command(
                self.wbc, motor_q, right_pose7=pose7.astype(np.float32), right_hand=right_hand, left_hand=self.hand_idle
            )
        self.q21 = cmd.q21
        deliver_to_sim(self.backend, self.lower, cmd)

        if self.tick == self.n_warm and self.backend.base_welded:
            self.backend.release_base()
            print("[ready] warmup complete — base released.", flush=True)

    def _phase(self) -> str:
        if self.tick < self.n_warm:
            return "WARMUP"
        if self.exec_i >= 0 and self.traj is not None:
            return self.traj.segment_at(self.exec_i).upper().replace("_", " ")
        if self.glide is not None:
            return f"RETURN HOME  ({min(self.glide_done / self.n_glide, 1.0) * 100:.0f}%)"
        if self.auto_reset_at is not None:
            return f"DONE  (object re-seats in {max(self.auto_reset_at - self.tick, 0) / HZ:.1f} s)"
        return "IDLE" + ("  (traj ready)" if self.traj is not None else "")

    def _update_view(self) -> None:
        mq = self.backend.read_state().motor_q
        pelvis = self.wbc.pelvis_pose7_W0(mq)
        set_pose(self.w0, world_T_W0_pose7(pelvis, self.backend.base_pose7()))
        self.robot.update(mq, pelvis)
        set_pose(self.object_node, self.backend.object_pose7())
        self.phase_lbl.value = self._phase()
        self.rt_marks.append((time.time(), self.tick))
        if len(self.rt_marks) > 20:
            self.rt_marks.pop(0)
            (t0, k0), (t1, k1) = self.rt_marks[0], self.rt_marks[-1]
            if t1 > t0:
                self.rt_lbl.value = f"{(k1 - k0) / HZ / (t1 - t0):.2f}x"

    def run(self) -> None:
        print(
            f"[viser] open http://localhost:{self.server.get_port()} — "
            "Capture + Infer, then Execute grasp. Ctrl+C to exit."
        )
        self.rate.resync()
        while True:
            self._poll_gui()
            self._control_tick()
            if self.tick % VIZ_EVERY == 0:
                self._update_view()
            self.tick += 1
            self.rate.sleep()


def main(args: Args) -> None:
    if args.scene not in OBJECTS:
        raise SystemExit(f"unknown scene {args.scene!r}; available: {', '.join(OBJECTS)}")
    try:
        GraspDemo(args).run()
    except KeyboardInterrupt:
        print("\n[exit] interrupted.")


if __name__ == "__main__":
    main(tyro.cli(Args))
