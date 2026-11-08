"""Chest-camera front end shared by the real entrypoints.

:class:`PerceptionFrontEnd` is the operator workflow before anything moves — capture, SAM2 mask to a
cleaned W0 cloud, policy, workspace check — with its viser panel. Nothing here commands the robot: joint
angles come in through a callable. Imports without pyrealsense2 or open3d; both load on first use.
"""

import enum
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import viser
from viser.extras import ViserUrdf

from hug import paths
from hug.control.config import HZ
from hug.grasp.policy import GraspPolicy
from hug.perception.object_cloud import (
    chest_camera_W0_T_cam,
    load_torso_T_cam,
    pc_cam_to_w0,
    unproject_depth_to_cam_xyz,
)
from hug.perception.sam2 import SamPipeline
from hug.utils import Rate, mat_to_pose7
from hug.viewer import RealRobotView, make_server, open_in_browser
from hug.wbc.pose_estimator import LinkPoseEstimator, motor_q_29_to_32

STAGE_NAMES = ("pregrasp", "grasp", "postgrasp")
STAGE_COLORS = ((0, 220, 220), (255, 220, 0), (220, 0, 220))  # cyan / yellow / magenta

Workspace = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
# Where a stage pose may lie in W0 (ahead of the pelvis, table to chest height); outside is a policy hallucination
WORKSPACE_W0: Workspace = ((0.15, 0.80), (-0.55, 0.55), (0.40, 1.30))
# Grasp hand (R_hand_base_link) to the nearest object point: 0.09-0.15 m over ~90 real captures; farther, it missed
GRASP_MAX_DIST_M = 0.20

# Viz cadences. The render is cheap; the live-cloud unproject strides the depth 4× (49k px) and pushes rarely.
URDF_PERIOD_S = 0.1
LIVE_PC_PERIOD_S = 0.2
LIVE_PC_DEPTH_STRIDE = 4
LIVE_PC_MAX_POINTS = 12_000


@dataclass
class ObjectCloudConfig:
    """Mask → object-cloud knobs, one spelling for every real entrypoint."""

    max_depth_m: float = 3.0  # drop depth pixels beyond this before unprojection
    sam_mask_erode_px: int = 3  # L515 mixed pixels at the mask edge unproject to flyers; 0 disables
    voxel_m: float = 0.005  # voxel-downsample edge (m)
    outlier_nb: int = 20  # statistical outlier removal: neighbours ...
    outlier_std: float = 1.0  # ... and σ threshold; open3d's 2.0 default lets L515 silhouette halos through
    dbscan_eps_m: float = 0.01  # keep the largest DBSCAN cluster; ~2× the voxel, well under an object's size
    dbscan_min_points: int = 10

    def clean(self, pc_W0: np.ndarray) -> Optional[np.ndarray]:
        """Voxel + statistical-outlier + DBSCAN-largest-cluster, with these knobs."""
        from hug.perception.realsense import clean_object_pc  # open3d, only here

        return clean_object_pc(
            pc_W0,
            voxel_m=self.voxel_m,
            nb=self.outlier_nb,
            std=self.outlier_std,
            dbscan_eps_m=self.dbscan_eps_m,
            dbscan_min_points=self.dbscan_min_points,
        )


# ------------------------------------------------------------------ frames


def read_aligned_rgbd(pipeline: Any, align: Any, depth_scale: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """One aligned L515 frame → ``(rgb, depth_m)``, or ``None`` if empty. Blocks on ``wait_for_frames``."""
    frames = align.process(pipeline.wait_for_frames())
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    rgb = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
    return rgb, depth_m


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode a boolean mask by ``px`` pixels (3×3 ellipse, ``px`` iterations)."""
    if px <= 0 or not mask.any():
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=px).astype(bool)


# ------------------------------------------------------------------ object cloud


@dataclass
class SegmentedObject:
    mask: np.ndarray  # (H, W) bool, after erosion
    pc_W0: np.ndarray  # (M, 3) after cleanup
    centroid_W0: np.ndarray  # (3,)


def segment_object(
    sam: SamPipeline,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    W0_T_cam: np.ndarray,
    cfg: ObjectCloudConfig,
) -> Optional[SegmentedObject]:
    """SAM2 mask → erosion → unproject → W0 → cleanup; ``None``, after saying why, if any step leaves nothing.

    Blocks on the browser: call it from a worker thread inside a control loop."""
    raw_mask = sam.segment(rgb, label="OBJECT: the one to grasp")
    mask = erode_mask(raw_mask, cfg.sam_mask_erode_px)
    n_mask = int(mask.sum())
    if n_mask == 0:
        print("[segment] empty mask after erosion; nothing to lift (reduce --cloud.sam-mask-erode-px?)")
        return None
    pc_cam, _ = unproject_depth_to_cam_xyz(depth_m, intrinsic, cfg.max_depth_m, mask=mask)
    if pc_cam.shape[0] == 0:
        print("[segment] mask had no valid depth (all pixels z=0 or beyond max_depth)")
        return None
    pc_raw = pc_cam_to_w0(pc_cam, W0_T_cam)
    pc = cfg.clean(pc_raw)
    if pc is None or pc.shape[0] == 0:
        print(f"[segment] cleanup dropped all points (raw={pc_raw.shape[0]}); too small or all noise — tighter mask?")
        return None
    centroid = pc.mean(axis=0)
    lo, hi = pc.min(axis=0), pc.max(axis=0)
    c, e = centroid, hi - lo
    size = f"{e[0]:.2f}×{e[1]:.2f}×{e[2]:.2f}"
    print(f"[segment] {pc.shape[0]} pts, centroid ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) m, size {size} m")
    return SegmentedObject(mask=mask, pc_W0=pc, centroid_W0=centroid)


def save_scene_bundle(
    path: Path,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    motor_q: np.ndarray,
    intrinsic: np.ndarray,
    depth_scale: float,
    torso_T_cam: np.ndarray,
    cfg: ObjectCloudConfig,
    timestamp_ns: int,
) -> None:
    """The ``.npz`` capture bundle: everything needed to replay the perception path offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rgb=rgb.astype(np.uint8),
        depth_m=depth_m.astype(np.float32),
        mask=mask.astype(bool),
        motor_q=np.asarray(motor_q, dtype=np.float64),
        intrinsic=np.asarray(intrinsic, dtype=np.float64),
        depth_scale=np.float32(depth_scale),
        torso_T_cam=torso_T_cam.astype(np.float64),
        max_depth_m=np.float32(cfg.max_depth_m),
        erode_px=np.int32(cfg.sam_mask_erode_px),
        timestamp_ns=np.int64(timestamp_ns),
    )


# ------------------------------------------------------------------ stages


def normalize_stages(stage_poses: list[np.ndarray], z_bias_m: float = 0.0) -> Optional[list[np.ndarray]]:
    """``parse_stages`` poses → float64 pose7s with unit quaternions and ``z_bias_m`` added.

    A degenerate quaternion rejects the whole plan (``None``, after saying why): the trajectory needs all three."""
    out = []
    for i, p in enumerate(stage_poses):
        pose = np.asarray(p, dtype=np.float64).copy()
        norm = float(np.linalg.norm(pose[3:7]))
        if norm < 1e-6:
            print(f"[policy] stage {i} ({STAGE_NAMES[i]}): degenerate quaternion (norm {norm:.1e}) — re-run the policy")
            return None
        pose[2] += z_bias_m
        pose[3:7] /= norm
        out.append(pose)
    return out


class StageHands:
    """The right Inspire hand at each predicted stage, coloured by stage, with the path between them."""

    def __init__(self, server: viser.ViserServer) -> None:
        self._server = server
        self._nodes: list[tuple[Any, ViserUrdf]] = []
        self._path: Any = None

    def draw(self, poses7: list[np.ndarray], hands_6: list[np.ndarray]) -> None:
        self.clear()
        for i, (pose, hand) in enumerate(zip(poses7, hands_6)):
            frame = self._server.scene.add_frame(
                f"/grasp/stage{i}",
                show_axes=True,
                axes_length=0.06,
                axes_radius=0.003,
                position=tuple(pose[:3]),
                wxyz=tuple(pose[3:7]),
            )
            urdf = ViserUrdf(
                self._server,
                Path(paths.R_HAND_INSPIRE_URDF),
                root_node_name=f"/grasp/stage{i}/hand",
                mesh_color_override=STAGE_COLORS[i],
            )
            urdf.update_cfg(np.asarray(hand, dtype=np.float64))
            self._nodes.append((frame, urdf))
        verts = np.array([p[:3] for p in poses7], dtype=np.float32)
        self._path = self._server.scene.add_line_segments(
            "/grasp/path",
            np.stack([verts[:-1], verts[1:]], axis=1),
            colors=(255, 255, 255),
            thickness=3.0,
            thickness_units="screen",
        )

    def clear(self) -> None:
        # frame.remove() cascades into the URDF nodes, so urdf.remove() warns "already removed"; viser needs both
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for frame, urdf in self._nodes:
                urdf.remove()
                frame.remove()
        self._nodes.clear()
        if self._path is not None:
            self._path.remove()
            self._path = None


# ------------------------------------------------------------------ checks


def check_workspace(pos_W0: np.ndarray, workspace: Workspace) -> tuple[bool, str]:
    """Is ``pos_W0`` inside the ``(x, y, z)`` range box? ``(ok, reason)``."""
    for name, val, (lo, hi) in zip("xyz", pos_W0[:3], workspace):
        if val < lo:
            return False, f"{name}={val:+.2f}<{lo:+.2f}"
        if val > hi:
            return False, f"{name}={val:+.2f}>{hi:+.2f}"
    return True, ""


def check_stage_ordering(poses7: list[np.ndarray], pc_W0: Optional[np.ndarray]) -> str:
    """Empty if the plan is plausible, else why: grasp at the object, pre-grasp no nearer it, lift not below grasp."""
    p0, p1, p2 = (p[:3] for p in poses7)
    if pc_W0 is not None:
        near = float(np.linalg.norm(pc_W0 - p1, axis=1).min())
        if near > GRASP_MAX_DIST_M:
            return f"grasp {near:.2f} m from the object (>{GRASP_MAX_DIST_M:.2f}): out of reach or a bad mask?"
        centroid_W0 = pc_W0.mean(axis=0)
        d0 = float(np.linalg.norm(p0 - centroid_W0))
        d1 = float(np.linalg.norm(p1 - centroid_W0))
        if d0 + 0.01 < d1:
            return f"pre-grasp closer to object than grasp ({d0:.2f}<{d1:.2f})"
    if p2[2] + 0.005 < p1[2]:
        return f"lift z<grasp z ({p2[2]:+.2f}<{p1[2]:+.2f})"
    return ""


@dataclass
class SafetyReport:
    """Per-stage bbox verdicts plus the ordering verdict. ``overall`` gates Execute."""

    stage_bbox: list[str]  # per stage: "OK" or the violated bound, e.g. "x=+0.85>+0.80"
    ordering: str  # "OK" or "FAIL: <reason>"
    overall: bool


def safety_check(poses7: list[np.ndarray], pc_W0: Optional[np.ndarray], workspace: Workspace) -> SafetyReport:
    """Workspace box per stage plus the stage ordering; cheap and instant, no IK.

    The operator's eyes on the stage hands are the rest of the safety net."""
    verdicts = [check_workspace(p, workspace) for p in poses7]
    reason = check_stage_ordering(poses7, pc_W0)
    return SafetyReport(
        stage_bbox=["OK" if ok else why for ok, why in verdicts],
        ordering="OK" if not reason else f"FAIL: {reason}",
        overall=all(ok for ok, _ in verdicts) and not reason,
    )


# ------------------------------------------------------------------ the workflow


class State(str, enum.Enum):
    """Operator workflow: the five shared perception steps, then three that exist only with a robot interface."""

    IDLE = "IDLE"
    CAPTURED = "CAPTURED"
    SEGMENTING = "SEGMENTING"  # SAM worker running, operator in the browser
    SEGMENTED = "SEGMENTED"
    PLANNED = "PLANNED"  # stages + safety verdict available
    EXECUTING = "EXECUTING"  # trial ticks running (preroll / trajectory / return home)
    GATE = "GATE"  # frozen at a trajectory boundary, waiting for the operator
    RETREAT = "RETREAT"  # abort glide back to READY

    @property
    def busy(self) -> bool:
        return self in (State.EXECUTING, State.GATE, State.RETREAT)


@dataclass
class PerceptionArgs:
    """CLI fields shared by the real perception entrypoints; a script's ``Args`` extends it."""

    torso_T_cam: Path = paths.TORSO_T_CAM  # chest-camera extrinsic; default: the shipped 60° mount's
    realsense_serial: Optional[str] = None  # pin one L515 if several are attached
    cloud: ObjectCloudConfig = field(default_factory=ObjectCloudConfig)  # --cloud.voxel-m ...
    policy_path: Optional[Path] = None  # grasp policy (.onnx); default: the shipped hug-grasp.onnx
    grasp_z_bias_m: float = 0.0  # m, added to every stage z
    viser_port: int = 8080
    sam_port: int = 8000  # SAM2 annotator tab
    open_browser: bool = True  # open the control tab now, the annotator on the first Segment; no-op without a display


@dataclass
class Capture:
    """One frozen RGB-D + proprioception snapshot, with the camera pose it was taken from."""

    rgb: np.ndarray
    depth_m: np.ndarray
    motor_q: np.ndarray
    W0_T_cam: np.ndarray
    ts_ns: int


_BASE_TAGS = ("capture", "segment", "policy", "reset")


class PerceptionFrontEnd:
    """The capture → segment → policy workflow, with its viser panel.

    Clicks queue tags; :meth:`drain_pending` applies them on the caller's loop, re-checking state per action."""

    def __init__(
        self,
        args: PerceptionArgs,
        motor_q: Callable[[], np.ndarray],
        estimator: Optional[LinkPoseEstimator] = None,
        right_hand_q: Optional[Callable[[], np.ndarray]] = None,
    ) -> None:
        from hug.perception.realsense import start_l515_pipeline  # pyrealsense2, only here

        self.args = args
        self.motor_q = motor_q
        self.right_hand_q = right_hand_q  # the 6 measured Inspire joints for the render; None draws the hand open
        self.estimator = estimator if estimator is not None else LinkPoseEstimator()
        self.torso_T_cam = load_torso_T_cam(args.torso_T_cam)
        print(f"[init] torso_T_cam t={self.torso_T_cam[:3, 3].round(3).tolist()} m")
        self.policy = GraspPolicy(args.policy_path)
        self.rs_pipe, _, self.intrinsic, self.depth_scale, self.rs_align = start_l515_pipeline(
            realsense_serial=args.realsense_serial, warmup_frames=10
        )
        # SAM starts now, not on the first Segment click, so the operator learns about the second tab at startup.
        self.sam = SamPipeline(port=args.sam_port)
        self.sam_url = self.sam.start()

        self.server = make_server(args.viser_port)
        self.viser_url = f"http://localhost:{self.server.get_port()}"
        self.robot = RealRobotView(self.server, self.torso_T_cam)
        self.stage_hands = StageHands(self.server)
        self._draw_workspace()
        self.rate = Rate(HZ)

        self.state = State.IDLE
        self.capture: Optional[Capture] = None
        self.segmented: Optional[SegmentedObject] = None
        self.stages: Optional[tuple[list[np.ndarray], list[np.ndarray]]] = None  # (poses7, hands_6)
        self.safety: Optional[SafetyReport] = None
        self.pending: set[str] = set()
        self._sam_done = threading.Event()
        self._annotator_opened = False  # the tab opens when first needed, not at startup
        self._sam_result: Optional[SegmentedObject] = None
        self._object_node: Any = None
        self._live_node: Any = None
        self._viz_t = {"urdf": 0.0, "cloud": 0.0}
        self._build_gui()
        self.set_state(State.IDLE)

    # ------------------------------------------------------------------ GUI

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.add_markdown(f"SAM2 annotator: [{self.sam_url}]({self.sam_url})")
        self.grasp_folder = gui.add_folder("Grasp")
        with self.grasp_folder:
            self.capture_btn = gui.add_button(
                "1 · Capture", order=10, hint="Freeze one RGB-D frame and the joint angles"
            )
            self.segment_btn = gui.add_button(
                "2 · Segment", order=20, hint="Click the object in the SAM2 tab, then Submit"
            )
            self.policy_btn = gui.add_button("3 · Run policy", order=30, hint="Re-click to resample the cloud")
            self.sample_cb = gui.add_checkbox("Sample latent (random grasp)", initial_value=False, order=35)
            self.reset_btn = gui.add_button("Reset", order=80, hint="Drop the capture and the plan")
        self.status_folder = gui.add_folder("Status")
        with self.status_folder:
            self.state_lbl = gui.add_text("state", initial_value="", disabled=True, order=10)
            self.camera_lbl = gui.add_text("camera", initial_value="—", disabled=True, order=20)
            self.object_lbl = gui.add_text("object", initial_value="—", disabled=True, order=30)
            self.height_lbl = gui.add_text("pelvis height (m)", initial_value="—", disabled=True, order=40)
        with gui.add_folder("Grasp check"):
            self.overall_lbl = gui.add_text("overall", initial_value="—", disabled=True)
            self.stage_lbls = [gui.add_text(name, initial_value="—", disabled=True) for name in STAGE_NAMES]
            self.ordering_lbl = gui.add_text("ordering", initial_value="—", disabled=True)
        for tag, btn in (
            ("capture", self.capture_btn),
            ("segment", self.segment_btn),
            ("policy", self.policy_btn),
            ("reset", self.reset_btn),
        ):
            btn.on_click(self.click(tag))

    def click(self, tag: str) -> Callable[[Any], None]:
        """A button callback that queues ``tag`` for :meth:`drain_pending`."""
        return lambda _evt: self.pending.add(tag)

    def set_state(self, state: State) -> None:
        self.state = state
        self.refresh_state_text()
        self._update_buttons()

    def refresh_state_text(self) -> None:
        self.state_lbl.value = self._state_text()

    def _state_text(self) -> str:
        """What is happening and what to do next; extended by the actuating script."""
        s = self.state
        if s == State.PLANNED and self.safety is not None:
            return "PLANNED · SAFE" if self.safety.overall else "PLANNED · BLOCKED — re-run the policy or re-segment"
        return {
            State.IDLE: "IDLE — click 1 · Capture",
            State.CAPTURED: "CAPTURED — click 2 · Segment",
            State.SEGMENTING: "SEGMENTING — click the object in the SAM2 tab, then Submit",
            State.SEGMENTED: "SEGMENTED — click 3 · Run policy",
        }.get(s, s.value)

    def _update_buttons(self) -> None:
        """Grey out what the state does not allow, so the operator cannot click out of order."""
        s = self.state
        self.capture_btn.disabled = s.busy or s == State.SEGMENTING
        self.segment_btn.disabled = s not in (State.CAPTURED, State.SEGMENTED, State.PLANNED)
        self.policy_btn.disabled = s not in (State.SEGMENTED, State.PLANNED)
        self.reset_btn.disabled = s.busy

    def _draw_workspace(self) -> None:
        """Wireframe of the workspace box: the 12 edges join corners differing in one coordinate."""
        (x0, x1), (y0, y1), (z0, z1) = WORKSPACE_W0
        corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], np.float32)
        edges = [(corners[a], corners[b]) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
        self.server.scene.add_line_segments(
            "/workspace", np.array(edges), colors=(120, 120, 120), thickness=1.5, thickness_units="screen"
        )

    def _draw_object_cloud(self, pc_W0: np.ndarray) -> None:
        if self._object_node is not None:
            self._object_node.remove()
        self._object_node = self.server.scene.add_point_cloud(
            "/object_cloud", pc_W0.astype(np.float32), colors=(255, 0, 0), point_size=0.012, point_shape="circle"
        )

    def open_tabs(self) -> None:
        """Print both URLs and, with a display, open the control tab; the annotator opens on the first Segment."""
        print(f"\n  control   {self.viser_url}\n  annotator {self.sam_url}\n")
        if self.args.open_browser:
            open_in_browser(self.viser_url)

    def _open_annotator_once(self) -> None:
        """First Segment: wait for the server (it loads SAM2 before binding its port), then open its tab."""
        if self._annotator_opened or not self.args.open_browser:
            return
        self._annotator_opened = True
        if self.sam.wait_until_ready():
            open_in_browser(self.sam_url)
        else:
            print(f"[sam2] server slow to start; open {self.sam_url} by hand.")

    # ------------------------------------------------------------ actions

    def drain_pending(self) -> set[str]:
        """Apply clicks and the SAM result, re-checking state per action; returns the tags this class does not own."""
        if self._sam_done.is_set():
            self._sam_done.clear()
            self._finish_segmentation()
        pending, self.pending = self.pending, set()
        others = {tag for tag in pending if tag not in _BASE_TAGS}
        s = self.state
        if "capture" in pending and not s.busy and s != State.SEGMENTING:
            self._do_capture()
        elif "segment" in pending and s in (State.CAPTURED, State.SEGMENTED, State.PLANNED):
            self._start_segmentation()
        elif "policy" in pending and s in (State.SEGMENTED, State.PLANNED):
            self._do_policy()
        elif "reset" in pending and not s.busy:
            self.reset()
        return others

    def _do_capture(self) -> None:
        frame = read_aligned_rgbd(self.rs_pipe, self.rs_align, self.depth_scale)
        self.rate.resync()  # a blocking camera read is not missed ticks
        if frame is None:
            print("[capture] empty frame — try again.")
            return
        self._clear_plan()
        motor_q = np.asarray(self.motor_q(), dtype=np.float64)
        # The camera pose is FK'd here, on the loop's thread, so the SAM worker never touches the estimator.
        W0_T_cam = chest_camera_W0_T_cam(self.estimator, motor_q, self.torso_T_cam)
        self.capture = Capture(frame[0], frame[1], motor_q, W0_T_cam, time.time_ns())
        print(f"[capture] frame at {self.capture.ts_ns} ns")
        self.set_state(State.CAPTURED)

    def _start_segmentation(self) -> None:
        """SAM blocks on browser input — a worker thread, or the caller's loop stalls."""
        snap = self.capture
        assert snap is not None
        self._sam_result = None
        self._sam_done.clear()
        self.set_state(State.SEGMENTING)

        def worker() -> None:
            try:
                self._open_annotator_once()
                seg = segment_object(self.sam, snap.rgb, snap.depth_m, self.intrinsic, snap.W0_T_cam, self.args.cloud)
                if seg is not None:  # every segmentation is a replayable test case
                    out = paths.CAPTURED_DIR / f"scene_{snap.ts_ns}.npz"
                    save_scene_bundle(
                        out,
                        rgb=snap.rgb,
                        depth_m=snap.depth_m,
                        mask=seg.mask,
                        motor_q=snap.motor_q,
                        intrinsic=self.intrinsic,
                        depth_scale=self.depth_scale,
                        torso_T_cam=self.torso_T_cam,
                        cfg=self.args.cloud,
                        timestamp_ns=snap.ts_ns,
                    )
                    print(f"[segment] saved {out}")
                self._sam_result = seg
            finally:
                self._sam_done.set()

        threading.Thread(target=worker, daemon=True).start()

    def _finish_segmentation(self) -> None:
        seg, self._sam_result = self._sam_result, None
        if self.state != State.SEGMENTING:  # a Reset or Capture landed meanwhile: the operator moved on
            return
        if seg is None:
            self.set_state(State.CAPTURED)
            return
        self.segmented = seg
        self._draw_object_cloud(seg.pc_W0)
        c = seg.centroid_W0
        self.object_lbl.value = f"{seg.pc_W0.shape[0]} pts · centroid ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})"
        self.set_state(State.SEGMENTED)

    def _do_policy(self) -> None:
        seg = self.segmented
        assert seg is not None
        t0 = time.time()
        stage_poses, stage_hands = self.policy.predict(seg.pc_W0, sample=self.sample_cb.value)
        self.rate.resync()  # inference is not missed ticks
        poses = normalize_stages(stage_poses, self.args.grasp_z_bias_m)
        if poses is None:
            return
        hands = [np.asarray(h, dtype=np.float64) for h in stage_hands]
        self.stages = (poses, hands)
        self.stage_hands.draw(poses, hands)
        self.safety = safety_check(poses, seg.pc_W0, WORKSPACE_W0)
        for lbl, verdict, p in zip(self.stage_lbls, self.safety.stage_bbox, poses):
            lbl.value = f"{verdict} · ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"
        self.ordering_lbl.value = self.safety.ordering
        self.overall_lbl.value = "SAFE" if self.safety.overall else "BLOCKED"
        why = "" if self.safety.ordering == "OK" else f" — {self.safety.ordering.removeprefix('FAIL: ')}"
        print(f"[policy] {self.overall_lbl.value}{why} ({(time.time() - t0) * 1e3:.0f} ms)")
        for name, verdict, p in zip(STAGE_NAMES, self.safety.stage_bbox, poses):
            print(f"  {name:9s} {verdict:4s} ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}) m")
        self.set_state(State.PLANNED)

    def _clear_plan(self) -> None:
        """Forget the segmentation and the plan (a new capture, or Reset)."""
        self.segmented = None
        self.stages = None
        self.safety = None
        self.stage_hands.clear()
        if self._object_node is not None:
            self._object_node.remove()
            self._object_node = None
        self.object_lbl.value = "—"
        self.overall_lbl.value = "—"
        for lbl in self.stage_lbls:
            lbl.value = "—"
        self.ordering_lbl.value = "—"

    def reset(self) -> None:
        self.capture = None
        self._clear_plan()
        self.set_state(State.IDLE)

    # ---------------------------------------------------------------- viz

    def viz_tick(self) -> None:
        """Robot render at 10 Hz, the live coloured scene cloud at 5 Hz."""
        now = time.time()
        if now - self._viz_t["urdf"] >= URDF_PERIOD_S:
            self._viz_t["urdf"] = now
            motor_q = np.asarray(self.motor_q(), dtype=np.float64)
            pelvis = mat_to_pose7(self.estimator.estimate(motor_q_29_to_32(motor_q), link_name="pelvis").matrix)
            self.robot.update(motor_q, pelvis, self.right_hand_q() if self.right_hand_q is not None else None)
            self.height_lbl.value = f"{pelvis[2]:.3f}"
            self._on_robot_viz(motor_q)
        if now - self._viz_t["cloud"] >= LIVE_PC_PERIOD_S:
            self._viz_t["cloud"] = now
            self._update_live_cloud()

    def _on_robot_viz(self, motor_q: np.ndarray) -> None:
        """Extra status the actuating script shows beside the render (nothing here)."""

    def _update_live_cloud(self) -> None:
        """Live scene context and the calibration check: the cloud should sit on the rendered robot and the floor."""
        frames = self.rs_pipe.poll_for_frames()
        if not frames:
            return
        frames = self.rs_align.process(frames)
        df, cf = frames.get_depth_frame(), frames.get_color_frame()
        if not df or not cf:
            return
        s = LIVE_PC_DEPTH_STRIDE
        depth_m = np.asanyarray(df.get_data())[::s, ::s].astype(np.float32) * self.depth_scale
        rgb = np.asanyarray(cf.get_data())[::s, ::s, ::-1]  # the colour stream is BGR
        K = self.intrinsic.copy()
        K[:2, :] /= s  # fx, fy, cx, cy scale with the stride
        pc_cam, colors = unproject_depth_to_cam_xyz(depth_m, K, self.args.cloud.max_depth_m, rgb=rgb)
        valid_pct = 100.0 * pc_cam.shape[0] / depth_m.size
        if pc_cam.shape[0] == 0 or colors is None:
            self.camera_lbl.value = "no valid depth"
            return
        if pc_cam.shape[0] > LIVE_PC_MAX_POINTS:
            idx = np.random.choice(pc_cam.shape[0], LIVE_PC_MAX_POINTS, replace=False)
            pc_cam, colors = pc_cam[idx], colors[idx]
        motor_q = np.asarray(self.motor_q(), dtype=np.float64)
        pc_W0 = pc_cam_to_w0(pc_cam, chest_camera_W0_T_cam(self.estimator, motor_q, self.torso_T_cam))
        if self._live_node is not None:
            self._live_node.remove()
        self._live_node = self.server.scene.add_point_cloud(
            "/live_cloud", pc_W0.astype(np.float32), colors.astype(np.uint8), point_size=0.005, point_shape="circle"
        )
        self.camera_lbl.value = f"{pc_W0.shape[0]} pts · depth valid {valid_pct:.0f}%"

    # ---------------------------------------------------------------- loop

    def run(self) -> None:
        """The non-actuating loop: clicks and the render, nothing else."""
        while True:
            self.drain_pending()
            self.viz_tick()
            self.rate.sleep()

    def close(self) -> None:
        for name, stop in (("sam", self.sam.shutdown), ("camera", self.rs_pipe.stop)):
            try:
                stop()
            except Exception as exc:  # noqa: BLE001 — best effort on the way out
                print(f"[exit] {name} shutdown: {exc}")
