# pyright: reportMissingImports=false
"""Chest-camera (L515) calibration on the G1, solved by robokit's hand-eye calibrator.

Take both robot hands off; the rope takes the weight, feet resting on the ground. The arms visit 12 shipped poses
(``--qpos-file`` for your own), then the SAM 2 tab (http://localhost:8000) shows each frame: mask both arms.
robokit solves the camera pose with ``torso_link`` as its mount, starting from the shipped extrinsic, and writes
``T_torso_camera.json`` and ``overlays/`` to ``results/calibration/<robot_id>/``. Capture and masks are cached
there; ``--redo segment`` re-masks, ``--redo capture`` starts over.

Usage:
    uv run --extra real-cu128 python scripts/real/calibration/g1_calibration.py --network-iface enp4s0
"""

import contextlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
import viser
import warp as wp
from robokit.helpers.hand_eye_calibration import HandEyeCalibrator, compute_robot_masks, presets
from robokit.robo import Robot
from robokit.xform.numpy.rotation_conversions import quaternion_to_matrix
from viser.extras import ViserUrdf

from hug import paths
from hug.perception.object_cloud import load_torso_T_cam
from hug.perception.realsense import start_l515_pipeline
from hug.perception.sam2 import SamPipeline
from hug.real.arm_presets import (
    ARM_DEFAULT_QPOS,
    ARM_JOINT_LIMITS_HIGH,
    ARM_JOINT_LIMITS_LOW,
    ARM_LIMIT_MARGIN,
    CALIBRATION_ARM_QPOS,
)
from hug.real.g1_motion import G1ArmMover, G1ArmMoverConfig
from hug.real.g1_robot import G1Robot, channel_factory_init
from hug.viewer import open_in_browser, real_robot_urdf

JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_JOINTS = [f"{side}_{j}_joint" for side in ("left", "right") for j in JOINTS]  # ARM_JOINT_INDICES order
ARM_LINKS = [n.replace("_joint", "_link") for n in ARM_JOINTS]  # what the solver renders against the masks
ARM_VEL, SETTLE_S, FRAME_DROPS = 0.3, 1.5, 5  # rad/s; the strap-hung torso swings ~2 s; frames buffered in a move


@dataclass
class Args:
    network_iface: str = "enp4s0"  # DDS interface to the G1
    robot_id: str = "default"  # results go to results/calibration/<robot_id>/
    redo: List[Literal["capture", "segment"]] = field(default_factory=list)  # rerun a cached stage
    qpos_file: Optional[Path] = None  # (N, 14) arm poses (.npy, e.g. teach_arm_qpos.py); default: the 12 shipped
    initial_torso_T_cam: Optional[Path] = paths.TORSO_T_CAM  # starting guess (.json); None drags one in viser
    realsense_serial: Optional[str] = None  # pick one L515 when several are attached


def capture(args: Args, robot: Robot) -> dict:
    """Move the arms through the poses; one colour frame per pose, with the encoder reading at that instant."""
    poses = CALIBRATION_ARM_QPOS if args.qpos_file is None else np.load(args.qpos_file).astype(np.float64)
    # a pose past a joint stop would trip G1ArmMover's settle check
    poses = np.clip(poses, ARM_JOINT_LIMITS_LOW + ARM_LIMIT_MARGIN, ARM_JOINT_LIMITS_HIGH - ARM_LIMIT_MARGIN)
    channel_factory_init(args.network_iface)
    g1 = G1Robot(control_dt=1.0 / 300.0, read_only=False, topic_prefix="rt")
    # 0.06 rad settle tol: pure PD without gravity feed-forward sags up to ~0.035 rad; FK reads the encoder, so no loss
    mover = G1ArmMover(g1, G1ArmMoverConfig(settle_q_tol=0.06))
    mover.hold_current()
    print(
        "[capture] arms on PD, legs limp: rope takes the weight, feet on the ground, robot hands off. Starting in 2 s."
    )
    time.sleep(2.0)
    mover.move_to_arm_qpos(ARM_DEFAULT_QPOS, max_velocity=ARM_VEL)
    pipeline, _, intrinsic, _, _ = start_l515_pipeline(realsense_serial=args.realsense_serial)
    idx = [list(robot.spec.actuated_joint_names).index(n) for n in ARM_JOINTS]
    images, full_q = [], np.zeros((len(poses), robot.num_actuated_joints))
    for i, q in enumerate(poses):
        print(f"[capture] pose {i + 1}/{len(poses)}")
        mover.move_to_arm_qpos(q, max_velocity=ARM_VEL)
        mover.hold(SETTLE_S)  # PD-hold, not sleep: an unfed command watchdog (~100-200 ms) drops the motors
        for _ in range(FRAME_DROPS):
            pipeline.wait_for_frames()
        mover.hold(0.05)  # re-pulse PD so the frame grab and the encoder read are within ~30 ms
        bgr = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        full_q[i, idx] = mover.get_arm_qpos()  # FK from the realized encoder reading, not the target
    mover.move_to_arm_qpos(ARM_DEFAULT_QPOS, max_velocity=ARM_VEL)
    g1.Close()
    pipeline.stop()
    return dict(images=np.stack(images), intrinsic=intrinsic, full_qpos=full_q)


def pick_camera_viser(q_full: np.ndarray, image: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Drag /camera onto the robot in viser for a mount the shipped extrinsic misses; returns OpenCV T_camera_pelvis."""
    with contextlib.redirect_stdout(io.StringIO()):  # viser's banner ignores verbose; the URL is printed below
        server = viser.ViserServer(verbose=False)
    server.scene.set_up_direction("+z")
    view = ViserUrdf(server, real_robot_urdf(None), root_node_name="/robot")  # no mount: that is what you place
    view.update_cfg(np.concatenate([q_full, np.zeros(6)]))  # the FK body's joints, then the hand's
    H, W = image.shape[:2]
    controls = server.scene.add_transform_controls(
        "/camera", scale=0.3, position=(0.05, 0.0, 0.4), wxyz=(0.5, -0.5, -0.5, 0.5)
    )
    fov = 2.0 * np.arctan2(H / 2.0, float(intrinsic[1, 1]))
    server.scene.add_camera_frustum("/camera/frustum", fov=fov, aspect=W / H, scale=0.2, image=image, variant="filled")
    done = server.gui.add_button("Use this pose")
    clicked = []
    done.on_click(lambda _: clicked.append(True))
    print("[viser] open http://localhost:8080 — drag /camera onto the robot's image, then 'Use this pose'.")
    while not clicked:
        time.sleep(0.05)
    T_pelvis_cam = np.eye(4)
    T_pelvis_cam[:3, :3] = quaternion_to_matrix(np.asarray(controls.wxyz, dtype=np.float64))
    T_pelvis_cam[:3, 3] = controls.position
    server.stop()
    return np.linalg.inv(T_pelvis_cam)


def main(args: Args) -> None:
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    out = paths.CALIBRATION_DIR / args.robot_id
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    robot = Robot.load(str(paths.G1_29DOF_HEAD_URDF), base_link_name="pelvis", load_meshes=True)

    recaptured = "capture" in args.redo or not (out / "capture.npz").exists()
    if recaptured:
        np.savez(out / "capture.npz", **capture(args, robot))
    data = np.load(out / "capture.npz")
    images, intrinsic, full_q = data["images"], data["intrinsic"], data["full_qpos"]

    if recaptured or "segment" in args.redo or not (out / "masks.npy").exists():
        sam = SamPipeline()
        url = sam.start()
        print(f"[sam2] {url} — mask both arms in each frame, then Submit.")
        if sam.wait_until_ready():
            open_in_browser(url)
        n = len(images)
        masks = np.stack([sam.segment(img, label=f"ARM {i + 1}/{n}: mask both arms") for i, img in enumerate(images)])
        sam.shutdown()
        np.save(out / "masks.npy", masks.astype(np.uint8))
    masks = np.load(out / "masks.npy")

    # torso_link carries the camera: the solve takes it as the mount, so the waist (upstream of it) cancels
    dev = torch.device("cuda")
    q_t = torch.from_numpy(full_q.astype(np.float32)).to(dev)
    T_pelvis_torso = robot.forward_kinematics_via_matrix_torch(q_t)[:, robot.link_names.index("torso_link")]
    T_torso_pelvis = torch.linalg.inv(T_pelvis_torso)
    if args.initial_torso_T_cam is not None:
        T_torso_camera_init = load_torso_T_cam(args.initial_torso_T_cam)
    else:  # the picker works in the URDF root: T_camera_torso = T_camera_pelvis @ T_pelvis_torso
        T_torso_camera_init = np.linalg.inv(
            pick_camera_viser(full_q[0], images[0], intrinsic) @ T_pelvis_torso[0].cpu().numpy()
        )
    H, W = images.shape[1:3]
    K = torch.from_numpy(intrinsic).float().to(dev)
    init = torch.from_numpy(np.linalg.inv(T_torso_camera_init)).float().to(dev)
    hec = HandEyeCalibrator(presets.rgb, robot, camera_intrinsic=K, height=H, width=W, render_link_names=ARM_LINKS)
    solved = hec.solve(init, target_masks=torch.from_numpy(masks).float().to(dev), q=q_t, T_mount_base=T_torso_pelvis)
    T_torso_camera = np.linalg.inv(solved.cpu().numpy().astype(np.float64))
    rows = ",\n".join(f"  {json.dumps(row)}" for row in T_torso_camera.tolist())  # one matrix row per line
    (out / "T_torso_camera.json").write_text(f'{{"T_torso_camera": [\n{rows}\n]}}\n')

    # overlays: the rendered arms before and after the solve next to the mask; "Solved" should sit on the mask
    renders = [
        compute_robot_masks(robot, q_t, K, T, H, W, T_torso_pelvis, ARM_LINKS).cpu().numpy() > 0.5
        for T in (init, solved)
    ]
    for i, img in enumerate(images):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        for ax, m, title in zip(
            axes, (renders[0][i], renders[1][i], masks[i] > 0), ("Initial guess", "Solved", "SAM 2 mask")
        ):
            shaded = img.copy()
            shaded[m] //= 4
            ax.imshow(shaded)
            ax.set_title(title, fontsize=18)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "overlays" / f"{i:02d}.png")
        plt.close(fig)
    moved = np.linalg.inv(T_torso_camera_init) @ T_torso_camera
    print(f"[solve] wrote {out / 'T_torso_camera.json'} ({np.linalg.norm(moved[:3, 3]) * 1000:.0f} mm from the guess)")
    print(f"[solve] check {out / 'overlays'}, then: preflight.py --torso-T-cam {out / 'T_torso_camera.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
