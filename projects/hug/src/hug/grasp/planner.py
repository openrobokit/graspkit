"""Grasp trajectory planner and the sim / real timing profiles.

The pick-and-place is pre-planned as dense ``(pose7, hand_6)`` waypoints at the control rate and replayed
frame by frame. Nine segments: approach, settle_pre_grasp, grasp, settle_grasp, post_grasp, settle_post_grasp,
lift, place, throw. Returning home is a joint-space glide (``control.step.glide_q21_commands``), not a segment.
"""

from dataclasses import dataclass

import numpy as np

from hug.utils import reparam, slerp_wxyz

SEGMENTS = (
    "approach",
    "settle_pre_grasp",
    "grasp",
    "settle_grasp",
    "post_grasp",
    "settle_post_grasp",
    "lift",
    "place",
    "throw",
)


@dataclass
class GraspTiming:
    """Planner parameters (seconds, metres). Sim and real differ only in the values."""

    approach: float = 4.0  # floor; stretched to drop / approach_descent_rate for deep reaches
    approach_descent_rate: float = 0.070  # m/s; faster fails on deep reaches, slower buys nothing
    settle_pre_grasp: float = 1.0
    grasp: float = 0.5
    settle_grasp: float = 1.0
    post_grasp: float = 0.5
    settle_post_grasp: float = 1.0  # every settle stays above the ~0.34 s Inspire finger time constant
    lift: float = 3.0
    lift_height: float = 0.12
    lift_max_z: float = 1.1
    place: float = 4.0
    place_rise: float = 0.05  # PLACE z = lift + this; an absolute z shakes the object loose at low tables
    throw: float = 1.0
    glide_home: float = 2.5  # the caller's joint-space glide after THROW; lands within 3 mm at any table height
    hand_close_factor: float = 15.0  # hand finishes its ramp at 1/factor of the segment, then holds
    interp_order: int = 5  # quintic: zero velocity and acceleration at every boundary


@dataclass
class SimGraspTiming(GraspTiming):
    """Roughly half the real profile, at the same lift success."""

    settle_pre_grasp: float = 0.5
    grasp: float = 0.3
    post_grasp: float = 0.2
    settle_post_grasp: float = 1.5
    lift_height: float = 0.10
    hand_close_factor: float = 1.0


@dataclass
class RealGraspTiming(GraspTiming):
    """The base profile, deliberately gentler than sim: on hardware peak joint velocity matters."""


@dataclass
class GraspTrajectory:
    """Pre-planned trajectory: ``poses`` ``[T, 7]`` wxyz, ``hands`` ``[T, 6]``, ``boundaries`` (10 frame indices)."""

    poses: np.ndarray
    hands: np.ndarray
    boundaries: list[int]  # boundaries[i] = first frame of SEGMENTS[i]; boundaries[9] == T
    hz: float

    @property
    def num_frames(self) -> int:
        return self.poses.shape[0]

    def get_frame(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx = max(0, min(idx, self.num_frames - 1))
        return self.poses[idx], self.hands[idx]

    def end(self, segment: str) -> int:
        """First frame after ``segment`` (one of :data:`SEGMENTS`)."""
        return self.boundaries[SEGMENTS.index(segment) + 1]

    def segment_at(self, idx: int) -> str:
        """Name of the segment frame ``idx`` belongs to (``"throw"`` past the end)."""
        for name in SEGMENTS:
            if idx < self.end(name):
                return name
        return SEGMENTS[-1]


def _interpolate_segment(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    start_hand: np.ndarray,
    end_hand: np.ndarray,
    num_frames: int,
    hand_speed_factor: float = 1.0,
    interp_order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose (lerp + slerp) and hand interpolation over one segment, time-reparametrised by ``interp_order``."""
    poses = np.zeros((num_frames, 7), dtype=np.float32)
    hands = np.zeros((num_frames, start_hand.shape[0]), dtype=np.float32)
    for i in range(num_frames):
        t_lin = i / max(num_frames - 1, 1)
        t = reparam(t_lin, interp_order)
        poses[i, :3] = start_pose[:3] + t * (end_pose[:3] - start_pose[:3])
        poses[i, 3:] = slerp_wxyz(start_pose[3:7], end_pose[3:7], t)
        t_hand = reparam(min(t_lin * hand_speed_factor, 1.0), interp_order)
        hands[i] = start_hand + t_hand * (end_hand - start_hand)
    return poses, hands


def plan_grasp_trajectory(
    idle_pose: np.ndarray,
    idle_hand: np.ndarray,
    stage_poses: list[np.ndarray],
    stage_hands: list[np.ndarray],
    *,
    place_pose: np.ndarray | None = None,
    timing: GraspTiming | None = None,
    hz: float = 50.0,
) -> GraspTrajectory:
    """Plan the full pick-and-place (wxyz poses).

    ``place_pose``: release xy/orientation, z = lift + ``place_rise`` (≤ ``lift_max_z``); ``None`` holds at the lift."""
    t = timing if timing is not None else GraspTiming()
    if t.approach_descent_rate <= 0.0:
        raise ValueError(f"approach_descent_rate must be > 0, got {t.approach_descent_rate}")

    approach_drop = max(0.0, float(idle_pose[2]) - float(stage_poses[0][2]))  # descent only
    approach_s = max(t.approach, approach_drop / t.approach_descent_rate)

    post_grasp_pose = np.asarray(stage_poses[2], dtype=np.float32)
    lift_pose = post_grasp_pose.copy()
    lift_pose[2] = min(post_grasp_pose[2] + t.lift_height, t.lift_max_z)
    lift_hand = np.asarray(stage_hands[2], dtype=np.float32)

    if place_pose is not None:
        place_pose_arr = np.array(place_pose, dtype=np.float32)  # copy: never write into the caller's array
        place_pose_arr[2] = min(lift_pose[2] + t.place_rise, t.lift_max_z)
    else:
        place_pose_arr = lift_pose.copy()
    place_hand = lift_hand.copy()
    throw_pose = place_pose_arr.copy()
    throw_hand = np.asarray(idle_hand, dtype=np.float32)

    # (start_pose, end_pose, start_hand, end_hand, duration, hand_speed_factor), in SEGMENTS order.
    segments = [
        (idle_pose, stage_poses[0], idle_hand, stage_hands[0], approach_s, 1.0),
        (stage_poses[0], stage_poses[0], stage_hands[0], stage_hands[0], t.settle_pre_grasp, 1.0),
        (stage_poses[0], stage_poses[1], stage_hands[0], stage_hands[1], t.grasp, t.hand_close_factor),
        (stage_poses[1], stage_poses[1], stage_hands[1], stage_hands[1], t.settle_grasp, 1.0),
        (stage_poses[1], stage_poses[2], stage_hands[1], stage_hands[2], t.post_grasp, t.hand_close_factor),
        (stage_poses[2], stage_poses[2], stage_hands[2], stage_hands[2], t.settle_post_grasp, 1.0),
        (stage_poses[2], lift_pose, stage_hands[2], lift_hand, t.lift, 1.0),
        (lift_pose, place_pose_arr, lift_hand, place_hand, t.place, 1.0),
        (place_pose_arr, throw_pose, place_hand, throw_hand, t.throw, 1.0),
    ]

    all_poses, all_hands, boundaries = [], [], [0]
    for start_p, end_p, start_h, end_h, duration, hand_factor in segments:
        num_frames = max(int(duration * hz), 2)
        poses, hands = _interpolate_segment(
            np.asarray(start_p, dtype=np.float32),
            np.asarray(end_p, dtype=np.float32),
            np.asarray(start_h, dtype=np.float32),
            np.asarray(end_h, dtype=np.float32),
            num_frames,
            hand_speed_factor=hand_factor,
            interp_order=t.interp_order,
        )
        all_poses.append(poses)
        all_hands.append(hands)
        boundaries.append(boundaries[-1] + num_frames)

    return GraspTrajectory(
        poses=np.concatenate(all_poses, axis=0),
        hands=np.concatenate(all_hands, axis=0),
        boundaries=boundaries,
        hz=hz,
    )
