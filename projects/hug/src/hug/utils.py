"""Small numeric helpers: filters, smooth interpolation, SE3/pose7 conversions.

Filters and SE3 helpers are lifted from humanoid_locomanip (``utils/filter.py``,
``utils/transform.py``). Quaternions are wxyz throughout.
"""

import time
from typing import Optional, Union

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------------ filters


class LPFilter:
    """First-order low-pass: ``y += alpha * (x - y)``."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y: Optional[np.ndarray] = None
        self.is_init = False

    def next(self, x: np.ndarray) -> np.ndarray:
        if not self.is_init or self.y is None:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self) -> None:
        self.y = None
        self.is_init = False


class SE3Filter:
    """SE3 low-pass via ``SE3.Interpolate(state, x, alpha)``."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._state: Optional[pin.SE3] = None
        self._initialized = False

    def reset(self) -> None:
        self._state = None
        self._initialized = False

    def update(self, x: pin.SE3) -> pin.SE3:
        if not self._initialized:
            self._state = pin.SE3(x)
            self._initialized = True
            return pin.SE3(self._state)
        self._state = pin.SE3.Interpolate(self._state, x, self.alpha)
        return pin.SE3(self._state)


class SE3RateLimiter:
    """Cap the per-update translation / rotation step toward a target SE3."""

    def __init__(self, max_trans_rate: float = 0.02, max_rot_rate: float = 0.05):
        self.max_trans_rate = max_trans_rate
        self.max_rot_rate = max_rot_rate
        self._state: Optional[pin.SE3] = None
        self._initialized = False

    def reset(self) -> None:
        self._state = None
        self._initialized = False

    def update(self, target: pin.SE3) -> pin.SE3:
        if not self._initialized:
            self._state = pin.SE3(target)
            self._initialized = True
            return pin.SE3(self._state)

        assert self._state is not None
        error_se3 = self._state.inverse() * target
        trans_error = error_se3.translation.copy()
        rot_error = pin.log3(error_se3.rotation)

        trans_norm = np.linalg.norm(trans_error)
        if trans_norm > self.max_trans_rate and trans_norm > 1e-9:
            trans_error = trans_error * (self.max_trans_rate / trans_norm)

        rot_norm = np.linalg.norm(rot_error)
        if rot_norm > self.max_rot_rate and rot_norm > 1e-9:
            rot_error = rot_error * (self.max_rot_rate / rot_norm)

        delta = pin.SE3(pin.exp3(rot_error), trans_error)
        self._state = self._state * delta
        return pin.SE3(self._state)


def clamp_se3(T: pin.SE3, max_trans: float, max_rot: float) -> pin.SE3:
    """Clamp an SE3's translation norm and rotation angle."""
    trans = T.translation.copy()
    trans_norm = np.linalg.norm(trans)
    if trans_norm > max_trans and trans_norm > 1e-9:
        trans = trans * (max_trans / trans_norm)

    rot = T.rotation
    axis_angle = pin.log3(rot)
    angle = np.linalg.norm(axis_angle)
    if angle > max_rot and angle > 1e-9:
        axis_angle = axis_angle * (max_rot / angle)
        rot = pin.exp3(axis_angle)

    return pin.SE3(rot, trans)


# ------------------------------------------------------- interpolation

TimeT = Union[float, np.ndarray]


def smoothstep(t: TimeT) -> TimeT:
    """Cubic ``3t² − 2t³``: zero velocity at both ends."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: TimeT) -> TimeT:
    """Quintic ``6t⁵ − 15t⁴ + 10t³``: zero velocity and acceleration at both ends."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def reparam(t: float, order: int = 3) -> float:
    """Time reparametrisation: ``order`` 1 = linear, 3 = smoothstep, 5 = smootherstep."""
    if order == 1:
        return float(np.clip(t, 0.0, 1.0))
    if order == 3:
        return float(smoothstep(t))
    if order == 5:
        return float(smootherstep(t))
    raise ValueError(f"interpolation order must be 1, 3, or 5; got {order!r}")


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * np.asarray(a) + t * np.asarray(b)


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between wxyz quaternions (float32 out)."""
    pq0 = pin.Quaternion(float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3]))
    pq1 = pin.Quaternion(float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3]))
    coeffs = pq0.slerp(float(t), pq1).coeffs()  # xyzw
    return np.array([coeffs[3], coeffs[0], coeffs[1], coeffs[2]], dtype=np.float32)


def smooth_lerp(a: np.ndarray, b: np.ndarray, t: float, *, order: int = 3) -> np.ndarray:
    """``lerp(a, b, reparam(t, order))``."""
    return lerp(a, b, reparam(t, order))


def blend(start, end, ratio):
    ratio = np.clip(ratio, 0, 1)
    return start * (1 - ratio) + end * ratio


def blend_waypoints(waypoints: list, ratio: float):
    """Piecewise-linear blend through ordered waypoints."""
    ratio = np.clip(ratio, 0, 1)
    n = len(waypoints) - 1
    segment = min(int(ratio * n), n - 1)
    local_ratio = ratio * n - segment
    return blend(waypoints[segment], waypoints[segment + 1], local_ratio)


def slew_pose7(
    cmd: np.ndarray,
    req: np.ndarray,
    dt: float,
    max_lin: float,
    max_ang: float,
) -> np.ndarray:
    """Step pose7 ``cmd`` toward ``req``, capped at ``max_lin`` m/s and ``max_ang`` rad/s."""
    out = req.copy()
    dpos = req[:3] - cmd[:3]
    n = float(np.linalg.norm(dpos))
    if n > 1e-9:
        out[:3] = cmd[:3] + dpos * (min(n, max_lin * dt) / n)
    r_cmd = R.from_quat([cmd[4], cmd[5], cmd[6], cmd[3]])
    r_req = R.from_quat([req[4], req[5], req[6], req[3]])
    rotvec = (r_req * r_cmd.inv()).as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    if ang > 1e-9:
        r_new = R.from_rotvec(rotvec * (min(ang, max_ang * dt) / ang)) * r_cmd
        xyzw = r_new.as_quat()
        out[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return out


# ----------------------------------------------------- SE3 / pose7 conversions


def pose7_to_mat(p7: np.ndarray) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` → 4x4 homogeneous matrix."""
    M = np.eye(4)
    M[:3, 3] = p7[:3]
    qw, qx, qy, qz = p7[3:7]
    M[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return M


def mat_to_pose7(M: np.ndarray) -> np.ndarray:
    """4x4 homogeneous → pose7 wxyz. Inverse of :func:`pose7_to_mat`."""
    xyzw = R.from_matrix(M[:3, :3]).as_quat()
    return np.array([M[0, 3], M[1, 3], M[2, 3], xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


# ------------------------------------------------------------------ loop pacing


class Rate:
    """Paces a control loop at ``hz``: call :meth:`sleep` once per tick.

    Sleeps to a running deadline; an overrun, or :meth:`resync` after a deliberate block, resets it to now."""

    def __init__(self, hz: float) -> None:
        self.dt = 1.0 / hz
        self.t_next = time.time()

    def resync(self) -> None:
        self.t_next = time.time()

    def sleep(self) -> None:
        self.t_next += self.dt
        remaining = self.t_next - time.time()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.t_next = time.time()
