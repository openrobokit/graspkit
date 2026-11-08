"""G1 arm joint limits and the chest-camera calibration poses.

Torch-free so it imports without the calibration extra. Every 14-vector is left arm then right arm,
each shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
(``G1JointIndex.LeftShoulderPitch..RightWristYaw``).
"""

import numpy as np
from numpy.typing import NDArray

# URDF limits (rad); asymmetric: shoulder_roll left [0, 1.55], right [-1.55, 0]; shoulder_yaw left ±1.3, right ±2.0
# fmt: off
ARM_JOINT_LIMITS_LOW: NDArray[np.float64] = np.array([-2.44, 0.0, -1.3, -1.05, -1.97, -1.61, -1.61,
                                                      -2.44, -1.55, -2.0, -1.05, -1.97, -1.61, -1.61], dtype=np.float64)
ARM_JOINT_LIMITS_HIGH: NDArray[np.float64] = np.array([1.44, 1.55, 1.3, 2.09, 1.97, 1.61, 1.61,
                                                       1.44, 0.0, 2.0, 2.09, 1.97, 1.61, 1.61], dtype=np.float64)
ARM_LIMIT_MARGIN: float = 0.05  # rad; shrinks the joint-limit clip box

# Where the calibration capture starts and parks.
ARM_DEFAULT_QPOS: NDArray[np.float64] = np.array([0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0,
                                                  0.5, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)

# The 12 calibration poses, in visiting order from ARM_DEFAULT_QPOS and back, for a robot with its hands off.
# Picked in SAPIEN from hand-tuned presets ±0.15 rad: >= 2 cm convex-hull clearance (arms, hands still on, body,
# a 6 cm sphere around the chest camera) at each pose and along each move; >= 25k arm pixels in view of the 60°
# mount's camera; spread over the image.
CALIBRATION_ARM_QPOS: NDArray[np.float64] = np.array([
    [-0.135,  0.050, -0.130, -0.138,  0.016, -0.095, -0.128,   0.125, -0.205, -0.122,  0.141,  0.050,  0.068,  0.019],
    [-0.417,  0.416, -0.548, -0.007,  0.311, -0.505,  0.032,   0.096, -0.050, -0.083, -0.016,  0.093, -0.067,  0.044],
    [-0.467,  0.590, -0.232,  0.407,  0.101, -0.464, -0.061,  -0.133, -0.050,  0.054, -0.064, -0.177, -0.008, -0.029],
    [-0.422,  0.406, -0.332,  0.601,  0.090, -0.408, -0.065,  -0.524, -0.230,  0.543, -0.424, -0.672, -0.280, -0.079],
    [-0.108,  0.224, -0.073,  1.153,  0.352,  0.044,  0.115,  -0.434, -0.810,  0.640, -0.070, -0.507, -0.050,  0.075],
    [-0.317,  0.439,  0.131,  0.977,  0.376, -0.056,  0.001,  -0.443, -0.654,  0.734,  0.195, -0.531, -0.032,  0.158],
    [-0.614,  0.369, -0.298,  0.288,  0.455, -0.253, -0.090,  -0.103, -0.275, -0.045,  0.962, -0.183,  0.009, -0.006],
    [-0.381,  0.151, -0.410,  0.127,  0.281, -0.248,  0.013,  -0.342, -0.305, -0.107,  0.996, -0.313,  0.175,  0.090],
    [-0.318,  0.424, -0.471,  0.125,  0.418, -0.235, -0.017,  -0.323, -0.823,  0.569,  0.131, -0.433, -0.021,  0.044],
    [-0.630,  0.295, -0.316,  0.059,  0.355, -0.429, -0.093,  -0.353, -0.700,  0.063,  0.027, -0.650, -0.223,  0.111],
    [-0.188,  0.358, -0.435, -0.329, -0.177, -0.303, -0.070,  -0.466, -0.452,  1.072,  0.146, -0.681, -0.011,  0.001],
    [-0.413,  0.100, -0.277, -0.596, -0.234, -0.248,  0.028,  -0.307, -0.744,  0.740, -0.080, -0.696, -0.059,  0.261],
], dtype=np.float64)
# fmt: on
