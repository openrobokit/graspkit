# Motion Planner Benchmark Results
Date: 2026-10-19 12:04

`warp`: optimize start → pre-grasp, then the approach, hold and lift at their step counts (main). `warp_joint`
(`enable_joint_optimization`): one robokit plan through start → pre-grasp → grasp → hold → lift. The hold keeps its
step count (1.5 s xarm, 0.6 s g1); robokit splits the other 31 frames by the time each segment needs and times
them within its joint limits. Arm acc/jerk at the 50 Hz control rate over each plan's motion; the arm hold is the
longest still stretch. g1's fingers plan at 4 rad/s, not the URDF's 0.5. Sim SR varies about ±1.5 points between runs.

## xarm7_xhand_right
Batch: 8 scenes × 128 grasps = 1024 trajs
| Metric | warp | warp_joint |
|---|---|---|
| Plan Time (s) | 12.62 | 9.55 |
| MP SR (pen <= 5 mm) | 0.965 | 0.965 |
| Sim SR (MP and lift) | 0.623 | 0.600 |
| Motion time (s) | 6.28 | 4.55 |
| Arm hold, min (s) | 1.52 | 1.50 |
| Arm hold, max (s) | 1.52 | 1.50 |
| Arm RMS acc (rad/s²) | 1.92 | 2.89 |
| Arm RMS jerk (rad/s³) | 136.1 | 201.8 |
| Arm peak acc (rad/s²) | 48.73 | 67.82 |
| Arm peak jerk (rad/s³) | 2448.9 | 3390.9 |

## g1_inspire_right
Batch: 8 scenes × 128 grasps = 1024 trajs
| Metric | warp | warp_joint |
|---|---|---|
| Plan Time (s) | 4.55 | 7.22 |
| MP SR (pen <= 5 mm) | 0.899 | 0.896 |
| Sim SR (MP and lift) | 0.053 | 0.046 |
| Motion time (s) | 2.18 | 3.03 |
| Arm hold, min (s) | 0.62 | 0.62 |
| Arm hold, max (s) | 0.62 | 0.62 |
| Arm RMS acc (rad/s²) | 26.42 | 5.27 |
| Arm RMS jerk (rad/s³) | 1947.6 | 365.6 |
| Arm peak acc (rad/s²) | 439.05 | 99.01 |
| Arm peak jerk (rad/s³) | 25103.7 | 4950.7 |
