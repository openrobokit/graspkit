# Grasp Optimization Algorithms: warp_gd and warp_lm

How the two grasp-optimizer backends work and when to use which. For running the stage see the
[grasp_optim tutorial](../guide/grasp_optim.md); for every parameter see the
[grasp optimization API](../reference/api/grasp_optim.md).

## The problem

Grasp synthesis is energy minimization over a batch of grasps. Each grasp optimizes

- the robot configuration `q`: hand joints, plus arm joints (fixed base) or a free SE(3) base
  pose (floating hand), and
- a discrete contact assignment: which `num_contact_points` of the robot's predefined contact
  candidates are active.

The energy is a weighted sum of residual terms (robokit `terms.dense.*`):

| Term | Pulls toward |
|---|---|
| `DistanceTask` | active contact points on the object surface |
| `ForceClosureTask` | contact normals that balance external wrenches (stability) |
| `SceneCollisionTask` | hand collision spheres out of the scene (object + table) |
| `SelfCollisionTask` | links out of each other |
| `PositionLimit` | joints inside their limits |
| `FingertipDistanceTask` (optional) | fingertip spheres onto the surface |
| `HandUpAngleTask` (optional) | hand-up axis at least `angle_z_min` from world +z |

Both backends share this formulation, the init samplers, and the Warp scene geometry (SDF
volumes / mesh BVHs). The energy is highly non-convex and the contact assignment is discrete,
so plain gradient descent gets stuck; each backend has its own escape mechanism.

## warp_gd: simulated-annealing gradient descent

`WarpGraspOptimizer` (`warp_optimizer.py`, backend `warp_gd`) runs each grasp as an independent annealing chain
for `num_iters` (~10k) iterations, DexGraspNet-style. One iteration:

1. RMSprop step on the continuous variables; step size decays over time, joints are clamped
   to limits.
2. With probability `switch_possibility`, propose a re-drawn contact set.
3. Evaluate the proposal's energy and gradient.
4. Metropolis accept: lower energy always wins; worse by `d` is accepted with probability
   `exp(-d / T)`, where the temperature `T` cools over time. Rejected grasps roll back.

High `T` early lets grasps explore pose and contact assignment jointly; as `T` cools the chain
settles into a low-energy basin. The whole iteration is captured as one CUDA graph, so the
long loop has almost no launch overhead. Works for floating and fixed-base robots alike, which
makes it the default backend.

## warp_lm: multi-seed Levenberg-Marquardt

`WarpGraspOptimizer` (backend `warp_lm`) treats the same energy as nonlinear least squares
and applies robokit's `LMOptimizer`: curvature-aware steps converge in tens of iterations
instead of thousands. A single LM run is purely local, so exploration comes from:

- **Multi-seed**: every init grasp spawns 24 jittered copies solved in parallel; the
  lowest-energy seed wins.
- **Contact resampling**: LM cannot differentiate the discrete contacts, so every 8 iterations
  each candidate is joined by a copy with a fresh random contact set; both take the next LM
  steps, and the one with the lower energy stays.

`warp_lm` supports floating bases only. To target an arm+hand robot, set
`enable_ik_scoring=True`: inits are pre-filtered by arm-IK reachability against the fixed-base
robot and final seeds are ranked by `energy + reachability penalty`, so the grasps
`GraspTrajBuilder` later IK-converts onto the arm are actually reachable.

## Choosing a backend

| | `warp_gd` | `warp_lm` |
|---|---|---|
| Method | RMSprop + simulated annealing | multi-seed Levenberg-Marquardt |
| Exploration | Metropolis accept over pose + contacts | 24 seeds + periodic contact resampling |
| Iterations | ~10000 first-order steps | 64 curvature-aware steps |
| Speed | baseline | typically 2-3x faster |
| Robots | floating and fixed base | floating base only (`enable_ik_scoring` bridges to arms) |

Rule of thumb: `warp_gd` for fixed-base robots or when maximum exploration matters; `warp_lm`
for floating-hand throughput, with `enable_ik_scoring` if the grasps will be converted to an
arm. Both are compared in `scripts/benchmark/grasp_opt/bench_grasp_optim.py`.
