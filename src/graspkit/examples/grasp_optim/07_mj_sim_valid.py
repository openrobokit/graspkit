"""Optimize Shadow-hand grasps for a YCB object, score in MuJoCo (force-closure), then live-replay winners in Viser."""

import math
import time

import torch

from graspkit.assets.objects import dex_diffuser
from graspkit.configs.env_config import MjEvalConfig
from graspkit.configs.experiment_config import MotionPlannerConfig
from graspkit.configs.robot.shadow_mj import shadow_right_mj
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.datasets.replay_data import ReplayData
from graspkit.envs import MjEnv
from graspkit.stages.grasp_optim import GraspOptimConfig, MeshDescription, SceneDescription, WarpGraspOptimizer
from graspkit.stages.motion_plan import build_motion_planner
from graspkit.utils.viser_utils import ReplayViewer

OBJECT = "mustard_bottle"
YCB_ROOT = dex_diffuser.DATA_DIR / "processed/ycb"
NUM_SAMPLES = 64
# TODO fmt off for this long list
_SHADOW_INIT = (
    0.0,
    0.3,
    0.4,
    0.4,
    0.0,
    0.3,
    0.4,
    0.4,
    0.0,
    0.3,
    0.4,
    0.4,
    0.0,
    0.0,
    0.3,
    0.4,
    0.4,
    0.2,
    0.8,
    0.2,
    0.3,
    0.0,
)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    grasp_optim_cfg = GraspOptimConfig(
        hand_qpos_mu=_SHADOW_INIT,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        sampler_type="floating_base",
        # Fingertip-close energy + sphere self-penetration penalty + pen filter — the working MuJoCo grasp recipe.
        distance_lower=0.08,
        distance_upper=0.15,
        w_fingertip_dis=30.0,
        w_spen=20.0,
        spen_threshold=0.0,
        spen_representation="sphere",
    )
    optimizer = WarpGraspOptimizer(grasp_optim_cfg, shadow_right_mj, device)
    # YCB collision.glb is a 3-part convex decomposition; MjEnv loads its scene geoms as the object's convex pieces.
    collision_glb = str(YCB_ROOT / OBJECT / "collision.glb")
    scene = SceneDescription(
        scene_id=OBJECT,
        target_actor_id=OBJECT,
        meshes={OBJECT: MeshDescription(actor_id=OBJECT, collision_mesh_path=collision_glb, scales=(1.0, 1.0, 1.0))},
    )
    grasp_qpos = optimizer.optimize([scene], num_samples_per_scene=NUM_SAMPLES)

    batch = GraspKitData(scene_ids=[scene.scene_id], scene_descriptions=[scene], grasp_qpos=grasp_qpos)
    motion_planner_cfg = MotionPlannerConfig(
        num_steps_pre_grasp=0, num_steps_post_grasp=0, pre_grasp_hand_grad_move=1.0, post_grasp_hand_grad_move=0.1
    )
    motion_planner = build_motion_planner(motion_planner_cfg, robot_config=shadow_right_mj, device=device)
    batch = motion_planner.plan(batch)
    mp_success = batch.pre_traj_penetration <= 0.005

    mj_cfg = MjEvalConfig()  # setting="fc": free-space force-closure test (no table)
    evaluator = MjEnv(mj_cfg, robot_config=shadow_right_mj)
    infos = evaluator.evaluate(batch, record=True)
    rollouts = evaluator.last_rollouts
    evaluator.close()
    eval_success = infos["success"].to(mp_success.device)

    print(f"motion-plan success: {mp_success.float().mean():.1%}")
    print(f"sim success:         {eval_success.float().mean():.1%}")
    print(f"overall success:     {(mp_success & eval_success).float().mean():.1%}")
    print(f"mean contacts={infos['contact_num'].mean():.1f}  delta_pos={infos['delta_pos'].mean():.3f}m")

    success_idx = set(torch.where((mp_success & eval_success)[0])[0].tolist())
    replays = [
        ReplayData.from_traj(r.qpos, r.obj_pose, shadow_right_mj, scene)
        for r in rollouts
        if r.grasp_index in success_idx
    ]
    assert replays, "no successful grasps to replay"
    viewer = ReplayViewer()
    viewer.set_replays(replays)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
