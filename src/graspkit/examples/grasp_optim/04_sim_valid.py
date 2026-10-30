"""Optimize xHand grasps for a YCB object, report sim success rate, and replay a successful trajectory in Viser."""

import math
import time

import torch

from graspkit.assets.objects import dex_diffuser
from graspkit.configs.env_config import MsEnvConfig
from graspkit.configs.experiment_config import EnvConfig, GraspOptimConfig, MotionPlannerConfig
from graspkit.configs.robot.xhand import xhand_right
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.datasets.replay_data import ReplayData
from graspkit.envs import build_env
from graspkit.stages.grasp_optim import MeshDescription, SceneDescription, WarpGraspOptimizer
from graspkit.stages.motion_plan import build_motion_planner
from graspkit.utils.viser_utils import ReplayViewer

OBJECT = "mustard_bottle"
YCB_ROOT = dex_diffuser.DATA_DIR / "processed/ycb"
NUM_SAMPLES = 128


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    grasp_optim_cfg = GraspOptimConfig(
        hand_qpos_mu=(1.0, 0.6, 0.3, 0.0, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6, 0.3),
        distance_upper=0.15,
        spen_threshold=0.025,
        front_theta_lower=-math.pi,
        front_theta_upper=math.pi,
        sampler_type="floating_base",
    )
    optimizer = WarpGraspOptimizer(grasp_optim_cfg, xhand_right, device)
    collision_glb = str(YCB_ROOT / OBJECT / "collision.glb")
    scene = SceneDescription(
        scene_id=OBJECT,
        target_actor_id=OBJECT,
        meshes={OBJECT: MeshDescription(actor_id=OBJECT, collision_mesh_path=collision_glb, scales=(1.0, 1.0, 1.0))},
    )
    grasp_qpos = optimizer.optimize([scene], num_samples_per_scene=NUM_SAMPLES)

    batch = GraspKitData(scene_ids=[scene.scene_id], scene_descriptions=[scene], grasp_qpos=grasp_qpos)
    motion_planner_cfg = MotionPlannerConfig(
        num_steps_pre_grasp=0, num_steps_post_grasp=0, pre_grasp_hand_grad_move=0.2, post_grasp_hand_grad_move=0.1
    )
    motion_planner = build_motion_planner(motion_planner_cfg, robot_config=xhand_right, device=device)
    batch = motion_planner.plan(batch)
    mp_success = batch.pre_traj_penetration <= 0.005

    evaluator_cfg = EnvConfig(
        ms_env_config=MsEnvConfig(robot_uid="xhand_right"),
        load_scenes=True,
        control_freq=50,
        object_mass=0.01,
        floating_object=True,
        num_envs=NUM_SAMPLES,
        num_seeds=1,
        enable_perturbation=True,
        perturbation_magnitude=5.0,
        perturbation_step_range=(-25, -5),
        num_perturbations=4,
    )
    evaluator = build_env(evaluator_cfg, xhand_right, device)
    infos = evaluator.evaluate(batch, record=True)
    rollouts = evaluator.last_rollouts
    eval_success = infos["success"]
    evaluator.close()

    print(f"motion-plan success: {mp_success.float().mean():.1%}")
    print(f"sim success:         {eval_success.float().mean():.1%}")
    print(f"overall success:     {(mp_success & eval_success).float().mean():.1%}")

    success_idx = set(torch.where((mp_success & eval_success)[0])[0].tolist())
    replays = [
        ReplayData.from_traj(r.qpos, r.obj_pose, xhand_right, scene) for r in rollouts if r.grasp_index in success_idx
    ]
    assert replays, "no successful samples to replay"
    viewer = ReplayViewer()
    viewer.set_replays(replays)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
