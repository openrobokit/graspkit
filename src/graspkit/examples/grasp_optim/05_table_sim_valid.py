"""Optimize Sharpa tabletop grasps (floating base + table); report sim success, replay a winning trajectory in Viser."""

import time

import torch

from graspkit.assets.objects import dex_diffuser
from graspkit.configs.env_config import MsEnvConfig
from graspkit.configs.experiment_config import EnvConfig, GraspOptimConfig, MotionPlannerConfig
from graspkit.configs.robot.sharpa import sharpa_right
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.datasets.replay_data import ReplayData
from graspkit.envs import build_env
from graspkit.stages.grasp_optim import BoxDescription, MeshDescription, SceneDescription, WarpGraspOptimizer
from graspkit.stages.motion_plan import build_motion_planner
from graspkit.utils.viser_utils import ReplayViewer

OBJECT = "mustard_bottle"
YCB_ROOT = dex_diffuser.DATA_DIR / "processed/ycb"
NUM_SAMPLES = 256


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    grasp_optim_cfg = GraspOptimConfig(
        optimizer_backend="warp_lm",
        num_contact_points=5,
        scene_geom_backend="mesh",
        w_angle_z=100.0,
        w_spen=20.0,
        spen_threshold=0.0,
    )
    optimizer = WarpGraspOptimizer(grasp_optim_cfg, sharpa_right, device)

    collision_glb = str(YCB_ROOT / OBJECT / "collision.glb")
    obj = MeshDescription(actor_id=OBJECT, collision_mesh_path=collision_glb, scales=(1.0, 1.0, 1.0))
    obj.position = (0.0, 0.0, -obj.get_bounds()[2][0])
    table = BoxDescription(actor_id="table", size=(0.3, 0.3, 0.1), position=(0.0, 0.0, -0.05))
    scene = SceneDescription(scene_id=OBJECT, target_actor_id=OBJECT, boxes={"table": table}, meshes={OBJECT: obj})
    grasp_qpos = optimizer.optimize([scene], num_samples_per_scene=NUM_SAMPLES)

    batch = GraspKitData(scene_ids=[scene.scene_id], scene_descriptions=[scene], grasp_qpos=grasp_qpos)
    motion_planner_cfg = MotionPlannerConfig(pre_grasp_hand_grad_move=1.0)
    motion_planner = build_motion_planner(motion_planner_cfg, robot_config=sharpa_right, device=device)
    batch = motion_planner.plan(batch)
    mp_success = batch.pre_traj_penetration <= 0.005

    evaluator_cfg = EnvConfig(
        ms_env_config=MsEnvConfig(robot_uid="sharpa_right_floating_base"),
        load_scenes=True,
        control_freq=50,
        object_mass=0.01,
        target_lift_dist=0.08,
        num_envs=NUM_SAMPLES,
        enable_perturbation=False,
        num_seeds=1,
    )
    evaluator = build_env(evaluator_cfg, sharpa_right, device)
    infos = evaluator.evaluate(batch, record=True)
    rollouts = evaluator.last_rollouts
    eval_success = infos["success"]
    evaluator.close()

    print(f"motion-plan success: {mp_success.float().mean():.1%}")
    print(f"sim success:         {eval_success.float().mean():.1%}")
    print(f"overall success:     {(mp_success & eval_success).float().mean():.1%}")

    success_idx = set(torch.where((mp_success & eval_success)[0])[0].tolist())
    replays = [
        ReplayData.from_traj(r.qpos, r.obj_pose, sharpa_right, scene) for r in rollouts if r.grasp_index in success_idx
    ]
    assert replays, "no successful samples to replay"
    viewer = ReplayViewer()
    viewer.set_replays(replays)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
