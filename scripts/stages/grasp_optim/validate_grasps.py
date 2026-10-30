"""Validate optimized grasps: motion planning + simulation, loads via GraspKitData, reports per-scene/aggregate success.

Usage:
uv run scripts/stages/grasp_optim/validate_grasps.py --experiment_name inspire_right_dex1b_v0_4_dev
uv run scripts/stages/grasp_optim/validate_grasps.py --experiment_name g1_inspire_right_dex1b_v0_4_dev
uv run scripts/stages/grasp_optim/validate_grasps.py --experiment_name g1_inspire_right_wbc_dex1b_v0_5_mini --coord-convert
uv run scripts/stages/grasp_optim/validate_grasps.py --experiment_name g1_inspire_right_wbc_dex1b_v0_4_1_mini --coord-convert

(check only: grasp-pose penetration into the target object, no motion planning / sim)
uv run scripts/stages/grasp_optim/validate_grasps.py --experiment_name g1_inspire_right_wbc_dex1b_v0_5_small --check-only
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
from dataclasses import dataclass, replace

import torch
import tyro
from rich.progress import track
from robokit.utils.ee_coord_utils import ee_coord_conversion
from robokit.xform.warp.torch_wrappers import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    rot_tl_to_tf_mat,
)

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import build_env
from graspkit.scene.scene_description import SceneDescription
from graspkit.robot.grasp_robot import GraspRobot, hand_coord_spec
from graspkit.stages.grasp_filters import compute_penetration_depth
from graspkit.stages.grasp_traj_build import build_planning_traj_builder
from graspkit.utils.io_utils import build_filesystem

torch.set_float32_matmul_precision("high")
logger = logging.getLogger("graspkit")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_grasps(
    cfg: ExperimentConfig,
    coord_convert: bool = False,
    scene_id: str | None = None,
    plain_eval: bool = True,
    save_low_success: bool = False,
    check_only: bool = False,
    pen_threshold: float = 0.005,
):
    robot_config = cfg.robot_config
    output_dir = cfg.grasp_output_dir
    motion_planner_cfg = cfg.traj_builder.motion_planner
    evaluator_cfg = cfg.grasp_optim.evaluator
    assert check_only or evaluator_cfg is not None, "validation requires grasp_optim.evaluator"

    # check_only skips motion planning + sim (penetration is pure FK + SDF); the physics path builds the traj builder.
    if check_only:
        robot_helper = GraspRobot.from_config(robot_config, device=device)
        traj_builder = None
        evaluator = None
        num_envs = 1024
    else:
        if plain_eval:
            evaluator_cfg = replace(evaluator_cfg, enable_perturbation=False, num_seeds=1)
        traj_builder = build_planning_traj_builder(
            cfg,
            device,
            motion_planner=replace(motion_planner_cfg, backend="torch"),
        )
        robot_helper = traj_builder.robot
        evaluator = build_env(evaluator_cfg, robot_config, device)
        num_envs = evaluator_cfg.num_envs

    coord_inv_tf = None
    if coord_convert:
        floating_spec = hand_coord_spec(cfg.robot_config)
        fixed_spec = hand_coord_spec(cfg.traj_builder.fixed_base_robot_config)
        if floating_spec != fixed_spec:
            coord_inv_tf = rot_tl_to_tf_mat(
                rot_mat=ee_coord_conversion(fixed_spec, floating_spec, return_tensors="pt").to(device)
            )

    label = "pass" if check_only else "success"
    merged_file = output_dir.rstrip("/") + ".h5"
    data_path = merged_file if build_filesystem(merged_file).exists(merged_file) else output_dir
    logger.info(f"grasps={data_path} scene_descs={cfg.scene_description_path}")
    scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(cfg.scene_description_path)

    # Iterate all grasp h5 files in the dir; if `--scene_id` is given, load only that one.
    load_ids = [scene_id] if scene_id is not None else None
    all_data = GraspKitData.load(data_path, scene_ids=load_ids, packed=True)
    assert all_data.num_scenes, f"no grasps found at {data_path}"
    missing = [sid for sid in all_data.scene_ids if sid not in scene_descriptions]
    assert not missing, f"grasp scenes with no matching scene_description: {missing[:5]}"
    all_data.scene_descriptions = [scene_descriptions[sid] for sid in all_data.scene_ids]

    # Sort scenes by grasp count desc and pack adaptively so each batch is ~num_envs padded slots.
    counts = all_data.num_grasps_per_scene  # CPU (B,)
    sorted_scene_indices = counts.argsort(descending=True).tolist()
    batches: list[list[int]] = []
    cursor = 0
    while cursor < len(sorted_scene_indices):
        max_k = int(counts[sorted_scene_indices[cursor]].item())
        batch_size = max(1, num_envs // max(max_k, 1))
        batches.append(sorted_scene_indices[cursor : cursor + batch_size])
        cursor += batch_size
    logger.info(
        f"Validating {all_data.num_scenes} scenes ({int(counts.sum().item())} grasps) in {len(batches)} batches"
    )

    total_valid = 0
    total_success = 0
    low_success_scenes: dict[str, SceneDescription] = {}

    for scene_indices in track(batches, description="validate", total=len(batches)):
        scenes_mask = torch.zeros(all_data.num_scenes, dtype=torch.bool)
        scenes_mask[torch.tensor(scene_indices)] = True
        packed = all_data.filter(scenes_mask)
        packed.pre_grasp_qpos = packed.pre_grasp_qpos.to(device)
        packed.grasp_qpos = packed.grasp_qpos.to(device)
        packed.post_grasp_qpos = packed.post_grasp_qpos.to(device)

        if coord_inv_tf is not None:
            for attr in ("pre_grasp_qpos", "grasp_qpos", "post_grasp_qpos"):
                qpos = getattr(packed, attr)
                pose_mat = rot_tl_to_tf_mat(
                    rot_mat=euler_angles_to_matrix(qpos[..., 3:6], axes="rxyz"),
                    tl=qpos[..., :3],
                )
                pose_mat = pose_mat @ coord_inv_tf
                new_euler = matrix_to_euler_angles(pose_mat[..., :3, :3], axes="rxyz")
                setattr(
                    packed,
                    attr,
                    torch.cat([pose_mat[..., :3, 3], new_euler, qpos[..., 6:]], dim=-1),
                )
        # Pad each scene up to batch-max K by duplicating its last grasp; masked out when aggregating success.
        batch_counts = packed.num_grasps_per_scene  # CPU (B,)
        num_scenes = packed.num_scenes
        max_k = int(batch_counts.max().item())
        repeats = torch.ones(int(packed.first_indices[-1].item()), dtype=torch.long)
        repeats[packed.first_indices[1:] - 1] = max_k - batch_counts + 1
        repeats = repeats.to(device)

        batch = GraspKitData(
            scene_ids=packed.scene_ids,
            scene_descriptions=packed.scene_descriptions,
            pre_grasp_qpos=packed.pre_grasp_qpos.repeat_interleave(repeats, dim=0).reshape(num_scenes, max_k, -1),
            grasp_qpos=packed.grasp_qpos.repeat_interleave(repeats, dim=0).reshape(num_scenes, max_k, -1),
            post_grasp_qpos=packed.post_grasp_qpos.repeat_interleave(repeats, dim=0).reshape(num_scenes, max_k, -1),
        )

        if check_only:
            pen = compute_penetration_depth(batch.scene_descriptions, batch.grasp_qpos, robot_helper)
            pass_mask = pen <= pen_threshold  # (B, max_k)
            for i, sid in enumerate(batch.scene_ids):
                n_i = int(batch_counts[i].item())
                scene_pass = int(pass_mask[i, :n_i].sum().item())
                total_valid += n_i
                total_success += scene_pass
                logger.info(
                    f"Scene {sid}: pen mean={pen[i, :n_i].mean() * 1000:.2f}mm, "
                    f"pass@{pen_threshold * 1000:.1f}mm={scene_pass}/{n_i} ({scene_pass / n_i:.1%})"
                )
            logger.info(f"Cumulative pass: {total_success}/{total_valid} ({total_success / total_valid:.1%})")
            continue

        batch, _ = traj_builder.build(batch)
        mp_success = batch.pre_traj_penetration <= 0.005

        eval_results = evaluator.evaluate(batch)
        success_mask = mp_success & eval_results["success"]  # (B, max_k)

        for i, sid in enumerate(batch.scene_ids):
            n_i = int(batch_counts[i].item())
            scene_success = int(success_mask[i, :n_i].sum().item())
            total_valid += n_i
            total_success += scene_success
            logger.info(f"Scene {sid}: {scene_success}/{n_i} ({scene_success / n_i:.1%})")
            if scene_success / n_i < 0.5:
                low_success_scenes[sid] = scene_descriptions[sid]

        logger.info(f"Cumulative: {total_success}/{total_valid} ({total_success / total_valid:.1%})")

    logger.info(f"Final {label}: {total_success}/{total_valid} ({total_success / total_valid:.1%})")
    if save_low_success and low_success_scenes:
        low_success_path = cfg.scene_description_path.rsplit(".", 1)[0] + "_low_success.msgpack"
        SceneDescription.save_scene_descriptions_msgpack(low_success_scenes, low_success_path)
        logger.info(f"Saved {len(low_success_scenes)} low-success scenes (<50%) to {low_success_path}")
    if evaluator is not None:
        evaluator.close()


if __name__ == "__main__":

    @dataclass
    class Args:
        experiment_name: str
        coord_convert: bool = False
        scene_id: str | None = None
        plain_eval: bool = True
        save_low_success: bool = False
        check_only: bool = False
        pen_threshold: float = 0.005

    args = tyro.cli(Args, use_underscores=True)
    validate_grasps(
        build_experiment_config(args.experiment_name),
        args.coord_convert,
        args.scene_id,
        args.plain_eval,
        args.save_low_success,
        args.check_only,
        args.pen_threshold,
    )
