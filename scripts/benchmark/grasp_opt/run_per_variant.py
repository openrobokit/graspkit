# ruff: noqa: E402
"""Run one (preset, backend, variant) combo as JSON, to dodge `--preset all`'s 540s timeout from accumulated state."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from helpers import (  # noqa: E402
    _build_graspkit_scenes,
    build_scene_pair,
    load_scenes,
    make_optim_configs,
    make_robot_configs,
    plan_with_graspkit,
    run_sim,
    setup_robot,
)
from robokit.helpers.grasp import ContactVar  # noqa: E402
from robokit.lie.se3 import se3_to_matrix  # noqa: E402, F401
from robokit.opt.optimizer import aggregate_residuals_to_costs_kernel  # noqa: E402
from robokit.opt.var_values import VarKey, VarValues  # noqa: E402
from robokit.utils.warp_utils import wp_vec7  # noqa: E402
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_quaternion  # noqa: E402

from graspkit.stages.grasp_filters import compute_penetration_depth  # noqa: E402
from graspkit.stages.grasp_optim import GraspOptimConfig as RkGraspOptimConfig  # noqa: E402
from graspkit.stages.grasp_optim import WarpGraspOptimizer  # noqa: E402
from graspkit.stages.grasp_optim.warp_optimizer import grasp_terms  # noqa: E402
from graspkit.stages.grasp_traj_build import (
    ik_convert_floating_to_fixed as rk_ik_convert_floating_to_fixed,  # noqa: E402
)

PRESETS = {
    "mano_right": ("floating", "floating_dev.msgpack", 128, "mano_right", None),
    "xhand_right": ("floating", "floating_dev.msgpack", 64, "xhand_right", None),
    "xarm7_xhand_right": ("fixed", "tabletop_dev.msgpack", 64, "xarm7_xhand_right", "xhand_right"),
    "g1_inspire_right": ("fixed", "tabletop_dev.msgpack", 64, "g1_inspire_right", "inspire_hand_right"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=list(PRESETS))
    ap.add_argument("--backend", required=True, choices=["mesh", "sdf_volume"])
    ap.add_argument("--variant", required=True, choices=["warp_gd", "warp_lm", "warp_lm_ik"])
    ap.add_argument("--num-scenes", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument(
        "--num-iters", type=int, default=None, help="Override per-preset cfg.optim_iters; None -> use preset value."
    )
    ap.add_argument("--num-contact-points", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-sim", action="store_true")
    ap.add_argument(
        "--sim-num-envs", type=int, default=None, help="Auto = num_scenes * n_samples (matches graspkit bench)."
    )
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    wp.init()
    device_str = "cuda:0" if wp.is_cuda_available() else "cpu"
    device = torch.device(device_str)

    task, msgpack, default_n, robot_name, partner_name = PRESETS[args.preset]
    n_samples = args.num_samples if args.num_samples is not None else default_n
    cfgs = make_robot_configs()
    optim_cfgs = make_optim_configs()
    cfg = cfgs[robot_name]
    optim_cfg = optim_cfgs[robot_name]
    partner_cfg = cfgs.get(partner_name) if partner_name else None
    partner_optim_cfg = optim_cfgs.get(partner_name) if partner_name else None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[{time.strftime('%H:%M:%S')}] loading scenes", flush=True)
    scenes = load_scenes(msgpack, num_scenes=args.num_scenes)
    # Geometry/robot for metrics below; filled in later (loading a robot before the optimizer's own load stalls warp).
    robot = cands = link_idx = None
    target_geom = scene_geom = init_target_geom = init_scene_geom = None
    bake_time = sdf_size = 0.0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    g_scenes = _build_graspkit_scenes(scenes)
    if task == "fixed" and args.variant in ("warp_lm", "warp_lm_ik"):
        # Fixed-base via robokit: optimize on the floating partner, then IK-convert to fixed (same path as --reference).
        ik_scoring = args.variant == "warp_lm_ik"
        num_iters = args.num_iters if args.num_iters is not None else partner_optim_cfg.optim_iters
        grasp_cfg = RkGraspOptimConfig(
            num_iters=num_iters,
            num_contact_points=args.num_contact_points,
            spen_threshold=partner_optim_cfg.optim_spen_threshold,
            distance_upper=partner_optim_cfg.init_distance_upper,
            hand_qpos_mu=partner_optim_cfg.init_hand_qpos_mu,
            scene_geom_backend=args.backend,
            optimizer_backend="warp_lm",
            enable_ik_scoring=ik_scoring,
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] build robokit warp_lm optimizer (floating, ik_scoring={ik_scoring})",
            flush=True,
        )
        optimizer = WarpGraspOptimizer(grasp_cfg, partner_cfg, device, fixed_base_robot_config=cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_optim = time.perf_counter()
        floating_qpos = optimizer.optimize(g_scenes, num_samples_per_scene=n_samples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        init_t, optim_t = 0.0, time.perf_counter() - t_optim
        print(
            f"[{time.strftime('%H:%M:%S')}] optimize done init={init_t:.1f}s optim={optim_t:.1f}s, IK-converting",
            flush=True,
        )
        robot, cands, link_idx = setup_robot(cfg, device_str)
        target_geom, scene_geom, init_target_geom, init_scene_geom, bake_time, sdf_size = build_scene_pair(
            scenes, args.backend, device_str
        )
        n_scene_opt = floating_qpos.shape[0]
        flat_floating = floating_qpos.reshape(n_scene_opt * n_samples, -1)
        num_grasps_per_scene = torch.full((n_scene_opt,), n_samples, dtype=torch.long, device=device)
        _, grasp_full, _, _ = rk_ik_convert_floating_to_fixed(
            g_scenes, num_grasps_per_scene, flat_floating, flat_floating, flat_floating, partner_cfg, robot
        )
        grasp_qpos = grasp_full.reshape(n_scene_opt, n_samples, -1)
    elif args.variant in ("warp_gd", "warp_lm"):
        # Canonical path: robokit grasp-optimizer lib; optimizer built first, metrics geometry below.
        num_iters = args.num_iters if args.num_iters is not None else optim_cfg.optim_iters
        grasp_cfg = RkGraspOptimConfig(
            num_iters=num_iters,
            num_contact_points=args.num_contact_points,
            spen_threshold=optim_cfg.optim_spen_threshold,
            distance_upper=optim_cfg.init_distance_upper,
            hand_qpos_mu=optim_cfg.init_hand_qpos_mu,
            scene_geom_backend=args.backend,
            optimizer_backend=args.variant,
            sampler_type="fixed_base" if task == "fixed" else "floating_base",
        )
        base_type = "fixed" if task == "fixed" else "floating"
        print(f"[{time.strftime('%H:%M:%S')}] build robokit optimizer ({args.variant}, {base_type})", flush=True)
        optimizer = WarpGraspOptimizer(grasp_cfg, cfg, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_optim = time.perf_counter()
        grasp_qpos = optimizer.optimize(g_scenes, num_samples_per_scene=n_samples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        init_t, optim_t = 0.0, time.perf_counter() - t_optim
        print(f"[{time.strftime('%H:%M:%S')}] optimize done init={init_t:.1f}s optim={optim_t:.1f}s", flush=True)
        robot, cands, link_idx = setup_robot(cfg, device_str)
        target_geom, scene_geom, init_target_geom, init_scene_geom, bake_time, sdf_size = build_scene_pair(
            scenes, args.backend, device_str
        )
    else:
        raise SystemExit(f"unsupported variant {args.variant!r} for task {task!r}")

    print(f"[{time.strftime('%H:%M:%S')}] compute penetration + energy via robokit terms", flush=True)
    n_scene, n_samples_actual = grasp_qpos.shape[:2]
    b = n_scene * n_samples_actual
    flat = grasp_qpos.reshape(b, -1).contiguous()
    nq = robot.num_actuated_joints
    last = flat.shape[-1]
    if last == nq:
        q_flat = flat
        base_mat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(b, 4, 4).contiguous().clone()
        base_mat[:, :3, 3] = torch.tensor(cfg.init_position, device=device, dtype=torch.float32)
    else:
        q_flat = flat[..., last - nq :].contiguous()
        base_xyz = flat[..., :3]
        base_rot = euler_angles_to_matrix(flat[..., 3:6], axes="rxyz")
        base_mat = torch.zeros(b, 4, 4, device=device, dtype=torch.float32)
        base_mat[:, :3, :3] = base_rot
        base_mat[:, :3, 3] = base_xyz
        base_mat[:, 3, 3] = 1.0
    base_quat = matrix_to_quaternion(base_mat[:, :3, :3])
    base_xyz_wxyz = torch.cat([base_mat[:, :3, 3], base_quat], dim=-1).contiguous()
    state = robot.state(
        q=wp.from_torch(q_flat, dtype=wp.float32),
        T_world_base=wp.from_torch(base_xyz_wxyz, dtype=wp_vec7),
    )
    # graspkit's energy: the LM terms on the final pose, with a fresh random contact set (drawn by `seed`)
    contacts = ContactVar(
        cands.numpy(),
        link_idx.numpy(),
        state.batch_size,
        args.num_contact_points,
        seed=args.seed,
        device=state.q.device,
    )
    var = VarValues({VarKey.ROBOT: state, VarKey.CONTACTS: contacts})
    energy_cfg = RkGraspOptimConfig(
        num_contact_points=args.num_contact_points, spen_threshold=optim_cfg.optim_spen_threshold
    )
    terms = grasp_terms(
        energy_cfg,
        state.robot,
        target_geom,
        scene_geom,
        lm=True,
    )
    pen = compute_penetration_depth(g_scenes, grasp_qpos, robot)
    mean_pen, max_pen = pen.mean().item(), pen.max().item()
    # Weighted LM residual cost 0.5*||r||^2 per term, mapped to the report's E_* columns.
    term_to_col = {
        "force_closure_task": "E_fc",
        "scene_distance_task": "E_dis",
        "scene_collision_task": "E_pen",
        "self_collision_task": "E_spen",
        "position_limit": "E_joints",
    }
    energies = {}
    for term in terms.values():
        term.precompute(var, need_gradient=False)
        residual = term.compute_weighted_residual(var)
        cost = wp.zeros((var.batch_size,), dtype=wp.float32, device=var.device)
        wp.launch(aggregate_residuals_to_costs_kernel, dim=var.batch_size, inputs=[residual, cost], device=var.device)
        energies[term.cost_name] = cost
    energy_terms = {
        term_to_col[name]: float(cost.numpy().mean()) for name, cost in energies.items() if name in term_to_col
    }
    total_energy = sum(energy_terms.values())

    row = {
        "init_time": init_t,
        "optim_time": optim_t,
        "total_time": init_t + optim_t,
        "mean_penetration": mean_pen,
        "max_penetration": max_pen,
        "mean_energy": total_energy,
        **energy_terms,
    }
    if args.backend == "sdf_volume":
        row["bake_time"] = bake_time
        row["sdf_file_size_mb"] = sdf_size

    if args.run_sim:
        full_traj, pre_pen, mp_t, mp_sr = plan_with_graspkit(grasp_qpos, cfg, scenes, device_str)
        row["mp_time"] = mp_t
        row["mp_sr"] = mp_sr
        # Auto-pick num_envs to match graspkit's bench convention (= num_scenes * n_samples).
        sim_n_envs = args.sim_num_envs if args.sim_num_envs is not None else args.num_scenes * n_samples
        sim_metrics = run_sim(full_traj, scenes, cfg, num_envs=sim_n_envs)
        row.update(sim_metrics)
        row["final_sr"] = (mp_sr / 100.0) * (sim_metrics["sim_sr"] / 100.0) * 100.0

    with open(args.out_json, "w") as f:
        json.dump(row, f, indent=2)
    print(f"WROTE {args.out_json}", flush=True)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
