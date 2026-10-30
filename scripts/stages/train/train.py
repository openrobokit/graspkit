"""Train a grasp policy on one GPU or with DDP (torchrun), or evaluate a checkpoint with `--trainer.eval-only`.

Usage:
(single gpu training)
uv run python scripts/stages/train/train.py g1_inspire_right_dex1b_mini_tabletop

(ddp training)
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run torchrun --standalone --nproc-per-node=2 scripts/stages/train/train.py g1_inspire_right_dex1b_mini_tabletop

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1,2,3,4
uv run torchrun --standalone --nproc-per-node=4 scripts/train/train.py xhand_right_dex1b_v0_5_mini_tabletop

(eval only, optionally --visualize to render the sim rollout)
uv run scripts/stages/train/train.py g1_inspire_right_dex1b_mini_tabletop --trainer.eval-only [--visualize] \
  --ckpt-path=logs/train/g1_inspire/dex1b_tabletop_mini/checkpoints/best_eval_success_rate.pt
"""

import graspkit  # noqa: I001  # must be first: sets numpy thread limits + loads .env

import dataclasses
from collections.abc import Callable
from collections import Counter
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, RandomSampler, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler

from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.datasets.utils import IterationBasedBatchSampler, worker_init_fn
from graspkit.stages.trainer import Trainer
from graspkit.models.grasp_policy import GraspPolicy, build_grasp_policy
from graspkit.utils.io_utils import ensure_local_filepath
from graspkit.utils.seed_utils import seed_everything
from graspkit.utils.stage_utils import write_completion_marker

logger = logging.getLogger("graspkit")

TRAIN_COMPLETE_MARKER = ".train_complete"


def get_iteration_based_dataloader(
    dataset: GraspDataset | Subset, exp_cfg: ExperimentConfig, generator: torch.Generator | None = None
) -> DataLoader:
    trainer_cfg = exp_cfg.trainer
    source, indices = (
        (dataset.dataset, dataset.indices) if isinstance(dataset, Subset) else (dataset, range(len(dataset)))
    )
    phases = [source.scene_ids[i].endswith("__feedback") for i in indices]
    counts = Counter(phases)
    if len(counts) == 2:
        sampler = WeightedRandomSampler(
            [1 / counts[phase] for phase in phases],
            len(dataset),
            generator=generator
            if generator is not None
            else torch.Generator().manual_seed(trainer_cfg.seed + (dist.get_rank() if dist.is_initialized() else 0)),
        )
    elif dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
    wld_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_rank_batch_size = trainer_cfg.batch_size // wld_size
    assert len(dataset) >= per_rank_batch_size, (
        f"Dataset has {len(dataset)} samples but per-rank batch_size is "
        f"{per_rank_batch_size} (batch_size={trainer_cfg.batch_size}, "
        f"world_size={wld_size}). Lower trainer.batch_size or expand the dataset."
    )
    batch_sampler = BatchSampler(sampler, batch_size=per_rank_batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, trainer_cfg.total_iters + 1)  # +1 for the last iteration
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=trainer_cfg.num_dataloader_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=trainer_cfg.seed),
        generator=torch.Generator().manual_seed(trainer_cfg.seed) if generator is not None else None,
    )


def train(exp_cfg: ExperimentConfig, policy_cls: Callable[..., GraspPolicy] = GraspPolicy):
    cfg = exp_cfg.trainer
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    if os.getenv("WORLD_SIZE", None) is not None:  # launched by torchrun
        init_process_group(backend="nccl", timeout=timedelta(hours=1))

    dataset_config = dataclasses.replace(
        exp_cfg.dataset,
        return_rgb=exp_cfg.dataset.return_rgb and exp_cfg.agent.vision_encoder in ("rgb_resnet", "rgb_resnet_views"),
    )
    if (
        exp_cfg.agent.vision_encoder in ("rgb_resnet", "rgb_resnet_views")
        and exp_cfg.ckpt_path is not None
        and torch.load(ensure_local_filepath(exp_cfg.ckpt_path), map_location="cpu", weights_only=True)[
            "policy_config"
        ]["vision_encoder"]
        == "pointnet"
    ):
        assert not dist.is_initialized() and cfg.evaluator is None and cfg.total_iters > 0
        source = torch.load(ensure_local_filepath(exp_cfg.ckpt_path), map_location="cuda:0", weights_only=True)
        teacher_cfg = GraspPolicyConfig(**source["policy_config"])
        assert teacher_cfg.vision_encoder == "pointnet" and teacher_cfg.grasp_evaluator_guidance_steps > 0
        teacher = build_grasp_policy(teacher_cfg, exp_cfg.robot_config, "cuda:0").eval().requires_grad_(False)
        teacher.load_state_dict(source["agent"], strict=True)
        student_cfg = dataclasses.replace(teacher_cfg, vision_encoder=exp_cfg.agent.vision_encoder)
        student = build_grasp_policy(student_cfg, exp_cfg.robot_config, "cuda:0").eval().requires_grad_(False)
        state = {key: value for key, value in source["agent"].items() if not key.startswith("vision_encoder.")}
        state.update({"vision_encoder." + key: value for key, value in student.vision_encoder.state_dict().items()})
        student.load_state_dict(state, strict=True)
        from torchvision.models import ResNet18_Weights

        pretrained = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
        repeats = student.vision_encoder.conv1.in_channels // 3
        pretrained["conv1.weight"] = pretrained["conv1.weight"].repeat(1, repeats, 1, 1) / repeats
        pretrained["fc.weight"] = student.vision_encoder.fc.weight.detach().cpu()
        pretrained["fc.bias"] = student.vision_encoder.fc.bias.detach().cpu()
        student.vision_encoder.load_state_dict(pretrained, strict=True)
        student.vision_encoder.requires_grad_(True)
        dataset = GraspDataset(
            dataclasses.replace(dataset_config, return_actions=False),
            exp_cfg.processed_data_path,
            exp_cfg.scene_description_path,
            split="train",
        )
        meshes = [
            dataset.scene_descriptions[sid].meshes[dataset.scene_descriptions[sid].target_actor_id].collision_mesh_path
            for sid in dataset.scene_ids
        ]
        objects, held = sorted(set(meshes)), set()
        for index in torch.randperm(len(objects), generator=torch.Generator().manual_seed(20260930)).tolist():
            held.add(objects[index])
            if sum(mesh in held for mesh in meshes) >= 64:
                break
        train_ids = [i for i, mesh in enumerate(meshes) if mesh not in held]
        valid_ids = [i for i, mesh in enumerate(meshes) if mesh in held]
        train_loader = get_iteration_based_dataloader(torch.utils.data.Subset(dataset, train_ids), exp_cfg)
        valid_loader = DataLoader(torch.utils.data.Subset(dataset, valid_ids), batch_size=64, num_workers=0)
        optimizer = torch.optim.AdamW(student.vision_encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scale = teacher.grasp_evaluator.deviation[: teacher_cfg.vision_encoder_feature_dim].clamp_min(1e-3)
        mean = torch.tensor([0.485, 0.456, 0.406] * repeats, device="cuda:0")[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225] * repeats, device="cuda:0")[None, :, None, None]
        out = Path(f"logs/train/{exp_cfg.name}")
        out.mkdir(parents=True, exist_ok=True)
        assert not (out / TRAIN_COMPLETE_MARKER).exists()
        best, begin = float("inf"), time.monotonic()
        with (out / "training.jsonl").open("w") as log:
            for step, batch in enumerate(train_loader):
                if step == cfg.total_iters:
                    break
                obs = batch["observations"]
                points = obs["pointcloud"].cuda().float() * teacher_cfg.pointcloud_scale
                rgb = F.interpolate(obs["rgb"].cuda().float(), size=(128, 128), mode="bilinear", align_corners=False)
                if student_cfg.vision_encoder == "rgb_resnet_views":
                    rgb = rgb.reshape(-1, 3, 128, 128)
                rgb = (rgb / 255 - mean) / std
                student.vision_encoder.train()
                with torch.no_grad():
                    target = teacher.vision_encoder(points)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = student.vision_encoder(rgb)
                    if student_cfg.vision_encoder == "rgb_resnet_views":
                        predicted = predicted.reshape(len(points), -1, predicted.shape[-1]).mean(1)
                loss = F.mse_loss(predicted.float() / scale, target / scale)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.vision_encoder.parameters(), 1.0)
                optimizer.step()
                if step == 0 or (step + 1) % cfg.eval_freq == 0 or step + 1 == cfg.total_iters:
                    student.eval()
                    validation = 0.0
                    with torch.no_grad():
                        for batch in valid_loader:
                            obs = batch["observations"]
                            points = obs["pointcloud"].cuda().float() * teacher_cfg.pointcloud_scale
                            rgb = F.interpolate(
                                obs["rgb"].cuda().float(), size=(128, 128), mode="bilinear", align_corners=False
                            )
                            if student_cfg.vision_encoder == "rgb_resnet_views":
                                rgb = rgb.reshape(-1, 3, 128, 128)
                            rgb = (rgb / 255 - mean) / std
                            predicted = student.vision_encoder(rgb)
                            if student_cfg.vision_encoder == "rgb_resnet_views":
                                predicted = predicted.reshape(len(points), -1, predicted.shape[-1]).mean(1)
                            validation += float(
                                F.mse_loss(predicted / scale, teacher.vision_encoder(points) / scale)
                            ) * len(points)
                    validation /= len(valid_ids)
                    row = {
                        "step": step + 1,
                        "training_mse": float(loss.detach()),
                        "validation_mse": validation,
                        "seconds": time.monotonic() - begin,
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    logger.info(row)
                    if validation < best:
                        best = validation
                        torch.save(
                            {
                                **source,
                                "agent": student.state_dict(),
                                "policy_config": dataclasses.asdict(student_cfg),
                                "processed_data_path": exp_cfg.processed_data_path,
                                "training": {
                                    **source["training"],
                                    "train_ids": sorted(
                                        set(source["training"]["train_ids"]) | {dataset.scene_ids[i] for i in train_ids}
                                    ),
                                    "rgb_distillation": {
                                        "updates": step + 1,
                                        "total_updates": cfg.total_iters,
                                        "trainer": dataclasses.asdict(cfg),
                                        "train_ids": [dataset.scene_ids[i] for i in train_ids],
                                        "validation_ids": [dataset.scene_ids[i] for i in valid_ids],
                                        "validation_mse": validation,
                                        "teacher": exp_cfg.ckpt_path,
                                        "encoder_initialization": "ImageNet_ResNet18",
                                        "supervision": "paired_training_RGB_and_depth_features",
                                        "inference_input": "RGB_only",
                                    },
                                },
                            },
                            out / "policy.pt",
                        )
        write_completion_marker(str(out / TRAIN_COMPLETE_MARKER))
        return
    train_dataset = GraspDataset(
        dataset_config,
        h5_path=exp_cfg.processed_data_path,
        scene_description_path=exp_cfg.scene_description_path,
        split="train",
        indices_only=cfg.cuda_graph,
    )
    assert len(train_dataset) > 0, f"No training data found in {exp_cfg.processed_data_path}"
    if (exp_cfg.agent.generative_model != "cvae" or exp_cfg.agent.normalize_cvae_actions) and exp_cfg.ckpt_path is None:
        assert set(train_dataset.h5_file["data"]) <= set(train_dataset.scene_descriptions)
        action_center = torch.as_tensor(train_dataset.h5_file["meta/action_mean"][:]).flatten()
        action_scale = torch.as_tensor(train_dataset.h5_file["meta/action_std"][:]).flatten().clamp_min(1e-3)
    train_dataloader = get_iteration_based_dataloader(train_dataset, exp_cfg)
    test_dataloader = None
    if cfg.evaluator is not None:
        eval_split = "train" if cfg.eval_on_train else "test"
        eval_dataset_config = dataclasses.replace(
            dataset_config, return_actions=False, return_graspness=False, cache_in_memory=False
        )
        test_dataset = GraspDataset(
            eval_dataset_config,
            h5_path=exp_cfg.processed_data_path,
            scene_description_path=exp_cfg.scene_description_path,
            split=eval_split,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_dataloader_workers,
            drop_last=False,
            shuffle=False,
        )
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        logger.info(f"Rank: {int(os.getenv('LOCAL_RANK', '0'))}, World size: {int(os.getenv('WORLD_SIZE', '1'))}")
        logger.info(f"Dataset size: train={len(train_dataset)}, evaluation={cfg.evaluator is not None}")
        log_dir = Path(f"logs/train/{exp_cfg.name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        training = {
            "policy_config": dataclasses.asdict(exp_cfg.agent),
            "trainer": dataclasses.asdict(cfg),
            "initial_checkpoint": exp_cfg.ckpt_path,
            "initialization": "random" if exp_cfg.ckpt_path is None else "checkpoint",
            "train_ids": train_dataset.scene_ids,
            "data_path": exp_cfg.processed_data_path,
        }
        if exp_cfg.agent.vision_encoder == "rgb_resnet_views" and exp_cfg.ckpt_path is None:
            training["initialization"] = "ImageNet_RGB_encoder_and_random_generator"
            training["encoder_initialization"] = "ImageNet_ResNet18"
        if exp_cfg.ckpt_path is not None:
            initial = torch.load(ensure_local_filepath(exp_cfg.ckpt_path), map_location="cpu", weights_only=True)
            training["train_ids"] = sorted(set(training["train_ids"]) | set(initial["training"]["train_ids"]))
        (log_dir / "training.json").write_text(json.dumps(training, indent=2) + "\n")

    trainer = Trainer(exp_cfg, policy_cls)
    if exp_cfg.agent.vision_encoder == "rgb_resnet_views" and exp_cfg.ckpt_path is None:
        from torchvision.models import ResNet18_Weights

        module = trainer.agent.module if dist.is_initialized() else trainer.agent
        pretrained = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
        pretrained["fc.weight"] = module.vision_encoder.fc.weight.detach().cpu()
        pretrained["fc.bias"] = module.vision_encoder.fc.bias.detach().cpu()
        module.vision_encoder.load_state_dict(pretrained, strict=True)
    if (exp_cfg.agent.generative_model != "cvae" or exp_cfg.agent.normalize_cvae_actions) and exp_cfg.ckpt_path is None:
        module = trainer.agent.module if dist.is_initialized() else trainer.agent
        module.generator.action_center.copy_(action_center)
        module.generator.action_scale.copy_(action_scale)
    if cfg.cuda_graph:
        trainer.train_graph(train_dataset, train_dataloader)
    else:
        trainer.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    if local_rank == 0:
        trainer.writer.close()
        for path in (log_dir / "checkpoints").glob("*.pt"):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            torch.save(
                {
                    **checkpoint,
                    "experiment": exp_cfg.name.split("__")[0],
                    "policy_config": training["policy_config"],
                    "scene_description_path": exp_cfg.scene_description_path,
                    "processed_data_path": exp_cfg.processed_data_path,
                    "training": training,
                },
                path,
            )

    if dist.is_initialized():
        destroy_process_group()

    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        write_completion_marker(str(Path(f"logs/train/{exp_cfg.name}") / TRAIN_COMPLETE_MARKER))


def eval(exp_cfg: ExperimentConfig):
    cfg = exp_cfg.trainer
    dataset_config = exp_cfg.dataset
    eval_split = "train" if cfg.eval_on_train else "test"
    eval_dataset_config = dataclasses.replace(
        dataset_config, return_actions=False, return_graspness=False, cache_in_memory=False
    )
    test_dataset = GraspDataset(
        eval_dataset_config,
        h5_path=exp_cfg.processed_data_path,
        scene_description_path=exp_cfg.scene_description_path,
        split=eval_split,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dataloader_workers,
        drop_last=False,
        shuffle=False,
    )
    logger.info(
        f"Evaluating on {len(test_dataset)} scenes from split '{eval_split}' (eval_on_train={cfg.eval_on_train})"
    )
    trainer = Trainer(exp_cfg)
    eval_results = trainer.evaluate(test_dataloader=test_dataloader, num_samples_per_scene=1)
    logger.info(f"SR: {torch.mean(eval_results['success'].float()):.4f}")
    logger.info(
        f"Penetration: mean={eval_results['penetration'].mean() * 1000:.2f}mm, "
        f"pass@{exp_cfg.trainer.eval_pen_threshold * 1000:.1f}mm="
        f"{eval_results['pen_pass'].float().mean():.2%}"
    )
    joint = (eval_results["success"] & eval_results["pen_pass"]).float().mean()
    logger.info(f"Joint (success & pen pass@{cfg.eval_pen_threshold * 1000:.0f}mm): {joint:.2%}")
    scene_ids = eval_results.pop("scene_ids")

    saved_results = {}
    for scene_idx, scene_id in enumerate(scene_ids):
        saved_results[scene_id] = {k: v[scene_idx].tolist() for k, v in eval_results.items()}
    eval_path = f"logs/eval/{exp_cfg.name}.json"
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(saved_results, f)


if __name__ == "__main__":
    graspkit.link()
    from graspkit.configs.experiment import cli_experiment

    exp_cfg: ExperimentConfig = cli_experiment()
    seed_everything(exp_cfg.trainer.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not exp_cfg.trainer.eval_only:
        train(exp_cfg)
    else:
        eval(exp_cfg)
