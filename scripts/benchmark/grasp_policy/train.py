"""Train native depth/RGB policies and their success scorers with a versioned recipe on a `collect.py` bank."""

import graspkit  # noqa: F401 I001

import sys
from graspkit.constants import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import h5py
import torch
import tyro

from graspkit.configs.experiment import build_experiment_config
from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.constants import DATA_ROOT
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.models.checkpoint import load_policy
from graspkit.utils.io_utils import build_filesystem
from graspkit.utils.seed_utils import seed_everything


def backup_outputs(out: Path, s3: str, robot: str, generator: str, sensor: str):
    if s3:
        fs = build_filesystem(s3)
        backed_up = []
        previous = (
            {entry["file"]: entry for entry in json.loads((out / "backup.json").read_text())}
            if (out / "backup.json").exists()
            else {}
        )
        suffix = "/smoke200" if out.name == "smoke200" else ""
        for path in sorted(out.iterdir(), key=lambda path: (path.name == "latest.json", path.name)):
            if path.is_file() and path.name != "backup.json":
                name = (
                    json.loads((out / "latest.json").read_text())["checkpoint"]
                    if path.name == "latest.pt"
                    else path.name
                )
                destination = f"{s3.rstrip('/')}/{robot}/{sensor}/{generator}{suffix}/{name}"
                entry = {"file": path.name, "s3": destination, "mtime_ns": path.stat().st_mtime_ns}
                if previous.get(path.name) != entry:
                    fs.put_file(str(path), destination)
                backed_up.append(entry)
        (out / "backup.json").write_text(json.dumps(backed_up, indent=2) + "\n")
        fs.put_file(
            str(out / "backup.json"),
            f"{s3.rstrip('/')}/{robot}/{sensor}/{generator}{suffix}/backup.json",
        )


def validate_inference(checkpoint: Path, cfg: ExperimentConfig) -> dict:
    policy, _, _ = load_policy(checkpoint, "cuda:0")
    data = GraspDataset(
        replace(cfg.dataset, return_actions=False, cache_in_memory=False),
        cfg.processed_data_path,
        cfg.scene_description_path,
    )
    observed = torch.utils.data.default_collate([data[index]["observations"] for index in range(3)])
    observed = {key: value.cuda() for key, value in observed.items()}
    data.h5_file.close()
    conditions = []
    hook = (
        policy.rgb_conditioning if policy.cfg.vision_encoder == "rgb_resnet_views" else policy.vision_encoder
    ).register_forward_hook(lambda module, arguments, output: conditions.append(output.detach()))
    with torch.no_grad():
        actions = policy.get_action(observed, 64)
        assert torch.isfinite(actions).all()
        unique = [len(torch.unique(values.flatten(1), dim=0)) for values in actions]
        assert unique == [64, 64, 64]
        if policy.cfg.grasp_evaluator_guidance_steps:
            assert len(conditions) == 1
            logits = policy.grasp_evaluator(
                conditions[0][:, None].expand(-1, 64, -1).flatten(0, 1),
                actions.flatten(0, 1).flatten(1),
            )
            assert torch.isfinite(logits).all()
    hook.remove()
    return {
        "unique_proposals": unique,
        "guidance_steps": policy.cfg.grasp_evaluator_guidance_steps,
        "scorer_logits_finite": bool(policy.cfg.grasp_evaluator_guidance_steps),
        "gpu_inference_verified": True,
        "finite_unique_64_proposals": True,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def main(
    robot: Literal["arx_x5", "panda_allegro", "agilex_aloha", "yam", "xarm7_sharpa"],
    generator: Literal["cvae", "flow_transformer"],
    stage: Literal["smoke", "actor", "reencode", "scorer", "train"],
    bank: Path,
    root: Path = DATA_ROOT / "runs/grasp_policy",
    sensor: Literal["depth", "rgb"] = "rgb",
    steps: int = 8000,
    scorer_steps: int = 10080,
    scorer_validation_scenes: int = 128,
    batch_size: int = 64,
    grad_clip_norm: float = 0.0,
    objective: Literal["auto", "native", "scaled"] = "auto",
    bundle_root: Path | None = None,
    bundle_s3: str = "",
    s3: str = "",
):
    """`bank` is a `collect.py --stage assemble` directory: data.h5, replay.pt and scenes.msgpack."""
    root, bank = root.resolve(), bank.resolve()
    out = root / "results" / robot / sensor / generator
    if stage == "smoke":
        steps = 200
        out = out / "smoke200"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seed_everything(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scaled_objective = objective == "scaled" or (
        objective == "auto" and (generator != "cvae" or robot in ("panda_allegro", "xarm7_sharpa"))
    )
    cfg = build_experiment_config(f"{robot}_unified_{sensor}")
    cfg = replace(
        cfg,
        name=f"{robot}_unified_{sensor}__{root.name}_{generator}_u{steps}",
        processed_data_path=str(bank / "data.h5"),
        scene_description_path=str(bank / "scenes.msgpack"),
        ckpt_path=None,
        agent=replace(
            cfg.agent,
            generative_model=generator,
            cvae_latent_dim=256,
            cvae_hidden_dim=256,
            vision_encoder_feature_dim=512,
            kl_weight=0.01,
            use_mean=False,
            grasp_evaluator_guidance_steps=0,
            enable_kinematic_loss=True,
            pointcloud_scale=10.0,
            normalize_cvae_actions=generator == "cvae",
            reconstruction_loss="phase_std_mse" if scaled_objective else "native",
            grasp_evaluator_hand_joint_features=True,
            grasp_evaluator_mask_constant_features=False,
        ),
        dataset=replace(
            cfg.dataset,
            num_actions=8,
            num_points=2048,
            return_actions=True,
            return_graspness=False,
            return_rgb=sensor == "rgb",
            return_object_pointcloud_only=sensor == "depth",
            cache_in_memory=True,
        ),
        trainer=replace(
            cfg.trainer,
            seed=0,
            batch_size=batch_size,
            total_iters=steps,
            save_freq=min(steps, 1000),
            log_freq=100,
            evaluator=None,
            track=False,
            num_dataloader_workers=0,
            lr=3e-4,
            weight_decay=1e-4,
            fast_step=False,
            grad_clip_norm=grad_clip_norm,
            cuda_graph=False,
        ),
    )
    with h5py.File(cfg.processed_data_path) as data:
        center = torch.tensor(data["meta/action_mean"][:]).flatten()
        scale = torch.tensor(data["meta/action_std"][:]).flatten().clamp_min(1e-3)
        sid = next(iter(data["data"]))
        obs = data["data"][sid]["obs"]
        if sensor == "rgb":
            assert obs["rgb"].shape[0] == 4 and obs["camera_poses"].shape == (4, 1, 7)
    recipe = {
        "agent": asdict(cfg.agent),
        "dataset": asdict(cfg.dataset),
        "trainer": asdict(cfg.trainer),
        "bank": str(bank),
        "sensor": sensor,
        "recipe_version": "graspkit_grasp_policy",
        "views": 4 if sensor == "rgb" else None,
        "view_pairing": "unchanged bank RGB/qpos/camera_poses",
        "actor_updates": steps + 1,
        "scorer_updates": scorer_steps,
        "objective": "per_phase_mse_with_joint_limit_recovery" if scaled_objective else "native",
        "objective_override": objective,
        "normalization": "training action mean/std, floor 1e-3",
    }
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

    if s3:
        fs = build_filesystem(s3)
        prefix = f"{s3.rstrip('/')}/{robot}/{sensor}/{generator}" + ("/smoke200" if stage == "smoke" else "")
        for name in (
            "actor.pt",
            "actor_validation.json",
            "actor_seconds.json",
            "replay.pt",
            "policy.pt",
            "latest.json",
        ):
            destination = f"{prefix}/{name}"
            if not (out / name).exists() and fs.exists(destination):
                fs.get_file(destination, str(out / name))
        if (out / "latest.json").exists() and not (out / "latest.pt").exists():
            committed = json.loads((out / "latest.json").read_text())
            fs.get_file(f"{prefix}/{committed['checkpoint']}", str(out / "latest.pt"))
    actor = out / "actor.pt"
    if stage in ("smoke", "actor", "train") and not actor.exists():
        from graspkit.stages.trainer import Trainer
        from scripts.stages.train.train import get_iteration_based_dataloader

        data = GraspDataset(cfg.dataset, cfg.processed_data_path, cfg.scene_description_path)
        trainer = Trainer(cfg)
        trainer.agent.generator.action_center.copy_(center.to(trainer.device))
        trainer.agent.generator.action_scale.copy_(scale.to(trainer.device))
        if sensor == "rgb":
            from torchvision.models import ResNet18_Weights

            pretrained = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
            pretrained["fc.weight"] = trainer.agent.vision_encoder.fc.weight.detach().cpu()
            pretrained["fc.bias"] = trainer.agent.vision_encoder.fc.bias.detach().cpu()
            trainer.agent.vision_encoder.load_state_dict(pretrained, strict=True)
        metadata = {
            "experiment": f"{robot}_unified_{sensor}",
            "policy_config": asdict(cfg.agent),
            "processed_data_path": cfg.processed_data_path,
            "scene_description_path": cfg.scene_description_path,
            "training": {
                "trainer": asdict(cfg.trainer),
                "train_ids": data.scene_ids,
                "encoder_initialization": "ImageNet_ResNet18" if sensor == "rgb" else "random",
                "objective": recipe["objective"],
                "objective_override": objective,
                "resume_data_order": "saved weighted-sampler generator; fresh sampler iteration",
            },
        }
        latest = out / "latest.pt"
        first = 0
        resumed = None
        if latest.exists():
            resumed = torch.load(latest, map_location="cuda:0", weights_only=True)
            assert resumed["policy_config"] == asdict(cfg.agent)
            first = resumed["iteration"] + 1
        assert first <= steps + 1
        loader = get_iteration_based_dataloader(
            data,
            replace(cfg, trainer=replace(cfg.trainer, total_iters=steps - first)),
            generator=torch.Generator().manual_seed(cfg.trainer.seed),
        )
        measured = time.monotonic()
        trainer.train(
            loader,
            checkpoint_dir=out,
            metadata=metadata,
            resume=resumed,
            publish=lambda path: backup_outputs(path, s3, robot, generator, sensor),
        )
        final = torch.load(latest, map_location="cpu", weights_only=True)
        assert final["iteration"] == steps
        torch.save(
            {
                key: value
                for key, value in final.items()
                if key
                in (
                    "agent",
                    "iteration",
                    "experiment",
                    "policy_config",
                    "processed_data_path",
                    "scene_description_path",
                    "training",
                )
            },
            actor,
        )
        (out / "actor_seconds.json").write_text(
            json.dumps(
                {
                    "seconds": time.monotonic() - started,
                    "updates": steps + 1,
                    "resumed_from_updates": first,
                    "measured_training_seconds": time.monotonic() - measured,
                    "updates_per_second": (steps + 1 - first) / (time.monotonic() - measured),
                }
            )
            + "\n"
        )
        trainer.writer.close()
        data.h5_file.close()
        del trainer, data, loader, final, resumed
        torch.cuda.empty_cache()
    if stage in ("smoke", "actor", "train"):
        (out / "actor_validation.json").write_text(json.dumps(validate_inference(actor, cfg)) + "\n")
        torch.cuda.empty_cache()
    backup_outputs(out, s3, robot, generator, sensor)
    replay_path = out / "replay.pt"
    if stage in ("reencode", "train") and not replay_path.exists():
        policy, _, saved = load_policy(actor, "cuda:0")
        replay = torch.load(bank / "replay.pt", map_location="cpu", weights_only=True)
        data = GraspDataset(
            replace(cfg.dataset, return_actions=False, cache_in_memory=False),
            cfg.processed_data_path,
            cfg.scene_description_path,
        )
        lookup = {sid: index for index, sid in enumerate(data.scene_ids)}
        assert set(replay["scene_ids"]) <= set(lookup)
        features, captured = [], []
        hook = (
            policy.rgb_conditioning if policy.cfg.vision_encoder == "rgb_resnet_views" else policy.vision_encoder
        ).register_forward_hook(lambda module, arguments, output: captured.append(output.detach().cpu()))
        with torch.no_grad():
            for offset in range(0, len(replay["scene_ids"]), 32):
                ids = replay["scene_ids"][offset : offset + 32]
                obs = torch.utils.data.default_collate([data[lookup[sid]]["observations"] for sid in ids])
                policy.get_action({key: value.cuda() for key, value in obs.items()}, 1)
                assert len(captured) == 1
                features.append(captured.pop())
                print(
                    f"{sensor} reencoded {offset + len(ids)}/{len(replay['scene_ids'])}",
                    flush=True,
                )
        hook.remove()
        data.h5_file.close()
        features = torch.cat(features)[:, None]
        assert features.shape == replay["features"].shape and torch.isfinite(features).all()
        torch.save(
            {
                **replay,
                "features": features,
            },
            replay_path.with_suffix(".part"),
        )
        replay_path.with_suffix(".part").replace(replay_path)
        del policy, data, features, replay, saved
        torch.cuda.empty_cache()
    backup_outputs(out, s3, robot, generator, sensor)
    if stage in ("scorer", "train") and not (out / "policy.pt").exists():
        from scripts.stages.train import train_grasp_evaluator

        (out / "scorer.inprogress.pt").unlink(missing_ok=True)
        train_grasp_evaluator.main(
            f"{robot}_unified_{sensor}",
            replay_path,
            actor,
            out / "scorer.inprogress.pt",
            iterations=scorer_steps,
            guidance_steps=8,
            guidance_radius=2.0,
            ranking_weight=1.0,
            validation_scenes=scorer_validation_scenes,
        )
        (out / "scorer.inprogress.pt").replace(out / "policy.pt")
        original = torch.load(actor, map_location="cpu", weights_only=True)
        policy, _, saved = load_policy(out / "policy.pt", "cuda:0")
        assert all(torch.equal(value, saved["agent"][key]) for key, value in original["agent"].items())
        assert all(torch.isfinite(value).all() for value in saved["agent"].values())
    if stage in ("scorer", "train"):
        from scripts.benchmark.grasp_policy.package import export_bundle

        validation = validate_inference(out / "policy.pt", cfg)
        (out / "policy_validation.json").write_text(json.dumps(validation) + "\n")
        validation.update(finite_unique_64_proposals=True, gpu_inference_verified=True)
        bundle = (root / "bundles" if bundle_root is None else bundle_root.resolve()) / robot / sensor / generator
        export_bundle(out / "policy.pt", actor, bundle, validation)
        backup_outputs(
            bundle,
            bundle_s3 or (s3.replace("/training", "/bundles") if s3 else ""),
            robot,
            generator,
            sensor,
        )
        (out / "train.complete").touch()
    (out / f"{stage}_seconds.json").write_text(json.dumps({"seconds": time.monotonic() - started}) + "\n")
    backup_outputs(out, s3, robot, generator, sensor)


if __name__ == "__main__":
    tyro.cli(main)
