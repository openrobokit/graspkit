"""Dev-first policy evaluation with fresh 100-scene processes."""

import graspkit  # noqa: F401 I001

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

import h5py
import msgspec
import numpy as np
import torch
import tyro

from graspkit.constants import DATA_ROOT, PROJECT_ROOT
from graspkit.models.grasp_policy import GraspPolicy


def main(
    stage: Literal["prepare", "run", "episode"],
    root: Path,
    scenes: Path = DATA_ROOT / "scene_description/dex1b/v0.5.0/tabletop_mini.msgpack",
    checkpoint: Path | None = None,
    robot: str = "arx_x5",
    sensor: Literal["depth", "rgb"] = "depth",
    generator: Literal["cvae", "flow_transformer"] = "cvae",
    simulator: Literal["maniskill", "isaac", "mujoco"] = "maniskill",
    loop: Literal["open", "closed"] = "open",
    split: Literal["dev", "test"] = "dev",
    banks: Path | None = None,
    policy_inventory: Path | None = None,
    artifact_root: str = "",
    offset: int = 0,
    count: int = 0,
    num_envs: int = 10,
    isaac_command: tuple[str, ...] = ("/isaac-sim/python.sh",),
    policy_setup: tyro.conf.Suppress[Callable[[GraspPolicy], None] | None] = None,
    diagnostic_observer: tyro.conf.Suppress[Callable[[str, dict], None] | None] = None,
):
    assert num_envs > 0
    root.mkdir(parents=True, exist_ok=True)
    source = msgspec.msgpack.decode(scenes.read_bytes())
    assert len(source["test"]) == 500 and not set(source["train"]) & set(source["test"])
    split_path = root / "evaluation_splits.json"
    if stage == "prepare":
        assert banks is not None and policy_inventory is not None
        protocol = root / "protocol.json"
        assert not protocol.exists(), "Prepare requires a fresh root; preserve existing frozen protocols."
        from graspkit.scene.scene_description import SceneDescription
        from graspkit.utils.io_utils import build_filesystem

        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.benchmark.grasp_policy.rgb_inputs import EXCLUDED_COLLISION_UUIDS

        descriptions = SceneDescription.load_scene_descriptions_msgpack(str(scenes))

        def object_key(scene: SceneDescription) -> str:
            path = scene.meshes[scene.target_actor_id].collision_mesh_path
            uuid = re.search(r"[a-f0-9]{32}", path)
            return uuid.group() if uuid is not None else str(Path(path).parent)

        trained = set()
        evidence = []
        for name in ("arx_x5", "panda_allegro", "agilex_aloha", "yam", "xarm7_sharpa"):
            path = banks / name / "history4096/train4096.h5"
            with h5py.File(path) as bank:
                ids = {key.split("__")[0] for key in bank["data"]}
            trained.update(ids)
            descriptions.update(SceneDescription.load_scene_descriptions_msgpack(str(path.parent / "scenes.msgpack")))
            evidence.append({"robot": name, "bank": str(path), "scenes": len(ids)})
        for filename in json.loads(policy_inventory.read_text()):
            saved = torch.load(filename, map_location="cpu", weights_only=True)
            for key in (
                "train_ids",
                "source_train_ids",
                "scorer_train_ids",
                "scorer_validation_ids",
            ):
                trained.update(sid.split("__")[0] for sid in saved["training"].get(key, []))
        excluded_objects = {object_key(descriptions[sid]) for sid in trained | set(source["test"])}
        candidates = sorted(
            source["train"],
            key=lambda sid: hashlib.sha256(f"dev:0:{sid}".encode()).hexdigest(),
        )
        dev, objects = [], set()
        for sid in candidates:
            obj = object_key(descriptions[sid])
            if obj not in excluded_objects | objects:
                dev.append(sid)
                objects.add(obj)
            if len(dev) == 20:
                break
        assert len(dev) == 20 and not set(dev) & trained
        proof = {
            "dev": sorted(dev),
            "test": sorted(source["test"]),
            "dev_scene_disjoint_from_training": True,
            "dev_object_disjoint_from_training_and_test": True,
            "training_banks": evidence,
            "dev_selection": "one scene per unseen object; SHA256 dev:0:scene ordering",
            "object_identity": "collision mesh object UUID, or mesh-parent path for named assets",
        }
        split_path.write_text(json.dumps(proof, indent=2) + "\n")
        settings = json.loads(Path(__file__).with_name("protocol.json").read_text())
        settings.update(dev_scene_ids=proof["dev"], s3_root=artifact_root.rstrip("/"))
        overlap = root / "compatibility/training_geometry_exclusions.json"
        overlap.parent.mkdir(parents=True, exist_ok=True)
        overlap.write_text(
            json.dumps(
                {
                    "excluded_collision_uuids": EXCLUDED_COLLISION_UUIDS,
                    "canonical_test_scene_ids": sorted(
                        sid for sid in source["test"] if object_key(descriptions[sid]) in EXCLUDED_COLLISION_UUIDS
                    ),
                    "source_repository": "graspkit",
                },
                indent=2,
            )
            + "\n"
        )
        protocol.write_text(json.dumps(settings, indent=2) + "\n")
        if artifact_root:
            fs = build_filesystem(artifact_root)
            assert not fs.exists(artifact_root.rstrip("/") + "/protocol.json"), "Preserve published protocols."
            for path in (split_path, overlap, protocol):
                remote = artifact_root.rstrip("/") + "/" + str(path.relative_to(root))
                fs.put_file(str(path), remote)
        print(json.dumps({"path": str(split_path), "dev": dev}), flush=True)
        return
    assert checkpoint is not None
    splits = json.loads(split_path.read_text())
    condition = Path(robot) / sensor / generator / simulator / loop
    selected = splits[split][offset : offset + (count or (100 if split == "test" else 20))]
    assert selected and len(selected) <= 100
    folder = root / "evaluations" / split / condition / f"{offset:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    ids_path = folder / "ids.json"
    ids_path.write_text(json.dumps(selected) + "\n")
    rows = folder / "rows.jsonl"
    trajectory = folder / "trajectory.pt"
    if stage == "episode":
        from graspkit.configs.env_config import IsaacEnvConfig, MjNativeEnvConfig
        from graspkit.models.checkpoint import load_policy

        policy, cfg, saved = load_policy(checkpoint, "cuda:0")
        assert saved["policy_config"]["generative_model"] == generator
        assert (sensor == "rgb") == policy.cfg.vision_encoder.startswith("rgb")
        evaluator = cfg.grasp_optim.evaluator
        if simulator == "isaac":
            evaluator = replace(evaluator, isaac_env_config=IsaacEnvConfig(worker_command=list(isaac_command)))
        if simulator == "mujoco":
            evaluator = replace(evaluator, mj_native_env_config=MjNativeEnvConfig())
        cfg = replace(
            cfg,
            grasp_optim=replace(cfg.grasp_optim, evaluator=evaluator),
            scene_description_path=str(scenes),
        )
        assert cfg.traj_builder.motion_planner.num_steps == 300
        assert cfg.traj_builder.motion_planner.num_steps_pre_grasp == 150
        assert cfg.grasp_optim.evaluator.target_lift_dist == 0.1
        policy.cfg = replace(policy.cfg, grasp_evaluator_guidance_steps=8, grasp_evaluator_guidance_radius=2.0)
        if policy_setup is not None:
            policy_setup(policy)
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.benchmark.grasp_policy.evaluate_policy import main as evaluate_policy

        phases = {sid: 0 for sid in selected}

        def observe(event: str, data: dict) -> None:
            if event == "render" and split == "dev":
                frame, sid = data["frame"], data["scene"].scene_id
                views = frame["camera_poses"].shape[1]
                np.savez(
                    folder / f"{sid}.phase{phases[sid]}.npz",
                    rgb=frame["rgb"][0].cpu().numpy(),
                    pointcloud=frame["xyz"][0].reshape(views, -1, 3).cpu().numpy(),
                    segmentation=frame["seg"][0].reshape(views, -1, 1).cpu().numpy(),
                    qpos=data["qpos"].cpu().numpy(),
                    cameras=frame["camera_poses"][0].cpu().numpy(),
                    object_pose=data["object_pose"].cpu().numpy(),
                    root_pose=data["root_pose"].cpu().numpy(),
                )
                phases[sid] += 1
            if diagnostic_observer is not None:
                diagnostic_observer(event, data)

        evaluate_policy(
            output=rows,
            checkpoint=checkpoint,
            split="train" if split == "dev" else "test",
            ids_path=ids_path,
            scene_description_path=scenes,
            mode=f"policy_{loop}_loop",
            selector="learned",
            guidance_steps=8,
            num_samples=64,
            num_envs=num_envs,
            seed=0,
            trajectories=trajectory,
            policy=policy,
            config=cfg,
            observer=observe,
        )
        entries = [json.loads(line) for line in rows.read_text().splitlines()]
        rows.write_text("".join(json.dumps(entry | {"simulator": simulator}) + "\n" for entry in entries))
        return
    from graspkit.utils.io_utils import build_filesystem

    remote = f"{artifact_root.rstrip('/')}/{folder.relative_to(root)}"
    fs = build_filesystem(artifact_root) if artifact_root else None
    marker = folder / "complete.json"
    if fs is not None and fs.exists(f"{remote}/complete.json"):
        proof = json.loads(fs.cat(f"{remote}/complete.json"))
        assert proof["scene_ids"] == selected
        for filename in proof["files"]:
            (folder / filename).write_bytes(fs.cat(f"{remote}/{filename}"))
        marker.write_text(json.dumps(proof, indent=2) + "\n")
    if marker.exists():
        proof = json.loads(marker.read_text())
        assert proof["scene_ids"] == selected
        print(json.dumps(dict(resumed=True, **proof)), flush=True)
        return
    rows.unlink(missing_ok=True)
    trajectory.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "episode",
        "--root",
        str(root),
        "--scenes",
        str(scenes),
        "--checkpoint",
        str(checkpoint),
        "--robot",
        robot,
        "--sensor",
        sensor,
        "--generator",
        generator,
        "--simulator",
        simulator,
        "--loop",
        loop,
        "--split",
        split,
        "--offset",
        str(offset),
        "--count",
        str(len(selected)),
        "--num-envs",
        str(num_envs),
        "--isaac-command",
        *isaac_command,
    ]
    started = time.monotonic()
    with (folder / "run.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(10)
            if fs is not None:
                for filename in ("run.log", "rows.jsonl"):
                    if (folder / filename).exists():
                        fs.put_file(str(folder / filename), f"{remote}/{filename}")
        if fs is not None:
            fs.put_file(str(folder / "run.log"), f"{remote}/run.log")
        assert process.returncode == 0, f"Evaluator exited {process.returncode}: {folder / 'run.log'}"
    entries = [json.loads(line) for line in rows.read_text().splitlines()]
    assert [entry["scene_id"] for entry in entries] == selected
    assert all(
        entry["control_steps"] == 300
        and entry["observation_count"] == 1 + (loop == "closed")
        and entry["model"] == generator
        for entry in entries
    )
    trajectories = torch.load(trajectory, map_location="cpu", weights_only=False)
    assert [entry["scene_id"] for entry in trajectories] == selected
    assert all(entry["qpos"].shape[0] == 300 and np.isfinite(entry["qpos"]).all() for entry in trajectories)
    files = [str(path.relative_to(folder)) for path in sorted(folder.rglob("*")) if path.is_file()]
    if fs is not None:
        for filename in files:
            fs.put_file(str(folder / filename), f"{remote}/{filename}")
    proof = {
        "checkpoint": str(checkpoint),
        "scene_ids": selected,
        "successes": sum(entry["success"] for entry in entries),
        "trials": len(entries),
        "files": files,
        "seconds": time.monotonic() - started,
        "simulator": simulator,
        "loop": loop,
        "split": split,
        "robot": robot,
        "input": sensor,
        "model": generator,
    }
    proof["num_envs"] = num_envs
    marker.write_text(json.dumps(proof, indent=2) + "\n")
    if fs is not None:
        fs.put_file(str(marker), f"{remote}/complete.json")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
    tyro.cli(main)
