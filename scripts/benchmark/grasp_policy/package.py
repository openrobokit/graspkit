"""Export a strictly loaded native policy and its training conventions."""

import graspkit  # noqa: F401 I001

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch
import tyro
from graspkit.models.checkpoint import load_policy
from graspkit.robot.grasp_robot import GraspRobot


def export_bundle(policy_path: Path, actor_path: Path, bundle: Path, validation: dict) -> dict:
    policy, cfg, saved = load_policy(policy_path)
    assert policy.cfg.grasp_evaluator_hand_joint_features and not policy.cfg.grasp_evaluator_mask_constant_features
    assert policy.cfg.generative_model != "cvae" or policy.cfg.normalize_cvae_actions
    recipe = json.loads((policy_path.parent / "recipe.json").read_text())
    dataset = dict(recipe["dataset"])
    if "cache_object_data" in dataset:
        dataset["cache_in_memory"] = dataset.pop("cache_object_data")
    cfg = replace(cfg, dataset=replace(cfg.dataset, **dataset))
    actor = torch.load(actor_path, map_location="cpu", weights_only=True)
    actor_keys = [key for key in saved["agent"] if not key.startswith("grasp_evaluator.")]
    assert all(torch.equal(actor["agent"][key], saved["agent"][key]) for key in actor_keys)
    assert all(torch.isfinite(value).all() for value in saved["agent"].values())
    helper = GraspRobot.from_config(cfg.robot_config)
    sensor = "rgb" if policy.cfg.vision_encoder.startswith("rgb") else "depth"
    robot = saved["experiment"].removesuffix(f"_unified_{sensor}")
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(policy_path, bundle / "policy.pt")
    shutil.copy2(actor_path, bundle / "actor.pt")
    shutil.copy2(policy_path.parent / "recipe.json", bundle / "recipe.json")
    train_ids = actor["training"]["train_ids"]
    (bundle / "training_ids.json").write_text(json.dumps(train_ids, indent=2) + "\n")
    conventions = {
        "joint_order": helper.actuated_joint_names,
        "hand_joint_indices": helper.hand_joint_indices.tolist(),
        "base_type": cfg.robot_config.base_type,
        "action_phases": ["pregrasp", "grasp", "close"],
        "pointcloud": {
            "num_points": cfg.dataset.num_points,
            "return_object_pointcloud_only": cfg.dataset.return_object_pointcloud_only,
            "coordinate_frame": "world",
            "units": "meters before configured pointcloud_scale",
        },
        "rgb": "Four matched views in native camera order; images and camera metadata stay paired; shared native preprocess_rgb",
        "configuration": asdict(cfg),
        "policy_config": asdict(policy.cfg),
        "training": {key: value for key, value in actor["training"].items() if key not in ("train_ids", "test_ids")},
    }
    (bundle / "config.json").write_text(json.dumps(conventions, default=str, indent=2) + "\n")
    audit = {
        **validation,
        "robot": robot,
        "sensor": sensor,
        "input": sensor,
        "generator": policy.cfg.generative_model,
        "strict_load": True,
        "actor_tensors_bit_identical_after_scorer": len(actor_keys),
        "actor_updates": actor["training"]["trainer"]["total_iters"] + 1,
        "scorer_updates": saved["iteration"],
        "scorer_run_updates": saved["training"]["iterations"],
        "scorer_selection": "best held-out replay validation checkpoint during the completed scorer run",
        "training_status": "complete",
        "new_simulator_evaluation": "pending",
        "loop_modes": {
            "open": "one observation, 300 executed steps",
            "closed": "fresh observation at step 75, replan 225 steps; same weights",
        },
        "local_bundle": str(bundle),
    }
    (bundle / "bundle.json").write_text(json.dumps(audit, indent=2) + "\n")
    (bundle / "README.md").write_text(
        "Load with the pinned GraspKit environment:\n\n```python\nfrom pathlib import Path\nfrom graspkit.models.checkpoint import load_policy\npolicy, experiment, checkpoint = load_policy(Path('policy.pt'), device='cuda:0')\nactions = policy.get_action(observations, 64)\n```\n\nThe hand-joint grasp evaluator and normalized inference are part of policy.pt; see config.json. Closed loop observes again at step 75 and replans 225 steps. New simulator results are tracked separately.\n"
    )
    return audit


def main(policy_path: Path, actor_path: Path, bundle: Path, validation: Path):
    export_bundle(policy_path, actor_path, bundle, json.loads(validation.read_text()))


if __name__ == "__main__":
    tyro.cli(main)
