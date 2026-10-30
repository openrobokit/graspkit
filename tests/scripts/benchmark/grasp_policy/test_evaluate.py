"""Tests for `scripts/benchmark/grasp_policy/evaluate.py` and its report."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import msgspec
import numpy as np
import pytest
import torch

from graspkit.constants import PROJECT_ROOT
from graspkit.scene.scene_description import SceneDescription
from scripts.benchmark.grasp_policy import evaluate, report
from scripts.benchmark.grasp_policy.rgb_inputs import EXCLUDED_COLLISION_UUIDS


class TestEvaluate:
    def test_isaac_batch_subprocess_and_provenance(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        ids = [f"dev_{index:03d}" for index in range(20)]
        content = msgspec.msgpack.encode({"train": ids, "test": [f"test_{index:03d}" for index in range(500)]})
        scenes = tmp_path / "scenes.msgpack"
        scenes.write_bytes(content)
        (tmp_path / "evaluation_splits.json").write_text(json.dumps({"dev": ids}))
        (tmp_path / "protocol.json").write_text(json.dumps({"seed": 0}))
        checkpoint = tmp_path / "policy.pt"
        checkpoint.write_bytes(b"checkpoint")
        folder = tmp_path / "evaluations/dev/arx_x5/depth/cvae/isaac/open/0000"
        commands = []

        class Episode:
            returncode = 0

            def __init__(self, command: list[str], **kwargs: object):
                commands.append(command)
                (folder / "rows.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {
                                "scene_id": sid,
                                "control_steps": 300,
                                "observation_count": 1,
                                "model": "cvae",
                                "success": False,
                            }
                        )
                        + "\n"
                        for sid in ids
                    )
                )
                torch.save([{"scene_id": sid, "qpos": np.zeros((300, 1))} for sid in ids], folder / "trajectory.pt")

            def poll(self) -> int:
                return 0

        monkeypatch.setattr(evaluate.subprocess, "Popen", Episode)
        evaluate.main(stage="run", root=tmp_path, scenes=scenes, checkpoint=checkpoint, simulator="isaac", num_envs=32)
        command = commands[0]
        assert command[command.index("--num-envs") + 1] == "32"
        assert command[command.index("--count") + 1] == "20"
        proof = json.loads((folder / "complete.json").read_text())
        assert proof["num_envs"] == 32
        assert proof["trials"] == 20 and proof["scene_ids"] == ids

    def test_fresh_protocol_prepare_and_report(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        template = json.loads((PROJECT_ROOT / "scripts/benchmark/grasp_policy/protocol.json").read_text())
        safe = [f"safe_{index}" for index in range(20)]
        test = [f"test_{index}" for index in range(500)]
        aliases = {
            "core_alias": "a" * 32,
            "test_alias": EXCLUDED_COLLISION_UUIDS[0],
            "scorer_alias_candidate": "b" * 32,
        }
        source = {"train": safe + list(aliases), "test": test}
        content = msgspec.msgpack.encode(source)
        scenes = tmp_path / "scenes.msgpack"
        scenes.write_bytes(content)

        def scene(uuid: str) -> SimpleNamespace:
            return SimpleNamespace(
                target_actor_id=0, meshes=[SimpleNamespace(collision_mesh_path=f"/objects/{uuid}/convex.obj")]
            )

        descriptions = {sid: scene(f"{index + 2000:032x}") for index, sid in enumerate(safe)}
        descriptions.update({sid: scene(f"{index + 1000:032x}") for index, sid in enumerate(test)})
        descriptions[test[0]] = scene(EXCLUDED_COLLISION_UUIDS[0])
        descriptions.update({sid: scene(uuid) for sid, uuid in aliases.items()})
        bank_descriptions = {"dexgraspnet_core_alias": scene("a" * 32), "scorer_validation_alias": scene("b" * 32)}
        monkeypatch.setattr(
            SceneDescription,
            "load_scene_descriptions_msgpack",
            lambda filename: dict(descriptions) if Path(filename) == scenes else dict(bank_descriptions),
        )
        banks = tmp_path / "banks"
        for robot in template["robots"]:
            bank = banks / robot / "history4096"
            bank.mkdir(parents=True)
            with h5py.File(bank / "train4096.h5", "w") as file:
                file.create_group("data/dexgraspnet_core_alias__feedback")
        actor = tmp_path / "inventory_actor.pt"
        torch.save({"training": {"scorer_validation_ids": ["scorer_validation_alias"]}}, actor)
        inventory = tmp_path / "inventory.json"
        inventory.write_text(json.dumps([str(actor)]))
        root = tmp_path / "run"
        evaluate.main(stage="prepare", root=root, scenes=scenes, banks=banks, policy_inventory=inventory)
        splits = json.loads((root / "evaluation_splits.json").read_text())
        assert splits["dev"] == sorted(safe) and splits["test"] == sorted(test)
        protocol_bytes = (root / "protocol.json").read_bytes()
        protocol = json.loads(protocol_bytes)
        assert (protocol["seed"], protocol["proposals"], protocol["control_steps"]) == (0, 64, 300)
        exclusions = json.loads((root / "compatibility/training_geometry_exclusions.json").read_text())
        assert exclusions["canonical_test_scene_ids"] == [test[0]]
        report.main(root=root)
        manifest = json.loads((root / "native_report/checkpoints.json").read_text())
        assert manifest["verified_policies"] == manifest["full_test_episodes"] == 0
        assert len(manifest["loop_entries"]) == 40
        with pytest.raises(AssertionError, match="fresh root"):
            evaluate.main(stage="prepare", root=root, scenes=scenes, banks=banks, policy_inventory=inventory)
        assert (root / "protocol.json").read_bytes() == protocol_bytes
