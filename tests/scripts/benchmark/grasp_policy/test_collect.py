"""Tests for `scripts/benchmark/grasp_policy/collect.py`."""

from pathlib import Path

import h5py
import numpy as np
import torch

from graspkit.scene.scene_description import MeshDescription, SceneDescription
from scripts.benchmark.grasp_policy import collect


class TestCollect:
    def test_collect_assemble_bank(self, tmp_path: Path):
        scenes = {
            f"scene{i}": SceneDescription(
                f"scene{i}", "object", meshes={"object": MeshDescription("object", f"{i % 3}.obj", (1, 1, 1))}
            )
            for i in range(6)
        }
        SceneDescription.save_scene_descriptions_msgpack(scenes, str(tmp_path / "scenes.msgpack"))
        rng = np.random.default_rng(0)
        written, folders = {}, []
        for round_index, (simulator, proposals) in enumerate((("maniskill", 64), ("isaac", 64), ("maniskill", 0))):
            folders.append(str(tmp_path / f"round{round_index}"))
            Path(folders[-1]).mkdir()
            with h5py.File(f"{folders[-1]}/0000-0001-00000.h5", "w") as shard:
                shard.attrs.update(robot="arx_x5", simulator=simulator, checkpoint="")
                for sid in scenes:
                    for phase in ("initial", "feedback"):
                        hidden = sid == "scene5" and phase == "feedback"
                        entry = {
                            "obs/pointcloud": rng.random((1, 8, 3), np.float32),
                            "obs/pointcloud_segmentation": np.full((1, 8, 1), 0 if hidden else 3, np.int16),
                            "obs/target_actor_segmentation_id": np.int64(3),
                            "obs/rgb": np.zeros((4, 2, 2, 3), np.uint8),
                            "obs/camera_poses": np.zeros((4, 1, 7), np.float32),
                            "obs/qpos": np.zeros((2, 7), np.float32),
                            "candidates": rng.random((4 + proposals, 3, 7), np.float32),
                            "success": (rng.random(4 + proposals) < 0.5) & (sid != "scene0"),
                            "source": np.array(["gt"] * 4 + ["policy"] * proposals, "S6"),
                            "displacement": np.zeros(3, np.float32),
                        }
                        for key, value in entry.items():
                            shard[f"{sid}__{phase}/{key}"] = value
                        written[f"{sid}__{simulator}{'1' if round_index == 2 else ''}__{phase}"] = entry
        bank = tmp_path / "bank"
        collect.main(
            "assemble",
            "arx_x5",
            str(tmp_path / "scenes.msgpack"),
            shards=tuple(folders),
            bank=bank,
            validation_objects=1,
        )
        kept = {
            state: entry for state, entry in written.items() if not state.startswith("scene5__") or "initial" in state
        }
        with h5py.File(bank / "data.h5") as data:
            assert set(data["data"]) == set(kept) and len(kept) == 33
            positives = []
            for state, entry in kept.items():
                assert np.array_equal(data["data"][state]["obs/qpos"][:], entry["obs/qpos"])
                assert ("actions" in data["data"][state]) == entry["success"].any()
                if entry["success"].any():
                    assert np.array_equal(data["data"][state]["actions"][:], entry["candidates"][entry["success"]])
                    positives.append(entry["candidates"][entry["success"]])
            np.testing.assert_allclose(data["meta/action_mean"][:], np.concatenate(positives).mean(0), rtol=1e-5)
            np.testing.assert_allclose(data["meta/action_std"][:], np.concatenate(positives).std(0), rtol=1e-4)
        replay = torch.load(bank / "replay.pt", weights_only=True)
        assert sorted(replay["scene_ids"]) == sorted(state for state in kept if "maniskill1" not in state)
        assert replay["actions"].shape == (22, 64, 21) and replay["features"].shape == (22, 1, 512)
        for row, state in enumerate(replay["scene_ids"]):
            assert torch.equal(replay["actions"][row], torch.from_numpy(kept[state]["candidates"][4:].reshape(64, 21)))
            assert torch.equal(replay["rewards"][row], torch.from_numpy(kept[state]["success"][4:]).float())
        mesh = {
            state: scenes[state.split("__")[0]].meshes["object"].collision_mesh_path for state in replay["scene_ids"]
        }
        held = {mesh[state] for state in replay["scorer_validation_ids"]}
        assert len(held) == 1 and replay["scorer_validation_ids"] == [state for state in mesh if mesh[state] in held]
        assert replay["reserved_validation_ids"] == []
        aliases = SceneDescription.load_scene_descriptions_msgpack(str(bank / "scenes.msgpack"), return_splits=True)
        assert aliases["test"] == {} and all(aliases["train"][state] == scenes[state.split("__")[0]] for state in kept)
