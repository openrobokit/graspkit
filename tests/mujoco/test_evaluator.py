"""Golden parity: MuJoCo evaluator matches DexGraspBench on vendored shadow_right `fc` fixtures (deterministic)."""

import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from graspkit.configs.env_config import MjEvalConfig

MUJOCO_EVAL_ROOT = Path(__file__).parents[1] / "fixtures/mujoco_eval"
REFERENCE_DIR = os.path.join(MUJOCO_EVAL_ROOT, "reference_eval")
GRASPDATA_DIR = os.path.join(MUJOCO_EVAL_ROOT, "graspdata")
MESH_ORDER = json.loads((MUJOCO_EVAL_ROOT / "mesh_order.json").read_text())
# One grasp per object/scale: 20 golden cases using only local fixtures.
SAMPLE = sorted(glob.glob(os.path.join(REFERENCE_DIR, "*/*/0.npy")))

assert len(SAMPLE) == 20, "Missing local MuJoCo golden fixtures"


@pytest.mark.hf_assets
@pytest.mark.skipif(not mujoco.__version__.startswith("3.4."), reason="goldens were recorded with MuJoCo 3.4")
class TestMjEvaluator:
    @pytest.mark.parametrize("ref_npy", SAMPLE, ids=lambda p: os.path.relpath(p, REFERENCE_DIR))
    def test_golden_shadow_fc(self, ref_npy: str, monkeypatch: pytest.MonkeyPatch):
        from graspkit.mujoco import MjEvaluator, dex_grasp  # the Shadow hand MJCF comes from robokit's HF assets

        rel = os.path.relpath(ref_npy, REFERENCE_DIR)
        grasp_data = np.load(os.path.join(GRASPDATA_DIR, rel), allow_pickle=True).item()
        ref = np.load(ref_npy, allow_pickle=True).item()

        # Mesh insertion order is part of these goldens, not the host filesystem's directory order.
        object_name = Path(rel).parts[0]
        mesh_dir = MUJOCO_EVAL_ROOT / "example_object" / object_name / "urdf/meshes"
        order = MESH_ORDER[object_name]
        assert len(order) == len(set(order)) and set(order) == set(os.listdir(mesh_dir))
        monkeypatch.setattr(dex_grasp, "os", SimpleNamespace(path=os.path, listdir={str(mesh_dir): order}.__getitem__))

        out = MjEvaluator(MjEvalConfig(obj_root=str(MUJOCO_EVAL_ROOT / "example_object"))).evaluate(grasp_data)

        assert bool(out["succ_flag"]) == bool(ref["succ_flag"])
        # RoboKit's slerp differs from scipy's by ~1e-16; the simulation grows that to ~4e-6 m / 2e-3 deg of object drift.
        atol = {"delta_pos": 1e-5, "delta_angle": 1e-2}
        for key in ("ho_pene", "self_pene", "contact_num", "contact_dist", "delta_pos", "delta_angle"):
            assert np.allclose(float(out[key]), float(ref[key]), atol=atol.get(key, 1e-9)), key
