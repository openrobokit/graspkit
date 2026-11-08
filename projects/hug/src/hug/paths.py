"""Every path the release uses.

Assets come from two HF datasets at the revisions graspkit pins (robokit: the robot description,
graspkit: the objects and models), downloaded once into ``ASSET_CACHE``. They resolve on attribute
access, never at import, so ``import hug`` stays offline.
"""

import logging
import os
from functools import cache
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS = REPO_ROOT / "results"
ASSET_CACHE = Path.home() / ".cache" / "hug"  # one real-file snapshot per dataset revision

# --------------------------------------------------------------------- HF homes: (repo, revision)
ROBOKIT = ("openrobokit/robokit", "461a28b5de95d89f139d2ca7125d787fdb70a745")
GRASPKIT = ("openrobokit/graspkit", "087390def8c4256a831744cfe579996b33f22f14")
_G1_INSPIRE = "robots/robot_description/humanoids/g1_inspire"  # not assembly/g1_inspire: a different robot


@cache
def _fetch(dataset: tuple[str, str], subdir: str) -> Path:
    """``subdir`` of ``dataset``, downloaded on the first call and read from ``ASSET_CACHE`` after."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # one line below instead of a bar per dataset
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)  # public datasets: no HF_TOKEN nag
    from huggingface_hub import snapshot_download

    repo, revision = dataset
    if not (ASSET_CACHE / repo / revision / subdir).exists():
        print(f"[assets] downloading {repo}/{subdir} into {ASSET_CACHE} (first use) ...")
    root = snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{subdir}/**"],
        local_dir=ASSET_CACHE / repo / revision,
    )
    return Path(root) / subdir


# Asset name -> how to fetch it. Resolved by __getattr__.
_LAZY: dict[str, Callable[[], Path]] = {
    # physics: 29 body DoF + both hands + the 3-DoF head (not a motor)
    "G1_INSPIRE_HEAD_URDF": lambda: _fetch(ROBOKIT, _G1_INSPIRE) / "urdf/g1_inspire_head.urdf",
    # FK / estimator: 32 joints (3 head between waist and arms), right hand only
    "G1_29DOF_HEAD_URDF": lambda: _fetch(ROBOKIT, _G1_INSPIRE) / "urdf/g1_29dof_head.urdf",
    # upper body + head, for the robokit IK
    "G1_INSPIRE_UPPER_HEAD_URDF": lambda: _fetch(ROBOKIT, _G1_INSPIRE) / "urdf/g1_inspire_upper_head.urdf",
    # right Inspire hand alone, for the ghost grasp overlays
    "R_HAND_INSPIRE_URDF": lambda: _fetch(ROBOKIT, _G1_INSPIRE) / "urdf/r_hand_inspire.urdf",
    # grasp targets, one directory per object
    "OBJECTS_DIR": lambda: _fetch(GRASPKIT, "data/object_data/hug_release"),
    # the CVAE grasp policy, sim and real
    "DEFAULT_GRASP_POLICY": lambda: _fetch(GRASPKIT, "policies/hug") / "hug-grasp.onnx",
    # GR00T's lower-body policies, beside it
    "GROOT_BALANCE_ONNX": lambda: _fetch(GRASPKIT, "policies/hug") / "GR00T-WholeBodyControl-Balance.onnx",
    "GROOT_WALK_ONNX": lambda: _fetch(GRASPKIT, "policies/hug") / "GR00T-WholeBodyControl-Walk.onnx",
}


def __getattr__(name: str) -> Path:
    """Resolve an asset, downloading it on first use."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    resolved = _LAZY[name]()
    globals()[name] = resolved  # every later lookup skips this function
    return resolved


# --------------------------------------------------------------------- repo data
TORSO_T_CAM = REPO_ROOT / "hardware" / "T_torso_camera_l515_60deg.json"  # 60° mount's extrinsic; grasp default

# --------------------------------------------------------------------- outputs
CAPTURED_DIR = RESULTS / "captured"  # one scene_<ts>.npz per segmentation (real/perception.py)
CALIBRATION_DIR = RESULTS / "calibration"  # one directory per robot id (calibration/g1_calibration.py)
