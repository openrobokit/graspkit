"""Released ARX X5 open-loop grasp policy checkpoints."""

from pathlib import Path

from robokit.assets import fetch

from graspkit.assets import REVISION
from graspkit.constants import GRASPKIT_HOME, HF_REPO_ID


def skill_policy_paths() -> tuple[Path, Path]:
    """The ARX X5 open-loop generator and ranker (`graspkit.policies.arx_x5_openloop`)."""
    rels = ["policies/skill/arx_x5/v3L_12k.pt", "policies/skill/arx_x5/ranker_div5.pt"]
    root = fetch(rels, REVISION, HF_REPO_ID, GRASPKIT_HOME)
    return root / rels[0], root / rels[1]
