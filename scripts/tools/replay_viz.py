"""Replay recorded rollouts (`.viz.h5`) in viser with rollout/frame sliders; viser+numpy only, loads fast.

uv run scripts/tools/replay_viz.py --path <out>.viz.h5
uv run scripts/tools/replay_viz.py --path <out>.viz.h5 --smoke
"""

import graspkit  # noqa: F401 I001  # must be first: sets numpy thread limits + loads .env

import logging
import time

import tyro

from graspkit.datasets.replay_data import ReplayData
from graspkit.utils.viser_utils import ReplayViewer

logger = logging.getLogger("graspkit")


def main(path: str, port: int = 8080, smoke: bool = False):
    replays = ReplayData.load(path)
    assert replays, f"no rollouts in {path}"

    viewer = ReplayViewer(port=0 if smoke else port)
    viewer.set_replays(replays)
    if smoke:
        stages = sorted({r.stage for r in replays})
        labels = sorted({r.label for r in replays})
        print(f"rollouts={len(replays)} stages={stages} labels={labels} shown={len(viewer.shown)}")
        return
    logger.info(f"viser on :{port} — {len(replays)} rollouts. Open the printed URL; Ctrl-C to stop.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main, use_underscores=True)
