"""Run `train_grasp_evaluator.py` on the replay's own scenes (not in the default file); pass `--scenes PATH` first.

Usage:
    uv run scripts/benchmark/grasp_policy/train_critic_scenes.py --scenes scenes.msgpack --experiment ... --replay ...
"""

import graspkit  # noqa: F401 I001

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import tyro

import graspkit.configs.experiment as experiments

assert sys.argv[1] == "--scenes"
scenes = sys.argv[2]
del sys.argv[1:3]
spec = importlib.util.spec_from_file_location(
    "train_grasp_evaluator", Path(__file__).parents[2] / "stages/train/train_grasp_evaluator.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = experiments.build_experiment_config
module.build_experiment_config = lambda name: replace(original(name), scene_description_path=scenes)
tyro.cli(module.main)
