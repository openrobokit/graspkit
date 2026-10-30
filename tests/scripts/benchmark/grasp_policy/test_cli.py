"""Command-line entry points of `scripts/benchmark/grasp_policy`: standalone runs, help and parsed parameters."""

import functools
import inspect
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import tyro

from graspkit.constants import DATA_ROOT, PROJECT_ROOT
from scripts.benchmark.grasp_policy import evaluate, evaluate_policy, package, report, train

SCRIPTS = PROJECT_ROOT / "scripts/benchmark/grasp_policy"


def parse(function: Callable, **values: object) -> dict:
    @functools.wraps(function)
    def capture(*args: object, **kwargs: object) -> dict:
        call = inspect.signature(function).bind(*args, **kwargs)
        call.apply_defaults()
        return call.arguments

    args = [item for key, value in values.items() for item in (f"--{key.replace('_', '-')}", str(value))]
    return tyro.cli(capture, args=args)


class TestCli:
    def test_standalone_imports(self, tmp_path: Path):
        code = """
import importlib, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'scripts/benchmark/grasp_policy'))
for name in ('train', 'evaluate', 'report', 'package', 'collect', 'rgb_inputs'):
    runpy.run_path(str(root / 'scripts/benchmark/grasp_policy' / (name + '.py')))
for name in ('graspkit.models.checkpoint', 'graspkit.isaac.job'):
    importlib.import_module(name)
paths = sys.path + [str(module.__dict__.get('__file__', '')) for module in sys.modules.values()]
assert not any('roboskills' in path.lower() for path in paths), paths
assert not any(name.startswith(('isaaclab', 'isaacsim', 'omni')) for name in sys.modules)
print('standalone imports: PASS')
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(PROJECT_ROOT)],
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES=""),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "standalone imports: PASS" in result.stdout

    @pytest.mark.parametrize(
        "entry, option",
        [
            ("train", "--bank"),
            ("evaluate", "--root"),
            ("report", "--root"),
            ("package", "--bundle"),
            ("collect", "--shards"),
        ],
    )
    def test_cli_help_outside_checkout(self, tmp_path: Path, entry: str, option: str):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / f"{entry}.py"), "--help"],
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH="", CUDA_VISIBLE_DEVICES=""),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert option in result.stdout

    @pytest.mark.parametrize("split", ["dev", "test"])
    @pytest.mark.parametrize("loop", ["open", "closed"])
    def test_production_cli_parameters(self, tmp_path: Path, split: str, loop: str):
        values = parse(
            evaluate.main,
            stage="run",
            root=tmp_path,
            scenes=tmp_path / "scenes.msgpack",
            checkpoint=tmp_path / "policy.pt",
            robot="yam",
            sensor="rgb",
            generator="flow_transformer",
            simulator="isaac",
            split=split,
            loop=loop,
            offset=100,
            count=8,
            num_envs=32,
        )
        assert values["root"] == tmp_path and values["checkpoint"] == tmp_path / "policy.pt"
        assert (values["split"], values["loop"], values["offset"], values["count"]) == (split, loop, 100, 8)
        assert (values["sensor"], values["generator"], values["simulator"]) == ("rgb", "flow_transformer", "isaac")
        assert values["isaac_command"] == ("/isaac-sim/python.sh",)
        assert values["num_envs"] == 32

    def test_native_evaluator_seed_parameters(self, tmp_path: Path):
        values = parse(
            evaluate_policy.main,
            output=tmp_path / "rows.jsonl",
            checkpoint=tmp_path / "policy.pt",
            seed=0,
            split="test",
            mode="policy_closed_loop",
            selector="learned",
            guidance_steps=8,
            num_samples=64,
        )
        assert values["seed"] == 0 and values["split"] == "test"
        assert values["mode"] == "policy_closed_loop" and values["selector"] == "learned"
        assert values["guidance_steps"] == 8 and values["num_samples"] == 64

    @pytest.mark.parametrize("sensor", ["depth", "rgb"])
    def test_explicit_training_recipe(self, tmp_path: Path, sensor: str):
        values = parse(
            train.main,
            root=tmp_path,
            bank=tmp_path / "bank",
            robot="agilex_aloha",
            generator="cvae",
            sensor=sensor,
            stage="train",
            steps=8000,
            scorer_steps=10080,
            objective="scaled",
            grad_clip_norm=1,
        )
        assert values["objective"] == "scaled" and values["grad_clip_norm"] == 1.0
        assert values["steps"] == 8000 and values["scorer_steps"] == 10080
        assert values["sensor"] == sensor

    def test_native_run_defaults(self):
        assert (
            parse(train.main, robot="arx_x5", generator="cvae", stage="smoke", bank="bank")["root"]
            == DATA_ROOT / "runs/grasp_policy"
        )
        assert parse(evaluate.main, stage="run", root="run")["num_envs"] == 10
        assert parse(report.main, root="run")["root"] == Path("run")

    def test_package_cli_paths(self, tmp_path: Path):
        paths = {
            key: tmp_path / value
            for key, value in {
                "policy_path": "policy.pt",
                "actor_path": "actor.pt",
                "bundle": "bundle",
                "validation": "validation.json",
            }.items()
        }
        values = parse(package.main, **paths)
        assert all(values[key] == path for key, path in paths.items())
