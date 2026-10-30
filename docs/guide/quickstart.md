# Quickstart

## Try it without installing

Run this to start the interactive grasp gallery:

```shell
uvx --from graspkit python -m graspkit.gallery.grasp_optim
```

Pick a hand, arm, and object. Press *Optimize* to generate grasps, then browse them with the
*Grasp* slider. The gallery caches results for the next visit.

## Optimize one grasp

After [installing](installation.md), run the smallest grasp optimization example:

```shell
uv run src/graspkit/examples/grasp_optim/01_basic.py
```

It optimizes xHand grasps for one YCB object and opens a [Viser](https://viser.studio) page to
browse the results. Grasp optimization needs a CUDA GPU.

## More examples

The scripts below are runnable. Paths are relative to the repository root.

| Directory | Scripts |
| --- | --- |
| [`src/graspkit/examples/grasp_optim`](../../src/graspkit/examples/grasp_optim/) | [`01_basic.py`](../../src/graspkit/examples/grasp_optim/01_basic.py), [`02_table.py`](../../src/graspkit/examples/grasp_optim/02_table.py), [`03_arm.py`](../../src/graspkit/examples/grasp_optim/03_arm.py), [`04_sim_valid.py`](../../src/graspkit/examples/grasp_optim/04_sim_valid.py), [`05_table_sim_valid.py`](../../src/graspkit/examples/grasp_optim/05_table_sim_valid.py), [`06_arm_sim_valid.py`](../../src/graspkit/examples/grasp_optim/06_arm_sim_valid.py), [`07_mj_sim_valid.py`](../../src/graspkit/examples/grasp_optim/07_mj_sim_valid.py) |
| [`scripts/launch.py`](../../scripts/launch.py) | Run an experiment from grasp optimization through policy training. See [The full pipeline](pipeline.md). |

Example usage:

```shell
uv run src/graspkit/examples/grasp_optim/02_table.py
```
