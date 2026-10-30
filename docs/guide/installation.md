# Installation

GraspKit needs Python 3.10–3.14; the `maniskill` extra needs 3.10–3.12 and `spconv` 3.10–3.13.
Grasp optimization, rendering, and training need a CUDA GPU.

=== "uv"
```shell
uv add graspkit
```
=== "pip"
```shell
pip install graspkit
```

## From source

=== "uv"
```shell
uv add git+https://github.com/openrobokit/graspkit.git
```
=== "pip"
```shell
pip install git+https://github.com/openrobokit/graspkit.git
```

To work on GraspKit itself, follow the [Development Guide](../dev.md#setup).

## PyTorch builds

`uv sync` installs the `dev` and `pt-cu126` groups by default. The PyTorch groups conflict with
each other, so select another one explicitly:

```shell
uv sync --no-default-groups --group pt-cu128
```

| Group | What it adds |
| --- | --- |
| `pt-cu126` | PyTorch + torchvision for CUDA 12.6 (default). |
| `pt-cu118`, `pt-cu128` | The same for CUDA 11.8 / 12.8. |
| `pt-cpu` | CPU-only PyTorch. |
| `dev` | ipython, pytest, `robot_descriptions` (default). |

## Extras

| Extra | What it adds |
| --- | --- |
| `maniskill` | ManiSkill, for simulation, rendering and grasp evaluation (Python 3.10–3.12). |
| `spconv` | spconv, for the default `spconv` vision encoder (Python 3.10–3.13). |
| `torchvision` | torchvision, for the `rgb_resnet` vision encoders. |
| `sam2` | SAM 2, for the segmentation step of [deployment](real_deployment.md). |
| `xarm` | `retargetingkit[realsense,xarm,ability]` + open3d, for the xArm rigs (Python 3.10). |
| `vega` | `retargetingkit[xarm,ability,dexmate]` + pyzmq + open3d, for the Vega humanoid. |
| `data-process` | coacd, fast-simplification, for asset processing. |

The `sam2` extra uses a wheel from a separate index:

=== "uv"
```shell
uv add "graspkit[sam2]" --index https://jianglongye.github.io/torch_packages_builder
```
=== "pip"
```shell
pip install "graspkit[sam2]" --extra-index-url https://jianglongye.github.io/torch_packages_builder
```

```shell
uv run --extra sam2 --extra vega scripts/deploy/deploy.py --help
```

## Assets and credentials

`import graspkit` downloads nothing. Assets are fetched when the module that needs them is
imported: graspkit's scenes, objects and calibration files come from the HuggingFace dataset
`openrobokit/graspkit` into `~/.graspkit/assets/<revision>`, and robot URDFs, meshes and
collision spheres come from `openrobokit/robokit` into `~/.robokit/assets/<rev>`.

Data and logs live in `~/.graspkit/data` and `~/.graspkit/logs`, created on import. `graspkit.link()`
symlinks both into the repo for browsing.

- A `.env` file with `GRASPKIT_S3_ROOT`, `FSSPEC_S3_ENDPOINT_URL`, `FSSPEC_S3_KEY` and `FSSPEC_S3_SECRET` is needed
  only for the `_s3` experiments; it is loaded on import.
- `GRASPKIT_EXCLUDED_GPUS` lists GPU ids that `scripts/launch.py` must never use, such as `4,5`.
- `graspkit.assets.REVISION` pins the asset revision; delete `~/.graspkit/assets/<revision>/.done/` to re-download.

## Verify

```shell
CUDA_VISIBLE_DEVICES=0 uv run pytest tests/
uv run src/graspkit/examples/grasp_optim/01_basic.py
```

The example opens a Viser page with optimized xHand grasps. Next: the [Quickstart](quickstart.md).
