<p align="center">
  <a href="https://github.com/openrobokit/graspkit"><img src="https://raw.githubusercontent.com/openrobokit/graspkit/main/docs/assets/graspkit-banner.webp" alt="GraspKit" width="100%"></a>
</p>

<p align="center">
  <a href="https://github.com/openrobokit/graspkit/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="https://github.com/openrobokit/graspkit/actions/workflows/build.yaml"><img src="https://img.shields.io/badge/CI-GitHub_Actions-blue" alt="GitHub Actions"></a>
  <a href="https://github.com/openrobokit/graspkit/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
</p>

GraspKit is an end-to-end toolkit for manipulation across various robot embodiments, including grippers
and dexterous hands.
It synthesizes grasps, plans trajectories, trains policies, and deploys them
in simulation or on real robots; it is built on top of [RoboKit](https://github.com/openrobokit/robokit).

## Quick links

- [Docs](https://github.com/openrobokit/graspkit/blob/main/docs/guide/overview.md)
- [Quickstart](https://github.com/openrobokit/graspkit/blob/main/docs/guide/quickstart.md)
- [Installation](https://github.com/openrobokit/graspkit/blob/main/docs/guide/installation.md)
- [Development](https://github.com/openrobokit/graspkit/blob/main/docs/dev.md)

## Installation

Requires Python 3.10–3.14 (the full pipeline with ManiSkill and spconv: 3.10–3.12):

```shell
uv add graspkit      # or: pip install graspkit
```

See [Installation](https://github.com/openrobokit/graspkit/blob/main/docs/guide/installation.md) for source installs, PyTorch options, and asset access.

## Features

<table width="100%">
  <tr>
    <td width="50%">
      <a href="https://github.com/openrobokit/graspkit/blob/main/docs/guide/grasp_optim.md"><img src="https://raw.githubusercontent.com/openrobokit/graspkit/main/docs/assets/features/grasp-optimization.webp" alt="1,024 dexterous grasps optimized in one GPU batch" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/graspkit/blob/main/docs/guide/grasp_optim.md">Fast grasp optimization</a></strong>
    </td>
    <td width="50%">
      <a href="https://github.com/openrobokit/graspkit/blob/main/docs/guide/training.md"><img src="https://raw.githubusercontent.com/openrobokit/graspkit/main/docs/assets/features/policy-learning.webp" alt="Object point cloud mapped by a grasp generator to a hand grasp" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/graspkit/blob/main/docs/guide/training.md">Policy learning</a></strong>
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/openrobokit/graspkit/blob/main/docs/reference/robot_hands.md"><img src="https://raw.githubusercontent.com/openrobokit/graspkit/main/docs/assets/features/embodiments.webp" alt="Different arms with dexterous hands grasping the same object in simulation" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/graspkit/blob/main/docs/reference/robot_hands.md">All arms, all hands</a></strong>
    </td>
    <td>
      <a href="https://github.com/openrobokit/graspkit/blob/main/docs/guide/real_deployment.md"><img src="https://raw.githubusercontent.com/openrobokit/graspkit/main/docs/assets/features/sim-to-real.webp" alt="Two real xArm robots with dexterous hands, diagonally split with their aligned simulation twin" width="100%"></a><br>
      <strong><a href="https://github.com/openrobokit/graspkit/blob/main/docs/guide/real_deployment.md">Real-world robustness</a></strong>
    </td>
  </tr>
</table>

Explore robot presets and optimized grasps in the [interactive gallery](https://github.com/openrobokit/graspkit/blob/main/docs/guide/quickstart.md#browse-grasps-in-the-gallery).

## How it works

For each scene, GraspKit optimizes and plans grasps on the GPU and renders the camera's point cloud.
A policy trained on these pairs maps point clouds to grasp trajectories, in simulation or on real robots.

| Stage | Role | Read more |
| --- | --- | --- |
| Grasp optimization | Generate grasps, plan motions, and optionally validate them in simulation. | [Grasp optimization](https://github.com/openrobokit/graspkit/blob/main/docs/guide/grasp_optim.md) |
| Rendering | Render point-cloud observations for the same scenes. | [Pipeline](https://github.com/openrobokit/graspkit/blob/main/docs/guide/pipeline.md) |
| Data processing | Join observations and grasps into policy training data. | [Pipeline](https://github.com/openrobokit/graspkit/blob/main/docs/guide/pipeline.md) |
| Training | Train a grasp policy on the processed data. | [Training](https://github.com/openrobokit/graspkit/blob/main/docs/guide/training.md) |
| Deployment | Run a trained policy in simulation or on a real robot. | [Sim](https://github.com/openrobokit/graspkit/blob/main/docs/guide/sim_deployment.md), [real](https://github.com/openrobokit/graspkit/blob/main/docs/guide/real_deployment.md) |

See the [core overview](https://github.com/openrobokit/graspkit/blob/main/docs/concepts/overview.md) for the full stage graph and artifacts.
