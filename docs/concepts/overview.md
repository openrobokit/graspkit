# Core Overview

How the stages fit together, what flows between them, and where robokit ends and GraspKit begins.

## The graph

```
scene descriptions (msgpack)            scripts/stages/scene_desc/*
        |
        +---------------------------+
        |                           |
   [render_data]              [optimize_grasps]              (sharded: scenes[rank::world_size])
   Renderer / ManiSkill        WarpGraspOptimizer
   kinematic replay,           -> GraspTrajBuilder: filters -> (IK convert) -> motion plan
   point-cloud obs             -> optional in-loop sim eval
        |                           |
   rendered_data_path          grasp_output_dir
   (h5 per scene)              (h5 per scene: pre / grasp / post qpos)
        +------------+--------------+
                     |
            [process_sim_data]  -> processed_data_path (obs + actions per scene)
                     |
                  [train]  GraspDataset -> Trainer (DDP through torchrun)
                     |
          logs/train/<name>/checkpoints/*.pt
```

`scripts/launch.py` dispatches these by name; the order is whatever `ExperimentConfig.steps` lists,
by default `render_data`, `optimize_grasps`, `process_sim_data`, `train`. A fifth step,
`sample_point_cloud`, exists for object-only clouds but no shipped experiment uses it.

## Stages

| Stage | In | Out | Config |
| --- | --- | --- | --- |
| `optimize_grasps` | scene descriptions | grasp h5 (pre/grasp/post qpos) | `GraspOptimConfig`, `GraspTrajBuilderConfig` |
| `render_data` | scene descriptions | rendered h5 (point clouds, segmentation, camera poses) | `RendererConfig` |
| `process_sim_data` | rendered h5 + grasp h5 | processed h5 (obs joined to actions) | `ExperimentConfig` |
| `train` | processed h5 | checkpoints | `TrainerConfig`, `DatasetConfig`, `GraspPolicyConfig` |

Grasp optimization is the compound stage: it optimizes, then filters (hand-axis cone,
contact-avoid, penetration, rest pose), optionally IK-converts floating grasps onto an arm, plans a
trajectory, and keeps only what a physics evaluator accepts.

## Resuming and sharding

Each stage writes a completion marker (`.optim_complete`, `.render_complete`,
`.process_sim_data_complete`, `.train_complete`) and skips itself when the marker exists. With
multiple ranks the marker is written per rank and promoted once every rank is done; `s3://` outputs
get no marker. `--restart-from <step>` deletes the marker and outputs for that step and everything
downstream, `--append` keeps the output and deletes only the marker. Grasp optimization additionally
resumes per scene: it re-runs scenes whose output is smaller than `min_grasps_threshold`.

Only grasp optimization uses every GPU, one process per card with scenes striped across ranks;
non-zero ranks sleep 20 s so rank 0 compiles the Warp kernels first. Rendering and processing pin to
the first GPU, and training goes through `torchrun`.

## Artifacts

`GraspKitData` is the tensor payload that crosses stages, in a padded `(B, K, ...)` or packed
`(N, ...)` layout, saved as HDF5. Scene descriptions are msgpack with a train/test split.
The processed file holds `data/<scene>/obs`, `data/<scene>/actions` and `meta/action_{mean,std}`;
the trainer consumes it through `GraspDataset`.

## Floating base versus fixed base

`RobotConfig.base_type` decides the action layout: a floating hand prepends six base DOFs, baked as
`[xyz, euler_rxyz, hand joints]`. A floating hand is always optimized with the floating-base
sampler, and the Levenberg-Marquardt backend is floating-only. Setting
`GraspTrajBuilderConfig.convert_to_fixed` filters those grasps and IK-converts them onto an arm
before planning; inside grasp optimization the conversion is always off, so the planner plans for
the robot being optimized.

## robokit versus GraspKit

robokit supplies the cost terms, optimizers, contact sampling, warp scenes, SDF baking, the robot
model and the motion planner. GraspKit owns the energy assembly, the init samplers, the grasp
filters, the scene-description adapter, the renderer, the datasets, the policy and the trainer.
Code graduates to robokit only when a second consumer exists.
