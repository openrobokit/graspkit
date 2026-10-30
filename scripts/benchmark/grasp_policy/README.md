# Policy Benchmark

Tabletop grasping with ARX X5A, Panda + Allegro, AgileX ALOHA, YAM, and xArm7 + Sharpa.

## Goal

Use one architecture and training recipe across all five embodiments, with separate RGB and depth models and benchmark cameras.
Final500 is authorized for the frozen shared recipe despite Dev misses. Merge remains gated on Final performance.

## Grasp policy v0.1 (2026-10-15)

All ten policies were retrained from fresh data with one shared recipe and evaluated on the 500 v0.5 TEST scenes in
ManiSkill, Isaac and MuJoCo, open and closed loop (60 conditions, 30,000 episodes). Success is the same in every
simulator: the object rises more than 10 cm and at least two fingertip/palm links press it with at least 0.5 N.

- Data: 4,742 v0.5 TRAIN scenes (20 dev objects and every test object held out). GT grasps from the grasp optimizer,
  then `collect.py` labels them in all three simulators from the initial state and from the displaced
  step-75 state (depth points and RGB views pair the initial and current frames).
- Rounds: GT (ManiSkill all scenes, MuJoCo and Isaac subsets); 64 proposals of an 8k-step depth actor (ManiSkill);
  64 proposals of the v2 depth and RGB actors (ManiSkill, Isaac and MuJoCo subsets).
- Training: flow transformer, 16,000 updates, batch 128, lr 3e-4, gradient clipping 1, frozen ImageNet batch-norm
  statistics for RGB; evaluator 10,080 updates with ranking weight 1; 64 proposals and eight guidance steps.
- ARX and ALOHA were retrained on 2026-10-16 with the same steps after the jaw-axis (#275) and arm-limit (#278)
  fixes (`runs/retrain_20261016`; banks of 24,370 and 24,170 states).

Cells give v0.1 success (%) with the bar in parentheses: the historical best for ManiSkill, and for Isaac the historical
policy re-scored in the current Isaac harness (the historical Isaac numbers came from the earlier CPU-PhysX
evaluator). MuJoCo has no earlier numbers. Bold cells miss their bar.

| Robot | Input | ManiSkill open | ManiSkill closed | Isaac open | Isaac closed | MuJoCo open | MuJoCo closed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARX X5A | Depth | 86.8 (77.6) | 89.4 (79.8) | 83.2 (70.2) | 80.4 (67.8) | 85.0 | 87.4 |
| ARX X5A | RGB | 81.8 (67.4) | 82.0 (65.8) | 75.0 (53.6) | 74.6 (46.8) | 76.6 | 78.8 |
| AgileX ALOHA | Depth | 87.4 (71.4) | 88.6 (76.8) | 80.6 (65.8) | 78.8 (61.8) | 88.6 | 88.4 |
| AgileX ALOHA | RGB | 80.8 (64.0) | 79.4 (59.4) | 67.8 (35.0) | 69.2 (32.0) | 78.6 | 78.8 |
| Panda + Allegro | Depth | 77.2 (73.8) | 74.6 (73.6) | 65.8 (62.6) | 67.8 (64.2) | 66.2 | 70.6 |
| Panda + Allegro | RGB | 72.6 (65.2) | 75.8 (64.8) | 61.4 (46.0) | 63.8 (42.2) | 62.0 | 62.8 |
| YAM | Depth | 92.6 (82.2) | 93.8 (83.8) | 82.6 (82.0) | 84.4 (79.8) | 89.6 | 92.4 |
| YAM | RGB | 86.4 (63.0) | 84.6 (65.4) | 73.2 (40.6) | 76.2 (36.0) | 78.0 | 79.4 |
| xArm7 + Sharpa | Depth | 82.8 (75.4) | 81.8 (75.6) | **59.8 (66.2)** | **56.8 (65.4)** | 70.4 | 77.6 |
| xArm7 + Sharpa | RGB | 73.8 (57.0) | 76.4 (62.4) | 51.8 (42.4) | 52.2 (42.8) | 61.2 | 64.6 |

Means over 20 conditions: ManiSkill 82.4%, Isaac 70.3%, MuJoCo 76.8%.

xArm7 + Sharpa depth misses in Isaac: most failures lift the object without two loaded fingertip/palm links.
Historical Isaac bests (CPU PhysX), for reference: ARX 71.2/61.8 depth, 52.4/42.2 RGB; ALOHA 66.6/63.2, 34.2/31.0;
Panda 70.2/62.6, 43.4/43.2; YAM 86.6/87.0, 40.6/35.8; Sharpa 81.2/81.6, 48.2/48.0 (open/closed).

## Shared recipe

Use this recipe for every embodiment; train RGB and depth models separately.

- Data: 4,096 TRAIN scenes with paired initial/current observations.
- Actor: existing flow transformer; train on grasps that succeed in native ManiSkill.
- Actor training: 8,001 initial updates, then 8,001 updates on successful grasps; preserve initial action normalization.
- Actor settings: batch 128, 64 actions, learning rate 0.0003, seed 0.
- Evaluator: existing action-conditioned MLP, trained fresh for 10,080 updates at learning rate 0.001 and ranking weight 1.
- An evaluator label is the simulator outcome of a proposed grasp: success = 1, failure = 0. Use both outcomes.
- Reencode evaluator observations with the trained actor; use the same 32 held training objects for evaluator selection.
- Depth encoder: PointNet. RGB encoder: ImageNet ResNet18 with ordered camera views. Both use the same flow transformer and evaluator.
- Evaluation: 64 proposals, eight refinement steps, one execution attempt. Freeze one checkpoint per robot/input for both loop modes.
- No PPO or retries.

Measured checkpoint sources: depth `8aaa6c67`, RGB `067baa7b`.

## Shared evaluation

Native ManiSkill; one approach/grasp/lift attempt, 64 proposals, and 300 control steps.
At step 75, displace the object by up to 15 mm in XY and 10° yaw.
Open loop uses the initial observation; closed loop adds one observation at step 75.
Training and evaluation pair initial/current snapshots; open loop duplicates the initial snapshot.
Depth uses target points in world coordinates; RGB uses full camera images.
Success requires >10 cm object rise and >0.5 N contact on at least two configured grasp links.
Dev and Final objects are disjoint from training objects. No candidate simulations during evaluation.
The preserved baseline uses three fixed cameras and different models/budgets.
Existing Final scenes have been evaluated; future comparisons on those scenes are exploratory.

## Dev

Same 100 scenes. Open loop uses the initial observation; closed loop observes again at step 75.
Both modes use the same checkpoint. Columns show success percentages.
The table uses the shared 4k recipe for all 20 conditions.
Mean success is **76.90%**, versus **87.25% baseline**; **18 conditions miss baseline**.
The previous 78.10% summary selected different budgets and evaluator variants per model.
All checkpoints and historical measurements remain preserved in [results.csv](results.csv).

| Robot | Input | Baseline open | Baseline closed | Recipe open | Recipe closed |
| --- | --- | ---: | ---: | ---: | ---: |
| ARX X5A | Depth | 89% | 96% | 83% | 92% |
| ARX X5A | RGB | 92% | 94% | 78% | 79% |
| Panda + Allegro | Depth | 86% | 84% | 65% | 73% |
| Panda + Allegro | RGB | 82% | 85% | 57% | 70% |
| AgileX ALOHA | Depth | 94% | 96% | 76% | 84% |
| AgileX ALOHA | RGB | 91% | 91% | 74% | 76% |
| YAM | Depth | 93% | 96% | 95% | 95% |
| YAM | RGB | 87% | 95% | 77% | 79% |
| xArm7 + Sharpa | Depth | 77% | 87% | 76% | 76% |
| xArm7 + Sharpa | RGB | 66% | 64% | 63% | 70% |

## Final

Final500 is complete for the frozen shared 4k recipe: ten checkpoints, twenty conditions, **7,356/10,000 successes (73.56%)**, versus **82.06% baseline**; **16 conditions miss baseline**. Each condition contains 500 one-attempt trials with 64 proposals and guidance 8. Actors remain at 16,002 updates and evaluators at 10,080; the original depth/RGB Dev sources and checkpoint are unchanged for both loops. Existing Final scenes had been evaluated previously, so this comparison is exploratory. Measurements and prior history remain in [results.csv](results.csv). Work is stopped; no training or merge was performed for this milestone.

| Robot | Input | Baseline open | Baseline closed | Recipe open | Recipe closed |
| --- | --- | ---: | ---: | ---: | ---: |
| ARX X5A | Depth | 86.0% | 89.0% | 83.2% | 87.0% |
| ARX X5A | RGB | 84.0% | 85.6% | 72.8% | 72.8% |
| Panda + Allegro | Depth | 80.0% | 82.6% | 63.2% | 68.8% |
| Panda + Allegro | RGB | 79.6% | 78.4% | 57.0% | 61.4% |
| AgileX ALOHA | Depth | 89.4% | 92.0% | 78.8% | 78.8% |
| AgileX ALOHA | RGB | 82.8% | 85.8% | 68.8% | 70.4% |
| YAM | Depth | 88.8% | 92.8% | 88.0% | 91.6% |
| YAM | RGB | 83.0% | 90.2% | 72.8% | 72.4% |
| xArm7 + Sharpa | Depth | 73.8% | 80.0% | 78.2% | 81.4% |
| xArm7 + Sharpa | RGB | 56.0% | 61.4% | 58.4% | 65.4% |

Failed attempts: 0 observation, 0 planning, 2467 lift and 177 strict-grasp-only. Mean trial cost: 8.969 s (8.761 s warm; first two ordered trials excluded per condition). Paired baseline comparison: 936 gained / 1786 lost.

## v0.5 Test

The same ten Final checkpoints and evaluation settings on the 500 v0.5 `tabletop_mini` TEST scenes (300 dexgraspnet test-split + 200 dexdiffuser, a held-out object family): **6,964/10,000 successes (69.64%)**, versus 73.56% on Final500. dexgraspnet 72.15%, dexdiffuser 65.88%. No objects overlap with training. No baseline was run on these scenes.

| Robot | Input | Final open | Final closed | v0.5 open | v0.5 closed |
| --- | --- | ---: | ---: | ---: | ---: |
| ARX X5A | Depth | 83.2% | 87.0% | 77.6% | 79.8% |
| ARX X5A | RGB | 72.8% | 72.8% | 67.4% | 65.8% |
| Panda + Allegro | Depth | 63.2% | 68.8% | 73.8% | 73.6% |
| Panda + Allegro | RGB | 57.0% | 61.4% | 65.2% | 64.8% |
| AgileX ALOHA | Depth | 78.8% | 78.8% | 71.4% | 76.8% |
| AgileX ALOHA | RGB | 68.8% | 70.4% | 64.0% | 59.4% |
| YAM | Depth | 88.0% | 91.6% | 82.2% | 83.4% |
| YAM | RGB | 72.8% | 72.4% | 59.0% | 60.4% |
| xArm7 + Sharpa | Depth | 78.2% | 81.4% | 75.4% | 75.6% |
| xArm7 + Sharpa | RGB | 58.4% | 65.4% | 54.8% | 62.4% |

Failed attempts: 2881 lift and 155 strict-grasp-only. Mean trial cost: 8.0 s.

## Cameras

ARX and ALOHA use one arm, its wrist camera, and the official head camera.
Panda uses DexJoCo's front and wrist cameras; these are single-arm adaptations of benchmark layouts.
YAM uses two cameras on opposite sides of its wrist. Sharpa uses one wrist camera and one external camera.
Wrist camera poses follow actual joints at each observation; arm and external-camera placement preserve the complete rig frame.
Capture uses 320×240 images; the shared RGB encoder uses 128×128 images.

Sources: [RoboDojo robot](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/blob/main/Assets/Robots/x5/robot_config.yml),
[RoboDojo cameras](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/env_cfg/camera/camera_config.yml),
[RoboTwin cameras](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/env_cfg/task_config/_camera_config.yml),
[DexJoCo task](https://github.com/brave-eai/dexjoco/blob/main/configs/multi_task/pick_bucket.yaml).
