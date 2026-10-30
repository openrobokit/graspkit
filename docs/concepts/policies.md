# Models and Policies

`GraspPolicy` in `src/graspkit/models/grasp_policy.py` maps an observation to a grasp:
three hand poses (pre-grasp, grasp, post-grasp)
that the trajectory builder turns into an executable motion.

Open and closed loop use the same policy. The execution controller decides when to observe and
predict again; the benchmark's closed-loop mode refreshes the observation and replans at step 75.

The policy selects `CVAE` from `models/cvae.py` or `FlowMatching` from
`models/flow_matching.py` using `generative_model` (`cvae`, `flow`, `flow_transformer`).
Both generators expose `forward`, `decode` and `sample`; access the selected component as
`policy.generator`. No shared base class is needed.

Saved configuration field names and the checkpoint's `cvae.*` parameter keys are unchanged.
The policy registers its generator under that historical checkpoint name, while callers
use `generator`. Existing state dictionaries load strictly without migration.
Imports use `GraspPolicy`, `GraspPolicyConfig` and `build_grasp_policy`; the training CLI
subcommand is `agent:grasp_policy_config`.

## The CVAE

```
point cloud (B, N, 3)
   -> [yaw canonicalization / augmentation] -> [centering] -> vision encoder (SpUNet or PointNet)
        |                                            |
        | global feature                             | per-point features
        |                                            +-> objectness head  (target vs background)
        |                                            +-> graspness head   (where to grasp)
        v
   CVAE encoder (training: ground-truth actions) -> z -> decoder -> (B, n, 3, d) actions
```

`d` is the hand's joint count for a fixed base; a floating base prepends a translation and a
rotation, `[xyz, rotation, hand joints]`, where the rotation is either Euler `rxyz` or a 6D
representation. The three queries are pre-grasp, grasp and post-grasp, in that order.

At inference the encoder is skipped: the graspness head picks the top-`k` points, and the CVAE
decodes a latent per point — the mean latent when `use_mean`, otherwise a sample scaled by
`cvae_temperature`. With `use_mean` the diversity comes from the distinct query points, not the
latent. `grasp_evaluator_guidance_steps` instead refines the latent by ascending a frozen grasp evaluator
inside a ball of radius `grasp_evaluator_guidance_radius`.

The vision encoders include a sparse-convolution U-Net (`vision_encoder="spconv"`, the default,
voxel size 5 mm), a BN-free PointNet and RGB CNN/ResNet variants. RGB construction and shared
training/inference preprocessing live in `models/vision_encoder/rgb.py`; the policy keeps
state/camera conditioning and generator selection. The CVAE is an MLP encoder/decoder over the flattened action
vector, conditioned on the vision feature, optionally with a learned prior.

Flow matching uses the same visual conditioning and action representation. Its velocity
network is an MLP (`flow`) or transformer (`flow_transformer`), trained along a linear
noise-to-action path. Sampling integrates eight Euler steps and restores physical joint
units using the bank's action mean/std. Flow returns zero latent moments through the shared
training interface, so its KL contribution is zero; joint reconstruction uses MSE.

## Losses

| Loss | Weight field | Penalizes |
| --- | --- | --- |
| `qpos` | `qpos_weight` | L1 on the reconstructed hand joints |
| `tl`, `rot` | `tl_weight`, `rot_weight` | L1 on floating-base translation and rotation |
| `kl` | `kl_weight` | Posterior against the standard or learned prior |
| `objectness` | `objectness_weight` | Per-point target-vs-background BCE, positives upweighted |
| `graspness` | `graspness_weight` | KL between predicted and labelled graspness distributions |
| `kinematic` | `kinematic_weight` | MSE between the FK link positions of predicted and ground-truth actions |
| `penetration` | `penetration_weight` | Collision-sphere overlap into the object beyond `penetration_threshold`, on prior samples |
| `prior_fingertip` | `prior_fingertip_weight` | Distance of fingertip spheres to the object surface |

The last two need scene geometry at training time; the trainer builds it only every
`penetration_every` steps and after `penetration_warmup_iters`.

## Representation flags

- `convert_rxyz_to_rot6d` — 6D rotation inside the CVAE, decoded back to Euler. `analytic_euler`
  picks the closed-form decode over the iterative one.
- `coupled_floating_base` — all three queries share the grasp rotation, and grasp and post-grasp
  share its translation.
- `center_pointcloud` — predict translations relative to the cloud centroid.
- `canonical_yaw` — rotate into a deterministic yaw frame derived from the cloud's second moment.
- `train_yaw_augmentation` — random training-time yaw applied to points and targets together.

## Data

`GraspDataset` reads the processed HDF5: per scene, one rendered view is drawn uniformly per access,
points are sampled (object-only or whole scene), and actions are sampled per grasp. With graspness
enabled, the per-action best points are placed first and the rest sampled around them. Graspness
labels come from the contact extractor's palm-cone weighting during `process_sim_data`. Fresh flow models use `meta/action_mean` and `meta/action_std` to normalize their action space.
CVAE training uses raw poses.

## Training

AdamW at `lr=1e-4`, weight decay `1e-4`, a 10x step decay two thirds of the way through, DDP when
distributed. Every step scans the gradients and skips the update if any rank saw a NaN or Inf.
Checkpoints hold `{"agent": state_dict}`; the provenance keys the benchmark expects (`experiment`,
`policy_config`, `training`) are added by the packaging scripts. Evaluation is described in
[Sim deployment](../guide/sim_deployment.md).

## From prediction to motion

A predicted grasp is not executable on its own. `GraspTrajBuilder` applies the filters (hand-axis
cone, contact-avoid, penetration, rest pose), optionally IK-converts a floating grasp onto an arm
with an orientation gate and an arm-range band, and plans the approach, closing, holding and lift
segments. The benchmark runs the filters; deployment skips them and relies on the IK gate, since a
real capture has no ground-truth object pose.
