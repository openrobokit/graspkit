from dataclasses import replace

from graspkit.configs.grasp_policy_config import GraspPolicyConfig

# Fixed-base presets
pointnet_basic = GraspPolicyConfig(
    vision_encoder="pointnet",
    kl_weight=0.01,
    use_mean=False,
)

pointnet_0_1_kl = GraspPolicyConfig(
    vision_encoder="pointnet",
    kl_weight=0.1,
    use_mean=False,
)
pointnet_0_1_kl_6drot = replace(pointnet_0_1_kl, convert_rxyz_to_rot6d=True, tl_weight=5.0)
pointnet_0_1_kl_kin_loss = replace(pointnet_0_1_kl, enable_kinematic_loss=True)

spconv_local_kin_loss = GraspPolicyConfig(
    vision_encoder="spconv",
    vision_encoder_global_feature=False,
    kl_weight=0.1,
    graspness_weight=0.1,
    enable_objectness_prediction=False,
    enable_kinematic_loss=True,
    use_mean=False,
)
spconv_layernorm_local_kin_loss = replace(spconv_local_kin_loss, vision_encoder_normalization="layernorm")

# Penetration-loss preset (pointnet baseline + soft penetration regularizer)
pointnet_0_1_kl_kin_loss_pen = replace(
    pointnet_0_1_kl_kin_loss,
    enable_penetration_loss=True,
    penetration_weight=4.0,
    penetration_threshold=0.005,
    penetration_every=8,
    penetration_warmup_iters=90000,
)

# Floating-base presets (add rot6d conversion and translation loss weight)
pointnet_0_1_kl_kin_loss_6drot = replace(pointnet_0_1_kl_kin_loss, convert_rxyz_to_rot6d=True, tl_weight=5.0)
spconv_local_kin_loss_6drot = replace(spconv_local_kin_loss, convert_rxyz_to_rot6d=True, tl_weight=5.0)
spconv_layernorm_local_kin_loss_6drot = replace(
    spconv_layernorm_local_kin_loss, convert_rxyz_to_rot6d=True, tl_weight=5.0
)

GRASP_POLICY_CONFIGS = {
    "pointnet_basic": pointnet_basic,
    "pointnet_0_1_kl": pointnet_0_1_kl,
    "pointnet_0_1_kl_6drot": pointnet_0_1_kl_6drot,
    "pointnet_0_1_kl_kin_loss": pointnet_0_1_kl_kin_loss,
    "pointnet_0_1_kl_kin_loss_6drot": pointnet_0_1_kl_kin_loss_6drot,
    "pointnet_0_1_kl_kin_loss_pen": pointnet_0_1_kl_kin_loss_pen,
    "spconv_local_kin_loss": spconv_local_kin_loss,
    "spconv_layernorm_local_kin_loss": spconv_layernorm_local_kin_loss,
    "spconv_local_kin_loss_6drot": spconv_local_kin_loss_6drot,
    "spconv_layernorm_local_kin_loss_6drot": spconv_layernorm_local_kin_loss_6drot,
}
