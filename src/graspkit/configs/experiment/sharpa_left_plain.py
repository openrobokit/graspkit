from graspkit.assets.scene_descriptions import dex1b_v0_5_0
from graspkit.configs.experiment_config import DatasetConfig, ExperimentConfig, TrainerConfig
from graspkit.configs.grasp_policy_config import GraspPolicyConfig
from graspkit.configs.robot.sharpa import sharpa_left
from graspkit.constants import DATA_ROOT

sharpa_left_plain = ExperimentConfig(
    name="sharpa_left_plain",
    steps=("train",),
    robot_config=sharpa_left,
    ckpt_path=None,
    agent=GraspPolicyConfig(
        vision_encoder="pointnet",
        convert_rxyz_to_rot6d=True,
        analytic_euler=True,
        center_pointcloud=True,
        pointcloud_scale=10.0,
        train_yaw_augmentation=True,
        cvae_latent_dim=32,
        kl_weight=0.5,
        tl_weight=20.0,
        enable_kinematic_loss=True,
        use_mean=False,
    ),
    dataset=DatasetConfig(num_points=1024, return_object_pointcloud_only=True, cache_in_memory=True),
    trainer=TrainerConfig(
        seed=29,
        evaluator=None,
        batch_size=16,
        lr=3e-4,
        total_iters=200000,
        save_freq=40000,
        num_dataloader_workers=4,
    ),
    train_single_gpu=True,
    scene_description_path=str(dex1b_v0_5_0.TABLETOP_MINI_PATH),
    processed_data_path=str(DATA_ROOT / "sim_data/sharpa_left_floating/native_train.h5"),
)
