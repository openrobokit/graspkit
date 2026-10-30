# Motion Planning

`GraspTrajBuilder` filters grasps, optionally IK-converts them onto an arm, then plans with
`build_motion_planner`: a `WarpPlanner` by default, or its base `TorchPlanner` when
`MotionPlannerConfig.backend` is `"torch"`.

::: graspkit.stages.grasp_traj_build.GraspTrajBuilder
::: graspkit.configs.experiment_config.GraspTrajBuilderConfig
::: graspkit.stages.motion_plan.WarpPlanner
::: graspkit.configs.experiment_config.MotionPlannerConfig
