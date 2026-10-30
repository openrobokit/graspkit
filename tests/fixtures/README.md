# Local test assets

`mujoco_eval/` retains 20 golden grasps and only their required object collision
pieces and mass metadata from the local DexGraspBench samples.

`mujoco_eval/mesh_order.json` records the convex-piece insertion order used by
those golden trajectories. The parity test supplies this order explicitly because
directory enumeration differs across filesystems and changes MuJoCo's numerical
rollouts. Expected values and the original tolerances are unchanged.
