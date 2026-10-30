# Data Format Specification

## Grasp Poses

### Per-scene files

One HDF5 file per scene:

- File name: `{scene_id}.h5`
- Datasets (float arrays):
  - `grasp_qpos`: shape `(N, D)`
  - `pre_grasp_qpos`: shape `(N, D)`
  - `post_grasp_qpos`: shape `(N, D)`

Where:

- `N` = number of stored grasps for this scene (varies per scene; may include appended results from multiple runs)
- `D` = dimension of the configuration. For floating base robots, this includes 3D position, 3D euler angles (rxyz), and hand joint positions.

### Merged file

- File name: `{output_dir}.h5`
- Structure: one group per scene `/{scene_id}/` containing the same datasets as per-scene files
