# Grasp Benchmark Results — mesh_vs_low_sdf (graspkit on robokit contact var and solvers; volcan RTX 3090, idle GPU)
Date: 2026-09-05 01:29
## xhand_right_dev_mesh_vs_low_sdf (floating task, 8 scenes × 64 samples = 512 grasps)
| Metric | mesh_warp_gd | mesh_warp_lm | low_sdf_warp_gd | low_sdf_warp_lm |
|---|---|---|---|---|
| Total Time (s) | 41.40 | 19.36 | 16.26 | 5.39 |
|   Init Time (s) | 0.00 | 0.00 | 0.00 | 0.00 |
|   Optim Time (s) | 41.40 | 19.36 | 16.26 | 5.39 |
| Mean Penetration (m) | 0.000109 | 0.000032 | 0.000232 | 0.000115 |
| Max Penetration (m) | 0.002760 | 0.002726 | 0.003704 | 0.002788 |
| Bake Time (s) | N/A | N/A | 0.07 | 0.07 |
| SDF File Size (MB) | N/A | N/A | 0.884 | 0.884 |
| MP Time (s) | 0.56 | 0.56 | 0.56 | 0.56 |
| Sim Time (s) | 29.13 | 28.79 | 29.33 | 27.99 |
| Motion Plan SR (%) | 99.61 | 98.83 | 99.41 | 98.24 |
| Simulation SR (%) | 36.52 | 67.58 | 40.43 | 77.54 |
| Final SR (%) | 36.38 | 66.79 | 40.19 | 76.18 |
| Mean Energy (total) | 13.8052 | 4.9424 | 13.1040 | 4.9389 |
|   E_fc | 3.7397 | 3.1054 | 3.7321 | 3.1070 |
|   E_dis | 8.9801 | 1.4237 | 8.3093 | 1.4090 |
|   E_pen | 0.8485 | 0.1760 | 0.8193 | 0.1751 |
|   E_spen | 0.2369 | 0.2373 | 0.2433 | 0.2478 |
|   E_joints | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Reference (graspkit main, pinned robokit 49a060f2) on the same GPU, same day
| Metric | mesh_warp_gd | mesh_warp_lm | low_sdf_warp_gd | low_sdf_warp_lm |
|---|---|---|---|---|
| Total Time (s) | 40.86 | 14.62 | 16.47 | 3.95 |
|   Init Time (s) | 0.00 | 0.00 | 0.00 | 0.00 |
|   Optim Time (s) | 40.86 | 14.62 | 16.47 | 3.95 |
| Mean Penetration (m) | 0.000122 | 0.000017 | 0.000255 | 0.000073 |
| Max Penetration (m) | 0.002034 | 0.001756 | 0.002753 | 0.002563 |
| Bake Time (s) | N/A | N/A | 0.07 | 0.07 |
| SDF File Size (MB) | N/A | N/A | 0.884 | 0.884 |
| MP Time (s) | 0.54 | 0.54 | 0.54 | 0.54 |
| Sim Time (s) | 32.53 | 29.22 | 29.42 | 29.20 |
| Motion Plan SR (%) | 100.00 | 99.61 | 100.00 | 98.44 |
| Simulation SR (%) | 40.82 | 66.80 | 40.43 | 73.83 |
| Final SR (%) | 40.82 | 66.54 | 40.43 | 72.67 |
| Mean Energy (total) | 11.2490 | 4.4072 | 10.6800 | 4.4161 |
|   E_fc | 3.7729 | 2.9599 | 3.7397 | 2.8880 |
|   E_dis | 6.5561 | 1.1337 | 6.0788 | 1.1954 |
|   E_pen | 0.6659 | 0.0849 | 0.6143 | 0.0847 |
|   E_spen | 0.2541 | 0.2288 | 0.2472 | 0.2480 |
|   E_joints | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
