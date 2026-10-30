# Grasp Benchmark Results — mesh_vs_low_sdf (robokit-standalone bench, graspkit optim)
Date: 2026-09-01 20:40

## mano_right_dev_mesh_vs_low_sdf (floating task, 8 scenes × 128 samples = 1024 grasps)
| Metric | mesh_warp_gd | mesh_warp_lm | low_sdf_warp_gd | low_sdf_warp_lm |
|---|---|---|---|---|
| Total Time (s) | 19.87 | 12.46 | 9.30 | 4.63 |
|   Init Time (s) | 0.00 | 0.00 | 0.00 | 0.00 |
|   Optim Time (s) | 19.87 | 12.46 | 9.30 | 4.63 |
| Mean Penetration (m) | 0.000711 | 0.000093 | 0.000830 | 0.000309 |
| Max Penetration (m) | 0.010753 | 0.003409 | 0.008155 | 0.005876 |
| Bake Time (s) | N/A | N/A | 0.16 | 0.11 |
| SDF File Size (MB) | N/A | N/A | 0.884 | 0.884 |
| MP Time (s) | 0.37 | 0.21 | 0.28 | 0.26 |
| Sim Time (s) | 29.49 | 26.37 | 29.25 | 27.76 |
| Motion Plan SR (%) | 99.02 | 93.85 | 97.46 | 87.60 |
| Simulation SR (%) | 85.25 | 93.07 | 88.57 | 94.04 |
| Final SR (%) | 84.42 | 87.34 | 86.33 | 82.38 |
| Mean Energy (total) | 15.4257 | 9.6206 | 14.8742 | 9.2947 |
|   E_fc | 3.8906 | 3.3264 | 3.8915 | 3.2904 |
|   E_dis | 11.2583 | 6.0557 | 10.7165 | 5.7694 |
|   E_pen | 0.1468 | 0.0362 | 0.1371 | 0.0344 |
|   E_spen | 0.1299 | 0.2022 | 0.1291 | 0.2005 |
|   E_joints | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## xhand_right_dev_mesh_vs_low_sdf (floating task, 8 scenes × 64 samples = 512 grasps)
| Metric | mesh_warp_gd | mesh_warp_lm | low_sdf_warp_gd | low_sdf_warp_lm |
|---|---|---|---|---|
| Total Time (s) | 57.98 | 23.67 | 23.86 | 5.44 |
|   Init Time (s) | 0.00 | 0.00 | 0.00 | 0.00 |
|   Optim Time (s) | 57.98 | 23.67 | 23.86 | 5.44 |
| Mean Penetration (m) | 0.000134 | 0.000011 | 0.000269 | 0.000063 |
| Max Penetration (m) | 0.002660 | 0.002377 | 0.002835 | 0.002601 |
| Bake Time (s) | N/A | N/A | 0.08 | 0.10 |
| SDF File Size (MB) | N/A | N/A | 0.884 | 0.884 |
| MP Time (s) | 0.55 | 0.53 | 0.61 | 0.68 |
| Sim Time (s) | 34.34 | 27.39 | 31.43 | 36.59 |
| Motion Plan SR (%) | 100.00 | 99.02 | 99.61 | 98.63 |
| Simulation SR (%) | 41.02 | 68.95 | 39.84 | 72.85 |
| Final SR (%) | 41.02 | 68.27 | 39.69 | 71.86 |
| Mean Energy (total) | 11.6156 | 4.4081 | 11.2981 | 4.3171 |
|   E_fc | 3.6418 | 2.9526 | 3.6628 | 2.8060 |
|   E_dis | 7.0743 | 1.1303 | 6.7425 | 1.1717 |
|   E_pen | 0.6507 | 0.0859 | 0.6343 | 0.0866 |
|   E_spen | 0.2487 | 0.2393 | 0.2585 | 0.2527 |
|   E_joints | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## xarm7_xhand_right_dev_mesh_vs_low_sdf (fixed task, 8 scenes × 64 samples = 512 grasps)
| Metric | mesh_warp_gd | mesh_warp_lm | mesh_warp_lm_ik | low_sdf_warp_gd | low_sdf_warp_lm | low_sdf_warp_lm_ik |
|---|---|---|---|---|---|---|
| Total Time (s) | 70.54 | 25.33 | 33.10 | 26.94 | 7.02 | 8.17 |
|   Init Time (s) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
|   Optim Time (s) | 70.54 | 25.33 | 33.10 | 26.94 | 7.02 | 8.17 |
| Mean Penetration (m) | 0.000310 | 0.004860 | 0.000061 | 0.000451 | 0.005361 | 0.000077 |
| Max Penetration (m) | 0.002685 | 0.083753 | 0.003634 | 0.003179 | 0.068765 | 0.003261 |
| Bake Time (s) | N/A | N/A | N/A | 0.12 | 0.11 | 0.10 |
| SDF File Size (MB) | N/A | N/A | N/A | 0.822 | 0.822 | 0.822 |
| MP Time (s) | 37.22 | 41.27 | 36.45 | 33.79 | 34.67 | 33.45 |
| Sim Time (s) | 48.15 | 60.97 | 50.68 | 59.71 | 63.17 | 58.90 |
| Motion Plan SR (%) | 95.12 | 72.07 | 86.91 | 95.51 | 70.90 | 85.74 |
| Simulation SR (%) | 71.29 | 63.67 | 82.23 | 77.54 | 66.02 | 86.52 |
| Final SR (%) | 67.81 | 45.89 | 71.47 | 74.06 | 46.80 | 74.19 |
| Mean Energy (total) | 10.1320 | 16.6313 | 5.6188 | 10.1913 | 20.0387 | 5.0056 |
|   E_fc | 3.9970 | 3.7432 | 3.4678 | 4.0198 | 3.8191 | 3.5152 |
|   E_dis | 5.2898 | 11.2538 | 1.7303 | 5.3567 | 14.1509 | 1.1077 |
|   E_pen | 0.5256 | 1.2369 | 0.1128 | 0.5127 | 1.6701 | 0.0745 |
|   E_spen | 0.3196 | 0.3974 | 0.3079 | 0.3020 | 0.3986 | 0.3082 |
|   E_joints | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## g1_inspire_right_dev_mesh_vs_low_sdf (fixed task, 8 scenes × 64 samples = 512 grasps)
| Metric | mesh_warp_gd | mesh_warp_lm | mesh_warp_lm_ik | low_sdf_warp_gd | low_sdf_warp_lm | low_sdf_warp_lm_ik |
|---|---|---|---|---|---|---|
| Total Time (s) | 44.11 | 10.88 | 12.65 | 15.41 | 3.19 | 5.09 |
|   Init Time (s) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
|   Optim Time (s) | 44.11 | 10.88 | 12.65 | 15.41 | 3.19 | 5.09 |
| Mean Penetration (m) | 0.001191 | 0.009617 | 0.000834 | 0.001670 | 0.009626 | 0.001074 |
| Max Penetration (m) | 0.007298 | 0.050407 | 0.013969 | 0.006876 | 0.051783 | 0.016443 |
| Bake Time (s) | N/A | N/A | N/A | 0.07 | 0.08 | 0.09 |
| SDF File Size (MB) | N/A | N/A | N/A | 0.822 | 0.822 | 0.822 |
| MP Time (s) | 12.02 | 8.63 | 8.35 | 7.97 | 8.19 | 8.12 |
| Sim Time (s) | 105.04 | 119.12 | 92.41 | 97.88 | 86.48 | 85.77 |
| Motion Plan SR (%) | 97.27 | 66.60 | 97.07 | 95.31 | 66.99 | 96.09 |
| Simulation SR (%) | 24.02 | 11.72 | 31.05 | 18.55 | 10.55 | 33.79 |
| Final SR (%) | 23.37 | 7.80 | 30.14 | 17.68 | 7.07 | 32.47 |
| Mean Energy (total) | 10.8919 | 127.7998 | 6.2349 | 10.3812 | 136.2042 | 5.8016 |
|   E_fc | 4.4957 | 5.6573 | 3.6901 | 4.4931 | 5.7496 | 3.7634 |
|   E_dis | 5.9712 | 115.4901 | 2.3930 | 5.4874 | 123.3574 | 1.9061 |
|   E_pen | 0.3103 | 6.5323 | 0.0685 | 0.2840 | 6.9809 | 0.0556 |
|   E_spen | 0.1147 | 0.1200 | 0.0832 | 0.1167 | 0.1163 | 0.0766 |
|   E_joints | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Regression investigation

- Final success improves in 17/20 settings. Regressions are xHand low-SDF GD (-1.25 points), xArm-xHand mesh GD (-2.00), and G1 low-SDF GD (-0.56).
- Restoring the pre-`ee1cd0d` closed-pose holding trajectory recovers xHand final success to 41.02%, 68.27%, 39.69%, and 71.86%; gradual closure remains Vega-only.
- Two matched, reversed-order Volcan runs show aggregate optimizer time +1.15%; individual settings range from -5.1% to +6.8%. There is no large timing regression, though the consistent ~6% slowdowns are real.
- Disabling robokit's newer finite-difference contact-normal derivative changes the matched current run from 4.48 s to 4.50 s, ruling it out as the timing cause.
- The July reference predates the per-scene collision mapping fix (`14bfac1`), so collision, penetration, and success values are not strictly apples-to-apples.
