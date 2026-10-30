"""Merge per-(preset, backend, variant) JSON results into one markdown matching graspkit's mesh_vs_low_sdf report shape.

Usage:
    uv run scripts/benchmark/grasp_opt/merge_results.py --in-dir /tmp/bench_results --out /tmp/merged.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

PRESETS = [
    ("mano_right", "floating", 128),
    ("xhand_right", "floating", 64),
    ("xarm7_xhand_right", "fixed", 64),
    ("g1_inspire_right", "fixed", 64),
]
BACKENDS = ["mesh", "sdf_volume"]
VARIANTS_BY_TASK = {"floating": ["warp_gd", "warp_lm"], "fixed": ["warp_gd", "warp_lm", "warp_lm_ik"]}

ROWS = [
    ("Total Time (s)", "total_time", ".2f"),
    ("  Init Time (s)", "init_time", ".2f"),
    ("  Optim Time (s)", "optim_time", ".2f"),
    ("Mean Penetration (m)", "mean_penetration", ".6f"),
    ("Max Penetration (m)", "max_penetration", ".6f"),
    ("Bake Time (s)", "bake_time", ".2f"),
    ("SDF File Size (MB)", "sdf_file_size_mb", ".3f"),
    ("MP Time (s)", "mp_time", ".2f"),
    ("Sim Time (s)", "sim_time", ".2f"),
    ("Motion Plan SR (%)", "mp_sr", ".2f"),
    ("Simulation SR (%)", "sim_sr", ".2f"),
    ("Final SR (%)", "final_sr", ".2f"),
    ("Mean Energy (total)", "mean_energy", ".4f"),
    ("  E_fc", "E_fc", ".4f"),
    ("  E_dis", "E_dis", ".4f"),
    ("  E_pen", "E_pen", ".4f"),
    ("  E_spen", "E_spen", ".4f"),
    ("  E_joints", "E_joints", ".4f"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/tmp/bench_results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-scenes", type=int, default=8)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    lines = [
        "# Grasp Benchmark Results — mesh_vs_low_sdf (robokit-standalone bench, graspkit optim)",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]

    for preset, task, n_samples in PRESETS:
        section = f"\n## {preset}_dev_mesh_vs_low_sdf ({task} task, {args.num_scenes} scenes × {n_samples} samples = {args.num_scenes * n_samples} grasps)"
        lines.append(section)
        method_cols: Dict[str, Dict] = {}
        for backend in BACKENDS:
            label_prefix = "mesh" if backend == "mesh" else "low_sdf"
            for variant in VARIANTS_BY_TASK[task]:
                key = f"{preset}__{backend}__{variant}"
                f = in_dir / f"{key}.json"
                col = f"{label_prefix}_{variant}"
                if f.exists():
                    method_cols[col] = json.loads(f.read_text())
                else:
                    method_cols[col] = {}

        methods = list(method_cols.keys())
        lines.append("| Metric | " + " | ".join(methods) + " |")
        lines.append("|---|" + "|".join("---" for _ in methods) + "|")
        for label, key, fmt in ROWS:
            vals = []
            for m in methods:
                v = method_cols[m].get(key)
                vals.append(f"{v:{fmt}}" if v is not None else "N/A")
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
