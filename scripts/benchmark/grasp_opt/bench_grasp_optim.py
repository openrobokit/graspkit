"""End-to-end orchestrator: runs each (preset, backend, variant) via run_per_variant.py, merges into markdown.

Usage:
    python3 scripts/benchmark/grasp_opt/bench_grasp_optim.py
    python3 scripts/benchmark/grasp_opt/bench_grasp_optim.py --preset xhand_right
"""

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

PRESETS = {
    "mano_right": ["warp_gd", "warp_lm"],
    "xhand_right": ["warp_gd", "warp_lm"],
    "xarm7_xhand_right": ["warp_gd", "warp_lm", "warp_lm_ik"],
    "g1_inspire_right": ["warp_gd", "warp_lm", "warp_lm_ik"],
}
BACKENDS = ["mesh", "sdf_volume"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="all", choices=["all", *PRESETS])
    ap.add_argument("--num-scenes", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=None)
    ap.add_argument("--num-contact-points", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/tmp/bench_results")
    ap.add_argument(
        "--out",
        default=str(THIS_DIR / "bench_grasp_optim_results.mesh_vs_low_sdf.md"),
    )
    ap.add_argument("--run-sim", action="store_true")
    ap.add_argument("--keep-existing", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    presets = list(PRESETS) if args.preset == "all" else [args.preset]

    for preset in presets:
        for backend in BACKENDS:
            for variant in PRESETS[preset]:
                json_path = out_dir / f"{preset}__{backend}__{variant}.json"
                if args.keep_existing and json_path.exists():
                    print(f"[skip exists] {json_path}", flush=True)
                    continue
                print(f"\n{'=' * 60}\n{preset} / {backend} / {variant}\n{'=' * 60}", flush=True)
                cmd = [
                    "uv",
                    "run",
                    "python",
                    str(THIS_DIR / "run_per_variant.py"),
                    "--preset",
                    preset,
                    "--backend",
                    backend,
                    "--variant",
                    variant,
                    "--num-scenes",
                    str(args.num_scenes),
                    "--num-contact-points",
                    str(args.num_contact_points),
                    "--seed",
                    str(args.seed),
                    "--out-json",
                    str(json_path),
                ]
                if args.num_samples is not None:
                    cmd += ["--num-samples", str(args.num_samples)]
                if args.run_sim:
                    cmd += ["--run-sim"]
                rc = subprocess.run(cmd, check=False).returncode
                if rc != 0:
                    print(f"WARNING: {preset}/{backend}/{variant} exited with {rc}", file=sys.stderr, flush=True)

    print(f"\n{'=' * 60}\nMerging into {args.out}\n{'=' * 60}", flush=True)
    subprocess.run(
        ["uv", "run", "--no-project", str(THIS_DIR / "merge_results.py"), "--in-dir", str(out_dir), "--out", args.out],
        check=True,
    )


if __name__ == "__main__":
    main()
