"""Collect bundles and evaluation rows for the full matrix."""

import graspkit  # noqa: F401 I001

import csv
import json
from itertools import product
from pathlib import Path

import tyro
from graspkit.utils.io_utils import build_filesystem


def main(
    root: Path,
    s3: str = "",
    sync: bool = False,
    sync_evaluations: bool = False,
    output: Path | None = None,
):
    output = root / "native_report" if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    assert not (sync or sync_evaluations) or s3
    fs = build_filesystem(s3) if s3 else None
    remote = s3.rstrip("/")
    if sync:
        fs.get_file(
            f"{remote}/compatibility/training_geometry_exclusions.json",
            str(root / "compatibility/training_geometry_exclusions.json"),
        )
        for uri in fs.find(f"{remote}/bundles"):
            destination = root / uri.removeprefix("s3://").removeprefix(remote.removeprefix("s3://") + "/")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                fs.get_file(uri, str(destination))
    if sync or sync_evaluations:
        for prefix in ("evaluations/dev", "evaluations/test", "evaluation_attempts"):
            for uri in fs.find(f"{remote}/{prefix}"):
                if uri.endswith("/failure.json"):
                    relative = uri.removeprefix("s3://").removeprefix(remote.removeprefix("s3://") + "/")
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    fs.get_file(uri, str(destination))
                if prefix == "evaluation_attempts" or not uri.endswith("/complete.json"):
                    continue
                relative = uri.removeprefix("s3://").removeprefix(remote.removeprefix("s3://") + "/")
                folder = root / Path(relative).parent
                folder.mkdir(parents=True, exist_ok=True)
                fs.get_file(uri, str(folder / "complete.json"))
                fs.get_file(uri.removesuffix("complete.json") + "rows.jsonl", str(folder / "rows.jsonl"))
    protocol = json.loads((root / "protocol.json").read_text())
    remote = protocol["s3_root"].rstrip("/") if not s3 else remote
    splits = json.loads((root / "evaluation_splits.json").read_text())
    overlap = json.loads((root / "compatibility/training_geometry_exclusions.json").read_text())
    canonical_aliases = set(overlap["canonical_test_scene_ids"])
    entries, table, failures = [], [], []
    verified_policies = 0
    for robot, sensor, generator in product(protocol["robots"], protocol["sensors"], protocol["generators"]):
        bundle = root / "bundles" / robot / sensor / generator
        present = (bundle / "bundle.json").exists()
        metadata = json.loads((bundle / "bundle.json").read_text()) if present else {}
        affected = set(metadata.get("test_object_overlap", {}).get("test_scene_ids", []))
        assert affected <= canonical_aliases
        if present:
            assert metadata["strict_load"]
            verified_policies += 1
        for loop in protocol["loops"]:
            entry = {
                "robot": robot,
                "input": sensor,
                "generator": generator,
                "loop": loop,
                "checkpoint_status": metadata.get("training_status", "missing"),
                "local_bundle": str(bundle) if present else None,
                "s3_bundle": f"{remote}/bundles/{robot}/{sensor}/{generator}" if present and remote else None,
                "loader": "graspkit.models.checkpoint.load_policy",
                "configuration": str(bundle / "config.json") if present else None,
                "training_test_alias_scene_ids": sorted(affected),
                "historical_selection": metadata.get("historical_selection"),
                "weights_shared_with_other_loop": True,
                "closed_loop": protocol["closed_loop"],
                "evaluations": [],
            }
            for simulator in protocol["simulators"]:
                for split in ("dev", "test"):
                    folder = root / "evaluations" / split / robot / sensor / generator / simulator / loop
                    rows, scene_ids = [], set()
                    for marker in sorted(folder.glob("*/complete.json")):
                        proof = json.loads(marker.read_text())
                        assert present and (
                            proof["robot"],
                            proof["input"],
                            proof["model"],
                            proof["simulator"],
                            proof["loop"],
                            proof["split"],
                        ) == (robot, sensor, generator, simulator, loop, split)
                        chunk = [json.loads(line) for line in (marker.parent / "rows.jsonl").read_text().splitlines()]
                        ids = [row["scene_id"] for row in chunk]
                        assert ids == proof["scene_ids"] and not scene_ids.intersection(ids)
                        assert len(chunk) == proof["trials"]
                        assert sum(bool(row["success"]) for row in chunk) == proof["successes"]
                        scene_ids.update(ids)
                        rows.extend(chunk)
                    assert scene_ids <= set(splits[split])
                    successes = sum(bool(row["success"]) for row in rows)
                    disjoint = [row for row in rows if row["scene_id"] not in affected]
                    expected = protocol["test_scenes_per_condition"] if split == "test" else protocol["dev_scene_count"]
                    result = {
                        "robot": robot,
                        "input": sensor,
                        "generator": generator,
                        "loop": loop,
                        "simulator": simulator,
                        "split": split,
                        "successes": successes,
                        "trials": len(rows),
                        "expected": expected,
                        "success_rate": successes / len(rows) if rows else None,
                        "complete": len(rows) == expected,
                        "object_disjoint_successes": sum(bool(row["success"]) for row in disjoint),
                        "object_disjoint_trials": len(disjoint),
                    }
                    entry["evaluations"].append(result)
                    table.append(result)
                    for log in folder.glob("*/failure.json"):
                        failures.append(json.loads(log.read_text()))
            entries.append(entry)
    failures_by_task = {failure["task_id"]: failure for failure in failures}
    for log in sorted((root / "evaluation_attempts").glob("*/failure.json")):
        failure = json.loads(log.read_text())
        if failure["task_id"] in failures_by_task:
            failures_by_task[failure["task_id"]]["preserved_attempt"] = failure
        else:
            failures_by_task[failure["task_id"]] = failure
    failures = list(failures_by_task.values())
    assert len(entries) == 40 and len(table) == 160
    manifest = {
        "verified_policies": verified_policies,
        "expected_policies": 20,
        "loop_entries": entries,
        "dev_conditions_complete": sum(row["complete"] for row in table if row["split"] == "dev"),
        "dev_episodes": sum(row["trials"] for row in table if row["split"] == "dev"),
        "full_test_conditions_complete": sum(row["complete"] for row in table if row["split"] == "test"),
        "full_test_episodes": sum(row["trials"] for row in table if row["split"] == "test"),
    }
    (output / "checkpoints.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        f"Policies: {verified_policies}/20. Test coverage: {manifest['full_test_episodes']}/40000 episodes.",
        "",
        "Closed loop: observe again at step 75, replan for 225 steps; weights shared with open loop.",
        protocol["evaluation_selection_note"],
        "",
        "| Robot | Input | Generator | Loop | Simulator | Successes/trials | SR |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    if any(entry["historical_selection"] for entry in entries):
        lines[3:3] = [
            "Historical depth recipes were selected on test outcomes and trained on eight object aliases affecting nine test scenes. Results are descriptive; object-disjoint counts are included in results.csv. New RGB training excludes those aliases.",
            "Isaac Sharpa results describe its measured backend dynamics: sampled full-test traces exceed named joint limits by over 1 radian. See [measured joint audit](../compatibility/sdk_named_joint_limits_audit.json). Contact-qualified success and commanded/measured trajectories are audited separately.",
            "Isaac RGB uses native geometry/materials and shared light intensity 600 selected only on dev images. ACES tone mapping remains enabled; residual brightness differences are documented in [pixel audit](../diagnostics/static_light_analysis/comparison.json).",
            "Earlier Isaac ALOHA depth chunks retain a partial camera visual; camera collision, control and robot-segment removal settings were unchanged. Every chunk records its source. RGB uses the repaired complete camera geometry.",
        ]
    for row in table:
        if row["split"] == "test":
            rate = f"{100 * row['success_rate']:.1f}%" if row["trials"] else "pending"
            lines.append(
                f"| {row['robot']} | {row['input']} | {row['generator']} | {row['loop']} | {row['simulator']} | {row['successes']}/{row['trials']} | {rate} |"
            )
    (output / "SR.md").write_text("\n".join(lines) + "\n")
    if s3:
        for filename in ("checkpoints.json", "results.csv", "SR.md", "failures.json"):
            destination = f"{remote}/native_report/{filename}"
            fs.put_file(str(output / filename), destination)
    print(
        json.dumps({key: value for key, value in manifest.items() if key != "loop_entries"}),
        flush=True,
    )


if __name__ == "__main__":
    tyro.cli(main)
