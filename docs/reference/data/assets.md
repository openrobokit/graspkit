# graspkit assets

graspkit's own assets (scene descriptions, object-mesh datasets) live in the HF dataset
[`openrobokit/graspkit`](https://huggingface.co/datasets/openrobokit/graspkit).
Robot assets (URDFs, spheres, contact points, self-collision ignores) live in **robokit's** dataset and
are consumed via modules such as `robokit.assets.robots.end_effectors.sharpa_hand`. `import graspkit`
downloads nothing. A robot module fetches its subtree into `~/.robokit/assets/<pinned-commit>/`;
graspkit assets use `~/.graspkit/assets/<graspkit.assets.REVISION>/`, fetched by the same `robokit.assets.fetch`.

`RobotConfig.self_collision_ignore_path` points to one robot's ignore YAML. Files named
`*.sphere.yml` and `*.capsule.yml` select the sibling matching the term's representation.
Grasp optimization defaults to spheres; select `spen_representation="capsule"` for capsule terms.
Motion planning and sphere filters always select sphere ignores. An enabled term warns at construction
if its matching file is unavailable; an invalid supplied path still raises. Runtime assemblies have
no ignore path unless one is supplied on their resulting config.

## Layout

```
calib/                 # per-rig camera extrinsics/intrinsics + joint offsets
data/scene_description/ # per-task scene msgpacks (dex1b/, ...)
data/object_data/       # mesh archives (dex_diffuser*, dexgraspnet, objaverse_dex1b_v2),
                        #   extracted at import; dexgraspnet_official_splits/ = train/test lists
```

`~/.graspkit/`: `assets/` = fetched HF files (what code reads), `hf_cache/` = HF blobs; `data/`+`logs/` = pipeline output
(S3-backed, not assets). `data/logs` live under home (not the repo), so graspkit works as a library too;
`graspkit.link()` symlinks them into your checkout for browsing.

## Consuming

```python
from graspkit.assets import calib
from graspkit.assets.scene_descriptions import dex1b_v0_4
from graspkit.assets.objects import dex_diffuser

dex1b_v0_4.FLOATING_MINI_PATH                       # importing fetches only v0.4
dex_diffuser.DATA_DIR / "processed/ycb"             # importing fetches + extracts the archive
calib.calib_file("xarm/26-06-21-camera_pose.json")  # fetches calib/, returns a Path
```

- **Objects** are archives: `objects.<src>` fetches + extracts into `DATA_ROOT/object_data/<src>` (marker-guarded) as `DATA_DIR`; `dexgraspnet_official_splits` is archive-free (`DIR`).
- **Scenes**: one module per dex1b version (`_DIR = fetch([...], REVISION, HF_REPO_ID, GRASPKIT_HOME)` + `*_PATH`). Add a scene = add a `*_PATH` + its `fetch` pattern.
- Ad-hoc: `robokit.assets.fetch(["data/scene_description/**"], REVISION, HF_REPO_ID, GRASPKIT_HOME) / "..."`. Bump `graspkit.assets.REVISION` after uploading new data.

Configs/experiments resolve by name, importing only what's needed (so only its assets download):

```python
from graspkit.configs.robot import build_robot_config
from graspkit.configs.experiment import build_experiment_config

build_robot_config("sharpa_right")                       # downloads only sharpa_right's robot assets
build_experiment_config("sharpa_right_dex1b_v0_4_mini_floating")
```

## Authoring

The dataset is too large to clone, so there's no local git checkout. Upload directly through the HF
API (uploads only new xet chunks; no git or git-xet plugin needed):

```shell
uv run scripts/tools/upload_to_hf.py data/scene_description/dex1b/v0.4/x.msgpack
```

It pushes the file/dir to the same repo-relative path (`upload_file`/`upload_folder`). Then expose it:
add a module/symbol under `src/graspkit/assets/scene_descriptions/` or `objects/`
(eager `_DIR = fetch([...], REVISION, HF_REPO_ID, GRASPKIT_HOME)` + typed `Path` symbols), keeping the
`data/` layout above, and bump `graspkit.assets.REVISION` to the new dataset commit.

To pull only a subset back, `fetch` with a pattern (or `snapshot_download(allow_patterns=[...])`):
never the whole repo.

Robot assets are authored in **robokit's** (small) dataset via its own git-xet checkout: see robokit.
