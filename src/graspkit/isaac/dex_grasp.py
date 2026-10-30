"""Isaac tabletop grasping for `worker.py` in ManiSkill's native appearance; import only after AppLauncher starts."""

import copy
import fcntl
import hashlib
import json
import math
from ast import literal_eval
from pathlib import Path

import h5py
import isaaclab.sim as sim_utils
import numpy as np
import omni.replicator.core as rep
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg, CameraData
from isaaclab.utils.math import quat_apply, quat_mul
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from isaacsim.core.simulation_manager import SimulationManager
from omni.physx import acquire_physx_interface
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from usdex.core import getValidPrimName

rep.orchestrator.set_capture_on_play(False)


def robot_usd(urdf: str, cache: str) -> str:
    """Import the URDF once per content and Isaac version; a USD written by one version may not open in another."""
    version = Path("/isaac-sim/VERSION").read_text().split("-")[0].strip()
    digest = hashlib.sha256(Path(urdf).read_bytes()).hexdigest()[:16]
    destination = Path(cache) / f"{digest}-{version}"
    destination.mkdir(parents=True, exist_ok=True)
    pointer = destination / "asset.txt"
    with (destination / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not pointer.exists():
            config = URDFImporterConfig(
                urdf_path=urdf,
                usd_path=str(destination),
                fix_base=True,
                merge_fixed_joints=False,
                joint_drive_type="force",
                joint_target_type="position",
                override_joint_stiffness=1e3,
                override_joint_damping=50.0,
                run_asset_transformer=True,
            )
            pointer.write_text(URDFImporter(config).import_urdf())
    return pointer.read_text()


def make_visual(
    stage: Usd.Stage, path: str, source: h5py.Group, material: dict | None = None, textures: dict | None = None
):
    """Vertex-colored mesh, or a glTF PBR material with its textures, under SAPIEN-like shader defaults."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    faces = source["faces"][...]
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(source["vertices"][...]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    surface = UsdShade.Material.Define(stage, path + "_material")
    shader = UsdShade.Shader.Define(stage, path + "_material/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5 if material is None else 1.0)
    if material is None:
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(source["colors"][...]))
        reader = UsdShade.Shader.Define(stage, path + "_material/color")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    else:
        mesh.CreateDoubleSidedAttr(material["doubleSided"])
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(source["normals"][...]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(source["uv"][...])
        )
        pbr = material["pbrMetallicRoughness"]
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(pbr["metallicFactor"])
        if "baseColorFactor" in pbr:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*pbr["baseColorFactor"][:3]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(pbr["roughnessFactor"])
        else:
            reader = UsdShade.Shader.Define(stage, path + "_material/uv")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            for name, texture, channel, value_type, color_space in (
                ("diffuseColor", pbr["baseColorTexture"], "rgb", Sdf.ValueTypeNames.Color3f, "sRGB"),
                ("roughness", pbr["metallicRoughnessTexture"], "g", Sdf.ValueTypeNames.Float, "raw"),
                ("normal", material["normalTexture"], "rgb", Sdf.ValueTypeNames.Normal3f, "raw"),
            ):
                image = UsdShade.Shader.Define(stage, path + "_material/" + name)
                image.CreateIdAttr("UsdUVTexture")
                image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(textures[str(texture["index"])])
                image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
                image.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                image.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                )
                if name == "normal":
                    image.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2))
                    image.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1))
                output_type = Sdf.ValueTypeNames.Float if channel == "g" else Sdf.ValueTypeNames.Float3
                shader.CreateInput(name, value_type).ConnectToSource(image.CreateOutput(channel, output_type))
    surface.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(surface)


def restore_native_materials(stage: Usd.Stage, path: str, materials: dict, mesh_materials: dict):
    """Apply SAPIEN's effective robot materials to the imported visual meshes without changing their geometry."""
    meshes = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Mesh)
        and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        and UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    ]
    applied = set()
    for mesh_path in meshes:
        prim = stage.GetPrimAtPath(mesh_path)
        while prim.IsInstanceProxy():
            ancestor = prim
            while ancestor.IsInstanceProxy():
                ancestor = ancestor.GetParent()
            ancestor.SetInstanceable(False)
            prim = stage.GetPrimAtPath(mesh_path)
        link = prim.GetParent()
        while link.GetName() not in materials:
            link = link.GetParent()
            assert link, mesh_path
        faces = len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
        candidates = materials[link.GetName()]
        if link.GetName() in mesh_materials:
            candidates = [mesh_materials[link.GetName()][prim.GetName()]]
            assert candidates[0]["expected_faces"] == faces
            applied.add((link.GetName(), prim.GetName()))
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        material.GetPrim().SetInstanceable(False)
        shader = UsdShade.Shader(material.GetSurfaceOutput().GetConnectedSource()[0])
        assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
        if len(candidates) > 1:
            candidates = [entry for entry in candidates if entry["faces"] == faces]
        assert candidates, (link.GetName(), mesh_path)
        if len(candidates) > 1:  # same face count: the imported base color identifies the SAPIEN material
            color = np.asarray(material.GetInput("diffuseColor").Get())
            distances = [np.square(np.asarray(entry["base_color"][:3]) - color).sum() for entry in candidates]
            assert sorted(distances)[1] > min(distances) + 1e-8 and min(distances) < 1e-12
            candidates = [candidates[int(np.argmin(distances))]]
        native = candidates[0]
        rgba = native["base_color"]
        if native["base_color_texture"]:
            texture = UsdShade.Shader(shader.GetInput("diffuseColor").GetConnectedSource()[0])
            assert texture.GetIdAttr().Get() == "UsdUVTexture"
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*rgba))
        else:
            color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            color.DisconnectSource()
            color.Set(Gf.Vec3f(*rgba[:3]))
        reflectance = np.sqrt(0.08 * native["specular"])
        values = {
            "roughness": native["roughness"],
            "metallic": native["metallic"],
            "ior": float((1 + reflectance) / (1 - reflectance)),
            "opacity": rgba[3],
        }
        for name, value in values.items():
            scalar = shader.CreateInput(name, Sdf.ValueTypeNames.Float)
            scalar.DisconnectSource()
            scalar.Set(value)
        emission = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
        emission.DisconnectSource()
        emission.Set(Gf.Vec3f(*native["emission"][:3]))
    assert applied == {(link, name) for link, names in mesh_materials.items() for name in names}


class IsaacDexGrasp:
    """`len(indices)` fixed-base tabletop envs on one GPU stage, built per reset; tensors stay on CUDA."""

    def __init__(self, job: h5py.File, indices: list[int], usd: str):
        n = self.n = len(indices)
        self.substeps = int(job.attrs["substeps"])
        self.dt = dt = 1.0 / (float(job.attrs["control_freq"]) * self.substeps)
        sim_utils.create_new_stage()
        sim = self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=dt,
                device="cuda:0",
                use_fabric=True,
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            )
        )
        # Stages are replaced without the editor's stop callback; Replicator renders only on request.
        sim._app_control_on_stop_handle.unsubscribe()
        sim._app_control_on_stop_handle = None
        sim.carb_settings.set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)
        sim.carb_settings.set_bool("/physics/disableContactProcessing", False)
        stage = sim.stage
        light = sim_utils.DomeLightCfg(intensity=600, visible_in_primary_ray=False)  # black background
        light.func("/World/Light", light)
        side = math.ceil(math.sqrt(n))
        origins = np.stack((np.arange(n) % side, np.arange(n) // side, np.zeros(n)), axis=1).astype(np.float32) * 2.5
        object_poses, table_poses, table_sizes = (
            job[key][indices].astype(np.float32) for key in ("obj_pose", "table_pose", "table_size")
        )
        grasp = sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        grasp.func("/World/Materials/grasp", grasp)
        textures = json.loads(job.attrs["table_textures"])
        with h5py.File(job.attrs["table_visual"]) as table:
            table_materials = json.loads(table.attrs["materials"])
            for i, index in enumerate(indices):
                env = f"/World/envs/env_{i}"
                UsdGeom.Xform.Define(stage, env).AddTranslateOp().Set(Gf.Vec3d(*origins[i].tolist()))
                box = UsdGeom.Cube.Define(stage, f"{env}/Table")
                box.CreateSizeAttr(1)
                box.AddTranslateOp().Set(Gf.Vec3d(*table_poses[i, :3].tolist()))
                box.AddOrientOp().Set(Gf.Quatf(*table_poses[i, 3:].tolist()))
                box.AddScaleOp().Set(Gf.Vec3f(*table_sizes[i].tolist()))
                UsdPhysics.CollisionAPI.Apply(box.GetPrim())
                box.MakeInvisible()
                # ManiSkill's textured table GLB, placed and scaled onto the collision box as its scene builder does.
                visual = UsdGeom.Xform.Define(stage, f"{env}/TableVisual")
                top = table_poses[i, :3].copy()
                top[2] -= table_sizes[i, 2] / 2
                visual.AddTranslateOp().Set(Gf.Vec3d(*top.tolist()))
                visual.AddOrientOp().Set(Gf.Quatf(*table_poses[i, 3:].tolist()))
                visual.AddRotateZOp().Set(90)
                scale = 1.75 * table_sizes[i, [1, 0, 2]] / np.asarray([1.209, 2.418, 0.9196429])
                visual.AddScaleOp().Set(Gf.Vec3f(*scale.tolist()))
                for piece in table["meshes"]:
                    group = table[f"meshes/{piece}"]
                    material = table_materials[int(group.attrs["material_index"])]
                    make_visual(stage, f"{env}/TableVisual/mesh_{piece}", group, material, textures)
                target = UsdGeom.Xform.Define(stage, f"{env}/Object")
                target.AddTranslateOp().Set(Gf.Vec3d(*object_poses[i, :3].tolist()))
                target.AddOrientOp().Set(Gf.Quatf(*object_poses[i, 3:].tolist()))
                UsdPhysics.RigidBodyAPI.Apply(target.GetPrim())
                UsdPhysics.MassAPI.Apply(target.GetPrim()).CreateMassAttr(float(job.attrs["object_mass"]))
                sim_utils.add_labels(target.GetPrim(), [f"target_{i}"], "class")
                mesh_index = int(job["mesh_idx"][index])
                for piece in job[f"meshes/{mesh_index}"]:  # convex pieces, one convex hull each
                    source = job[f"meshes/{mesh_index}/{piece}"]
                    mesh = UsdGeom.Mesh.Define(stage, f"{env}/Object/collision_{piece}")
                    faces = source["faces"][...]
                    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(source["vertices"][...]))
                    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
                    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
                    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
                    sim_utils.modify_collision_properties(
                        str(mesh.GetPath()), sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
                    )
                    mesh.MakeInvisible()
                make_visual(stage, f"{env}/Object/visual", job[f"visuals/{mesh_index}"])
                sim_utils.bind_physics_material(f"{env}/Object", "/World/Materials/grasp")
        names, hand = list(job.attrs["joint_names"]), list(job.attrs["hand_joints"])
        qpos = job["qpos"][indices, 0]
        root = job["root_pose"][indices[0]]
        robot = self.robot = Articulation(
            ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=30,
                        solver_velocity_iteration_count=1,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(root[:3]),
                    rot=tuple(root[3:]),
                    joint_pos={
                        getValidPrimName(name): float(value) for name, value in zip(names, qpos[0], strict=True)
                    },
                ),
                actuators={
                    "arm": ImplicitActuatorCfg(
                        joint_names_expr=[getValidPrimName(name) for name in names if name not in hand],
                        stiffness=1e6,
                        damping=1e5,
                        effort_limit_sim=500,
                        velocity_limit_sim=100,
                    ),
                    "hand": ImplicitActuatorCfg(
                        joint_names_expr=[getValidPrimName(name) for name in hand],
                        stiffness=1e3,
                        damping=50,
                        effort_limit_sim=50,
                        velocity_limit_sim=100,
                        friction=0.05,
                    ),
                },
            )
        )
        materials = {}
        for entry in json.loads(job.attrs["materials"]):
            materials.setdefault(getValidPrimName(entry["link"]), []).append(entry)
        mesh_materials = {
            getValidPrimName(link): {getValidPrimName(name): entry for name, entry in groups.items()}
            for link, groups in json.loads(job.attrs["camera_materials"]).items()
        }
        for i in range(n):
            restore_native_materials(stage, f"/World/envs/env_{i}/Robot", materials, mesh_materials)
            sim_utils.add_labels(stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot"), [f"robot_{i}"], "class")
        self.target = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Object", spawn=None))
        sim_utils.make_uninstanceable("/World/envs", stage)
        sim_utils.modify_collision_properties(
            "/World/envs", sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
        )
        contacts = []  # per contact link: its body in every env, reporting forces against that env's object
        for link in job.attrs["contact_links"]:
            matching = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith("/World/envs/env_0/Robot/")
                and prim.GetName() == getValidPrimName(link)
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            assert len(matching) == 1, (link, matching)
            contacts.append([matching[0].replace("env_0", f"env_{i}") for i in range(n)])
            for path in contacts[-1]:
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.0)
                if link in job.attrs["tip_links"]:
                    sim_utils.bind_physics_material(path, "/World/Materials/grasp")
                    sim_utils.modify_collision_properties(
                        path,
                        sim_utils.CollisionPropertiesCfg(torsional_patch_radius=0.05, min_torsional_patch_radius=0.04),
                    )
        self.camera_rows = json.loads(job.attrs["cameras"])
        # A few camera rigs visit the envs in rounds, so the physics batch size does not scale camera setup.
        self.rigs = min(n, 8)
        for r in range(self.rigs):
            UsdGeom.Xform.Define(stage, f"/World/Rigs/rig_{r}")
        self.cameras = [
            Camera(
                CameraCfg(
                    prim_path=f"/World/Rigs/rig_.*/Camera{index}",
                    width=row["width"],
                    height=row["height"],
                    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                        np.asarray(row["intrinsic"]).reshape(-1).tolist(),
                        row["width"],
                        row["height"],
                        clipping_range=(0.005, 5.0),
                    ),
                    data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
                    colorize_semantic_segmentation=True,
                    update_latest_camera_pose=True,
                )
            )
            for index, row in enumerate(self.camera_rows)
        ]
        physics = acquire_physx_interface()
        physics.overwrite_gpu_setting(1)
        sim.reset()
        assert physics.get_overwrite_gpu_setting() == 1
        self.sensors = [
            SimulationManager.get_physics_sim_view().create_rigid_contact_view(
                paths,
                filter_patterns=[[f"/World/envs/env_{i}/Object"] for i in range(n)],
                max_contact_data_count=n * 32,
            )
            for paths in contacts
        ]
        assert all(sensor.sensor_count == n and sensor.filter_count == 1 for sensor in self.sensors)
        device = self.device = sim.device
        self.org = torch.from_numpy(origins).to(device)
        self.tables = torch.from_numpy(table_poses[:, :3]).to(device)
        self.bounds = torch.from_numpy(np.asarray(job.attrs["point_bounds"], np.float32)).to(device)
        self.lift_dist = float(job.attrs["lift_dist"])
        self.joints = [robot.joint_names.index(getValidPrimName(name)) for name in names]
        self.policy_joints = [
            robot.joint_names.index(getValidPrimName(name)) for name in job.attrs["policy_joint_names"]
        ]
        self.command = robot.data.default_joint_pos.clone()
        self.command[:, self.joints] = torch.from_numpy(qpos).to(device)
        robot.write_joint_state_to_sim(self.command, torch.zeros_like(self.command))
        pose = torch.from_numpy(object_poses).to(device)
        pose[:, :3] += self.org
        self.target.write_root_pose_to_sim(pose)
        self.target.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))
        robot.set_joint_position_target(self.command)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        self.target.update(dt)
        self.initial_z = torch.from_numpy(object_poses[:, 2]).to(device)
        self.external_force = False

    def step_qpos(self, qpos: np.ndarray):
        """Hold full-joint position targets (n, J) for one control step."""
        self.command[:, self.joints] = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        for _ in range(self.substeps):
            self.robot.set_joint_position_target(self.command)
            self.robot.write_data_to_sim()
            if self.external_force:
                self.target.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.dt)
            self.target.update(self.dt)

    def get_qpos(self) -> np.ndarray:
        return self.robot.data.joint_pos[:, self.policy_joints].cpu().numpy()

    def get_object_pose(self) -> np.ndarray:
        return torch.cat((self.target.data.root_pos_w - self.org, self.target.data.root_quat_w), 1).cpu().numpy()

    def set_object_pose(self, pose: np.ndarray):
        pose = torch.as_tensor(pose, dtype=torch.float32, device=self.device).clone()
        pose[:, :3] += self.org
        self.target.write_root_pose_to_sim(pose)
        self.target.update(self.dt)

    def get_info(self) -> dict[str, np.ndarray]:
        """Lifted past `lift_dist` while at least two contact links press the object with 0.5 N or more."""
        lift = self.target.data.root_pos_w[:, 2] - self.org[:, 2] - self.initial_z
        forces = torch.stack(
            [sensor.get_contact_force_matrix(self.dt)[:, 0].norm(dim=-1) for sensor in self.sensors], 1
        )
        contacts = (forces >= 0.5).sum(dim=1)
        lifted, grasped = lift > self.lift_dist, contacts >= 2
        info = {"success": lifted & grasped, "is_lifted": lifted, "is_strict_grasped": grasped, "lift_dz": lift}
        return {key: value.cpu().numpy() for key, value in info.items()}

    def capture(self) -> list[CameraData]:
        """Render every env with the rigs, one round of envs at a time; returns each camera's data for all envs."""
        poses = []
        for row in self.camera_rows:
            local = torch.tensor(row["pose"], dtype=torch.float32, device=self.device).repeat(self.n, 1)
            if row["mount"] is None:
                poses.append((local[:, :3] + self.org, local[:, 3:]))
            else:
                body = self.robot.data.body_pose_w[:, self.robot.body_names.index(getValidPrimName(row["mount"]))]
                poses.append((body[:, :3] + quat_apply(body[:, 3:], local[:, :3]), quat_mul(body[:, 3:], local[:, 3:])))
        rounds = []
        for start in range(0, self.n, self.rigs):
            visit = torch.arange(start, start + self.rigs, device=self.device).clamp(max=self.n - 1)
            for camera, (positions, rotations) in zip(self.cameras, poses):
                camera.set_world_poses(positions[visit], rotations[visit], convention="world")
            self.sim.forward()
            rep.orchestrator.step(rt_subframes=3, delta_time=0.0, pause_timeline=False, wait_for_render=True)
            for camera in self.cameras:
                camera.update(self.dt, force_recompute=True)
            rounds.append([copy.deepcopy(camera.data) for camera in self.cameras])
        return [
            CameraData(
                pos_w=torch.cat([views[c].pos_w for views in rounds])[: self.n],
                quat_w_world=torch.cat([views[c].quat_w_world for views in rounds])[: self.n],
                image_shape=rounds[0][c].image_shape,
                intrinsic_matrices=torch.cat([views[c].intrinsic_matrices for views in rounds])[: self.n],
                output={
                    key: torch.cat([views[c].output[key] for views in rounds])[: self.n] for key in rounds[0][c].output
                },
                info=[info for views in rounds for info in views[c].info][: self.n],
            )
            for c in range(len(self.cameras))
        ]

    def observe(self) -> dict[str, np.ndarray]:
        """Per env/camera: RGB at requested intrinsics; env-frame workspace points without the robot, target-flagged."""
        captured = self.capture()
        rgb, points, targets, counts, cameras = [], [], [], [], []
        for i in range(self.n):
            for data, row in zip(captured, self.camera_rows):
                height, width = data.image_shape
                intrinsic = torch.tensor(row["intrinsic"], dtype=torch.float32, device=self.device)
                actual = data.intrinsic_matrices[i]
                v_grid, u_grid = torch.meshgrid(
                    torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij"
                )
                u_source = (u_grid - intrinsic[0, 2]) * actual[0, 0] / intrinsic[0, 0] + actual[0, 2]
                v_source = (v_grid - intrinsic[1, 2]) * actual[1, 1] / intrinsic[1, 1] + actual[1, 2]
                grid = torch.stack((2 * u_source / (width - 1) - 1, 2 * v_source / (height - 1) - 1), dim=-1)[None]

                def remap(image: torch.Tensor, mode: str, grid: torch.Tensor = grid) -> torch.Tensor:
                    sampled = torch.nn.functional.grid_sample(
                        image.permute(2, 0, 1)[None].float(), grid, mode=mode, align_corners=True
                    )
                    return sampled[0].permute(1, 2, 0)

                rgb.append(remap(data.output["rgb"][i, ..., :3], "bilinear").round().clamp(0, 255).byte())
                segment = remap(data.output["semantic_segmentation"][i], "nearest").int()
                depth = remap(data.output["distance_to_image_plane"][i], "nearest")[..., 0]
                labels = data.info[i]["semantic_segmentation"]["idToLabels"]
                robot, target = (
                    (
                        segment[:, :, None]
                        == torch.tensor(
                            [literal_eval(key) for key, label in labels.items() if label.get("class") == name],
                            dtype=segment.dtype,
                            device=self.device,
                        ).reshape(1, 1, -1, 4)
                    )
                    .all(dim=-1)
                    .any(dim=-1)
                    for name in (f"robot_{i}", f"target_{i}")
                )
                v, u = torch.where(torch.isfinite(depth) & (depth > 0) & ~robot)
                z = depth[v, u]
                xyz = torch.stack(
                    (
                        (torch.round(u_source[v, u]) + 0.5 - actual[0, 2]) * z / actual[0, 0],
                        (torch.round(v_source[v, u]) + 0.5 - actual[1, 2]) * z / actual[1, 1],
                        z,
                    ),
                    dim=1,
                )
                world = quat_apply(data.quat_w_ros[i].expand(len(xyz), -1), xyz) + data.pos_w[i] - self.org[i]
                bounds = self.bounds + self.tables[i, :, None]
                keep = ((world >= bounds[:, 0]) & (world <= bounds[:, 1])).all(dim=1)
                points.append(world[keep])
                targets.append(target[v, u][keep])
                counts.append(int(keep.sum()))
                cameras.append(torch.cat((data.pos_w[i] - self.org[i], data.quat_w_world[i])))
        v = len(self.camera_rows)
        return {
            "rgb": torch.stack(rgb).reshape(self.n, v, *rgb[0].shape).cpu().numpy(),
            "points": torch.cat(points).cpu().numpy(),
            "target": torch.cat(targets).cpu().numpy(),
            "counts": np.asarray(counts).reshape(self.n, v),
            "cameras": torch.stack(cameras).reshape(self.n, v, 7).cpu().numpy(),
        }

    def run(self, qpos: np.ndarray, record: bool, forces: np.ndarray | None) -> dict[str, np.ndarray]:
        """Stream full-joint targets (n, T, J), optionally pushing the objects with world forces (T, n, 3)."""
        measured, poses = [], []
        for step in range(qpos.shape[1]):
            if forces is not None:
                force = torch.as_tensor(forces[step], dtype=torch.float32, device=self.device)[:, None]
                self.target.set_external_force_and_torque(force, torch.zeros_like(force), is_global=True)
                self.external_force = True
            self.step_qpos(qpos[:, step])
            if record:
                measured.append(self.get_qpos())
                poses.append(self.get_object_pose())
        result = self.get_info()
        if record:
            result.update(qpos=np.stack(measured, 1), obj_pose=np.stack(poses, 1))
        return result

    def close(self):
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
