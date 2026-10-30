"""Shared staged Viser interface for sim and real deployment."""

import threading
import time
from typing import Callable

import numpy as np
import torch
import trimesh
import viser
import yourdfpy
from robokit.xform.warp.torch_wrappers import euler_angles_to_matrix, matrix_to_quaternion
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.types import Device
from viser.extras import ViserUrdf

from graspkit.constants import OBJECT_COLOR, ROBOT_COLOR, TABLE_COLOR
from graspkit.robot.grasp_robot import GraspRobot
from graspkit.scene.scene_description import SceneDescription
from graspkit.utils.image_utils import screen_to_image_pixel

STAGES = ("Capture", "Segment", "Check Grasp/Trajectory", "Deploy", "Release")


class DeployVisualizer:
    def __init__(
        self,
        robot_helper: GraspRobot,
        policy_robot: GraspRobot,
        device: Device,
        scenes: dict[str, SceneDescription],
        auto: bool = False,
        sam_model: str = "facebook/sam2.1-hiera-base-plus",
        observe_cloud: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None,
        read_qpos: Callable[[], torch.Tensor] | None = None,
    ):
        assert device.type == "cuda", "SAM2 deployment requires CUDA"
        self.robot_helper = robot_helper
        self.policy_robot = policy_robot
        self.device = device
        self.auto = auto
        print(f"loading SAM2: {sam_model}")
        self.predictor = SAM2ImagePredictor.from_pretrained(sam_model, device=str(device))
        self.server = viser.ViserServer()
        self.root = self.server.scene.add_frame("/robot", show_axes=False)
        self.root.position = tuple(robot_helper.root_pose_7d[:3].cpu().numpy())
        self.root.wxyz = tuple(robot_helper.root_pose_7d[3:].cpu().numpy())
        config = robot_helper.config
        urdf = yourdfpy.URDF.load(
            config.urdf_or_mjcf_path,
            build_scene_graph=True,
            load_meshes=True,
            **({"mesh_dir": config.mesh_dir} if config.mesh_dir else {}),
        )
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot", mesh_color_override=ROBOT_COLOR)
        full = GraspRobot.from_config(config, device=device, prune=False)
        init = full.init_qpos.cpu().numpy()
        self.urdf_init = np.array(
            [init[full.actuated_joint_names.index(n)] for n in urdf.actuated_joint_names], dtype=np.float32
        )
        self.chain_to_urdf = np.array([urdf.actuated_joint_names.index(n) for n in robot_helper.actuated_joint_names])

        scene_ids = list(scenes)
        self.scenes = scenes
        self.scene_dropdown = self.server.gui.add_dropdown("Scene", scene_ids, initial_value=scene_ids[0])
        self.scene_dropdown.on_update(lambda _e: self.show_scene(self.scenes[self.scene_id]))
        self.rgb = self.server.gui.add_image(np.zeros((1, 1, 3), dtype=np.uint8), label="Captured RGB")
        self.prompt_mode = self.server.gui.add_button_group("SAM2 Prompt", ("Positive", "Negative"), visible=False)
        self.undo_prompt = self.server.gui.add_button("Undo Prompt", visible=False)
        self.clear_prompts = self.server.gui.add_button("Clear Prompts", visible=False)
        self.submit_mask = self.server.gui.add_button("Submit Mask", disabled=True, visible=False)
        self.undo_prompt.on_click(self._undo_prompt)
        self.clear_prompts.on_click(self._clear_prompts)
        self.submit_mask.on_click(self._submit_mask)
        self.traj_slider = self.server.gui.add_slider("Trajectory Step", min=0, max=1, step=1, initial_value=0)
        self.traj_slider.on_update(lambda _e: self._on_slider())
        self.grasp_slider = self.server.gui.add_slider("Grasp", min=0, max=1, step=1, initial_value=0, visible=False)
        self.grasp_slider.on_update(lambda _e: self._on_grasp())
        self.observe_cloud = observe_cloud
        self.read_qpos = read_qpos
        with self.server.gui.add_folder("Calibration check"):
            self.show_observed = self.server.gui.add_checkbox("Show observed cloud", initial_value=False)
            self.show_observed.on_update(lambda _e: self._refresh_observed())
            self.refresh_observed = self.server.gui.add_button("Refresh observed cloud")
            self.refresh_observed.on_click(lambda _e: self._refresh_observed())
            # Viser fires this on its own thread, so stay off the GPU: a CUDA op would kill the planner's graph capture.
            self.refresh_qpos = self.server.gui.add_button("Refresh measured qpos")
            self.refresh_qpos.on_click(lambda _e: self._refresh_qpos())
            self.qpos_source = self.server.gui.add_dropdown(
                "Robot qpos", ("measured", "commanded"), initial_value="measured"
            )
            # Redraw on switch: the callback only fires while the arm moves, but you inspect once it has stopped.
            self.qpos_source.on_update(lambda _e: self._redraw_qpos())
        with self.server.gui.add_folder("Deploy"):
            self.deploy_until = self.server.gui.add_slider("Deploy Until Step", min=0, max=1, step=1, initial_value=1)
        self.last_measured: torch.Tensor | None = None
        self.last_commanded: torch.Tensor | None = None
        self.click_event = threading.Event()
        self.clicked_stage = ""
        self.stage_buttons: dict[str, viser.GuiButtonHandle] = {}
        for stage in STAGES:
            self.stage_buttons[stage] = self.server.gui.add_button(stage, disabled=stage != "Capture")
            self.stage_buttons[stage].on_click(lambda _e, name=stage: self._click_stage(name))
        self.traj = self.trajs = self.feasible_qpos = None
        self.deploying = False
        self.capture_rgb: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.prompt_points: list[tuple[int, int]] = []
        self.prompt_labels: list[int] = []
        self.segment_event = threading.Event()
        self.predict_lock = threading.Lock()
        self.show_qpos(robot_helper.init_qpos.to(device))
        self.show_scene(self.scenes[self.scene_id])

    @property
    def scene_id(self) -> str:
        assert self.scene_dropdown is not None
        return self.scene_dropdown.value

    def _on_slider(self) -> None:
        # deploy drives the slider while showing measured qpos; replaying planned qpos here would shake the robot
        if self.traj is not None and not self.deploying:
            self.show_qpos(self.traj[min(int(self.traj_slider.value), self.traj.shape[0] - 1)])

    def _on_grasp(self) -> None:
        k = int(self.grasp_slider.value)
        self.traj = self.trajs[k]
        self.traj_slider.max = self.traj.shape[0] - 1
        self.show_qpos(self.feasible_qpos[k])

    def _click_stage(self, stage: str) -> None:
        for button in self.stage_buttons.values():
            button.disabled = True
        self.scene_dropdown.disabled = True
        self.clicked_stage = stage
        self.click_event.set()

    def _predict_mask(self) -> None:
        assert self.capture_rgb is not None
        self.submit_mask.disabled = True
        if self.prompt_points:
            with self.predict_lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, _, _ = self.predictor.predict(
                    point_coords=np.asarray(self.prompt_points, dtype=np.float32),
                    point_labels=np.asarray(self.prompt_labels, dtype=np.int32),
                    multimask_output=False,
                )
            self.mask = masks[0].astype(bool)
            self.submit_mask.disabled = not self.mask.any()
        else:
            self.mask = np.zeros(self.capture_rgb.shape[:2], dtype=bool)

        preview = self.capture_rgb.copy()
        preview[self.mask] = (0.45 * preview[self.mask] + 0.55 * np.array((130, 200, 230), dtype=np.float32)).astype(
            np.uint8
        )
        radius = max(3, min(preview.shape[:2]) // 100)
        yy, xx = np.ogrid[: preview.shape[0], : preview.shape[1]]
        for (x, y), label in zip(self.prompt_points, self.prompt_labels):
            dot = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            preview[dot] = (40, 220, 40) if label else (240, 40, 40)
        self.rgb.image = preview
        self.server.scene.set_background_image(preview, format="png")

    def _on_pointer(self, event: viser.ScenePointerEvent) -> None:
        assert self.capture_rgb is not None
        self.prompt_points.append(screen_to_image_pixel(event.screen_pos[0], self.capture_rgb.shape[:2]))
        self.prompt_labels.append(int(self.prompt_mode.value == "Positive"))
        self._predict_mask()

    def _undo_prompt(self, _event: viser.GuiEvent) -> None:
        if self.prompt_points:
            self.prompt_points.pop()
            self.prompt_labels.pop()
            self._predict_mask()

    def _clear_prompts(self, _event: viser.GuiEvent) -> None:
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self._predict_mask()

    def _submit_mask(self, _event: viser.GuiEvent) -> None:
        assert self.mask is not None and self.mask.any()
        self.segment_event.set()

    def wait(self, stage: str, abortable: bool = True) -> bool:
        """Block until the user advances; False if they clicked Capture to abandon (pass `abortable=False` mid-move)."""
        self.deploying = False  # paused: hand the slider back to the user for scrubbing
        offer_abort = abortable and stage != "Capture"
        for name, button in self.stage_buttons.items():
            button.disabled = name != stage and not (offer_abort and name == "Capture")
        self.scene_dropdown.disabled = stage != "Capture"
        hint = " (or 'Capture' to abandon this cycle)" if offer_abort else ""
        print(f"viser: click '{stage}' to continue{hint}")
        if self.auto:
            return True
        self.click_event.wait()
        self.click_event.clear()
        return self.clicked_stage == stage

    def show_qpos(self, qpos: torch.Tensor) -> None:
        if self.robot_helper.config.base_type == "floating":
            quat = matrix_to_quaternion(euler_angles_to_matrix(qpos[None, 3:6], "rxyz"))[0]
            self.root.position, self.root.wxyz = tuple(qpos[:3].cpu().numpy()), tuple(quat.cpu().numpy())
            qpos = qpos[6:]
        cfg = self.urdf_init.copy()
        cfg[self.chain_to_urdf] = qpos.cpu().numpy()
        self.urdf_vis.update_cfg(cfg)

    def _refresh_qpos(self) -> None:
        if self.read_qpos is not None:
            self.last_measured = self.read_qpos()[0]
            self._redraw_qpos()

    def _redraw_qpos(self) -> None:
        qpos = self.last_commanded if self.qpos_source.value == "commanded" else self.last_measured
        if qpos is not None:
            self.show_qpos(qpos)

    def _refresh_observed(self) -> None:
        # Re-read joint state: deploy samples it before the arm settles, so a stale model looks like bad calibration.
        self._refresh_qpos()
        self.server.scene.remove_by_name("/observed")
        if not self.show_observed.value or self.observe_cloud is None:
            return
        points, colors = self.observe_cloud()
        self.server.scene.add_point_cloud("/observed", points, colors, point_size=0.002)
        print(f"observed cloud: {points.shape[0]} points")

    def show_capture(self, rgb: np.ndarray) -> None:
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        self.server.scene.remove_by_name("/cloud")  # last cycle's object, stale against this capture
        self.grasp_slider.visible = False
        self.rgb.image = rgb

    def segment(self, rgb: np.ndarray, auto_prompt: tuple[int, int] | None = None) -> np.ndarray:
        self.capture_rgb = np.ascontiguousarray(rgb)
        self.predictor.set_image(self.capture_rgb)
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self.segment_event.clear()
        self.server.scene.set_global_visibility(False)
        for control in (self.prompt_mode, self.undo_prompt, self.clear_prompts, self.submit_mask):
            control.visible = True
        self._predict_mask()
        self.server.scene.on_pointer_event("click")(self._on_pointer)
        if auto_prompt is not None:
            self.prompt_points.append(auto_prompt)
            self.prompt_labels.append(1)
            self._predict_mask()
            self.segment_event.set()
        print("viser: add SAM2 prompts, then click 'Submit Mask'")
        self.segment_event.wait()
        self.server.scene.remove_pointer_callback()
        for control in (self.prompt_mode, self.undo_prompt, self.clear_prompts, self.submit_mask):
            control.visible = False
        self.server.scene.set_background_image(None)
        self.server.scene.set_global_visibility(True)
        assert self.mask is not None and self.mask.any()
        return self.mask.copy()

    def show_scene(self, scene: SceneDescription) -> None:
        self.server.scene.remove_by_name("/scene")
        for actor_id, box_desc in scene.boxes.items():
            box = trimesh.creation.box(box_desc.size)
            self.server.scene.add_mesh_simple(
                f"/scene/{actor_id}",
                np.asarray(box.vertices),
                np.asarray(box.faces),
                color=TABLE_COLOR,
                flat_shading=True,
                position=box_desc.position,
                wxyz=box_desc.quaternion,
            )
        for actor_id, mesh_desc in scene.meshes.items():
            obj_mesh = trimesh.load(mesh_desc.get_visual_mesh_path(), force="mesh", process=False)
            handle = self.server.scene.add_mesh_simple(
                f"/scene/{actor_id}",
                np.asarray(obj_mesh.vertices) * np.asarray(mesh_desc.scales),
                np.asarray(obj_mesh.faces),
                color=OBJECT_COLOR if actor_id == scene.target_actor_id else TABLE_COLOR,
                flat_shading=True,
                position=mesh_desc.position,
                wxyz=mesh_desc.quaternion,
            )
            if actor_id == scene.target_actor_id:
                self.target_mesh = handle

    def show_segment(self, scene: SceneDescription, pts: torch.Tensor) -> None:
        self.traj = self.trajs = None
        self.show_scene(scene)
        self.server.scene.add_point_cloud("/cloud", pts[0].cpu().numpy(), (255, 60, 60), point_size=0.003)

    def show_plan(self, candidates: torch.Tensor, feasible: torch.Tensor, trajs: torch.Tensor) -> None:
        self.feasible_qpos = feasible
        self.trajs = trajs
        self.grasp_slider.max = trajs.shape[0] - 1
        self.grasp_slider.value = 0
        self.grasp_slider.visible = True
        self._on_grasp()
        self.deploy_until.max = self.traj.shape[0] - 1
        self.deploy_until.value = self.traj.shape[0] - 1

    def show_deploy_qpos(
        self,
        step: int,
        qpos: torch.Tensor,
        frame_delay: float,
        obj_pose: torch.Tensor | None = None,
        commanded: torch.Tensor | None = None,
    ) -> None:
        # The stale cloud is kept: where the hand lands against where the object was seen is the calibration check.
        self.deploying = True
        self.traj_slider.value = step
        self.last_measured = qpos
        self.last_commanded = commanded
        if self.qpos_source.value == "commanded" and commanded is not None:
            qpos = commanded
        self.show_qpos(qpos)
        if obj_pose is not None:
            pose = obj_pose.cpu().numpy()
            self.target_mesh.position, self.target_mesh.wxyz = tuple(pose[:3]), tuple(pose[3:])
        self.server.flush()
        time.sleep(frame_delay)

    def close(self) -> None:
        self.server.stop()
