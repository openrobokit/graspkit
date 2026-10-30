# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
"""FingertipDistanceTask: one residual per fingertip link, its closest collision sphere's gap to the object."""

from typing import Literal, Optional

import numpy as np
import warp as wp
from robokit.geom.scene import WarpScene
from robokit.opt.var_values import VarValues
from robokit.terms.dense.scene_collision_task import SceneCollisionTask
from robokit.terms.task import ResidualTask

from graspkit.robot.grasp_robot import GraspRobot


@wp.kernel
def _argmin_kernel(residuals: wp.array2d(dtype=wp.float32), out: wp.array1d(dtype=wp.int32)):
    b = wp.tid()
    best = int(0)
    for j in range(1, residuals.shape[1]):
        if residuals[b, j] < residuals[b, best]:
            best = j
    out[b] = best


@wp.kernel
def _gather_residual_kernel(
    residuals: wp.array2d(dtype=wp.float32),
    argmin: wp.array1d(dtype=wp.int32),
    row: int,
    out: wp.array2d(dtype=wp.float32),
):
    b = wp.tid()
    out[b, row] = residuals[b, argmin[b]]


@wp.kernel
def _gather_jacobian_kernel(
    jacobian: wp.array3d(dtype=wp.float32),
    argmin: wp.array1d(dtype=wp.int32),
    row: int,
    out: wp.array3d(dtype=wp.float32),
):
    b, d = wp.tid()
    out[b, row, d] = jacobian[b, argmin[b], d]


class FingertipDistanceTask(ResidualTask):
    """`weight * min_i |sdf(c_i) - r_i|` over one fingertip link's spheres, so only its closest sphere must touch."""

    def __init__(
        self,
        robot: GraspRobot,
        scene: WarpScene,
        scene_indices: wp.array,
        link_name: str,
        weight: float,
        residual_mode: Literal["abs", "sqrt_abs"] = "abs",
        residual_eps: float = 1e-6,
    ):
        link = robot.spec.link_names.index(link_name)
        spheres = np.flatnonzero(robot.spec.collision_spheres_link_indices == link).astype(np.int32)
        assert len(spheres) > 0, f"fingertip link {link_name!r} carries no collision spheres"
        self.link_name = link_name
        self.spheres = SceneCollisionTask(
            robot=robot,
            scene=scene,
            scene_indices=scene_indices,
            sphere_indices=spheres,
            weight=weight,
            penalty="surface_distance",
            residual_mode=residual_mode,
            residual_eps=residual_eps,
        )
        self._residuals: Optional[wp.array] = None
        self._jacobian: Optional[wp.array] = None
        self._argmin: Optional[wp.array] = None

    @property
    def cost_name(self) -> str:
        return f"fingertip/{self.link_name}"

    @property
    def residual_dim(self) -> int:
        return 1

    def precompute(self, var_values: VarValues, need_gradient: bool = False):
        self.spheres.precompute(var_values, need_gradient)

    def _closest(self, var_values: VarValues) -> None:
        """Every sphere's residual into `_residuals` (on the tape, so its gradient flows) and the argmin per grasp."""
        b, n, device = var_values.batch_size, self.spheres.residual_dim, var_values.device
        if self._residuals is None:
            self._residuals = wp.zeros((b, n), dtype=wp.float32, device=device, requires_grad=True)
            self._argmin = wp.zeros((b,), dtype=wp.int32, device=device)
        self.spheres.compute_weighted_residual(var_values, out_residual=self._residuals, row_offset=0)
        wp.launch(_argmin_kernel, dim=b, inputs=[self._residuals], outputs=[self._argmin], device=device)

    def compute_weighted_residual(
        self, var_values: VarValues, out_residual: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        self._closest(var_values)
        b, device = var_values.batch_size, var_values.device
        if out_residual is None:
            out_residual, row_offset = wp.zeros((b, 1), dtype=wp.float32, device=device), 0
        wp.launch(
            _gather_residual_kernel,
            dim=b,
            inputs=[self._residuals, self._argmin, row_offset],
            outputs=[out_residual],
            device=device,
        )
        return out_residual

    def compute_weighted_jacobian_analytic(
        self, var_values: VarValues, out_jacobian: Optional[wp.array] = None, row_offset: int = 0
    ) -> wp.array:
        """The argmin sphere's row; reuses the scene query of the `compute_weighted_residual` call on `var_values`."""
        b, dofs, device = var_values.batch_size, var_values.tangent_dim, var_values.device
        if self._jacobian is None:
            self._jacobian = wp.zeros((b, self.spheres.residual_dim, dofs), dtype=wp.float32, device=device)
        if out_jacobian is None:
            out_jacobian, row_offset = wp.zeros((b, 1, dofs), dtype=wp.float32, device=device), 0
        self.spheres.compute_weighted_jacobian_analytic(var_values, out_jacobian=self._jacobian, row_offset=0)
        wp.launch(
            _gather_jacobian_kernel,
            dim=(b, dofs),
            inputs=[self._jacobian, self._argmin, row_offset],
            outputs=[out_jacobian],
            device=device,
        )
        return out_jacobian
