import torch
from robokit.xform.utils import matrix_to_rotation_6d
from torch import nn
from torch.nn import functional as F

from graspkit.configs.robot_config import RobotConfig
from graspkit.models.cvae import CVAE
from graspkit.models.flow_matching import FlowMatching
from graspkit.robot.grasp_robot import GraspRobot


class GraspEvaluator(nn.Module):
    """Learned grasp evaluator: success logits from visual features, fingertip positions, palm rotation and joints."""

    def __init__(
        self,
        feature_dim: int,
        robot_config: RobotConfig,
        num_queries: int,
        local_features: bool = False,
        anchor_state: bool = False,
        hand_joint_features: bool = False,
        mask_constant_features: bool = True,
    ):
        super().__init__()
        self.robot_config, self.num_queries = robot_config, num_queries
        self.local_features = local_features
        self.anchor_state = anchor_state
        self.hand_joint_features = hand_joint_features
        self.mask_constant_features = mask_constant_features
        assert not anchor_state or (local_features and robot_config.base_type == "fixed")
        robot = GraspRobot.from_config(robot_config)
        joint_dim = len(robot.hand_joint_indices) if hand_joint_features else robot.num_actuated_joints
        phase_dim = 3 * len(robot_config.fingertip_link_names) + 6 + joint_dim
        input_dim = feature_dim + (num_queries + int(anchor_state)) * phase_dim
        self.register_buffer("center", torch.zeros(input_dim))
        self.register_buffer("deviation", torch.ones(input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        if anchor_state:
            nn.init.zeros_(self.network[0].weight[:, -phase_dim:])
        if local_features:
            self.token_projection = nn.Linear(feature_dim, 256)
            self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Use the same robot FK features for replay training, ranking, and latent refinement."""
        robot = GraspRobot.from_config(self.robot_config, device=action.device)
        phase = action.unflatten(-1, (self.num_queries, -1))
        poses = robot.forward_kinematics_torch(phase)
        joints = phase[..., -robot.num_actuated_joints :].clamp(
            robot.actuated_joint_limits_torch[:, 0], robot.actuated_joint_limits_torch[:, 1]
        )
        if self.hand_joint_features:
            joints = joints[..., robot.hand_joint_indices]
        tips = [robot.spec.link_names.index(name) for name in self.robot_config.fingertip_link_names]
        palm = robot.spec.link_names.index(self.robot_config.palm_link_name)
        return torch.cat(
            (
                poses[..., tips, :3, 3].flatten(-2),
                matrix_to_rotation_6d(poses[..., palm, :3, :3]),
                joints,
            ),
            -1,
        ).flatten(-2)

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        # Floored deviations represent constant replay features, not learned variation.
        normalized = (inputs - self.center) / self.deviation
        if self.mask_constant_features:
            normalized = normalized.masked_fill(self.deviation <= 1e-4, 0)
        return normalized

    def forward(
        self,
        condition: torch.Tensor,
        action: torch.Tensor,
        tokens: torch.Tensor | None = None,
        encoded_action: bool = False,
        anchor_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (tokens is not None) == self.local_features
        assert (anchor_state is not None) == self.anchor_state
        inputs = torch.cat([condition, action if encoded_action else self.encode_action(action)], -1)
        if self.anchor_state:
            assert anchor_state.shape == (len(inputs), 1, self.center.numel() - inputs.shape[-1])
            inputs = torch.cat((inputs, anchor_state.expand(-1, inputs.shape[1], -1)), dim=-1)
        inputs = self.normalize(inputs)
        if not self.local_features:
            return self.network(inputs)
        assert inputs.ndim == tokens.ndim == 3 and len(inputs) == len(tokens) and tokens.shape[1] == 64
        query = self.network[0](inputs)
        keys = self.token_projection(tokens)
        context = self.attention(query, keys, keys, need_weights=False)[0]
        return self.network[1:](query + context)

    def refine(
        self,
        generator: CVAE | FlowMatching,
        latent: torch.Tensor,
        condition: torch.Tensor,
        steps: int,
        radius: float,
        tokens: torch.Tensor | None = None,
        anchor_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Refine one latent draw inside an L2 ball, without resampling or simulation."""
        assert not self.training and steps > 0 and radius > 0
        assert (tokens is not None) == self.local_features
        assert (anchor_state is not None) == self.anchor_state
        # Callers may use inference_mode; normal tensors are needed for latent-only gradients.
        with torch.inference_mode(False), torch.enable_grad():
            origin, condition = latent.detach().clone(), condition.detach().clone()
            if self.local_features:
                tokens = tokens.detach().clone()
            if self.anchor_state:
                anchor_state = anchor_state.detach().clone()
            value = origin
            for _ in range(steps):
                value = value.detach().requires_grad_(True)
                action = generator.decode(value, condition)
                score = (
                    self(
                        condition.unflatten(0, (len(tokens), -1)),
                        action.unflatten(0, (len(tokens), -1)),
                        tokens,
                        anchor_state=anchor_state,
                    )
                    if self.local_features
                    else self(condition, action)
                )
                gradient = torch.autograd.grad(score.sum(), value)[0]
                displacement = value.detach() + radius / 5 * F.normalize(gradient, dim=-1) - origin
                value = origin + displacement * (radius / displacement.norm(dim=-1, keepdim=True).clamp_min(radius))
            with torch.no_grad():
                return generator.decode(value, condition)
