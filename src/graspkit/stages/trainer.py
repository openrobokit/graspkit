import json
import logging
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from graspkit.configs.experiment_config import ExperimentConfig
from graspkit.datasets.grasp_dataset import GraspDataset
from graspkit.datasets.graspkit_data import GraspKitData
from graspkit.envs import BaseEnv, build_env
from graspkit.models.grasp_policy import GraspPolicy, build_grasp_policy
from graspkit.scene.scene_description import SceneDescription
from graspkit.scene.scene_geom import SceneGeom
from graspkit.stages.grasp_filters import compute_penetration_depth
from graspkit.stages.grasp_traj_build import GraspTrajBuilder
from graspkit.utils.io_utils import ensure_local_filepath

logger = logging.getLogger("graspkit")


class Trainer:
    """Trains a `GraspPolicy` on `GraspDataset` batches, with DDP, checkpoints and periodic simulation evaluation."""

    def __init__(self, exp_cfg: ExperimentConfig, policy_cls: Callable[..., GraspPolicy] = GraspPolicy):
        self.cfg = exp_cfg.trainer
        self.exp_cfg = exp_cfg
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

        if self.local_rank == 0:
            if self.cfg.track:
                import wandb

                wandb.init(
                    project=self.cfg.wandb_project_name,
                    sync_tensorboard=True,
                    config=vars(self.cfg),
                    name=exp_cfg.name,
                    save_code=True,
                )

            self.writer = SummaryWriter(f"logs/train/{exp_cfg.name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n{}".format(
                    "\n".join([f"|{key}|{value}|" for key, value in vars(self.cfg).items()])
                ),
            )

            # Eval runs through the traj builder: its config decides the base (floating or IK-fixed) used below.
            self.traj_builder: GraspTrajBuilder | None = None
            self.evaluator: BaseEnv | None = None
        if exp_cfg.agent.enable_penetration_loss:
            self.scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(exp_cfg.scene_description_path)

        # agent setup
        self.agent = build_grasp_policy(exp_cfg.agent, exp_cfg.robot_config, self.device, policy_cls)
        if self.exp_cfg.ckpt_path is not None:
            if self.local_rank == 0:
                logger.info(f"Loading agent from {self.exp_cfg.ckpt_path}")
            self.agent.load_state_dict(
                torch.load(ensure_local_filepath(self.exp_cfg.ckpt_path), map_location=self.device)["agent"]
            )
        if dist.is_initialized():
            self.agent = DDP(self.agent, device_ids=[self.local_rank], output_device=self.local_rank)

        # optimizer setup
        param_dicts = [{"params": [p for n, p in self.agent.named_parameters() if p.requires_grad]}]
        fused = self.cfg.fast_step and self.device.type == "cuda"
        if fused:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.optimizer = optim.AdamW(param_dicts, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, fused=fused)

        # LR drop by a factor of 10 after lr_drop iters
        lr_drop = int((2 / 3) * self.cfg.total_iters)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_drop)

    def _save_checkpoint(self, filename: str):
        module_to_save = self.agent.module if dist.is_initialized() else self.agent
        os.makedirs(f"logs/train/{self.exp_cfg.name}/checkpoints", exist_ok=True)
        torch.save({"agent": module_to_save.state_dict()}, f"logs/train/{self.exp_cfg.name}/checkpoints/{filename}")

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader | None = None,
        checkpoint_dir: Path | None = None,
        metadata: dict | None = None,
        resume: dict | None = None,
        publish: Callable[[Path], None] | None = None,
    ):
        """Train normally, or save resumable single-device milestones and publish them."""
        first = 0
        measured = time.monotonic()
        if checkpoint_dir is not None:
            assert not dist.is_initialized() and self.cfg.evaluator is None and metadata is not None
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            sampler = train_dataloader.batch_sampler.batch_sampler.sampler
            assert sampler.generator is not None
            if resume is not None:
                self.agent.load_state_dict(resume["agent"], strict=True)
                self.optimizer.load_state_dict(resume["optimizer"])
                self.lr_scheduler.load_state_dict(resume["scheduler"])
                first = resume["iteration"] + 1
                torch.set_rng_state(resume["torch_rng"].cpu())
                if self.device.type == "cuda":
                    torch.cuda.set_rng_state(resume["cuda_rng"].cpu(), self.device)
                random.setstate(resume["python_rng"])
                state = resume["numpy_rng"]
                np.random.set_state((state[0], state[1].cpu().numpy().astype(np.uint32), *state[2:]))
                sampler.generator.set_state(resume["sampler_rng"].cpu())
        self.agent.train()
        best_eval_metrics = defaultdict(float)
        if self.local_rank == 0 and checkpoint_dir is None:
            self._save_checkpoint("initial.pt")

        for iteration, data_batch in enumerate(train_dataloader, first):
            # copy data from cpu to gpu
            obs_batch_dict = {k: v.to(self.device, non_blocking=True) for k, v in data_batch["observations"].items()}
            act_batch = data_batch["actions"].to(self.device, non_blocking=True)

            # Rebuild the (expensive) penetration target geometry every penetration_every steps, else clear it.
            if self.exp_cfg.agent.enable_penetration_loss:
                module = self.agent.module if dist.is_initialized() else self.agent
                if (
                    iteration >= self.exp_cfg.agent.penetration_warmup_iters
                    and iteration % self.exp_cfg.agent.penetration_every == 0
                ):
                    module._scene_geom = SceneGeom(
                        [self.scene_descriptions[s_id] for s_id in data_batch["metas"]["scene_id"]], device=self.device
                    )
                else:
                    module._scene_geom = None

            # forward and compute loss
            loss_dict = self.agent(obs_batch_dict, act_batch)
            total_loss = loss_dict["loss"]

            # backward
            self.optimizer.zero_grad()
            total_loss.backward()

            found_nan = False
            if self.cfg.fast_step or checkpoint_dir is not None:
                # A non-finite norm also catches overflow in otherwise finite gradient entries.
                grads = [p.grad for p in self.agent.parameters() if p.grad is not None]
                norms = torch._foreach_norm(grads)
                found_nan = not bool(torch.isfinite(torch.stack(norms)).all())
            else:
                for p in self.agent.parameters():  # NOTE: I do not understand why, but this is necessary
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        found_nan = True
                        break
            if dist.is_initialized():
                found_nan_tensor = torch.tensor(1.0 if found_nan else 0.0, device=self.device)
                dist.all_reduce(found_nan_tensor, op=dist.ReduceOp.MAX)
                found_nan = found_nan_tensor > 0.5
            if checkpoint_dir is not None and (found_nan or not torch.isfinite(total_loss)):
                diagnostic = {
                    "iteration": iteration,
                    "losses": {key: str(float(value.detach())) for key, value in loss_dict.items()},
                    "nonfinite_gradients": [
                        name
                        for name, parameter in self.agent.named_parameters()
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    ],
                    "gradient_norms": {
                        name: str(float(norm))
                        for name, norm in zip(
                            [name for name, p in self.agent.named_parameters() if p.grad is not None],
                            norms,
                            strict=True,
                        )
                    },
                }
                (checkpoint_dir / "gradient_failure.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
                torch.save(
                    {**metadata, "agent": self.agent.state_dict(), "batch": data_batch},
                    checkpoint_dir / "gradient_failure.pt",
                )
                if publish is not None:
                    publish(checkpoint_dir)
                raise FloatingPointError(json.dumps(diagnostic))
            if not found_nan:
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.parameters(),
                        self.cfg.grad_clip_norm,
                        error_if_nonfinite=checkpoint_dir is not None,
                    )
                self.optimizer.step()
                self.lr_scheduler.step()  # step lr scheduler every batch, unlike standard pytorch behavior
            else:
                logger.warning(f"NaN gradient detected at iteration {iteration}. Skipping optimizer step.")
                self.optimizer.zero_grad()  # clear grads if NaN detected

            # record training metrics
            if iteration % self.cfg.log_freq == 0:
                if dist.is_initialized():
                    for k, v in loss_dict.items():
                        dist.all_reduce(v, op=dist.ReduceOp.AVG)
                if self.local_rank == 0:
                    loss_dict_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                    logger.info(f"Iteration {iteration}, {loss_dict_str}")
                    if self.cfg.track:
                        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], iteration)
                        for k, v in loss_dict.items():
                            self.writer.add_scalar(f"losses/{k}", v.item(), iteration)

            if checkpoint_dir is not None:
                if iteration % self.cfg.log_freq == 0 or iteration == self.cfg.total_iters:
                    row = {
                        "iteration": iteration,
                        "loss": float(total_loss.detach()),
                        "loss_components": {key: float(value.detach()) for key, value in loss_dict.items()},
                        "seconds": time.monotonic() - measured,
                        "completed_updates": iteration + 1,
                    }
                    with (checkpoint_dir / "training.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                if (iteration + 1) % self.cfg.save_freq == 0 or iteration == self.cfg.total_iters:
                    state = np.random.get_state()
                    checkpoint = {
                        **metadata,
                        "agent": self.agent.state_dict(),
                        "iteration": iteration,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.lr_scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                        "numpy_rng": (state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]),
                        "python_rng": random.getstate(),
                        "sampler_rng": sampler.generator.get_state(),
                    }
                    latest = checkpoint_dir / "latest.pt"
                    torch.save(checkpoint, latest.with_suffix(".part"))
                    latest.with_suffix(".part").replace(latest)
                    pointer = checkpoint_dir / "latest.json"
                    pointer.with_suffix(".part").write_text(
                        json.dumps(
                            {
                                "updates": iteration + 1,
                                "checkpoint": f"resume/iter_{iteration}.pt",
                            }
                        )
                        + "\n"
                    )
                    pointer.with_suffix(".part").replace(pointer)
                    if publish is not None:
                        publish(checkpoint_dir)

            # skip iter 0: random/loaded weights, and real-only runs may lack the ManiSkill evaluator's scene geometry
            if self.cfg.evaluator is not None and iteration > 0 and iteration % self.cfg.eval_freq == 0:
                if self.local_rank == 0:
                    eval_results = self.evaluate(test_dataloader)
                    sim_success = eval_results["success"]
                    sr = sim_success.sum() / sim_success.numel()
                    pen_pass = eval_results["pen_pass"]
                    pen_rate = pen_pass.sum() / pen_pass.numel()
                    logger.info(f"Eval success rate: {sr:.2%} | pen pass rate: {pen_rate:.2%}")
                    with open(f"logs/train/{self.exp_cfg.name}/eval_success.jsonl", "a") as f:
                        f.write(
                            json.dumps({"iter": iteration, "success_rate": float(sr), "pen_pass_rate": float(pen_rate)})
                            + "\n"
                        )
                    if self.cfg.track:
                        self.writer.add_scalar("eval/success_rate", sr, iteration)
                        self.writer.add_scalar("eval/pen_pass_rate", pen_rate, iteration)
                    if sr > best_eval_metrics["success_rate"]:
                        best_eval_metrics["success_rate"] = sr
                        self._save_checkpoint("best_eval_success_rate.pt")
                if dist.is_initialized():
                    dist.barrier()

            # save checkpoint
            if checkpoint_dir is None and iteration % self.cfg.save_freq == 0:
                if self.local_rank == 0:
                    self._save_checkpoint(f"iter_{iteration}.pt")
                if dist.is_initialized():
                    dist.barrier()

    def train_graph(self, dataset: GraspDataset, index_loader: DataLoader):
        """`train` for `TrainerConfig.cuda_graph`: the same updates, with the step captured once as a CUDA graph."""
        import gc

        import warp as wp

        assert (
            not dist.is_initialized() and self.cfg.evaluator is None and not self.exp_cfg.agent.enable_penetration_loss
        )
        dev = self.device
        b = dataset.to_device(dev)
        n_points = b["points"].shape[2]
        params = [p for p in self.agent.parameters() if p.requires_grad]
        lr = torch.tensor(self.cfg.lr, device=dev)
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=self.cfg.weight_decay, fused=True, capturable=True)
        lr_drop = int((2 / 3) * self.cfg.total_iters)
        batch = self.cfg.batch_size
        s_entry = torch.zeros(batch, dtype=torch.long, device=dev)
        s_draw = torch.zeros(batch, dtype=torch.long, device=dev)
        s_rows = torch.zeros(batch, dataset.cfg.num_actions, dtype=torch.long, device=dev)

        def step():
            r = b["render"][s_entry]
            pts = b["points"][r, s_draw].float() + b["center"][r, s_draw][:, None]
            obs = {"pointcloud": pts, "objectness": b["target_id"][r][:, None, None].expand(-1, n_points, 1)}
            out = self.agent(obs, b["flat_actions"][s_rows])
            out["loss"].backward()
            grads = [p.grad for p in params]
            self.optimizer.found_inf = (~torch.isfinite(torch.stack(torch._foreach_norm(grads))).all()).float()
            self.optimizer.grad_scale = None
            self.optimizer.step()
            return out

        def load(data):
            s_entry.copy_(data["entry"], non_blocking=True)
            s_draw.copy_(data["draw"], non_blocking=True)
            s_rows.copy_(data["rows"], non_blocking=True)

        self.agent.train()
        if self.local_rank == 0:
            self._save_checkpoint("initial.pt")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        loader = iter(index_loader)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graph, out, iteration = torch.cuda.CUDAGraph(), None, 0
        with wp.ScopedStream(wp.stream_from_torch(side)):  # warp's kernels on the capture stream
            with torch.cuda.stream(side):
                for _ in range(3):  # warm-up updates (counted): allocations and lazy setup happen outside the capture
                    load(next(loader))
                    self.optimizer.zero_grad(set_to_none=True)
                    out = step()
                    self._graph_bookkeeping(iteration, out, lr, lr_drop)
                    iteration += 1
            torch.cuda.current_stream().wait_stream(side)
            self.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            # relaxed: warp frees host-side bookkeeping on its own stream while the step is captured
            with torch.cuda.graph(graph, stream=side, capture_error_mode="relaxed"):
                out = step()
        for data in loader:
            if iteration > self.cfg.total_iters:
                break
            load(data)
            graph.replay()
            self._graph_bookkeeping(iteration, out, lr, lr_drop)
            iteration += 1

    def _graph_bookkeeping(self, iteration: int, loss_dict: dict, lr: torch.Tensor, lr_drop: int):
        if iteration + 1 == lr_drop:  # StepLR(step_size=lr_drop, gamma 0.1), stepped after each update
            lr.mul_(0.1)
        if iteration % self.cfg.log_freq == 0 and self.local_rank == 0:
            logger.info(f"Iteration {iteration}, " + ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items()))
            if self.cfg.track:
                self.writer.add_scalar("charts/learning_rate", float(lr), iteration)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f"losses/{k}", v.item(), iteration)
        if iteration % self.cfg.save_freq == 0 and self.local_rank == 0:
            self._save_checkpoint(f"iter_{iteration}.pt")

    @torch.no_grad()
    def evaluate(self, test_dataloader: DataLoader, num_samples_per_scene: int = 1):
        assert self.cfg.evaluator is not None
        if self.traj_builder is None:
            self.cfg.evaluator.num_envs = (
                self.cfg.eval_num_envs if self.cfg.eval_only else min(self.cfg.batch_size, self.cfg.eval_num_envs)
            )
            self.scene_descriptions = SceneDescription.load_scene_descriptions_msgpack(
                self.exp_cfg.scene_description_path
            )
            self.traj_builder = GraspTrajBuilder(self.exp_cfg.traj_builder, self.device)
            self.evaluator = build_env(self.cfg.evaluator, self.traj_builder.robot.config, self.device)
            torch.cuda.set_device(self.local_rank)
        self.agent.eval()
        module = self.agent.module if dist.is_initialized() else self.agent
        eval_results = defaultdict(list)
        scene_ids = []
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= self.cfg.max_eval_batches:
                logger.info(f"Reached max eval batches: {self.cfg.max_eval_batches}")
                break
            obs_batch_dict = batch["observations"]
            obs_batch_dict = {k: v.to(self.device, non_blocking=True) for k, v in obs_batch_dict.items()}
            batch_scene_descriptions = [self.scene_descriptions[s_id] for s_id in batch["metas"]["scene_id"]]

            batch_actions = module.get_action(obs_batch_dict, num_samples_per_scene=num_samples_per_scene)
            eval_batch = GraspKitData(
                scene_ids=[d.scene_id for d in batch_scene_descriptions],
                scene_descriptions=batch_scene_descriptions,
                grasp_qpos=batch_actions[..., 1, :],
                pre_grasp_qpos=batch_actions[..., 0, :],
                post_grasp_qpos=batch_actions[..., 2, :],
            )
            eval_batch, _ = self.traj_builder.build(eval_batch)
            batch_eval_results = self.evaluator.evaluate(eval_batch)
            pen = compute_penetration_depth(batch_scene_descriptions, eval_batch.grasp_qpos, self.traj_builder.robot)
            batch_eval_results["penetration"] = pen
            batch_eval_results["pen_pass"] = pen <= self.cfg.eval_pen_threshold
            for k, v in batch_eval_results.items():
                eval_results[k].append(v)
            scene_ids.extend(batch["metas"]["scene_id"])
        self.agent.train()
        ret_dict = {k: torch.cat(v) for k, v in eval_results.items()}
        ret_dict["scene_ids"] = scene_ids
        return ret_dict
