import os
import random

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class IterationBasedBatchSampler(Sampler):
    """Resample a batch sampler until the requested iteration count."""

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # Set the epoch so distributed samplers change their data split.
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """Seed a PyTorch data-loader worker."""
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
