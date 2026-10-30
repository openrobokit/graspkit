import os
import random

import numpy as np
import torch


# Ref: https://github.com/open-mmlab/mmcv/blob/bfdd1f9c4281316f4f654672f0e9dbe1b0624ff0/mmcv/runner/utils.py#L74
def seed_everything(seed: int, deterministic: bool = False):
    """Seed all random number generators.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Set CUDNN deterministic mode (cudnn.deterministic=True, benchmark=False); default False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
