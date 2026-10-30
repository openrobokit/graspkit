import logging
from contextlib import contextmanager
from typing import Iterator

import torch

logger = logging.getLogger("graspkit")


@contextmanager
def cuda_timeit(name: str) -> Iterator[None]:
    """Time a block of code that may involve CUDA ops, using CUDA events for accurate measurement."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    yield
    end_event.record()
    torch.cuda.synchronize()
    logger.info(f"{name} costs {start_event.elapsed_time(end_event) / 1e3:.5f} s")
