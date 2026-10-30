import logging
import os
from pathlib import Path

logger = logging.getLogger("graspkit")


def write_completion_marker(
    marker_path: str,
    world_size: int | None = None,
    rank: int | None = None,
) -> None:
    if not marker_path or marker_path.startswith("s3://"):
        return
    if world_size is not None and world_size > 1:
        assert rank is not None
        per_rank = f"{marker_path}_w{world_size}_r{rank}"
        Path(per_rank).touch()
        logger.info(f"Wrote per-rank marker: {per_rank}")
        all_done = all(os.path.exists(f"{marker_path}_w{world_size}_r{r}") for r in range(world_size))
    else:
        all_done = True
    if all_done:
        Path(marker_path).touch()
        logger.info(f"Wrote completion marker: {marker_path}")
