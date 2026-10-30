import os
import subprocess
from typing import Optional

EXCLUDED_GPUS = {int(gpu_id) for gpu_id in os.environ.get("GRASPKIT_EXCLUDED_GPUS", "").split(",") if gpu_id}


def resolve_gpus(gpus: Optional[str], num_gpus: int, gpu_memory_threshold_mb: int) -> list[int]:
    if gpus is not None:
        return [int(gpu_id) for gpu_id in gpus.split(",") if int(gpu_id) not in EXCLUDED_GPUS]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    free_gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        gpu_id = int(parts[0].strip())
        if int(parts[1].strip()) < gpu_memory_threshold_mb and gpu_id not in EXCLUDED_GPUS:
            free_gpus.append(gpu_id)
    assert len(free_gpus) >= num_gpus, (
        f"Requested {num_gpus} GPUs but only {len(free_gpus)} are free "
        f"(threshold={gpu_memory_threshold_mb}MB). Free GPUs: {free_gpus}"
    )
    return free_gpus[:num_gpus]
