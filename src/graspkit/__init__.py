import logging
import os
import warnings

# Thread limits for numpy/BLAS; must be set before any numpy-touching import, so scripts should import graspkit first.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")

import dotenv

dotenv.load_dotenv()

import rich
from rich.logging import RichHandler

rich.reconfigure(log_path=False)

# ignore glfw warnings for headless servers
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="CUDA reports that you have.*fork_rng",
    module="torch.random",
)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Failed to find Vulkan ICD file")

logger = logging.getLogger("graspkit")
logger.propagate = False
logger.setLevel("INFO")
if not logger.handlers:
    logger.addHandler(RichHandler(console=rich.get_console(), show_path=False, log_time_format="[%X]"))

from graspkit.constants import link  # noqa: E402  re-export so `graspkit.link()` works

__all__ = ["link"]
