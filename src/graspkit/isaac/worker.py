"""Isaac process behind `IsaacEnv`, on Isaac's Python without graspkit; arrays travel as packets, not NumPy pickles."""

import argparse
import gc
import os
import socket
import sys
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np


def pack_array(value: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    return value.dtype.str, value.shape, value.tobytes()


def unpack_array(packet: tuple[str, tuple[int, ...], bytes]) -> np.ndarray:
    dtype, shape, data = packet
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--usd-cache", required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True,
        enable_cameras=True,
        device="cuda:0",
        create_new_stage=False,
        disable_viewport_updates=True,
        limit_cpu_threads=args.threads,
        renderer="RaytracedLighting",
        kit_args=f"--/plugins/carb.tasking.plugin/threadCount={args.threads} "
        f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={args.threads} --/persistent/physics/numThreads=2",
    ).app
    # Stages are replaced headless, without the editor's USD observers.
    app.app.get_extension_manager().set_extension_enabled_immediate("omni.kit.window.stage", False)
    sys.path.insert(0, str(Path(__file__).parent))
    import h5py
    import torch
    from dex_grasp import IsaacDexGrasp, robot_usd

    torch.backends.cuda.matmul.allow_tf32 = False
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.connect(args.socket)
    connection = Connection(channel.detach())
    scene = None
    while True:
        request = {key: unpack_array(v) if isinstance(v, tuple) else v for key, v in connection.recv().items()}
        operation, result = request["operation"], None
        if operation in ("reset", "evaluate"):
            if scene is not None:  # free the old cameras before the new stage attaches its render products
                scene.close()
                scene = None
                gc.collect()
            with h5py.File(request["job"]) as job:
                scene = IsaacDexGrasp(job, request["indices"], robot_usd(str(job.attrs["urdf"]), args.usd_cache))
                if operation == "evaluate":
                    result = scene.run(job["qpos"][request["indices"]], request["record"], request["forces"])
        elif operation == "step_qpos":
            scene.step_qpos(request["qpos"])
        elif operation == "set_object_pose":
            scene.set_object_pose(request["pose"])
        elif operation in ("get_qpos", "get_object_pose", "get_info", "observe"):
            result = getattr(scene, operation)()
        elif operation == "close":
            if scene is not None:
                scene.close()
            connection.send(None)
            connection.close()
            os._exit(0)  # Kit's headless teardown can hang; the worker owns the whole runtime.
        else:
            raise ValueError(operation)
        if isinstance(result, dict):
            result = {key: pack_array(value) for key, value in result.items()}
        elif result is not None:
            result = pack_array(result)
        connection.send(result)


if __name__ == "__main__":
    main()
