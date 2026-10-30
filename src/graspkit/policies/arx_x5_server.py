"""The ARX X5 open-loop grasp policy (`arx_x5_openloop`) behind a unix socket, for a RoboDojo grasp backend,
which starts it on first use:

    uv run python -m graspkit.policies.arx_x5_server --socket S [--ckpt CKPT] [--ranker RANKER] [--rank R] [--threads N]

CKPT and RANKER default to the published generator and ranker (`graspkit.assets.policies.skill_policy_paths`). --threads:
torch's CPU threads (default one per core), fewer where other work keeps the cores busy (on the CPU the reachability
check then runs its threads in lockstep, each waiting on the slowest; 2 threads gave 5 threads' grasps to 1e-6).

Wire: JSON lines. {"op": "info"} or {"op": "propose", "object": (N, 3) world points of the object, "scene": (M, 3)
points around it or null, "table_z": z, "n": grasps, "seed": s} -> {"ok": true, "T": (n, 3, 4, 4) world TCP poses of
the pre-grasp / grasp / hold, "u": (n, 3) jaw opening fractions (1 open, 0 closed), "score": (n,) higher first,
"seconds": s}. The TCP frame: +x along the fingers (the hand's travel), jaws open along +-y, origin between the pads.
A failing request ends the server; the backend starts a new one on its next call.
"""

import argparse
import hashlib
import json
import os
import select
import socket
import time
from pathlib import Path

import numpy as np
import torch

from graspkit.assets.policies import skill_policy_paths
from graspkit.policies.arx_x5_openloop import OPEN_Q, load_policy, plan_openloop_grasps

LINK6_TCP = np.array([[1, 0, 0, 0.145], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])  # ARX X5 link6 -> TCP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--ckpt", help="the generator checkpoint (default: B7's)")
    p.add_argument("--ranker", help="the success ranker checkpoint (default: B7's)")
    p.add_argument("--rank", default="ranker_reach", choices=("helper", "none", "ranker", "ranker_reach"),
                   help="plan_openloop_grasps' ordering")  # fmt: skip
    p.add_argument("--threads", type=int, help="torch's CPU threads (default: one per core)")
    p.add_argument("--exit-with-parent", action="store_true")
    a = p.parse_args()
    ckpt, ranker = skill_policy_paths()
    ckpt, ranker = str(a.ckpt or ckpt), str(a.ranker or ranker)
    if a.threads:
        torch.set_num_threads(a.threads)
    os.environ["GRASP_POLICY_RANKER"] = ranker  # plan_openloop_grasps reads the ranker checkpoint from the env
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    load_policy(ckpt, device)  # cached: loaded once, a bad checkpoint fails here
    sha = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()[:16] for k, v in (("ckpt", ckpt), ("ranker", ranker))}
    info = {"ckpt": ckpt, "rank": a.rank, "ranker": ranker, "pid": os.getpid(), "sha256": sha}

    path = Path(a.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen(4)
    print("ready", json.dumps(info), flush=True)
    parent = os.getppid()
    while not (a.exit_with_parent and os.getppid() != parent):  # the episode server is gone
        if not select.select([srv], [], [], 5.0)[0]:
            continue
        conn, _ = srv.accept()
        with conn, conn.makefile("rw") as f:
            for line in f:
                t0, req = time.monotonic(), json.loads(line)
                out = dict(ok=True, **info)
                if req["op"] == "propose":
                    scene = None if not req.get("scene") else np.asarray(req["scene"], float)
                    got = plan_openloop_grasps(np.asarray(req["object"], float), float(req["table_z"]), scene, ckpt,
                                               num_samples=int(req["n"]), seed=int(req.get("seed", 0)), device=device,
                                               rank=a.rank)  # fmt: skip
                    T = np.stack([np.stack([Tp, Tg, Tg]) for Tg, _, Tp in got]) @ LINK6_TCP
                    u = [[1.0, q / OPEN_Q, 0.0] for _, q, _ in got]  # open on the way in, at q, closed to hold
                    out = {"ok": True, "T": T.tolist(), "u": u, "score": (-np.arange(len(got), dtype=float)).tolist()}
                out["seconds"] = round(time.monotonic() - t0, 4)
                f.write(json.dumps(out) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
