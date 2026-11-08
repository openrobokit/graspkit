"""Real grasp preflight: the perception half of ``real/grasp.py``, with no way to move the robot.

Same panel, same steps — 1 · Capture, 2 · Segment (in the SAM2 tab), 3 · Run policy —
and the same workspace checks, on a read-only DDS subscription: nothing here can
command a joint. Use it to validate a fresh calibration (the live cloud should sit on
the rendered robot and on the floor) and to see what the policy predicts before the
actuating script runs. Every segmentation also saves ``results/captured/scene_<ts>.npz``,
the bundle that replays the perception path offline.

Usage:
    uv run --extra real-cu128 python scripts/real/preflight.py --network-iface enp4s0
"""

from dataclasses import dataclass

import numpy as np
import tyro

from hug.real.g1_robot import G1Robot, channel_factory_init
from hug.real.perception import PerceptionArgs, PerceptionFrontEnd


@dataclass
class Args(PerceptionArgs):
    network_iface: str = "enp4s0"  # DDS network interface (the real robot is domain 0, prefix 'rt')


def main(args: Args) -> None:
    channel_factory_init(args.network_iface)
    g1 = G1Robot(read_only=True)  # subscribes to lowstate only; raises if the robot is silent for 5 s
    front = PerceptionFrontEnd(args, motor_q=lambda: np.asarray(g1.GetMotorQ(), dtype=np.float64))
    front.open_tabs()
    print("  ready — click 1 · Capture. Ctrl+C exits.\n")
    try:
        front.run()
    except KeyboardInterrupt:
        print("\n[exit] interrupted.")
    finally:
        front.close()
        g1.Close()


if __name__ == "__main__":
    main(tyro.cli(Args))
