"""Inspire hand check over serial: no robot, no arm PD, safe any time.

``read`` prints the URDF qpos at ~20 Hz; ``ramp`` drives the hand open ↔ half-closed at 50 Hz. Ctrl+C stops.
If the port does not open, see ``docs/inspire_usb_debug.md``.

Usage:
    uv run --extra real-cu128 python scripts/real/check_hand.py --mode ramp [--serial-port /dev/ttyUSB0]
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from hug.real.g1_inspire_interface import G1InspireConfig
from hug.real.inspire_hand import InspireHand


def read(serial_port: str) -> None:
    """Read-only: print URDF qpos at ~20 Hz."""
    hand = InspireHand(serial_port=serial_port, id=1)
    try:
        while True:
            print(hand.GetPosition())
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[exit] closing.")
    finally:
        hand.close()


def ramp(serial_port: str) -> None:
    """Linearly ramp the hand back and forth between q_open and q_mid."""
    hand = InspireHand(serial_port=serial_port, id=1)
    q_open = np.zeros(6)
    q_mid = np.array([0.6, 0.25, 0.85, 0.85, 0.85, 0.85])
    steps = 100  # ~2 s per leg at 50 Hz
    try:
        while True:
            for i in range(steps):
                r = i / steps
                q = (1 - r) * q_open + r * q_mid
                hand.SetPosition(q)
                print("cmd:", q, "obs:", hand.GetPosition())
                time.sleep(0.02)
            for i in range(steps):
                r = i / steps
                q = (1 - r) * q_mid + r * q_open
                hand.SetPosition(q)
                print("cmd:", q, "obs:", hand.GetPosition())
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("[exit] closing.")
    finally:
        hand.close()


@dataclass
class Args:
    mode: Literal["read", "ramp"] = "read"  # read: print qpos only; ramp: move the fingers
    serial_port: str = G1InspireConfig.right_hand_serial_port  # serial device path


def main(args: Args) -> None:
    np.set_printoptions(precision=3, suppress=True)
    {"read": read, "ramp": ramp}[args.mode](args.serial_port)


if __name__ == "__main__":
    main(tyro.cli(Args))
