"""G1 robot wrapper over unitree_sdk2_python (DDS).

Call ``channel_factory_init`` before constructing ``G1Robot``.
"""

import os
import struct
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.core import channel as _sdk_channel
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

G1_NUM_MOTOR = 29

# CRC-32/MPEG-2 byte table (polynomial 0x04C11DB7, MSB first): the SDK's crc32_core, in Python.
_CRC_TABLE: list[int] = []
for _i in range(256):
    _c = _i << 24
    for _ in range(8):
        _c = ((_c << 1) ^ 0x04C11DB7) if _c & 0x80000000 else (_c << 1)
    _CRC_TABLE.append(_c & 0xFFFFFFFF)


class _CRC(CRC):  # type: ignore[misc]  # the SDK is untyped
    """The SDK's lowcmd CRC without its prebuilt ``crc_*.so``, which the wheel built from the git repo leaves out."""

    def __init__(self) -> None:
        try:
            super().__init__()
        except OSError:  # no utils/lib/crc_*.so in the installed package: take the SDK's own pure-Python branch
            self.platform = "python"

    def _crc_py(self, data) -> int:
        """Table-driven (0.2 ms per lowcmd); the SDK's bit loop is 20x slower."""
        crc = 0xFFFFFFFF
        for word in data:
            for b in int(word).to_bytes(4, "big"):
                crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[(crc >> 24) ^ b]
        return crc


# The SDK hardcodes /tmp/cdds.LOG; CycloneDDS aborts init when another uid owns that file, so one trace per uid.
_CDDS_LOG = os.path.join(tempfile.gettempdir(), f"cdds-{os.getuid()}.LOG")


def channel_factory_init(network_interface: str, domain_id: int = 0) -> None:
    """Initialise the DDS channel factory. Every real entrypoint goes through here."""
    # rebind on channel, not channel_config: channel.py imported the name, so it holds its own reference
    _sdk_channel.ChannelConfigHasInterface = _sdk_channel.ChannelConfigHasInterface.replace("/tmp/cdds.LOG", _CDDS_LOG)
    if network_interface:
        ChannelFactoryInitialize(domain_id, network_interface)
    else:
        ChannelFactoryInitialize(domain_id)


@dataclass
class RemoteState:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    R1: int = 0
    L1: int = 0
    Start: int = 0
    Select: int = 0
    R2: int = 0
    L2: int = 0
    F1: int = 0
    F3: int = 0
    A: int = 0
    B: int = 0
    X: int = 0
    Y: int = 0
    Up: int = 0
    Right: int = 0
    Down: int = 0
    Left: int = 0


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistA = 13
    WaistPitch = 14
    WaistB = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


class Mode:
    PR = 0
    AB = 1


def _parse_wireless_remote(data) -> RemoteState:
    rs = RemoteState()
    rs.lx = struct.unpack("<f", bytes(data[4:8]))[0]
    rs.rx = struct.unpack("<f", bytes(data[8:12]))[0]
    rs.ry = struct.unpack("<f", bytes(data[12:16]))[0]
    rs.ly = struct.unpack("<f", bytes(data[20:24]))[0]
    b1, b2 = data[2], data[3]
    rs.R1 = (b1 >> 0) & 1
    rs.L1 = (b1 >> 1) & 1
    rs.Start = (b1 >> 2) & 1
    rs.Select = (b1 >> 3) & 1
    rs.R2 = (b1 >> 4) & 1
    rs.L2 = (b1 >> 5) & 1
    rs.F1 = (b1 >> 6) & 1
    rs.F3 = (b1 >> 7) & 1
    rs.A = (b2 >> 0) & 1
    rs.B = (b2 >> 1) & 1
    rs.X = (b2 >> 2) & 1
    rs.Y = (b2 >> 3) & 1
    rs.Up = (b2 >> 4) & 1
    rs.Right = (b2 >> 5) & 1
    rs.Down = (b2 >> 6) & 1
    rs.Left = (b2 >> 7) & 1
    return rs


class G1Robot:
    """High-level wrapper for the Unitree G1 (29-DoF) over the SDK's DDS interface."""

    def __init__(
        self,
        control_dt: float = 1.0 / 300.0,
        read_only: bool = False,
        topic_prefix: str = "rt",
    ) -> None:
        """Connect and wait up to 5 s for the first lowstate.

        ``read_only`` skips the MotionSwitcher release and the lowcmd publisher; ``Step`` then only paces."""
        self._nj = G1_NUM_MOTOR
        self._control_dt = control_dt
        self._read_only = read_only
        self._topic_prefix = topic_prefix

        self._motor_angles: NDArray[np.float64] = np.zeros(self._nj)
        self._motor_velocities: NDArray[np.float64] = np.zeros(self._nj)
        self._motor_taus: NDArray[np.float64] = np.zeros(self._nj)
        self._motor_temperatures: NDArray[np.float64] = np.zeros(self._nj)
        self._base_quat: NDArray[np.float64] = np.zeros(4)
        self._base_rpy: NDArray[np.float64] = np.zeros(3)
        self._ang_vel: NDArray[np.float64] = np.zeros(3)
        self._accel: NDArray[np.float64] = np.zeros(3)
        self._remote_state = RemoteState()
        self._low_state: Optional[LowState_] = None

        self._mode_machine: int = 0
        self._mode_pr: int = Mode.PR
        self._got_first_state: bool = False

        self._q_des: NDArray[np.float64] = np.zeros(self._nj)
        self._qd_des: NDArray[np.float64] = np.zeros(self._nj)
        self._tau_ff: NDArray[np.float64] = np.zeros(self._nj)
        self._kp: NDArray[np.float64] = np.zeros(self._nj)
        self._kd: NDArray[np.float64] = np.full(self._nj, 2.0)

        self._low_cmd: LowCmd_ = unitree_hg_msg_dds__LowCmd_()
        self._crc = _CRC()

        if not self._read_only:
            self._msc = MotionSwitcherClient()
            self._msc.SetTimeout(5.0)
            self._msc.Init()
            # CheckMode returns None when nothing responds; treat that as nothing to release
            _, result = self._msc.CheckMode()
            while result and result.get("name"):
                self._msc.ReleaseMode()
                _, result = self._msc.CheckMode()
                time.sleep(1)

            self._lowcmd_pub = ChannelPublisher(f"{self._topic_prefix}/lowcmd", LowCmd_)
            self._lowcmd_pub.Init()

        self._lowstate_sub = ChannelSubscriber(f"{self._topic_prefix}/lowstate", LowState_)
        self._lowstate_sub.Init(self._on_low_state, 10)

        for _ in range(500):
            if self._got_first_state:
                break
            time.sleep(0.01)
        if not self._got_first_state:
            raise RuntimeError(
                "Did not receive low_state from G1 within 5 s — "
                "is the robot reachable on the configured network interface?"
            )

    def _on_low_state(self, msg: LowState_) -> None:
        self._low_state = msg
        if not self._got_first_state:
            self._mode_machine = msg.mode_machine
            self._got_first_state = True
        for i in range(self._nj):
            ms = msg.motor_state[i]
            self._motor_angles[i] = ms.q
            self._motor_velocities[i] = ms.dq
            self._motor_taus[i] = ms.tau_est
            self._motor_temperatures[i] = ms.temperature[0]
        self._base_rpy[:] = msg.imu_state.rpy[:]
        self._base_quat[:] = msg.imu_state.quaternion[:]
        self._ang_vel[:] = msg.imu_state.gyroscope[:]
        self._accel[:] = msg.imu_state.accelerometer[:]
        self._remote_state = _parse_wireless_remote(msg.wireless_remote)

    def Step(self, do_sleep: bool = True) -> None:
        if not self._read_only:
            self._low_cmd.mode_pr = self._mode_pr
            self._low_cmd.mode_machine = self._mode_machine
            for i in range(self._nj):
                mc = self._low_cmd.motor_cmd[i]
                mc.mode = 1
                mc.q = float(self._q_des[i])
                mc.dq = float(self._qd_des[i])
                mc.tau = float(self._tau_ff[i])
                mc.kp = float(self._kp[i])
                mc.kd = float(self._kd[i])
            self._low_cmd.crc = self._crc.Crc(self._low_cmd)
            self._lowcmd_pub.Write(self._low_cmd)
        if do_sleep:
            time.sleep(self._control_dt)

    def SetJointCmd(
        self,
        q: NDArray[np.float64],
        dq: NDArray[np.float64],
        tau: NDArray[np.float64],
        kp: NDArray[np.float64],
        kd: NDArray[np.float64],
        mask: Optional[NDArray[np.bool_]] = None,
        selector: Optional[NDArray[np.intp]] = None,
    ) -> None:
        if mask is not None and selector is not None:
            raise ValueError("Specify mask OR selector, not both")
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
        elif selector is not None:
            m = np.zeros(self._nj, dtype=bool)
            m[selector] = True
        else:
            m = np.ones(self._nj, dtype=bool)
        self._q_des[m] = np.asarray(q)[m]
        self._qd_des[m] = np.asarray(dq)[m]
        self._tau_ff[m] = np.asarray(tau)[m]
        self._kp[m] = np.asarray(kp)[m]
        self._kd[m] = np.asarray(kd)[m]

    def SetPartialJointCmd(
        self,
        joint_indices: NDArray[np.intp],
        q: NDArray[np.float64],
        dq: NDArray[np.float64],
        tau: NDArray[np.float64],
        kp: NDArray[np.float64],
        kd: NDArray[np.float64],
    ) -> None:
        self._q_des[joint_indices] = q
        self._qd_des[joint_indices] = dq
        self._tau_ff[joint_indices] = tau
        self._kp[joint_indices] = kp
        self._kd[joint_indices] = kd

    def GetMotorQ(self) -> NDArray[np.float64]:
        return self._motor_angles.copy()

    def GetMotorVel(self) -> NDArray[np.float64]:
        return self._motor_velocities.copy()

    def GetMotorTau(self) -> NDArray[np.float64]:
        return self._motor_taus.copy()

    def GetTemperatures(self) -> NDArray[np.float64]:
        return self._motor_temperatures.copy()

    def GetBaseQuat(self) -> NDArray[np.float64]:
        return self._base_quat.copy()

    def GetBaseRPY(self) -> NDArray[np.float64]:
        return self._base_rpy.copy()

    def GetAngVel(self) -> NDArray[np.float64]:
        return self._ang_vel.copy()

    def GetAccel(self) -> NDArray[np.float64]:
        return self._accel.copy()

    def GetRemoteState(self) -> RemoteState:
        return self._remote_state

    def Close(self) -> None:
        print("[g1] damping mode.", flush=True)
        self.SetJointCmd(
            np.zeros(self._nj),
            np.zeros(self._nj),
            np.zeros(self._nj),
            np.zeros(self._nj),
            np.ones(self._nj) * 1.0,
        )
        self.Step(do_sleep=False)
