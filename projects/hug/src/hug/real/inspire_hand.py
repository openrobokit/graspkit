"""Inspire dexterous hand serial driver.

A background thread polls the hand at the serial port's natural rate; ``SetPosition`` / ``GetPosition``
never block. ``HandMapper`` converts URDF radians (thumb_rotate first) to the protocol's [0, 1] values
(little finger first, open/close reversed).
"""

import atexit
import logging
import threading
import time
from typing import Optional

import numpy as np
import serial
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


class HandMapper:
    """URDF (rad) ↔ serial (0–1, reversed open/close, little-finger-first) mapping."""

    def __init__(self) -> None:
        self.thumb_rotate_range = (-0.1, 1.3)
        self.thumb_bend_range = (0.0, 0.5)
        self.finger_range = (0.0, 1.7)

    def urdf_to_serial(self, q_urdf: NDArray[np.float64]) -> NDArray[np.float64]:
        thumb_rotate = 1.0 - (q_urdf[0] - self.thumb_rotate_range[0]) / (
            self.thumb_rotate_range[1] - self.thumb_rotate_range[0]
        )
        thumb_bend = 1.0 - (q_urdf[1] - self.thumb_bend_range[0]) / (
            self.thumb_bend_range[1] - self.thumb_bend_range[0]
        )
        index = 1.0 - (q_urdf[2] - self.finger_range[0]) / (self.finger_range[1] - self.finger_range[0])
        middle = 1.0 - (q_urdf[3] - self.finger_range[0]) / (self.finger_range[1] - self.finger_range[0])
        ring = 1.0 - (q_urdf[4] - self.finger_range[0]) / (self.finger_range[1] - self.finger_range[0])
        little = 1.0 - (q_urdf[5] - self.finger_range[0]) / (self.finger_range[1] - self.finger_range[0])
        return np.clip(
            np.array([little, ring, middle, index, thumb_bend, thumb_rotate]),
            0.0,
            1.0,
        )

    def serial_to_urdf(self, q_serial: NDArray[np.float64]) -> NDArray[np.float64]:
        q_serial = np.clip(q_serial, -0.1, 1.1)
        little = (1.0 - q_serial[0]) * (self.finger_range[1] - self.finger_range[0]) + self.finger_range[0]
        ring = (1.0 - q_serial[1]) * (self.finger_range[1] - self.finger_range[0]) + self.finger_range[0]
        middle = (1.0 - q_serial[2]) * (self.finger_range[1] - self.finger_range[0]) + self.finger_range[0]
        index = (1.0 - q_serial[3]) * (self.finger_range[1] - self.finger_range[0]) + self.finger_range[0]
        thumb_bend = (1.0 - q_serial[4]) * (
            self.thumb_bend_range[1] - self.thumb_bend_range[0]
        ) + self.thumb_bend_range[0]
        thumb_rotate = (1.0 - q_serial[5]) * (
            self.thumb_rotate_range[1] - self.thumb_rotate_range[0]
        ) + self.thumb_rotate_range[0]
        return np.array([thumb_rotate, thumb_bend, index, middle, ring, little])


class InspireHand:
    """Inspire 6-DOF hand over serial.

    q (URDF, rad): thumb_rotate [-0.1, 1.3], thumb_bend [0, 0.5], then index, middle, ring, little [0, 1.7]."""

    def __init__(
        self,
        serial_port: str = "/dev/ttyUSB0",
        id: int = 1,
        baudrate: int = 115200,
    ) -> None:
        self.ser: Optional[serial.Serial] = serial.Serial(
            port=serial_port,
            baudrate=baudrate,
            timeout=0.05,
            write_timeout=0.1,
            exclusive=True,
        )
        if not self.ser.is_open:
            raise ConnectionError(f"Failed to connect to {serial_port}")
        self.id = id
        self.qpos: NDArray[np.float64] = np.ones(6)  # serial 1 = open: the first GetPosition before a reply reads open
        self.qcmd: NDArray[np.float64] = np.zeros(6)
        self.first_qcmd = threading.Event()
        self.mapper = HandMapper()
        self.bg_should_stop = threading.Event()
        self.bg_thread = threading.Thread(target=self._exec_loop, daemon=True)
        self.bg_thread.start()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _checksum(self, data: bytes | bytearray) -> int:
        return sum(data[2:-1]) & 0xFF

    def _exec_loop(self) -> None:
        try:
            while not self.bg_should_stop.is_set():
                qpos = self._GetPosition()
                if qpos is not None:
                    self.qpos = qpos
                if self.first_qcmd.is_set():
                    self._SetPosition(self.qcmd)
        except serial.SerialException as exc:
            print(f"[hand] serial link lost ({exc}): the hand no longer moves. See docs/inspire_usb_debug.md.")
        self._cleanup()

    def SetPosition(self, q: NDArray[np.float64]) -> None:
        assert len(q) == 6, "q must be a 6-element array"
        self.qcmd[:] = self.mapper.urdf_to_serial(q)
        self.first_qcmd.set()

    def GetPosition(self) -> NDArray[np.float64]:
        return self.mapper.serial_to_urdf(self.qpos.copy())

    def close(self) -> None:
        self.bg_should_stop.set()
        self.bg_thread.join(timeout=1.0)

    def _send_and_receive(self, cmd: bytes | bytearray, recv_len: int = 9) -> bytes:
        assert self.ser is not None
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(0.005)
        self.ser.write(cmd)
        result: bytes = self.ser.read(recv_len)
        return result

    def _SetPosition(self, q: NDArray[np.float64]) -> Optional[bytes]:
        q_int16 = np.clip((q * 1000).astype(np.int16), 0, 1000)
        cmd = bytearray(20)
        cmd[0:7] = [0xEB, 0x90, self.id, 0x0F, 0x12, 0xCE, 0x05]
        for i in range(6):
            cmd[7 + 2 * i] = q_int16[i] & 0xFF
            cmd[8 + 2 * i] = (q_int16[i] >> 8) & 0xFF
        cmd[19] = self._checksum(cmd)
        retval = self._send_and_receive(cmd)
        if len(retval) != 9:
            _logger.warning("SetPosition failed, retval len: %d", len(retval))
            return None
        return retval

    def _GetPosition(self) -> Optional[NDArray[np.float64]]:
        cmd = bytearray([0xEB, 0x90, self.id, 0x04, 0x11, 0x0A, 0x06, 0x0C, 0x00])
        cmd[-1] = self._checksum(cmd)
        response = self._send_and_receive(cmd, 20)
        if len(response) != 20:
            _logger.warning("GetPosition failed, response len: %d", len(response))
            return None
        if not (response[0] == 0x90 and response[1] == 0xEB and response[2] == self.id):
            _logger.warning("GetPosition failed, response header: %s", response[:3])
            return None
        if response[-1] != self._checksum(response):
            _logger.warning(
                "GetPosition failed, response checksum: %d, calculated: %d",
                response[-1],
                self._checksum(response),
            )
            return None
        q = np.zeros(6)
        for i in range(6):
            val = int.from_bytes(response[7 + 2 * i : 9 + 2 * i], "little", signed=False)
            q[i] = val / 1000.0
        return q
