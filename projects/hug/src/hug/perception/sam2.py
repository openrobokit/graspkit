"""Lifecycle wrapper for robokit's SAM2 annotation server (``robokit.helpers.hand_eye_calibration.sam2_web``).

:class:`SamPipeline` spawns the server once per process and reuses it. The operator labels in a browser tab at
``http://localhost:8000``, and every ``segment`` call blocks until Submit. robokit loads on first use: its
calibration package imports torch, which only the real extras install.
"""

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


class SamPipeline:
    """Spawns robokit's SAM2 server once per process and wraps its blocking ``segment`` client.

    The browser tab at ``http://localhost:<port>`` must be open, or a segmentation never returns."""

    def __init__(self, port: int = 8000) -> None:
        self.port = port
        self.url = f"http://localhost:{port}"
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()

    def start(self) -> str:
        """Spawn the server now and return its URL, so the operator learns about the second tab at startup."""
        self._ensure_started()
        return self.url

    def wait_until_ready(self, timeout_s: float = 180.0) -> bool:
        """Block until the HTTP port accepts connections, or ``timeout_s``; SAM2 loads before uvicorn binds."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.5)
                if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                    return True
            time.sleep(0.5)
        return False

    def _ensure_started(self) -> None:
        from robokit.helpers.hand_eye_calibration import sam2_web

        with self.lock:
            if self.proc is not None:
                return
            self.proc = subprocess.Popen(
                [sys.executable, str(Path(sam2_web.__file__).parent / "server.py"), "--port", str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            threading.Thread(target=self._relay_output, args=(self.proc.stdout,), daemon=True).start()

    @staticmethod
    def _relay_output(stream) -> None:
        """Forward the server's output minus uvicorn's INFO lines (access log on stdout); errors still come through."""
        for line in stream:
            if not line.startswith("INFO:"):
                sys.stdout.write(line)
                sys.stdout.flush()

    def segment(self, rgb: np.ndarray, label: str = "") -> np.ndarray:
        """Block until the operator submits a mask; ``label`` is shown in the tab. Returns ``(H, W) bool``."""
        from robokit.helpers.hand_eye_calibration.sam2_web import segment

        self._ensure_started()
        return np.asarray(segment(rgb, label=label)).astype(bool)

    def shutdown(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
