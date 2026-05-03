# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layered pipeline tests to isolate which DimSim/dimos stage breaks in CI.

Three classes, each one layer up:

  TestDimSimBinaryOnly   bare ~/.dimsim/bin/dimsim — proves the simulator,
                         headless Chrome, WebGL, and LCM multicast all work
                         on the runner without dimos in the picture.
  TestDimosSimBasic      dimos --simulation run sim-basic — adds dimos's
                         Module/Blueprint stack and the bridge wrapper but
                         no nav/agent stack.
  (TestSimNav lives in test_dimsim_nav.py — the full end-to-end test.)

Each fixture captures the subprocess's stdout/stderr to a tempfile and dumps
it on teardown so failures in CI carry the diagnostic context that DEVNULL
was hiding.
"""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
import websocket

from dimos.e2e_tests._subprocess_log import dump_log, launch_with_streaming_log
from dimos.e2e_tests.lcm_spy import LcmSpy

BRIDGE_PORT = 8090


# ── Helpers ──────────────────────────────────────────────────────────────────


def _force_kill_port(port: int) -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
        )
        for pid in result.stdout.strip().split():
            if pid:
                try:
                    os.kill(int(pid), 9)
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass


def _wait_for_port(port: int, timeout: float = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _wait_for_port_free(port: int, timeout: float = 10) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                time.sleep(1)
        except OSError:
            return True
    return False


def _dimsim_binary() -> Path:
    return Path.home() / ".dimsim" / "bin" / "dimsim"


def _ensure_dimsim_binary() -> Path:
    """Ensure ~/.dimsim/bin/dimsim exists. Skip the test if not present and
    no `dimsim` is on PATH — auto-install lives in dimos.robot.sim.bridge and
    we don't want to copy that here."""
    binary = _dimsim_binary()
    if binary.exists():
        return binary
    fallback = shutil.which("dimsim")
    if fallback:
        return Path(fallback)
    pytest.skip(
        f"dimsim binary not found at {binary} or on PATH — run `dimos --simulation "
        "run sim-basic` once to trigger auto-install, then re-run."
    )


# ── Layer 1: bare dimsim binary ──────────────────────────────────────────────


@pytest.fixture(scope="class")
def dimsim_only():
    """Launch ~/.dimsim/bin/dimsim dev directly — no dimos involvement."""
    binary = _ensure_dimsim_binary()

    _force_kill_port(BRIDGE_PORT)
    assert _wait_for_port_free(BRIDGE_PORT, timeout=10), (
        f"Port {BRIDGE_PORT} still busy after force-kill"
    )

    proc, log_path = launch_with_streaming_log(
        "dimsim_binary",
        [
            str(binary),
            "dev",
            "--scene",
            "apt",
            "--port",
            str(BRIDGE_PORT),
            "--headless",
            "--render",
            "cpu",
        ],
    )

    try:
        if not _wait_for_port(BRIDGE_PORT, timeout=120):
            dump_log("dimsim binary", log_path)
            pytest.fail(f"dimsim bridge port {BRIDGE_PORT} never opened")
        yield proc
    finally:
        try:
            os.killpg(proc.pid, 15)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
        dump_log("dimsim binary", log_path)
        log_path.unlink(missing_ok=True)
        _force_kill_port(BRIDGE_PORT)


@pytest.fixture(scope="class")
def dimsim_only_spy(dimsim_only):
    spy = LcmSpy()
    spy.save_topic("/color_image#sensor_msgs.Image")
    spy.save_topic("/odom#geometry_msgs.PoseStamped")
    spy.start()
    try:
        yield spy
    finally:
        spy.stop()


class TestDimSimBinaryOnly:
    """Stage 1: bare dimsim. If this fails the simulator itself is broken on
    the runner — Chrome/WebGL/multicast — and dimos can't help.
    """

    def test_http_index_served(self, dimsim_only) -> None:
        with urllib.request.urlopen(
            f"http://localhost:{BRIDGE_PORT}/", timeout=5
        ) as resp:
            assert resp.status == 200
            body = resp.read(2048).decode("utf-8", errors="replace")
            assert "<html" in body.lower(), "index.html missing — dimsim assets not unpacked?"

    def test_websocket_handshake(self, dimsim_only) -> None:
        ws = websocket.WebSocket()
        ws.settimeout(10)
        try:
            ws.connect(f"ws://localhost:{BRIDGE_PORT}")
        finally:
            ws.close()

    def test_color_image_publishes(self, dimsim_only_spy) -> None:
        """Headless Chrome must launch, init WebGL, and produce JPEG frames
        that the bridge multicasts. This is the stage that's been failing in CI.
        """
        dimsim_only_spy.wait_for_saved_topic(
            "/color_image#sensor_msgs.Image", timeout=120.0
        )
        msgs = dimsim_only_spy.messages.get("/color_image#sensor_msgs.Image", [])
        assert len(msgs) > 0


# ── Layer 2: dimos sim-basic ─────────────────────────────────────────────────


@pytest.fixture(scope="class")
def dimos_sim_basic():
    """Launch dimos sim-basic — adds the dimos blueprint stack on top of dimsim."""
    _force_kill_port(BRIDGE_PORT)
    assert _wait_for_port_free(BRIDGE_PORT, timeout=10), (
        f"Port {BRIDGE_PORT} still busy after force-kill"
    )

    venv_bin = str(Path(sys.prefix) / "bin")
    env = {
        **os.environ,
        "DIMSIM_HEADLESS": "1",
        "DIMSIM_RENDER": os.environ.get("DIMSIM_RENDER", "cpu"),
        "DIMSIM_VERBOSE": "1",
        "PYTHONUNBUFFERED": "1",
        "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
    }

    proc, log_path = launch_with_streaming_log(
        "dimos_sim_basic",
        ["dimos", "--simulation", "run", "sim-basic"],
        env=env,
    )

    try:
        if not _wait_for_port(BRIDGE_PORT, timeout=120):
            dump_log("dimos sim-basic", log_path)
            pytest.fail(f"dimos sim-basic never opened port {BRIDGE_PORT}")
        yield proc
    finally:
        try:
            os.killpg(proc.pid, 15)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
        dump_log("dimos sim-basic", log_path)
        log_path.unlink(missing_ok=True)
        _force_kill_port(BRIDGE_PORT)


@pytest.fixture(scope="class")
def dimos_sim_basic_spy(dimos_sim_basic):
    spy = LcmSpy()
    spy.save_topic("/color_image#sensor_msgs.Image")
    spy.save_topic("/odom#geometry_msgs.PoseStamped")
    spy.start()
    try:
        yield spy
    finally:
        spy.stop()


class TestDimosSimBasic:
    """Stage 2: dimos sim-basic. Only meaningful if Layer 1 passes — if it
    does, this isolates anything that the dimos blueprint stack itself adds
    or breaks (transports, TF, headless gating).
    """

    def test_color_image_publishes(self, dimos_sim_basic_spy) -> None:
        dimos_sim_basic_spy.wait_for_saved_topic(
            "/color_image#sensor_msgs.Image", timeout=120.0
        )

    def test_odom_publishes(self, dimos_sim_basic_spy) -> None:
        dimos_sim_basic_spy.wait_for_saved_topic(
            "/odom#geometry_msgs.PoseStamped", timeout=200.0
        )
