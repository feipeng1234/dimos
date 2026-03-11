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

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import platform
import threading
import traceback

import lcm

from dimos.protocol.service.spec import Service
from dimos.protocol.service.system_configurator import configure_system, lcm_configurators
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DEFAULT_LCM_HOST = "239.255.76.67"
_DEFAULT_LCM_PORT = "7667"
# LCM_DEFAULT_URL is used by LCM (we didn't pick that env var name)
_DEFAULT_LCM_URL = os.getenv(
    "LCM_DEFAULT_URL", f"udpm://{_DEFAULT_LCM_HOST}:{_DEFAULT_LCM_PORT}?ttl=0"
)


def autoconf(check_only: bool = False) -> None:
    checks = lcm_configurators()
    if not checks:
        logger.error(f"System configuration not supported on {platform.system()}")
        return
    configure_system(checks, check_only=check_only)


@dataclass
class LCMConfig:
    ttl: int = 0
    url: str | None = None
    autoconf: bool = True
    lcm: lcm.LCM | None = None

    def __post_init__(self) -> None:
        if self.url is None:
            self.url = _DEFAULT_LCM_URL


_LCM_LOOP_TIMEOUT = 50


class NotStartedError(Exception):
    """Raised when LCM is accessed before start() or after stop()."""


# this class just sets up cpp LCM instance
# and runs its handle loop in a thread
# higher order stuff is done by pubsub/impl/lcmpubsub.py
class LCMService(Service[LCMConfig]):
    default_config = LCMConfig
    _l: lcm.LCM | None
    _stop_event: threading.Event
    _l_lock: threading.Lock
    _thread: threading.Thread | None
    _call_thread_pool: ThreadPoolExecutor | None = None
    _call_thread_pool_lock: threading.RLock = threading.RLock()

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)

        # autoconf must run in __init__ (not start()) so that stdin prompts
        # are asked during blueprint build, before the module moves to a worker
        try:
            autoconf(check_only=not self.config.autoconf)
        except Exception as e:
            print(f"Error checking system configuration: {e}")

        self._l: lcm.LCM | None = None
        self._l_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._owns_lcm_obj = False

    @property
    def l(self) -> lcm.LCM:  # noqa: E743
        if not self._l:
            raise NotStartedError(
                """LCM either not started or already stopped. Call .start() before accessing the .l property."""
            )

        return self._l

    @l.setter
    def l(self, value: lcm.LCM) -> None:  # noqa: E743
        self._l = value

    def __getstate__(self):  # type: ignore[no-untyped-def]
        """Exclude unpicklable runtime attributes when serializing."""
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        state.pop("_l", None)
        state.pop("_stop_event", None)
        state.pop("_thread", None)
        state.pop("_l_lock", None)
        state.pop("_call_thread_pool", None)
        state.pop("_call_thread_pool_lock", None)
        return state

    def __setstate__(self, state) -> None:  # type: ignore[no-untyped-def]
        """Restore object from pickled state."""
        self.__dict__.update(state)
        # Reinitialize runtime attributes
        self._l = None
        self._stop_event = threading.Event()
        self._thread = None
        self._call_thread_pool = None
        self._call_thread_pool_lock = threading.RLock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._l = (
            self._l
            or self.config.lcm
            or (lcm.LCM(self.config.url) if self.config.url else lcm.LCM())
        )
        self._owns_lcm_obj = not isinstance(self.config.lcm, lcm.LCM)
        self._thread = threading.Thread(target=self._lcm_loop)
        self._thread.daemon = True
        self._thread.start()

    def _lcm_loop(self) -> None:
        """LCM message handling loop."""

        while not self._stop_event.is_set():
            try:
                # no lock because the only thing that would change self._l is .stop()/.start()/setstate and this thread won't be running while those are happening
                self.l.handle_timeout(_LCM_LOOP_TIMEOUT)
            except Exception as e:
                stack_trace = traceback.format_exc()
                logger.error(f"Error in LCM handling: {e}\n{stack_trace}")

    def stop(self) -> None:
        """Stop the LCM loop."""
        self._stop_event.set()
        if self._thread is not None:
            # Only join if we're not the LCM thread (avoid "cannot join current thread")
            if threading.current_thread() != self._thread:
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    logger.warning("LCM thread did not stop cleanly within timeout")

        if self._owns_lcm_obj:
            del self._l
            self._l = None  # as an indicator that the LCM is closed

        with self._call_thread_pool_lock:
            if self._call_thread_pool:
                # Check if we're being called from within the thread pool
                # If so, we can't wait for shutdown (would cause "cannot join current thread")
                current_thread = threading.current_thread()
                is_pool_thread = False

                # Check if current thread is one of the pool's threads
                # ThreadPoolExecutor threads have names like "ThreadPoolExecutor-N_M"
                if hasattr(self._call_thread_pool, "_threads"):
                    is_pool_thread = current_thread in self._call_thread_pool._threads
                elif "ThreadPoolExecutor" in current_thread.name:
                    # Fallback: check thread name pattern
                    is_pool_thread = True

                # Don't wait if we're in a pool thread to avoid deadlock
                self._call_thread_pool.shutdown(wait=not is_pool_thread)
                self._call_thread_pool = None

    def _get_call_thread_pool(self) -> ThreadPoolExecutor:
        with self._call_thread_pool_lock:
            if self._call_thread_pool is None:
                self._call_thread_pool = ThreadPoolExecutor(max_workers=4)
            return self._call_thread_pool
