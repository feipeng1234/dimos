# Copyright 2026 Dimensional Inc.
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

from collections.abc import Iterator

import pytest

from dimos.core.tests.stress_test_module import StressTestModule
from dimos.porcelain.dimos import Dimos


@pytest.fixture
def app():
    instance = Dimos()
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture(scope="session")
def running_app() -> Iterator[Dimos]:
    """Session-scoped: shared across every test in this xdist worker.

    Tests that mutate state (`.stop()`, `.run()`, `.restart()`) must
    use the function-scoped `running_app_fresh` for isolation.
    """
    instance = Dimos(n_workers=1)
    instance.run(StressTestModule)
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture
def running_app_fresh() -> Iterator[Dimos]:
    """Function-scoped — for tests that need a private Dimos lifecycle."""
    instance = Dimos(n_workers=1)
    instance.run(StressTestModule)
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture(scope="session")
def _session_rpyc_port(running_app: Dimos) -> int:
    """The rpyc port for the session-scoped `running_app`.

    Started once so multiple clients in the session can share it.
    """
    return running_app._coordinator.start_rpyc_service()


@pytest.fixture(scope="session")
def client(running_app: Dimos, _session_rpyc_port: int) -> Iterator[Dimos]:
    """Session-scoped rpyc client paired with the session-scoped `running_app`."""
    instance = Dimos.connect(host="localhost", port=_session_rpyc_port)
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture
def client_fresh(running_app_fresh: Dimos) -> Iterator[Dimos]:
    """Function-scoped client paired with `running_app_fresh`.

    Use this only when the test mutates server-side state (`.run()`,
    `.restart()`) and needs a private cluster for isolation.
    """
    port = running_app_fresh._coordinator.start_rpyc_service()
    instance = Dimos.connect(host="localhost", port=port)
    try:
        yield instance
    finally:
        instance.stop()


@pytest.fixture
def temp_client(running_app: Dimos, _session_rpyc_port: int) -> Iterator[Dimos]:
    """Function-scoped lightweight rpyc client to the session
    `running_app`. No fresh cluster spawn — just a new rpyc connection
    (cheap). Use this for tests that call `.stop()` on the client to
    verify client lifecycle without affecting the server.
    """
    instance = Dimos.connect(host="localhost", port=_session_rpyc_port)
    try:
        yield instance
    finally:
        if instance.is_running:
            instance.stop()
