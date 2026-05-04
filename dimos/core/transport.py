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

import threading
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from dimos.core.stream import In, Out, Stream, Transport
from dimos.msgs.protocol import DimosMsg
from dimos.utils import colors

try:
    import cyclonedds as _cyclonedds  # noqa: F401

    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False

from dimos.protocol.pubsub.impl.lcmpubsub import LCM, PickleLCM, Topic as LCMTopic
from dimos.protocol.pubsub.impl.rospubsub import DimosROS, ROSTopic
from dimos.protocol.pubsub.impl.shmpubsub import BytesSharedMemory, PickleSharedMemory

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def _qualname(cls: type) -> str:
    """Fully-qualified name suitable for round-tripping through resolve_msg_type."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_type(qualname: str) -> type | None:
    """Inverse of _qualname; returns None if the type cannot be imported."""
    import importlib

    module_path, _, class_name = qualname.rpartition(".")
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None
    return getattr(module, class_name, None)


# TODO
# Transports need to be rewritten and simplified,
#
# there is no need for them to get a reference to "a stream" on publish/subscribe calls
# this is a legacy from dask transports.
#
# new transport should literally have 2 functions (next to start/stop)
# "send(msg)" and "receive(callback)" and that's all
#
# we can also consider pubsubs conforming directly to Transport specs
# and removing PubSubTransport glue entirely
#
# Why not ONLY pubsubs without Transport abstraction?
#
# General idea for transports (and why they exist at all)
# is that they can be * anything * like
#
# a web camera rtsp stream for Image, audio stream from mic, etc
# http binary streams, tcp connections etc


class PubSubTransport(Transport[T]):
    topic: Any

    def __init__(self, topic: Any) -> None:
        self.topic = topic

    def __str__(self) -> str:
        return (
            colors.green(f"{self.__class__.__name__}(")
            + colors.blue(self.topic)
            + colors.green(")")
        )


class pLCMTransport(PubSubTransport[T]):
    _started: bool = False

    def __init__(self, topic: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(topic)
        self.lcm = PickleLCM(**kwargs)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (pLCMTransport, (self.topic,))

    def __getstate__(self) -> dict[str, Any]:
        return {"kind": "plcm", "topic": self.topic}

    def broadcast(self, _: Out[T] | None, msg: T) -> None:
        if not self._started:
            self.start()

        self.lcm.publish(self.topic, msg)

    def subscribe(
        self, callback: Callable[[T], Any], selfstream: Stream[T] | None = None
    ) -> Callable[[], None]:
        if not self._started:
            self.start()
        return self.lcm.subscribe(LCMTopic(self.topic), lambda msg, topic: callback(msg))

    def start(self) -> None:
        self.lcm.start()
        self._started = True

    def stop(self) -> None:
        self.lcm.stop()
        self._started = False


class LCMTransport(PubSubTransport[T]):
    _started: bool = False

    def __init__(self, topic: str, type: type, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(LCMTopic(topic, type))
        if not hasattr(self, "lcm"):
            self.lcm = LCM(**kwargs)

    def start(self) -> None:
        self.lcm.start()
        self._started = True

    def stop(self) -> None:
        self.lcm.stop()
        self._started = False

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (LCMTransport, (self.topic.topic, self.topic.lcm_type))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "kind": "lcm",
            "topic": self.topic.topic,
            "lcm_type": _qualname(self.topic.lcm_type) if self.topic.lcm_type else None,
        }

    def broadcast(self, _, msg) -> None:  # type: ignore[no-untyped-def]
        if not self._started:
            self.start()

        self.lcm.publish(self.topic, msg)

    def subscribe(self, callback: Callable[[T], None], selfstream: In[T] = None) -> None:  # type: ignore[assignment, override]
        if not self._started:
            self.start()
        return self.lcm.subscribe(self.topic, lambda msg, topic: callback(msg))  # type: ignore[arg-type, return-value]


class JpegLcmTransport(LCMTransport):  # type: ignore[type-arg]
    def __init__(self, topic: str, type: type, **kwargs) -> None:  # type: ignore[no-untyped-def]
        from dimos.protocol.pubsub.impl.jpeg_lcm import (
            JpegLCM,
        )  # ~330ms: deferred to avoid pulling in Image/cv2/rerun

        self.lcm = JpegLCM(**kwargs)  # type: ignore[assignment]
        super().__init__(topic, type)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (JpegLcmTransport, (self.topic.topic, self.topic.lcm_type))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "kind": "jpeg_lcm",
            "topic": self.topic.topic,
            "lcm_type": _qualname(self.topic.lcm_type) if self.topic.lcm_type else None,
        }

    def start(self) -> None:
        self.lcm.start()
        self._started = True

    def stop(self) -> None:
        self.lcm.stop()
        self._started = False


class pSHMTransport(PubSubTransport[T]):
    _started: bool = False

    def __init__(self, topic: str, default_capacity: int | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(topic)
        if default_capacity is not None:
            kwargs["default_capacity"] = default_capacity
        self.shm = PickleSharedMemory(**kwargs)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (pSHMTransport, (self.topic, self.shm.config.default_capacity))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "kind": "pshm",
            "topic": self.topic,
            "capacity": self.shm.config.default_capacity,
        }

    def broadcast(self, _, msg) -> None:  # type: ignore[no-untyped-def]
        if not self._started:
            self.start()

        self.shm.publish(self.topic, msg)

    def subscribe(self, callback: Callable[[T], None], selfstream: In[T] = None) -> None:  # type: ignore[assignment, override]
        if not self._started:
            self.start()
        return self.shm.subscribe(self.topic, lambda msg, topic: callback(msg))  # type: ignore[return-value]

    def start(self) -> None:
        self.shm.start()
        self._started = True

    def stop(self) -> None:
        self.shm.stop()
        self._started = False


class SHMTransport(PubSubTransport[T]):
    _started: bool = False

    def __init__(self, topic: str, default_capacity: int | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(topic)
        if default_capacity is not None:
            kwargs["default_capacity"] = default_capacity
        self.shm = BytesSharedMemory(**kwargs)

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (SHMTransport, (self.topic, self.shm.config.default_capacity))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "kind": "shm",
            "topic": self.topic,
            "capacity": self.shm.config.default_capacity,
        }

    def broadcast(self, _, msg) -> None:  # type: ignore[no-untyped-def]
        if not self._started:
            self.start()

        self.shm.publish(self.topic, msg)

    def subscribe(self, callback: Callable[[T], None], selfstream: In[T] | None = None) -> None:  # type: ignore[override]
        if not self._started:
            self.start()
        return self.shm.subscribe(self.topic, lambda msg, topic: callback(msg))  # type: ignore[arg-type, return-value]

    def start(self) -> None:
        self.shm.start()
        self._started = True

    def stop(self) -> None:
        self.shm.stop()
        self._started = False


class JpegShmTransport(PubSubTransport[T]):
    _started: bool = False

    def __init__(  # type: ignore[no-untyped-def]
        self,
        topic: str,
        quality: int = 75,
        default_capacity: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(topic)
        from dimos.protocol.pubsub.impl.jpeg_shm import (
            JpegSharedMemory,
        )  # deferred to avoid pulling in Image/cv2/rerun

        if default_capacity is not None:
            kwargs["default_capacity"] = default_capacity
        self.shm = JpegSharedMemory(quality=quality, **kwargs)
        self.quality = quality

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (
            JpegShmTransport,
            (self.topic, self.quality, self.shm.config.default_capacity),
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "kind": "jpeg_shm",
            "topic": self.topic,
            "quality": self.quality,
            "capacity": self.shm.config.default_capacity,
        }

    def broadcast(self, _, msg) -> None:  # type: ignore[no-untyped-def]
        if not self._started:
            self.start()

        self.shm.publish(self.topic, msg)

    def subscribe(self, callback: Callable[[T], None], selfstream: In[T] | None = None) -> None:  # type: ignore[override]
        if not self._started:
            self.start()
        return self.shm.subscribe(self.topic, lambda msg, topic: callback(msg))  # type: ignore[arg-type, return-value]

    def start(self) -> None:
        self.shm.start()
        self._started = True

    def stop(self) -> None:
        self.shm.stop()
        self._started = False


class ROSTransport(PubSubTransport[DimosMsg]):
    _ros: DimosROS | None = None

    def __init__(self, topic: str, msg_type: type[DimosMsg], **kwargs: Any) -> None:
        super().__init__(ROSTopic(topic, msg_type))
        self._kwargs = kwargs

    def __reduce__(self) -> tuple[Any, ...]:
        return (ROSTransport, (self.topic.topic, self.topic.msg_type))

    def __getstate__(self) -> dict[str, Any]:
        return {
            "kind": "ros",
            "topic": self.topic.topic,
            "msg_type": _qualname(self.topic.msg_type),
        }

    def broadcast(self, _: Out[DimosMsg], msg: DimosMsg) -> None:
        if self._ros is None:
            self.start()
            assert self._ros is not None  # for type narrowing
        self._ros.publish(self.topic, msg)

    def subscribe(
        self, callback: Callable[[DimosMsg], Any], selfstream: Stream[DimosMsg] | None = None
    ) -> Callable[[], None]:
        if self._ros is None:
            self.start()
            assert self._ros is not None  # for type narrowing
        return self._ros.subscribe(self.topic, lambda msg, topic: callback(msg))

    def start(self) -> None:
        if self._ros is None:
            self._ros = DimosROS(**self._kwargs)
            self._ros.start()

    def stop(self) -> None:
        if self._ros is not None:
            self._ros.stop()
            self._ros = None


if DDS_AVAILABLE:
    from dimos.protocol.pubsub.impl.ddspubsub import DDS, Topic as DDSTopic

    class DDSTransport(PubSubTransport[T]):
        def __init__(self, topic: str, type: type, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(DDSTopic(topic, type))
            self.dds = DDS(**kwargs)
            self._started: bool = False
            self._start_lock = threading.RLock()

        def __getstate__(self) -> dict[str, Any]:
            return {
                "kind": "dds",
                "topic": self.topic.topic,
                "type": _qualname(self.topic.type),
            }

        def start(self) -> None:
            with self._start_lock:
                if not self._started:
                    self.dds.start()
                    self._started = True

        def stop(self) -> None:
            with self._start_lock:
                if self._started:
                    self.dds.stop()
                    self._started = False

        def broadcast(self, _, msg) -> None:  # type: ignore[no-untyped-def]
            with self._start_lock:
                if not self._started:
                    self.start()
                self.dds.publish(self.topic, msg)

        def subscribe(
            self, callback: Callable[[T], None], selfstream: Stream[T] | None = None
        ) -> Callable[[], None]:
            with self._start_lock:
                if not self._started:
                    self.start()
                return self.dds.subscribe(self.topic, lambda msg, topic: callback(msg))


class ZenohTransport(PubSubTransport[T]): ...


def deserialize_transport(state: dict[str, Any]) -> PubSubTransport[Any] | None:
    """Inverse of `transport.__getstate__()`.

    Returns None if the state cannot be reconstructed (e.g. msg type can no longer
    be imported, or the kind is unknown). Caller is responsible for handling None.
    """
    kind = state.get("kind")
    if kind == "plcm":
        return pLCMTransport(state["topic"])
    if kind == "lcm":
        msg_type = _resolve_type(state["lcm_type"]) if state.get("lcm_type") else None
        if msg_type is None:
            return None
        return LCMTransport(state["topic"], msg_type)
    if kind == "jpeg_lcm":
        msg_type = _resolve_type(state["lcm_type"]) if state.get("lcm_type") else None
        if msg_type is None:
            return None
        return JpegLcmTransport(state["topic"], msg_type)
    if kind == "pshm":
        return pSHMTransport(state["topic"], default_capacity=state["capacity"])
    if kind == "shm":
        return SHMTransport(state["topic"], default_capacity=state["capacity"])
    if kind == "jpeg_shm":
        return JpegShmTransport(
            state["topic"],
            quality=state["quality"],
            default_capacity=state["capacity"],
        )
    if kind == "ros":
        msg_type = _resolve_type(state["msg_type"])
        if msg_type is None:
            return None
        return ROSTransport(state["topic"], msg_type)
    if kind == "dds" and DDS_AVAILABLE:
        msg_type = _resolve_type(state["type"])
        if msg_type is None:
            return None
        return DDSTransport(state["topic"], msg_type)  # type: ignore[name-defined]
    return None
