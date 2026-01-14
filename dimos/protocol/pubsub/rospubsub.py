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

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import threading
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rclpy = None  # type: ignore[assignment]
    SingleThreadedExecutor = None  # type: ignore[assignment, misc]
    Node = None  # type: ignore[assignment, misc]

from dimos.protocol.pubsub.spec import MsgT, PubSub, PubSubEncoderMixin, TopicT


# Type definitions for LCM and ROS messages
@runtime_checkable
class LCMMessage(Protocol):
    """Protocol for LCM message types (from dimos_lcm or lcm_msgs)."""

    msg_name: str
    __slots__: tuple[str, ...]


@runtime_checkable
class ROSMessage(Protocol):
    """Protocol for ROS message types."""

    def get_fields_and_field_types(self) -> dict[str, str]: ...


# Type aliases for clarity
Primitive: TypeAlias = int | float | str | bool | bytes | None
LCMMessageT = TypeVar("LCMMessageT", bound=LCMMessage)
ROSMessageT = TypeVar("ROSMessageT", bound=ROSMessage)

# Type caches for dynamic import optimization
_ros_type_cache: dict[str, type[ROSMessage]] = {}
_dimos_type_cache: dict[str, type[LCMMessage]] = {}

# Special type mappings for ROS1→ROS2 differences
# LCM uses ROS1-style message layout, ROS2 moved some types
_LCM_TO_ROS_PACKAGE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # (lcm_package, lcm_class) -> (ros_package, ros_class)
    ("std_msgs", "Time"): ("builtin_interfaces", "Time"),
    ("std_msgs", "Duration"): ("builtin_interfaces", "Duration"),
}


def _is_primitive(value: object) -> bool:
    """Check if value is a primitive type (not a message)."""
    return isinstance(value, (int, float, str, bool, bytes, type(None)))


def _get_ros_type(dimos_msg: LCMMessage) -> type[ROSMessage]:
    """Get the corresponding ROS type for a dimos message.

    Uses the module path of the dimos_lcm/lcm_msgs type to derive the ROS package.
    e.g., dimos_lcm.geometry_msgs.Vector3 -> geometry_msgs.msg.Vector3
         lcm_msgs.std_msgs.Header -> std_msgs.msg.Header
         lcm_msgs.std_msgs.Time -> builtin_interfaces.msg.Time (ROS2 special case)
    """
    dimos_type = type(dimos_msg)
    module_name = dimos_type.__module__  # e.g., "dimos_lcm.geometry_msgs.Vector3"
    class_name = dimos_type.__name__  # e.g., "Vector3"

    cache_key = f"{module_name}.{class_name}"
    if cache_key in _ros_type_cache:
        return _ros_type_cache[cache_key]

    # Parse module name to get package
    # Format: "dimos_lcm.package.ClassName" or "lcm_msgs.package.ClassName"
    parts = module_name.split(".")
    if len(parts) < 2 or parts[0] not in ("dimos_lcm", "lcm_msgs"):
        raise ValueError(f"Unexpected LCM module format: {module_name}")
    package = parts[1]  # e.g., "geometry_msgs", "std_msgs"

    # Check for special ROS1→ROS2 package remapping
    if (package, class_name) in _LCM_TO_ROS_PACKAGE_MAP:
        package, class_name = _LCM_TO_ROS_PACKAGE_MAP[(package, class_name)]

    # Import from ROS: geometry_msgs.msg.Vector3
    ros_module = importlib.import_module(f"{package}.msg")
    ros_type = getattr(ros_module, class_name)

    _ros_type_cache[cache_key] = ros_type
    return ros_type


def _get_dimos_type(ros_msg: ROSMessage) -> type[LCMMessage]:
    """Get the corresponding dimos_lcm type for a ROS message.

    Uses module path and class name to dynamically import the dimos_lcm type.
    e.g., geometry_msgs.msg._vector3.Vector3 -> dimos_lcm.geometry_msgs.Vector3
    """
    ros_type = type(ros_msg)
    module_name = ros_type.__module__  # e.g., "geometry_msgs.msg._vector3"
    class_name = ros_type.__name__  # e.g., "Vector3"

    cache_key = f"{module_name}.{class_name}"
    if cache_key in _dimos_type_cache:
        return _dimos_type_cache[cache_key]

    # Parse module to get package: "geometry_msgs.msg._vector3" -> "geometry_msgs"
    # Format is either "package.msg" or "package.msg._typename"
    parts = module_name.split(".")
    if len(parts) < 2 or (parts[1] != "msg" and "msg" not in parts):
        raise ValueError(f"Unexpected ROS module format: {module_name}")
    package = parts[0]  # e.g., "geometry_msgs"

    # Import from dimos_lcm: dimos_lcm.geometry_msgs.Vector3
    dimos_module = importlib.import_module(f"dimos_lcm.{package}")
    dimos_type = getattr(dimos_module, class_name)

    _dimos_type_cache[cache_key] = dimos_type
    return dimos_type


def _get_field_names(msg: LCMMessage | ROSMessage) -> list[str]:
    """Get field names from a message using __slots__ or get_fields_and_field_types."""
    # Try ROS-style introspection first
    if hasattr(msg, "get_fields_and_field_types"):
        return list(msg.get_fields_and_field_types().keys())
    # Fall back to __slots__ (LCM-style)
    if hasattr(msg, "__slots__"):
        return list(msg.__slots__)
    # Last resort: use __dict__
    return list(vars(msg).keys())


def _convert_to_ros(
    dimos_value: LCMMessage | Primitive | list[Any] | tuple[Any, ...],
) -> ROSMessage | Primitive | list[Any]:
    """Recursively convert a dimos/LCM value to ROS format."""
    # Primitives pass through
    if _is_primitive(dimos_value):
        return dimos_value  # type: ignore[return-value]

    # Handle lists/arrays
    if isinstance(dimos_value, (list, tuple)):
        return [_convert_to_ros(item) for item in dimos_value]

    # Handle message objects with msg_name (LCM messages)
    if isinstance(dimos_value, LCMMessage):
        ros_type = _get_ros_type(dimos_value)
        ros_msg = ros_type()
        ros_field_names = set(_get_field_names(ros_msg))

        for field_name in _get_field_names(dimos_value):
            # Skip fields that don't exist in ROS type (e.g., Header.seq in ROS2)
            if field_name not in ros_field_names:
                continue
            dimos_field_value = getattr(dimos_value, field_name)
            ros_field_value = _convert_to_ros(dimos_field_value)
            setattr(ros_msg, field_name, ros_field_value)

        return ros_msg

    # Unknown type, try to pass through
    return dimos_value  # type: ignore[return-value]


def _convert_to_dimos(
    ros_value: ROSMessage | Primitive | list[Any] | tuple[Any, ...],
) -> LCMMessage | Primitive | list[Any] | float:
    """Recursively convert a ROS value to dimos/LCM format."""
    # Primitives pass through
    if _is_primitive(ros_value):
        return ros_value  # type: ignore[return-value]

    # Handle lists/arrays
    if isinstance(ros_value, (list, tuple)):
        return [_convert_to_dimos(item) for item in ros_value]

    # Handle numpy arrays (common in ROS)
    if hasattr(ros_value, "tolist"):
        return ros_value.tolist()  # type: ignore[union-attr]

    # Handle ROS message objects
    ros_module = type(ros_value).__module__
    if ".msg" in ros_module or ros_module.startswith("builtin_interfaces"):
        # Special handling for builtin_interfaces.msg.Time -> convert to float
        if ros_module == "builtin_interfaces.msg._time" or type(ros_value).__name__ == "Time":
            return ros_value.sec + ros_value.nanosec / 1e9  # type: ignore[union-attr]

        dimos_type = _get_dimos_type(ros_value)  # type: ignore[arg-type]
        field_names = _get_field_names(ros_value)  # type: ignore[arg-type]

        # Construct dimos message with converted fields
        field_values: dict[str, Any] = {}
        for field_name in field_names:
            ros_field_value = getattr(ros_value, field_name)
            field_values[field_name] = _convert_to_dimos(ros_field_value)

        # Try keyword construction first, fall back to default + setattr
        try:
            return dimos_type(**field_values)
        except TypeError:
            dimos_msg = dimos_type()
            for field_name, value in field_values.items():
                setattr(dimos_msg, field_name, value)
            return dimos_msg

    # Unknown type, try to pass through
    return ros_value  # type: ignore[return-value]


@dataclass
class ROSTopic:
    """Topic descriptor for ROS pubsub."""

    topic: str
    ros_type: type
    qos: "QoSProfile | None" = None  # Optional per-topic QoS override


class RawROS(PubSub[ROSTopic, Any]):
    """ROS 2 PubSub implementation following the PubSub spec.

    This allows direct comparison of ROS messaging performance against
    native LCM and other pubsub implementations.
    """

    def __init__(
        self, node_name: str = "dimos_ros_pubsub", qos: "QoSProfile | None" = None
    ) -> None:
        """Initialize the ROS pubsub.

        Args:
            node_name: Name for the ROS node
            qos: Optional QoS profile (defaults to BEST_EFFORT for throughput)
        """
        if not ROS_AVAILABLE:
            raise ImportError("rclpy is not installed. ROS pubsub requires ROS 2.")

        self._node_name = node_name
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._running = False

        # Track publishers and subscriptions
        self._publishers: dict[str, Any] = {}
        self._subscriptions: dict[str, list[tuple[Any, Callable[[Any, ROSTopic], None]]]] = {}
        self._lock = threading.Lock()

        # QoS profile - use provided or default to best-effort for throughput
        if qos is not None:
            self._qos = qos
        else:
            self._qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )

    def start(self) -> None:
        """Start the ROS node and executor."""
        if self._running:
            return

        if not rclpy.ok():
            rclpy.init()

        self._node = Node(self._node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._running = True
        self._spin_thread = threading.Thread(target=self._spin, name="ros_pubsub_spin")
        self._spin_thread.start()

    def stop(self) -> None:
        """Stop the ROS node and clean up."""
        if not self._running:
            return

        self._running = False

        # Wake up the executor so spin thread can exit
        if self._executor:
            self._executor.wake()

        # Wait for spin thread to finish
        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)

        if self._executor:
            self._executor.shutdown()

        if self._node:
            self._node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        self._publishers.clear()
        self._subscriptions.clear()
        self._spin_thread = None

    def _spin(self) -> None:
        """Background thread for spinning the ROS executor."""
        while self._running and self._executor:
            self._executor.spin_once(timeout_sec=0)  # Non-blocking for max throughput

    def _get_or_create_publisher(self, topic: ROSTopic) -> Any:
        """Get existing publisher or create a new one."""
        if topic.topic not in self._publishers:
            qos = topic.qos if topic.qos is not None else self._qos
            self._publishers[topic.topic] = self._node.create_publisher(
                topic.ros_type, topic.topic, qos
            )
        return self._publishers[topic.topic]

    def publish(self, topic: ROSTopic, message: Any) -> None:
        """Publish a message to a ROS topic.

        Args:
            topic: ROSTopic descriptor with topic name and message type
            message: ROS message to publish
        """
        if not self._running or not self._node:
            return

        publisher = self._get_or_create_publisher(topic)
        publisher.publish(message)

    def subscribe(
        self, topic: ROSTopic, callback: Callable[[Any, ROSTopic], None]
    ) -> Callable[[], None]:
        """Subscribe to a ROS topic with a callback.

        Args:
            topic: ROSTopic descriptor with topic name and message type
            callback: Function called with (message, topic) when message received

        Returns:
            Unsubscribe function
        """
        if not self._running or not self._node:
            raise RuntimeError("ROS pubsub not started")

        with self._lock:

            def ros_callback(msg: Any) -> None:
                callback(msg, topic)

            qos = topic.qos if topic.qos is not None else self._qos
            subscription = self._node.create_subscription(
                topic.ros_type, topic.topic, ros_callback, qos
            )

            if topic.topic not in self._subscriptions:
                self._subscriptions[topic.topic] = []
            self._subscriptions[topic.topic].append((subscription, callback))

            def unsubscribe() -> None:
                with self._lock:
                    if topic.topic in self._subscriptions:
                        self._subscriptions[topic.topic] = [
                            (sub, cb)
                            for sub, cb in self._subscriptions[topic.topic]
                            if cb is not callback
                        ]
                        if self._node:
                            self._node.destroy_subscription(subscription)

            return unsubscribe


class LCM2ROSMixin(PubSubEncoderMixin[TopicT, MsgT]):
    """Mixin that converts between dimos_lcm (LCM-based) and ROS messages.

    This enables seamless interop: publish LCM messages to ROS topics
    and receive ROS messages as LCM messages.
    """

    def encode(self, msg: MsgT, *_: TopicT) -> ROSMessage:
        """Convert a dimos_lcm message to its equivalent ROS message.

        Args:
            msg: An LCM message (e.g., dimos_lcm.geometry_msgs.Vector3)

        Returns:
            The corresponding ROS message (e.g., geometry_msgs.msg.Vector3)
        """
        return _convert_to_ros(msg)  # type: ignore[return-value]

    def decode(self, msg: ROSMessage, _: TopicT | None = None) -> MsgT:
        """Convert a ROS message to its equivalent dimos_lcm message.

        Args:
            msg: A ROS message (e.g., geometry_msgs.msg.Vector3)

        Returns:
            The corresponding LCM message (e.g., dimos_lcm.geometry_msgs.Vector3)
        """
        return _convert_to_dimos(msg)  # type: ignore[return-value]


class DimosROS(
    RawROS,
    LCM2ROSMixin[ROSTopic, Any],
):
    """ROS PubSub with automatic dimos.msgs ↔ ROS message conversion."""

    pass


ROS = DimosROS
