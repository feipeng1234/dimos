# How to Integrate a New Robot with DimOS

This guide walks you through integrating any robot — humanoid, quadruped, drone, wheeled base, or manipulator arm — with DimOS. It's based on real integration experiences with the Unitree Go2 / G1 / B1, Galaxea R1 Pro, Booster K1, AgileX Piper, UFactory xArm, OpenArm, and others.

**Time estimate**: 1–5 days depending on your robot's interface quality.

**What you'll build**:
- A **Connection Module** that owns your robot's SDK / network connection and exposes its sensors + command sinks as DimOS streams.
- A small amount of **coordinator wiring** that bridges your Module into the `ControlCoordinator` via the generic `transport_lcm` adapter.
- **Blueprints** that compose the system with perception, navigation, and agent capabilities.

> **Branch note**: examples target the current `dev` branch APIs. Create integration PRs from `dev` and target `dev`. Don't copy snippets from older feature branches without adapting them to the current Module, Blueprint, and ControlCoordinator APIs.

---

## The Recommended Architecture

DimOS has converged on a single pattern for robot integration, anchored by the Unitree G1 (whole-body) and Go2 (mobile base). **Use this pattern for any new robot unless you have a strong reason not to.**

```
┌──────────────────────────────────────────────────────────────┐
│  Connection Module          (one per robot)                  │
│  Owns:                                                        │
│    • The vendor connection (SDK / DDS / WebRTC / WebSocket)  │
│    • Sensor Out streams  (camera, lidar, odom, imu, joints)  │
│    • Command In streams  (cmd_vel, motor_command, …)         │
│    • @skill methods exposed to the agent                     │
└──────────────────────────┬───────────────────────────────────┘
                           │  pub/sub over LCM (or DDS)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Transport Adapter        (generic, you DON'T write this)    │
│    adapter_type="transport_lcm"                              │
│  Picked by HardwareType:                                     │
│    HardwareType.BASE         → twist transport adapter       │
│    HardwareType.WHOLE_BODY   → whole-body transport adapter  │
│    HardwareType.MANIPULATOR  → no transport adapter yet —    │
│                                use a direct adapter (Arms).  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  ControlCoordinator + Tasks                                  │
│  High-rate loop (e.g. 500 Hz on G1, 100 Hz default).         │
│  Arbitrates tasks: servo / velocity / trajectory.            │
└──────────────────────────────────────────────────────────────┘

                  ▲ assembled by ▲

┌──────────────────────────────────────────────────────────────┐
│  Coordinator Blueprint                                       │
│  autoconnect(YourConnection.blueprint(),                     │
│              ControlCoordinator.blueprint(...))              │
│   .remappings([...])    # fix stream-name collisions         │
│   .transports({...})    # wire Module streams ↔ LCM topics   │
└──────────────────────────────────────────────────────────────┘
```

**Why this shape?** The Connection Module is the only thing that has to know how to talk to your specific robot. The transport adapter is a generic shim — you reuse `transport_lcm`, never write your own. The `ControlCoordinator` gives you a uniform task model across all robots. Splitting along these lines lets you swap robots, swap simulators, run distributed across processes, and reuse the perception / navigation / agent stack unchanged.

**Reference implementations** — read these alongside this guide:

| Pattern | File | What to learn from it |
|--------|------|------|
| Whole-body, joint-level | `dimos/robot/unitree/g1/blueprints/basic/unitree_g1_coordinator.py` | Joint-level motor IO at 500 Hz, motor_states + IMU wiring |
| Twist base | `dimos/robot/unitree/go2/blueprints/basic/unitree_go2_coordinator.py` | Mobile-base velocity control with cmd_vel + odom + remappings |
| Connection Module — base | `dimos/robot/unitree/go2/connection.py` | Stream declarations, @skill exposure, lifecycle |
| Connection Module — whole-body | `dimos/robot/unitree/g1/wholebody_connection.py` | motor_command In + motor_states/imu Out at high rate |

### When to *not* use this pattern

**Manipulator arms.** Manipulator adapters are *much* more complex than base or whole-body ones (they carry grippers, cartesian pose, force/torque, control-mode switching, enable/disable, and error clearing on top of joint IO), so designing a clean stream-based contract for them is non-trivial and hasn't been done yet. xArm, AgileX Piper, and OpenArm use the **direct adapter** pattern in the meantime — the adapter opens the SDK in-process and the `ControlCoordinator` calls it directly, no Module in between. See [Direct Adapter Pattern (Arms)](#direct-adapter-pattern-arms).

**One-off prototyping.** If you're just trying to see the robot move, a standalone Connection Module without a coordinator works fine — you simply skip Phase 5. But anything that ships should go through the coordinator.

---

## Before You Start

### What you need from your robot vendor

| Item | Why | Priority |
|------|-----|----------|
| Python SDK or ROS 2 interface | Primary control channel | Required |
| Network access (ethernet / WiFi) | Connect your dev machine to the robot | Required |
| Camera stream (RTSP / ROS / WebSocket) | Visual perception | Required for agentic use |
| Odometry data (position + orientation) | Navigation and mapping | Required for navigation |
| LiDAR point cloud | Obstacle avoidance and SLAM | Required for navigation |
| IMU data | Localization accuracy | Recommended |
| URDF model file | Motion planning and collision checking | Required for arm planning |
| API documentation | Know what commands the robot accepts | Very helpful |
| SSH access to the robot's onboard computer | Debug, check topics, install software | Very helpful |

> **Tip**: if your robot doesn't provide odometry, LiDAR, or IMU natively, add external sensors (Livox MID-360 for LiDAR + IMU, Intel RealSense for depth, etc.). DimOS has built-in modules for these — see [External Sensors](#external-sensors).

### What you need on your dev machine

- Ubuntu 22.04 / 24.04 (recommended) or macOS.
- Python 3.12.
- DimOS installed. For repo development: `uv sync --all-extras --no-extra dds`. For package installs, pick the extras your robot needs, e.g. `uv pip install 'dimos[base,unitree,manipulation]'`.
- Your robot's vendor SDK installed.

---

## The Integration Flow

Every robot integration follows the same phases regardless of form factor:

```
Phase 1: Connect              Can you ping the robot? Can you see its API?
   │
Phase 2: Read sensors         Can you receive camera, joint state, odometry?
   │
Phase 3: Send commands        Can you make the robot move?
   │
Phase 4: Connection Module    Wrap connection + sensors + commands in a Module
   │
Phase 5: Coordinator wiring   Pick the transport adapter, write the blueprint
   │
Phase 6: Test on hardware     Validate end-to-end through DimOS
   │
Phase 7: Layer on capability  Smart (nav) and agentic (LLM) blueprints
```

The first three phases are vendor-side validation — you're proving things work *outside* DimOS. Only after all three pass do you start writing DimOS code.

---

## Phase 1: Connect to the Robot

Always the first thing you do, and always where the first surprises happen.

### 1a. Establish network connectivity

```bash
# If the robot's IP is unknown, try common defaults:
ping 192.168.123.1      # Common Unitree default
ping 192.168.1.1        # Common default gateway

# If ping doesn't work, scan the subnet:
sudo arp-scan --interface=eth0 192.168.123.0/24

# Or watch for traffic:
sudo tcpdump -i eth0 -n | head -20
```

> **Real example (R1 Pro)**: the robot had no known IP on ethernet. We used `tcpdump` and `arp -a` to discover it, then assigned a static IP via netplan.

### 1b. Verify you can talk to the robot's API

**ROS 2:**
```bash
source /opt/ros/humble/setup.bash         # or jazzy
export ROS_DOMAIN_ID=0                    # match the robot's domain ID
ros2 topic list --no-daemon

# You should see things like /cmd_vel, /joint_states, /camera/image_raw, /odom, /scan or /points.
```

**Python SDK:**
```python
from your_robot_sdk import RobotClient

robot = RobotClient("192.168.1.100")
robot.connect()
print(robot.get_status())
robot.disconnect()
```

**Custom UDP/TCP/WebSocket:**
```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b'\x00\x01', ("192.168.1.100", 8000))
data, addr = sock.recvfrom(1024)
print(f"Got response: {data.hex()}")
```

### Common Phase 1 problems

| Problem | Solution |
|---------|----------|
| Can't ping the robot | Check cable, check subnet (robot might be on `192.168.123.x` not `192.168.1.x`) |
| ROS 2 topics not visible | Check `ROS_DOMAIN_ID`. Check `ROS_LOCALHOST_ONLY` is not set to 1 on the robot |
| Topics visible but empty | DDS middleware mismatch — both sides must use the same RMW (FastDDS or CycloneDDS) |
| Topics visible from robot but not from laptop | FastDDS sending multicast on the wrong interface — create a FastDDS XML profile bound to the correct NIC |

> **Real example (R1 Pro)**: five separate network issues had to be solved before topics flowed: `ROS_LOCALHOST_ONLY=1` in the robot's bashrc, CycloneDDS/FastDDS EDP incompatibility, FastDDS wrong interface, `interfaceWhiteList` renamed in FastDDS 3.x, and a misleading discovery server. Each one silently broke topic visibility.

---

## Phase 2: Read Sensor Data

Verify each sensor stream **outside DimOS** first. You're proving the data exists; the Connection Module will own the actual subscriptions in Phase 4.

### 2a. Camera

**ROS 2 topic:**
```python
import rclpy
from sensor_msgs.msg import CompressedImage

rclpy.init()
node = rclpy.create_node('camera_test')
def cb(msg): print(f"Got frame: {len(msg.data)} bytes")
node.create_subscription(CompressedImage, '/camera/image/compressed', cb, 10)
rclpy.spin_once(node, timeout_sec=5.0)
```

**RTSP:**
```python
import av
container = av.open("rtsp://192.168.1.100:8554/video1")
for frame in container.decode(video=0):
    print(f"Got frame: {frame.to_ndarray(format='bgr24').shape}")
    break
```

**WebSocket:**
```python
import asyncio, websockets

async def test():
    async with websockets.connect("ws://192.168.1.100:8080/video") as ws:
        data = await ws.recv()
        print(f"Got frame: {len(data)} bytes")

asyncio.run(test())
```

### 2b. Odometry

```bash
ros2 topic echo /odom --once
# Want: position (x, y, z) and orientation (quaternion)
```

If the robot doesn't publish odometry:
1. **External SLAM** — add a LiDAR (e.g. MID-360) and use FAST-LIO2 (already in DimOS).
2. **Dead reckoning** — integrate velocity commands over time (inaccurate, fallback only).

### 2c. LiDAR / Point Cloud

```bash
ros2 topic echo /scan --once          # 2D laser
ros2 topic echo /points --once        # 3D cloud
```

### 2d. Joint States (arms / humanoids)

```bash
ros2 topic echo /joint_states --once
# Want: name[], position[], velocity[], effort[]
```

For joint-level whole-body control (G1-style), you also need:
- **IMU** — usually `/imu` or part of the SDK's state message.
- **Per-joint feedback at the same rate as your control loop** (G1 runs at 500 Hz). If your SDK only exposes 100 Hz state for a 500 Hz controller, you're going to have a bad time.

---

## Phase 3: Send Motion Commands

Now make the robot move. **Start small** — send a tiny velocity for a short duration, with a hand on the e-stop.

### Mobile bases (wheeled, legged, quadruped)

**ROS 2:**
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.1}, angular: {z: 0.0}}" --once
```

**Python SDK:**
```python
robot.move(linear_x=0.1, angular_z=0.0)
time.sleep(1.0)
robot.stop()
```

### Manipulator arms

```python
current = robot.get_joint_positions()
print(f"Current joints: {current}")

target = list(current)
target[0] += 0.1   # radians — verify your SDK's units
robot.set_joint_positions(target)
```

### Joint-level whole-body

If your robot accepts per-joint `(q, dq, kp, kd, tau)` commands (Unitree-style low-level):
```python
# Stand-still pseudocode — send zero feedforward and weak gains
for i in range(N):
    cmd[i] = (q_current[i], 0.0, kp_low, kd_low, 0.0)
sdk.send_motor_command(cmd)
```

Verify at low gains *before* turning them up. The sentinels for "no command on this DOF" are `POS_STOP=2.146e9` and `VEL_STOP=16000.0` — same as Unitree SDK.

### Common Phase 3 problems

| Problem | Solution |
|---------|----------|
| Command sent but robot doesn't move | Look for "gates" — subscriber-count requirements, braking flags, mode bits. Some robots need several conditions satisfied simultaneously. |
| Robot moves erratically | Unit mismatch (degrees vs radians, mm vs m, normalized vs absolute). Convert at the boundary. |
| Robot moves once then stops | Some SDKs require continuous commands at a minimum rate (e.g. 20 Hz). Add a watchdog/control loop. |
| Need to e-stop | Know how *before* you test motion. Hardware e-stop button is best. |

> **Real example (R1 Pro)**: the chassis had three hidden gates that all had to be unlocked simultaneously — subscriber count, braking-mode flag, and an acceleration limit that defaulted to zero. Multiple sessions of investigation including binary disassembly of the control node.

> **Real example (M20)**: velocity commands had to be normalized to `[-1, 1]` for UDP mode but sent as absolute m/s for DDS navigation mode. Two completely different code paths for the same robot.

---

## Phase 4: Write the Connection Module

This is the only file you write per-robot. It owns the whole vendor surface: open the connection, stream sensor data out as DimOS streams, accept commands in, and expose `@skill` methods for the agent.

Place it at `dimos/robot/yourvendor/yourmodel/connection.py`. There are two flavors depending on *what level you control the robot at*.

### Flavor A — Mobile base / quadruped (`HardwareType.BASE`)

Use this if the robot accepts twist-style velocity commands. Reference: `GO2Connection`.

```python
"""YourRobot connection module — mobile base."""

from threading import Event, Thread
from typing import Any
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.spec.perception import Camera


class YourRobotConfig(ModuleConfig):
    ip: str = "192.168.1.100"
    camera_port: int = 8554


class YourRobotConnection(Module, Camera):
    """Connection module for YourRobot. Owns SDK + sensors + cmd_vel."""

    config: YourRobotConfig

    # Sensor streams OUT of the robot
    color_image: Out[Image]
    camera_info: Out[CameraInfo]
    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]

    # Command stream INTO the robot
    cmd_vel: In[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sdk = None
        self._stop = Event()

    @rpc
    def start(self) -> None:
        super().start()
        self._stop.clear()

        from your_robot_sdk import RobotClient
        self._sdk = RobotClient(self.config.ip)
        self._sdk.connect()

        self._camera_thread = Thread(target=self._stream_camera, daemon=True)
        self._camera_thread.start()

        self._disposables.add(Disposable(self.cmd_vel.subscribe(self._on_cmd_vel)))

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._sdk:
            self._sdk.stop()
            self._sdk.disconnect()
            self._sdk = None
        if getattr(self, "_camera_thread", None) and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=1.0)
        super().stop()

    def _stream_camera(self) -> None:
        import av
        container = av.open(f"rtsp://{self.config.ip}:{self.config.camera_port}/video1")
        for frame in container.decode(video=0):
            if self._stop.is_set():
                break
            img = frame.to_ndarray(format='rgb24')
            self.color_image.publish(Image.from_numpy(
                img, format=ImageFormat.RGB, frame_id="camera_optical",
            ))

    def _on_cmd_vel(self, twist: Twist) -> None:
        if self._sdk:
            self._sdk.move(
                linear_x=twist.linear.x,
                linear_y=twist.linear.y,
                angular_z=twist.angular.z,
            )

    # Agent-callable skills
    @skill
    def walk(self, x: float, y: float = 0.0, yaw: float = 0.0) -> str:
        """Walk in the specified direction.

        Args:
            x: Forward speed (m/s).
            y: Lateral speed (m/s).
            yaw: Rotation speed (rad/s).
        """
        self._on_cmd_vel(Twist(
            linear=Vector3(x, y, 0.0), angular=Vector3(0.0, 0.0, yaw),
        ))
        return f"Walking: x={x}, y={y}, yaw={yaw}"

    @skill
    def stop_moving(self) -> str:
        """Stop all motion."""
        self._on_cmd_vel(Twist())
        return "Stopped."
```

### Flavor B — Joint-level whole-body (`HardwareType.WHOLE_BODY`)

Use this if you want low-level per-joint `(q, dq, kp, kd, tau)` control. Reference: `G1WholeBodyConnection`.

The shape is the same as Flavor A but the streams differ:

```python
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray

class YourHumanoidConnection(Module, Camera):
    # Sensor streams OUT
    motor_states: Out[JointState]      # per-joint q, dq, tau at control rate
    imu: Out[Imu]                      # quaternion + gyro + accel
    color_image: Out[Image]            # optional, if onboard cameras
    camera_info: Out[CameraInfo]

    # Command stream IN
    motor_command: In[MotorCommandArray]   # per-joint (q, dq, kp, kd, tau)

    @rpc
    def start(self) -> None:
        super().start()
        # 1. Open the vendor SDK / DDS channel
        # 2. Spawn a high-rate thread that publishes motor_states + imu
        # 3. Subscribe to motor_command and forward to the SDK
        ...
```

The transport adapter in Phase 5 expects exactly these stream names (`motor_states`, `imu`, `motor_command`). Match them.

### Sensors and ROS-based robots

If your robot publishes sensors via ROS 2, **do the rclpy subscription inside the Connection Module** and republish onto your DimOS Out streams. Keep all robot-specific code in one file. (`ROSTransport` at the blueprint level still exists — it's useful for sensors-as-modules and external sensors — but the canonical place for *your robot's* sensors is the Connection Module.)

---

## Phase 5: Wire the Coordinator

The transport adapter is generic — you do **not** write one per robot. You pick `adapter_type="transport_lcm"` in your blueprint, and the `ControlCoordinator` instantiates the right one based on `HardwareType`:

| HardwareType | Transport adapter | Subscribes to | Publishes to |
|--|--|--|--|
| `BASE` | `dimos/hardware/drive_trains/transport/adapter.py` | `/{hardware_id}/odom` | `/{hardware_id}/cmd_vel` |
| `WHOLE_BODY` | `dimos/hardware/whole_body/transport/adapter.py` | `/{hardware_id}/motor_states`, `/{hardware_id}/imu` | `/{hardware_id}/motor_command` |
| `MANIPULATOR` | (none yet — use direct adapter, see below) | — | — |

Your job in the blueprint is to wire those LCM topics to the Module's stream names with `.transports({...})`.

### 5a. Coordinator blueprint — whole-body (G1)

```python
"""Unitree G1 ControlCoordinator: connection + servo task via LCM bridge."""

import os
from dimos.control.components import HardwareComponent, HardwareType, make_humanoid_joints
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.MotorCommandArray import MotorCommandArray
from dimos.robot.unitree.g1.wholebody_connection import G1WholeBodyConnection

_g1_joints = make_humanoid_joints("g1")

unitree_g1_coordinator = (
    autoconnect(
        G1WholeBodyConnection.blueprint(
            release_sport_mode=True,
            network_interface=os.getenv("ROBOT_INTERFACE", ""),
        ),
        ControlCoordinator.blueprint(
            tick_rate=500,
            hardware=[
                HardwareComponent(
                    hardware_id="g1",
                    hardware_type=HardwareType.WHOLE_BODY,
                    joints=_g1_joints,
                    adapter_type="transport_lcm",
                ),
            ],
            tasks=[
                TaskConfig(name="servo_g1", type="servo",
                           joint_names=_g1_joints, priority=10),
            ],
        ),
    )
    .transports({
        ("motor_states", JointState):  LCMTransport("/g1/motor_states",  JointState),
        ("imu",          Imu):         LCMTransport("/g1/imu",           Imu),
        ("motor_command", MotorCommandArray): LCMTransport("/g1/motor_command", MotorCommandArray),
        ("joint_state",  JointState):  LCMTransport("/coordinator/joint_state", JointState),
        ("joint_command", JointState): LCMTransport("/g1/joint_command", JointState),
    })
)
```

### 5b. Coordinator blueprint — mobile base (Go2)

```python
"""Unitree Go2 ControlCoordinator: GO2Connection + velocity task via LCM bridge."""

from dimos.control.components import HardwareComponent, HardwareType, make_twist_base_joints
from dimos.control.coordinator import ControlCoordinator, TaskConfig
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.go2.connection import GO2Connection

_go2_joints = make_twist_base_joints("go2")

unitree_go2_coordinator = (
    autoconnect(
        GO2Connection.blueprint(),
        ControlCoordinator.blueprint(
            hardware=[
                HardwareComponent(
                    hardware_id="go2",
                    hardware_type=HardwareType.BASE,
                    joints=_go2_joints,
                    adapter_type="transport_lcm",
                ),
            ],
            tasks=[
                TaskConfig(name="vel_go2", type="velocity",
                           joint_names=_go2_joints, priority=10),
            ],
        ),
    )
    # Module's cmd_vel/odom collide with coordinator's — rename them
    .remappings([
        (GO2Connection, "cmd_vel", "go2_cmd_vel"),
        (GO2Connection, "odom",    "go2_odom"),
    ])
    .transports({
        ("cmd_vel",        Twist):       LCMTransport("/cmd_vel", Twist),
        ("twist_command",  Twist):       LCMTransport("/cmd_vel", Twist),
        ("go2_cmd_vel",    Twist):       LCMTransport("/go2/cmd_vel", Twist),
        ("go2_odom",       PoseStamped): LCMTransport("/go2/odom",    PoseStamped),
        ("joint_state",    JointState):  LCMTransport("/coordinator/joint_state", JointState),
    })
    .global_config(obstacle_avoidance=False)
)
```

### Things to notice

- **`tick_rate`** — `500` for joint-level whole-body, default (~100) for twist bases. Check your robot's actual control bandwidth before going high.
- **`TaskConfig.type`** — `"servo"` for whole-body joint control, `"velocity"` for twist bases, `"trajectory"` for arm waypoint following.
- **`.remappings(...)`** — needed when your Module's stream names collide with coordinator-side names (Go2's `cmd_vel`/`odom` clash with the coordinator's). Whole-body modules avoid this by using distinct names (`motor_states`, `motor_command`).
- **`.transports(...)`** — every stream that needs to leave the in-process bus gets an LCM topic. The transport adapter and the Module talk through these topics.
- **`network_interface`** — Cyclonedds needs to be pinned to a NIC on multi-NIC hosts. Pass it through env (`ROBOT_INTERFACE=eth0 dimos run …`).

---

## Direct Adapter Pattern (Arms)

Manipulator arms (xArm, AgileX Piper, OpenArm) currently bypass the Module + Transport pipeline. Instead the adapter opens the SDK in-process and the `ControlCoordinator` calls into it directly. This is the legacy pattern — it works and is well-supported, but expect it to be migrated to the canonical pipeline once a `TransportManipulatorAdapter` lands.

Place your adapter at `dimos/hardware/manipulators/yourarm/adapter.py`:

```python
"""YourArm hardware adapter — implements ManipulatorAdapter via duck typing."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dimos.hardware.manipulators.registry import AdapterRegistry

from dimos.hardware.manipulators.spec import ControlMode, JointLimits, ManipulatorInfo


class YourArmAdapter:
    def __init__(self, address: str, dof: int = 6) -> None:
        self._address = address
        self._dof = dof
        self._sdk = None
        self._mode = ControlMode.POSITION

    def connect(self) -> bool:
        try:
            from yourarm_sdk import YourArmSDK
            self._sdk = YourArmSDK(self._address)
            self._sdk.connect()
            return self._sdk.is_alive()
        except ImportError:
            print("ERROR: yourarm-sdk not installed")
            return False

    def disconnect(self) -> None:
        if self._sdk:
            self._sdk.disconnect()
            self._sdk = None

    def is_connected(self) -> bool:
        return self._sdk is not None and self._sdk.is_alive()

    def get_info(self) -> ManipulatorInfo:
        return ManipulatorInfo(vendor="YourVendor", model="YourModel", dof=self._dof)

    def get_dof(self) -> int:
        return self._dof

    def get_limits(self) -> JointLimits:
        return JointLimits(
            position_lower=[-math.pi] * self._dof,
            position_upper=[ math.pi] * self._dof,
            velocity_max=[   math.pi] * self._dof,
        )

    def set_control_mode(self, mode: ControlMode) -> bool:
        self._mode = mode
        return True

    def get_control_mode(self) -> ControlMode:
        return self._mode

    # SI units (rad, rad/s, Nm) at the boundary
    def read_joint_positions(self) -> list[float]:
        raw = self._sdk.get_joint_positions()
        return [math.radians(p) for p in raw[:self._dof]]   # SDK in degrees → rad

    def read_joint_velocities(self) -> list[float]:
        return [0.0] * self._dof

    def read_joint_efforts(self) -> list[float]:
        return [0.0] * self._dof

    def read_state(self) -> dict[str, int]:
        return {"mode": 0, "state": 0}

    def read_error(self) -> tuple[int, str]:
        return (0, "")

    def write_joint_positions(self, positions: list[float], velocity: float = 1.0) -> bool:
        return self._sdk.set_joint_positions([math.degrees(p) for p in positions])

    def write_joint_velocities(self, velocities: list[float]) -> bool:
        return False

    def write_stop(self) -> bool:
        return self._sdk.emergency_stop()

    def write_enable(self, enable: bool) -> bool:
        return self._sdk.enable_motors(enable)

    def read_enabled(self) -> bool:
        return self._sdk.motors_enabled() if self._sdk else False

    def write_clear_errors(self) -> bool:
        return self._sdk.clear_errors() if self._sdk else False

    # Optional capabilities
    def read_cartesian_position(self) -> dict[str, float] | None: return None
    def write_cartesian_position(self, pose, velocity=1.0) -> bool: return False
    def read_gripper_position(self) -> float | None: return None
    def write_gripper_position(self, position: float) -> bool: return False
    def read_force_torque(self) -> list[float] | None: return None


def register(registry: "AdapterRegistry") -> None:
    registry.register("yourarm", YourArmAdapter)
```

The blueprint then references `adapter_type="yourarm"` in the `HardwareComponent`, just like `transport_lcm` for whole-body / base — same `ControlCoordinator`, just a different adapter class behind the scenes.

> **Why two patterns?** Arms have synchronous high-frequency joint IO that benefits from being in-process with the coordinator's tick loop. The transport-bridge pattern adds a ~ms of LCM latency that's fine for whole-body humanoids (already at 500 Hz with high-rate motor channels) but unhelpful for arms. Once we settle the right interface for `MotorCommandArray`-style arm control, the manipulator transport adapter will land and arms will move to the canonical pattern.

---

## Phase 6: Test on Hardware

Build up incrementally — five sequential tests, each building on the previous:

| Test | What it validates | Safe? |
|------|-------------------|-------|
| 1. Discovery | Can you see the robot's API/topics? | Yes (read-only) |
| 2. Sensor read | Can you receive camera/joint/odometry data? | Yes (read-only) |
| 3. Small motion | Does a tiny velocity command work? | Mostly safe |
| 4. Full motion | Do complex commands work correctly? | Use caution |
| 5. DimOS integration | Does the full Module + Coordinator work? | Use caution |

**Always test read-only operations first.** Example skeleton:

```python
"""Test 1: connectivity + sensors."""

def main() -> bool:
    from your_robot_sdk import RobotClient

    robot = RobotClient("192.168.1.100")
    assert robot.connect(), "Failed to connect"

    print(f"Status: {robot.get_status()}")
    frame = robot.get_camera_frame()
    print(f"Camera: {frame.shape if frame is not None else 'None'}")

    robot.disconnect()
    return True

if __name__ == "__main__":
    print("PASS" if main() else "FAIL")
```

Then validate the DimOS path:

```bash
# Auto-generates dimos/robot/all_blueprints.py
uv run pytest dimos/robot/test_all_blueprints_generation.py

# Run end to end
ROBOT_INTERFACE=eth0 dimos run yourmodel-coordinator
```

---

## Phase 7: Layer on Capability

Once the basic coordinator blueprint runs, layer on capability by composing more blueprints with `autoconnect`. Build them up in the order: **basic → smart → agentic**.

### Smart blueprint — adds navigation

```python
from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels import VoxelGridMapper
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner

yourmodel_smart = autoconnect(
    yourmodel_coordinator,
    VoxelGridMapper.blueprint(),
    CostMapper.blueprint(),
    ReplanningAStarPlanner.blueprint(),
).global_config(n_workers=4, robot_model="yourmodel")
```

### Agentic blueprint — adds MCP tools + LLM client

```python
from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer

YOURMODEL_SYSTEM_PROMPT = """
You are controlling YourRobot. Use the available tools safely and briefly
explain physical actions before executing them.
"""

yourmodel_agentic = autoconnect(
    yourmodel_smart,
    McpServer.blueprint(),
    McpClient.blueprint(system_prompt=YOURMODEL_SYSTEM_PROMPT),
).global_config(n_workers=8, robot_model="yourmodel")
```

The `@skill` methods you defined on the Connection Module are exposed automatically.

---

## External Sensors

If your robot doesn't ship LiDAR / odometry / IMU / depth, you can add external sensors. DimOS has built-in modules for the common ones.

> Use the **raw driver** (Mid360) **or** the **integrated SLAM** module (FastLio2) — not both in the same blueprint, unless you intentionally want two independent Livox consumers.

### Livox MID-360 (LiDAR + IMU)

Most common external sensor. Native C++ driver + FAST-LIO2 config ready to go.

**Raw LiDAR + IMU:**
```python
from dimos.hardware.sensors.lidar.livox.module import Mid360

yourmodel_with_lidar = autoconnect(
    yourmodel_coordinator,
    Mid360.blueprint(host_ip="192.168.1.5", lidar_ip="192.168.1.155"),
    VoxelGridMapper.blueprint(),
    CostMapper.blueprint(),
)
```
Gives you `lidar: Out[PointCloud2]` (~10 Hz) and `imu: Out[Imu]` (~200 Hz).

**FAST-LIO2 SLAM:**
```python
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2

yourmodel_with_slam = autoconnect(
    yourmodel_coordinator,
    FastLio2.blueprint(host_ip="192.168.1.5", lidar_ip="192.168.1.155"),
    VoxelGridMapper.blueprint(),
    CostMapper.blueprint(),
)
```
Gives you `lidar: Out[PointCloud2]`, `odometry: Out[nav_msgs.Odometry]`, optional `global_map: Out[PointCloud2]`.

> **Navigation note**: the native nav stack consumes `odom: PoseStamped`. FAST-LIO2 publishes `nav_msgs.Odometry` — add a small conversion module or have your robot's Connection Module publish a `PoseStamped` to feed `ReplanningAStarPlanner` directly.

### Intel RealSense (depth)

```python
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera

yourmodel_with_realsense = autoconnect(
    yourmodel_coordinator,
    RealSenseCamera.blueprint(
        width=848, height=480, fps=15,
        enable_depth=True,
        base_frame_id="base_link",
    ),
)
```
Gives color + depth `Image` (depth aligned to color), `CameraInfo` for both, optional `PointCloud2`.

> **Real example (xArm grasping)**: the xArm manipulation blueprint mounts a RealSense on the end-effector with a calibrated transform for eye-in-hand grasping. See `dimos/manipulation/blueprints.py`.

### ZED (stereo + tracking)

```python
from dimos.hardware.sensors.camera.zed.camera import ZEDCamera

yourmodel_with_zed = autoconnect(
    yourmodel_coordinator,
    ZEDCamera.blueprint(
        width=1280, height=720, fps=15,
        depth_mode="NEURAL",
        enable_tracking=True,
        enable_imu_fusion=True,
        base_frame_id="base_link",
    ),
)
```
Gives color + depth `Image`, `CameraInfo`, built-in visual-inertial odometry, optional positional tracking with area memory.

### Mounting

1. **Mount rigidly** — vibration causes noisy data.
2. **Know the transform** — sensor pose relative to `base_link`.
3. **Power** — most sensors need USB or ethernet plus separate power.
4. **Bandwidth** — LiDAR + cameras can saturate a USB hub. Use separate USB controllers or ethernet.

---

## Form Factor Reference

The integration flow is the same for every form factor; the emphasis differs.

### Quadrupeds (Go2, M20)

- Gait switching (walk, trot, stair-climb), motion states (stand, sit, lie down) — expose as `@skill`.
- Usually SDK-based velocity control, not ROS.
- Often need heartbeat/keepalive messages.
- **Pattern**: Connection Module + `HardwareType.BASE` + `transport_lcm`.

### Humanoids (G1, R1 Pro)

- Multiple subsystems (arms, chassis/legs, torso, grippers).
- ControlCoordinator orchestrates them at high rate (G1 servo at 500 Hz).
- Joint-level control for arms, velocity for locomotion. May need URDF for arm planning.
- **Pattern (G1-style)**: single whole-body Connection Module + `HardwareType.WHOLE_BODY` + `transport_lcm`.
- **Pattern (R1 Pro-style, legacy)**: separate adapters per subsystem + coordinator. Will migrate.

### Wheeled bases / AMRs

- Simplest: just `cmd_vel` + sensors.
- Often have ROS 2 built-in.
- **Pattern**: same as quadrupeds — `BASE` + `transport_lcm`.

### Manipulator arms (xArm, Piper, OpenArm)

- Joint position / velocity control + gripper.
- URDF for motion planning (Drake).
- Unit conversion is critical (degrees vs radians, mm vs m).
- **Pattern**: direct adapter (legacy until `TransportManipulatorAdapter` lands).

### Drones

- MAVLink protocol (standard across many drones).
- 3D velocity (including altitude), GPS localization, flight modes / arming.
- **Pattern**: Connection Module + `HardwareType.BASE` (extended) + `transport_lcm`.

---

## File Checklist

### Module + Coordinator path (recommended)

- [ ] `dimos/robot/yourvendor/yourmodel/connection.py` — Connection Module
- [ ] `dimos/robot/yourvendor/yourmodel/blueprints/basic/yourmodel_coordinator.py` — Coordinator blueprint
- [ ] `dimos/robot/yourvendor/yourmodel/blueprints/smart/...` — Optional nav/perception variants
- [ ] `dimos/robot/yourvendor/yourmodel/blueprints/agentic/...` — Optional LLM/MCP variants
- [ ] `pyproject.toml` — vendor SDK in optional dependencies (if needed)
- [ ] `scripts/yourmodel_test/` — Test scripts (recommended)

### Direct adapter path (arms only)

- [ ] `dimos/hardware/manipulators/yourarm/adapter.py` — Adapter + `register()` hook
- [ ] `dimos/robot/yourvendor/yourmodel/blueprints.py` — Coordinator blueprint with `adapter_type="yourarm"`
- [ ] `pyproject.toml` — vendor SDK in optional dependencies

### Verification

- [ ] `uv run pytest dimos/robot/test_all_blueprints_generation.py` passes.
- [ ] `dimos run yourmodel-coordinator` starts and shows the camera feed (and motor_states / odom on LCM).
- [ ] Robot responds to a test motion command.

> **Note on `__init__.py`**: dimos uses PEP 420 namespace packages. Don't create `__init__.py` files for new packages.

---

## Lessons from Real Integrations

Hard-won lessons from integrating 10+ robots with DimOS.

### Network issues will eat most of your time

Every single integration burned days on connectivity. Budget for it.

- **Always use `--no-daemon`** when running `ros2 topic list` — the daemon is unreliable for cross-machine discovery.
- **Pin DDS middleware**. If the robot uses FastDDS, use FastDDS on your laptop too. CycloneDDS / FastDDS interop is not guaranteed.
- **Create a FastDDS XML profile** that binds to the correct NIC. Laptops with WiFi + ethernet + VPN confuse DDS multicast.
- **Pin Cyclonedds NIC** on multi-NIC hosts via `ROBOT_INTERFACE=eth0`.
- **Check `ROS_LOCALHOST_ONLY`** on the robot. Vendors sometimes ship with this set to 1.

### Test incrementally

Don't try to build the full agentic blueprint on day one. Phases 1 → 7, in order. Get one camera frame before you try to send a command. Send a command before you wrap it in DimOS.

### Units will bite you

DimOS uses SI everywhere. Your robot's SDK probably doesn't.

| Quantity | DimOS | Common SDK units |
|----------|-------|------------------|
| Angles | radians | degrees |
| Distance | meters | millimeters |
| Velocity | m/s | mm/s, or normalized [-1, 1] |
| Angular velocity | rad/s | deg/s |
| Force | Newtons | grams, kg |
| Torque | Nm | mNm |

Convert at the adapter / Connection Module boundary. Never let non-SI units leak into DimOS.

### Document your "gates"

Many robots have hidden prerequisites for accepting commands (subscriber counts, mode flags, brake states). The R1 Pro had three. The M20 needed a specific gait mode. Document them in a README in your test-scripts directory — the next person will thank you.

### Sensor dropout is real under load

When running cameras + LiDAR + a high-rate control loop, streams can drop. Common causes:
- OS socket buffer overflow (`sysctl net.core.rmem_max`).
- DDS receive thread starvation under high-frequency control traffic.
- GIL contention in Python decoding large frames on the spin thread.

Solutions that have worked:
- Move `bytes(msg.data)` copy off the spin thread into a worker.
- Use a separate `rclpy.Context` for sensor subscriptions (isolated DDS participant).
- `queue.Queue(maxsize=1)` for latest-frame semantics.

### Keep stream names distinct from coordinator names

Your Connection Module's stream names will collide with coordinator-side names if you reuse generic names like `cmd_vel` or `odom`. Either prefix them in the Module (`go2_cmd_vel`) or add `.remappings([...])` in the blueprint. Whole-body modules sidestep this with `motor_states` / `motor_command` / `imu`.

### Match `tick_rate` to actual SDK bandwidth

A coordinator at 500 Hz only helps if the Connection Module can keep up. If your SDK only exposes 100 Hz state, run the coordinator at 100 Hz and don't pretend.

---

## Quick Reference: Existing Integrations

| Robot | Type | Location | Connection | Pattern | Good reference for… |
|-------|------|----------|------------|---------|---------------------|
| Unitree Go2 | Quadruped | `dimos/robot/unitree/go2/` | WebRTC SDK | **Module + Coordinator** | Canonical mobile-base pattern, skills, remappings |
| Unitree G1 | Humanoid | `dimos/robot/unitree/g1/` | DDS / SDK | **Module + Coordinator** | Canonical whole-body pattern, joint-level control at 500 Hz |
| Unitree B1 | Quadruped | `dimos/robot/unitree/b1/` | ROS 2 | Module (legacy) | ROS-sensor wiring inside Module |
| Booster K1 | Humanoid | `dimos/robot/booster/k1/` (branch: `miguel/booster`) | RPC / WebSocket | Module (legacy) | Simple Connection Module |
| Deep Robotics M20 | Quadruped | `dimos/robot/deep_robotics/m20/` (branch: `afik/feat/m20`) | UDP + ROS 2 | Module | Dual-mode connection, protocol parsing |
| Galaxea R1 Pro | Humanoid | `dimos/hardware/*/r1pro/` (branch) | ROS 2 | Direct adapters (legacy) | Adapter pattern, multi-subsystem, troubleshooting |
| UFactory xArm | Arm | `dimos/hardware/manipulators/xarm/` | TCP/IP SDK | Direct adapter | Arm adapter, motion planning |
| AgileX Piper | Arm | `dimos/hardware/manipulators/piper/` | CAN bus | Direct adapter | Arm adapter, gripper |
| OpenArm | Bimanual arm | `dimos/hardware/manipulators/openarm/` | Damiao CAN | Direct adapter | Bimanual setup, CAN protocol |

> The **G1 and Go2 coordinators on `dev`** are the most complete references for the canonical pattern. The **R1 Pro README** at `scripts/r1pro_test/README.md` (branch `task/mustafa/r1pro-dual-arm-testing`) is the best troubleshooting reference for ROS 2 / DDS issues.
