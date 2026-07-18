"""
The ROS 2 side of the environment: publish setpoints, cache the latest state.

This is deliberately thin. It owns no policy, no reward, no episode logic --
it is a synchronous view onto an asynchronous stream, and nothing more.

Threading model
---------------
rclpy wants to spin; Gymnasium wants to block on ``step()``. We reconcile them
by spinning a ``SingleThreadedExecutor`` on a daemon thread and having callbacks
write into a mutex-guarded snapshot. ``step()`` then reads a consistent copy.

Frame conventions
-----------------
Stonefish is NED: +X forward/north, +Y starboard/east, +Z **down**. So depth is
+z, and "go deeper" is +heave. Every sign in this package follows that; if you
see a stray minus somewhere it is a bug, not a convention.

The one thing you must confirm
-------------------------------
``nav_msgs/Odometry`` nominally reports ``twist`` in ``child_frame_id`` (body),
but simulators vary and this exact point has already bitten this project once.
``odom_twist_frame`` therefore defaults to ``"world"`` and is a config knob, not
a hardcoded assumption. Your ``ardusub_json_bridge.py`` already contains the
resolved answer for your build -- copy it from there rather than trusting this
default. ``scripts/check_conventions.py`` will also just measure it for you.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

import numpy as np

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import FluidPressure, Imu
    from std_msgs.msg import Float64MultiArray
    _HAVE_ROS = True
except ImportError:  # pragma: no cover - lets the math be unit-tested off-robot
    _HAVE_ROS = False
    Node = object  # type: ignore

__all__ = ["VehicleState", "RosLink", "quat_to_rpy", "quat_to_matrix", "wrap_pi"]


def wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix (body -> world)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion (x, y, z, w) -> fixed-axis roll, pitch, yaw."""
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


@dataclass
class VehicleState:
    """A consistent snapshot of the vehicle at one instant.

    ``position`` and ``orientation`` are ground truth from the simulator's
    odometry sensor. They are legitimate for computing reward and termination;
    they are **not** legitimate as policy input, because the real vehicle has no
    such sensor. The env enforces that split -- see ``AuvBaseEnv``.
    """

    stamp: float = 0.0  # message stamp [s]
    wall_time: float = 0.0  # local receipt time [s], for real-time-factor checks
    position: np.ndarray = None  # (3,) world NED [m]
    orientation: np.ndarray = None  # (4,) quaternion xyzw, body -> world
    lin_vel_body: np.ndarray = None  # (3,) body frame [m/s]
    ang_vel_body: np.ndarray = None  # (3,) body frame [rad/s]
    depth_pressure: float = 0.0  # from the pressure sensor [m], if available
    seq: int = 0  # odometry messages received so far

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = np.zeros(3)
        if self.orientation is None:
            self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        if self.lin_vel_body is None:
            self.lin_vel_body = np.zeros(3)
        if self.ang_vel_body is None:
            self.ang_vel_body = np.zeros(3)

    @property
    def rpy(self) -> tuple[float, float, float]:
        return quat_to_rpy(self.orientation)

    @property
    def depth(self) -> float:
        """Depth below surface [m]. NED, so this is simply +z."""
        return float(self.position[2])

    def copy(self) -> "VehicleState":
        return replace(
            self,
            position=self.position.copy(),
            orientation=self.orientation.copy(),
            lin_vel_body=self.lin_vel_body.copy(),
            ang_vel_body=self.ang_vel_body.copy(),
        )


class RosLink:
    """Owns the rclpy node, the executor thread, and the state snapshot."""

    def __init__(
        self,
        robot_name: str = "sauvc_auv",
        setpoint_topic: str | None = None,
        odom_topic: str | None = None,
        imu_topic: str | None = None,
        pressure_topic: str | None = None,
        n_thrusters: int = 8,
        odom_twist_frame: str = "world",
        node_name: str | None = None,
        domain_id: int | None = None,
    ) -> None:
        if not _HAVE_ROS:
            raise ImportError(
                "rclpy is not importable. Source your ROS 2 overlay first:\n"
                "  source /opt/ros/jazzy/setup.bash && "
                "source ~/Robotics_Job/sauvc_ws/install/setup.bash"
            )
        if odom_twist_frame not in ("body", "world"):
            raise ValueError("odom_twist_frame must be 'body' or 'world'")

        self.robot_name = robot_name
        self.odom_twist_frame = odom_twist_frame
        self.n_thrusters = n_thrusters

        base = f"/{robot_name}"
        self.setpoint_topic = setpoint_topic or f"{base}/thruster_setpoints"
        self.odom_topic = odom_topic or f"{base}/odometry"
        self.imu_topic = imu_topic or f"{base}/imu"
        self.pressure_topic = pressure_topic or f"{base}/pressure"

        if domain_id is not None:
            import os

            os.environ["ROS_DOMAIN_ID"] = str(domain_id)

        if not rclpy.ok():
            rclpy.init()

        self._node = Node(node_name or f"sauvc_gym_link_{id(self):x}")
        self._lock = threading.Lock()
        self._state = VehicleState()
        self._seq = 0

        # Sensor data: keep the newest, drop the rest. A stale queue is worse
        # than a dropped message for a controller.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub = self._node.create_publisher(Float64MultiArray, self.setpoint_topic, 10)
        self._node.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
        self._node.create_subscription(Imu, self.imu_topic, self._on_imu, qos)
        self._node.create_subscription(
            FluidPressure, self.pressure_topic, self._on_pressure, qos
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._closed = False
        self._thread.start()

    # ------------------------------------------------------------- internals

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception:  # executor torn down under us during close()
            pass

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        quat = np.array([o.x, o.y, o.z, o.w])
        lin = np.array([v.x, v.y, v.z])
        if self.odom_twist_frame == "world":
            lin = quat_to_matrix(quat).T @ lin  # world -> body

        with self._lock:
            self._seq += 1
            self._state = replace(
                self._state,
                stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                wall_time=time.monotonic(),
                position=np.array([p.x, p.y, p.z]),
                orientation=quat,
                lin_vel_body=lin,
                ang_vel_body=np.array([w.x, w.y, w.z]),
                seq=self._seq,
            )

    def _on_imu(self, msg) -> None:
        # Odometry already carries angular rate; the IMU is the *deployable*
        # source for it, so prefer it when both are present.
        with self._lock:
            self._state = replace(
                self._state,
                ang_vel_body=np.array(
                    [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                ),
            )

    def _on_pressure(self, msg) -> None:
        # Stonefish reports absolute pressure in pascals.
        depth = (float(msg.fluid_pressure) - 101325.0) / (1025.0 * 9.80665)
        with self._lock:
            self._state = replace(self._state, depth_pressure=depth)

    # ----------------------------------------------------------------- public

    @property
    def node(self):
        return self._node

    def get_state(self) -> VehicleState:
        with self._lock:
            return self._state.copy()

    def send_setpoints(self, setpoints: np.ndarray) -> None:
        """Publish thruster setpoints, clipped to the legal range."""
        setpoints = np.asarray(setpoints, dtype=float).reshape(-1)
        if setpoints.shape[0] != self.n_thrusters:
            raise ValueError(
                f"expected {self.n_thrusters} setpoints, got {setpoints.shape[0]}"
            )
        msg = Float64MultiArray()
        msg.data = [float(v) for v in np.clip(setpoints, -1.0, 1.0)]
        self._pub.publish(msg)

    def stop(self) -> None:
        """All thrusters to zero. Called on close and between episodes."""
        self.send_setpoints(np.zeros(self.n_thrusters))

    def wait_for_data(self, timeout: float = 30.0) -> bool:
        """Block until at least one odometry message has arrived."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq > 0:
                    return True
            time.sleep(0.02)
        return False

    def wait_for_new_state(self, since_seq: int, timeout: float) -> VehicleState:
        """Block until odometry newer than ``since_seq`` arrives, or timeout.

        Synchronising on message arrival rather than on the wall clock means the
        env tracks the simulator's actual pace. If Stonefish drops below
        real time -- which it will, with cameras enabled -- we wait for it
        instead of silently stepping on stale data.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq > since_seq:
                    return self._state.copy()
            time.sleep(0.001)
        return self.get_state()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
        except Exception:
            pass
        self._executor.shutdown()
        self._node.destroy_node()
        self._thread.join(timeout=2.0)
