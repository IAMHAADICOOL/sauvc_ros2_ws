#!/usr/bin/env python3
"""
depth_altitude_node.py — Phase 1. Reads a pressure sensor, publishes depth and altitude.

Publishes:
  /depth     geometry_msgs/PoseWithCovarianceStamped   (pose.position.z = -depth)
  /altitude  std_msgs/Float32                          (pool_depth - depth, meters)

Assumes a Bluerobotics MS5837 (Bar30/Bar02) on I2C. If your sensor differs, replace the
`read_depth()` function only — everything else stands. `pip install ms5837` (or clone
bluerobotics/ms5837-python).

Parameters:
  use_floor_profile   [bool] TRUE  = SAUVC mode: altitude uses the V-shaped floor
                             profile below, evaluated at the EKF's x-estimate.
                             FALSE = flat-pool test mode: altitude = pool_depth - depth,
                             for practice pools with a constant-depth bottom.
  pool_depth          [m]  the constant floor depth used whenever the profile is OFF
                           (and as a safety fallback if the profile arrays are empty).
                           Set this to YOUR practice pool's measured depth when testing.
  floor_profile_x     [m]  along-pool breakpoints for the floor profile (from the
                           rulebook side-view figure; MEASURE/CONFIRM at the venue,
                           rulebook dimensions carry a blanket +/-5% tolerance)
  floor_profile_depth [m]  floor depth at each breakpoint. Default is the 2026
                           qualification-arena V-profile: 1.2 m at both walls,
                           1.6 m at mid-length of the 25 m pool.
  fluid_density[kg/m^3] 997 freshwater / 1029 salt (SAUVC pools are freshwater)
  i2c_bus      Jetson I2C bus number (check with `i2cdetect -l`; often 1 or 7/8 on Jetson)
  sensor_model 'bar02' or 'bar30' — MUST match your actual board (see datasheet: only
               Bar02 publishes a "Relative Accuracy" figure; Bar30 does not, and has a
               much larger — but mostly self-cancelling, see below — Absolute Accuracy).
  depth_var    (m^2) measurement noise variance fed to the EKF. THE DEFAULT BELOW IS A
               PLACEHOLDER, not derived from the datasheet — replace it with your Phase 1
               stationary-hold std, squared, per SETUP.md Phase 1 / PIPELINE.md Phase 2b.
               Datasheet "Resolution" is a quantization floor, not noise. "Absolute
               Accuracy" (huge — e.g. Bar30's ±200mbar/204cm) is calibration OFFSET, not
               sample noise, and is mostly cancelled by the surface-zero below — it isn't
               the number to use here either, though a big temperature swing between
               zeroing (in air) and diving (in water) can leave some of it uncancelled.

Sloped-floor handling: the SAUVC pool floor is NOT flat (V-profile, 1.2 m at the walls
to 1.6 m mid-length), so altitude = floor_depth(x) - vehicle_depth, where x comes from
/odometry/filtered. This creates a mild feedback loop (altitude -> flow velocity -> EKF
x -> altitude), which is benign because the slope is only ~3.2%: a 1 m error in x costs
~3 cm of altitude (~3-5%% velocity scale error), vs 25-80%% for a flat-floor assumption.
x starts exactly known (start zone at the wall) and is re-anchored by the gate
x-correction mid-run. Before the first odometry message, x is assumed 0 (the start).

Surface-zeroing: on startup, with the vehicle floating at the surface, the first 3 s of
readings are averaged as the zero offset. Power on / launch the node at the surface.
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


class DepthAltitudeNode(Node):
    def __init__(self):
        super().__init__('depth_altitude_node')
        self.declare_parameter('pool_depth', 1.4)   # used whenever the profile is OFF
        self.declare_parameter('use_floor_profile', True)
        self.declare_parameter('floor_profile_x', [0.0, 12.5, 25.0])
        self.declare_parameter('floor_profile_depth', [1.2, 1.6, 1.2])
        self.declare_parameter('fluid_density', 997.0)
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('sensor_model', 'bar30')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('depth_var', 0.0004)   # PLACEHOLDER — replace with your
                                                       # own measured std^2 (see above)

        self.pool_depth = self.get_parameter('pool_depth').value
        self.depth_var = self.get_parameter('depth_var').value
        self.use_profile = self.get_parameter('use_floor_profile').value
        self.profile_x = np.array(self.get_parameter('floor_profile_x').value, float)
        self.profile_d = np.array(self.get_parameter('floor_profile_depth').value, float)
        if len(self.profile_x) != len(self.profile_d):
            raise ValueError('floor_profile_x and floor_profile_depth must be same length')
        if self.use_profile and len(self.profile_x) == 0:
            self.get_logger().warn('use_floor_profile=true but the profile is empty — '
                                   'falling back to constant pool_depth')
            self.use_profile = False
        mode = (f'PROFILE mode (V-floor, {len(self.profile_x)} breakpoints)'
                if self.use_profile else
                f'FLAT mode (constant pool_depth={self.pool_depth} m)')
        self.get_logger().info(f'altitude source: {mode}')
        self.x_est = 0.0          # start zone is at the wall: x=0 exactly known at launch
        self.have_odom = False

        import ms5837
        model = self.get_parameter('sensor_model').value.lower()
        cls = {'bar02': ms5837.MS5837_02BA, 'bar30': ms5837.MS5837_30BA}.get(model)
        if cls is None:
            raise ValueError(f"sensor_model must be 'bar02' or 'bar30', got {model!r}")
        self.sensor = cls(self.get_parameter('i2c_bus').value)
        if not self.sensor.init():
            raise RuntimeError('MS5837 init failed — check I2C wiring/bus number')
        self.sensor.setFluidDensity(self.get_parameter('fluid_density').value)

        # Surface zero: average 3 s of readings at startup.
        zs = []
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if self.sensor.read():
                zs.append(self.sensor.depth())
            time.sleep(0.05)
        self.zero = sum(zs) / max(len(zs), 1)
        self.get_logger().info(f'surface zero = {self.zero:.3f} m ({len(zs)} samples)')

        self.pub_depth = self.create_publisher(PoseWithCovarianceStamped, '/depth', 10)
        self.pub_alt = self.create_publisher(Float32, '/altitude', 10)
        self.pub_floor = self.create_publisher(Float32, '/floor_depth', 10)  # debugging
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 10)
        self.window = []          # small median filter against pressure spikes
        period = 1.0 / self.get_parameter('rate_hz').value
        self.create_timer(period, self.tick)

    def on_odom(self, msg):
        self.x_est = msg.pose.pose.position.x
        self.have_odom = True

    def floor_depth(self):
        """Floor depth below the vehicle.
        FLAT mode (use_floor_profile=false): constant pool_depth — for testing in a
        practice pool with a flat bottom.
        PROFILE mode (use_floor_profile=true): piecewise-linear interpolation of the
        SAUVC V-profile at the EKF's along-pool x-estimate. np.interp clamps beyond the
        profile ends (flat extrapolation), the right conservative behavior if x drifts
        slightly past a wall. Before the first odometry message, x=0 (start wall)."""
        if not self.use_profile:
            return self.pool_depth
        x = self.x_est if self.have_odom else 0.0
        return float(np.interp(x, self.profile_x, self.profile_d))

    def read_depth(self):
        if not self.sensor.read():
            return None
        return self.sensor.depth() - self.zero

    def tick(self):
        d = self.read_depth()
        if d is None:
            return
        self.window.append(d)
        if len(self.window) > 5:
            self.window.pop(0)
        d = sorted(self.window)[len(self.window) // 2]   # median of 5

        m = PoseWithCovarianceStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.pose.pose.position.z = -d                       # REP-103: z up, depth is negative
        cov = [0.0] * 36
        cov[14] = self.depth_var                          # z variance
        m.pose.covariance = cov
        self.pub_depth.publish(m)

        fd = self.floor_depth()
        alt = Float32()
        alt.data = max(fd - d, 0.0)
        self.pub_alt.publish(alt)
        self.pub_floor.publish(Float32(data=fd))


def main():
    rclpy.init()
    rclpy.spin(DepthAltitudeNode())


if __name__ == '__main__':
    main()
