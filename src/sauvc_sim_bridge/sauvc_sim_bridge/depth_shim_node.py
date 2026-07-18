#!/usr/bin/env python3
"""depth_shim_node — makes Stonefish's pressure sensor look like the Bar30 driver.

Sub: /sauvc_auv/pressure  sensor_msgs/FluidPressure   (absolute, Pa)
     /odometry/filtered   nav_msgs/Odometry           (for the floor-profile x lookup)
Pub: /depth               geometry_msgs/PoseWithCovarianceStamped   (pose.position.z = -depth)
     /altitude            std_msgs/Float32
     /floor_depth         std_msgs/Float32            (debug)

This is the sim twin of sauvc_drivers/depth_altitude_node.py. It publishes byte-identical
message types on identical topics, so sauvc_localization's flow_velocity_node and the
robot_localization EKF cannot tell the difference. Everything downstream of /depth and
/altitude is unmodified.

SURFACE-ZEROING IS OFF BY DEFAULT HERE, AND THAT *IS* THE PARITY-PRESERVING CHOICE
-----------------------------------------------------------------------------------
Measured, not assumed (`ros2 topic echo /sauvc_auv/pressure --once`, vehicle floating):

    fluid_pressure: -16.717976936682707
    variance: 400.0

Two things follow.

(a) Stonefish publishes **GAUGE pressure relative to the free surface**. There is no
    101325 Pa atmospheric term -- and it is not clamped at zero, which is why the value
    is NEGATIVE: -16.7 Pa / (rho*g) = -1.7 mm, i.e. the sensor is sitting 1.7 mm above
    the waterline. So `P/(rho*g)` is ALREADY true depth below the surface.

    On hardware the chain is different: the Bar30 reads ABSOLUTE pressure, `ms5837.depth()`
    subtracts the library's atmospheric constant, and the surface-zero mops up the
    residual mismatch plus the sensor's calibration offset. The zeroing exists to remove
    two error sources that **do not exist in this simulator**.

    Parity is about matching the POST-zero semantics, not the pre-zero mechanism. Both
    paths must end at "metres below the free surface". The real driver needs zeroing to
    get there; this node is already there. Zeroing here would only add noise.

(b) The default is `zero_secs: 0.0` for that reason alone -- not because zeroing would
    break. It is worth being precise here, because an earlier version of this comment
    overstated the case: in practice you launch this node against an ALREADY-RUNNING sim,
    by which time the vehicle has floated up and settled, and a zeroing window would
    capture ~0 and be harmless.

    The settle-detector below is therefore INSURANCE, not a bug fix. It earns its keep in
    exactly one situation: a combined launch file that starts Stonefish AND this node
    together. The scenarios spawn at `start_position="... 0.3"` -- 0.3 m DEEP in NED --
    and the vehicle floats up on its ~+1.2 kgf net buoyancy, so a naive window from t=0
    WOULD average the ascent and bake in ~11 cm. `sim_drivers.launch.py` does not do that
    today (you launch the simulator separately), but `sim_phase6_full` might, and the
    failure would be silent.

Set `zero_secs` > 0 only if you specifically want to exercise the zeroing code path. If
you do, this node waits for the reading to SETTLE first (see `_settled`) instead of
trusting t=0.

WHAT IS DELIBERATELY IDENTICAL TO THE REAL DRIVER
  * Median-of-5 window against spikes.
  * z published as NEGATIVE depth (REP-103, z up), frame_id 'odom'.
  * Altitude from the shared FloorProfile at the EKF's x-estimate, with x=0 assumed
    before the first odometry message (the start zone is at the wall).
  * The same mild feedback loop (altitude -> flow -> EKF x -> altitude), benign for the
    same reason: the 3.2% slope costs ~3 cm of altitude per 1 m of x error.

WHAT DIFFERS, AND WHY
  * `depth_var` here is the SIM's noise, not the Bar30's -- and it is DERIVED from the
    message rather than hand-set. Confirmed against the live echo: `variance: 400.0`
    == 20^2, exactly the scene's <noise pressure="20.0"/> in Pa. So
        sigma_depth = sigma_P / (rho*g) = 20 / (1000*9.81) = 2.04 mm
        depth_var   = 4.157e-06 m^2
    That is **96x tighter** than the real driver's 4.0e-04 placeholder. Feeding the
    Bar30's pessimism to the sim EKF would tell it to distrust a sensor that is in fact
    near-perfect, and you would tune the filter against a lie -- then carry those gains
    to a pool where the sensor really is that noisy. Set `depth_var_override` >= 0 only
    if you deliberately want the real sensor's pessimism here.
  * No I2C, no ms5837, no sensor_model. Those describe hardware that does not exist here.

KNOWN SHARED BIAS (not introduced by the sim -- flagging it because it is now visible)
  The scene mounts the pressure sensor at origin xyz="0.0 0.0 -0.10", i.e. 10 cm ABOVE
  the body origin in FRD. /depth therefore reports the SENSOR's depth, which the EKF
  fuses as base_link's z -- a persistent ~10 cm offset. The real driver has exactly the
  same property, so it cancels between sim and hardware IFF the real Bar30 sits ~10 cm
  above the CAD origin too. Worth measuring once. It is constant, so depth-holding
  absorbs it into the setpoint; it matters only if you ever compare z against ground
  truth (e.g. /sauvc_auv/odometry) and wonder about a fixed 10 cm discrepancy.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import FluidPressure
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


class _FloorProfile:
    """Self-contained copy of depth_altitude_node's floor logic.

    This is duplicated ON PURPOSE. The real driver keeps `floor_depth()` inline as an
    instance method; there is no shared module to import (an earlier version of this shim
    tried to import sauvc_localization.floor_profile, which does not exist, and that is
    what crashed depth_shim_node on launch). Rather than refactor a WORKING hardware node
    to satisfy the sim, the shim carries its own copy. The two are simple enough that
    drift is unlikely, and the numbers live in the launch file / scene either way.

    np.interp clamps beyond the profile ends (flat extrapolation) — the same conservative
    behaviour as the driver when x drifts slightly past a wall.
    """

    def __init__(self, use_profile, pool_depth, profile_x, profile_depth):
        self.use_profile = bool(use_profile)
        self.pool_depth = float(pool_depth)
        self.profile_x = np.asarray(profile_x, dtype=float)
        self.profile_d = np.asarray(profile_depth, dtype=float)
        if self.profile_x.size != self.profile_d.size:
            raise ValueError('floor_profile_x and floor_profile_depth must be same length')
        if self.use_profile and self.profile_x.size == 0:
            self.use_profile = False
        if self.use_profile and np.any(np.diff(self.profile_x) <= 0):
            raise ValueError('floor_profile_x must be strictly increasing')

    def describe(self):
        return (f'PROFILE mode (V-floor, {self.profile_x.size} breakpoints)'
                if self.use_profile else
                f'FLAT mode (constant pool_depth={self.pool_depth} m)')

    def depth_at(self, x):
        if not self.use_profile:
            return self.pool_depth
        return float(np.interp(float(x), self.profile_x, self.profile_d))

    def altitude(self, x, vehicle_depth):
        return max(self.depth_at(x) - float(vehicle_depth), 0.0)


class DepthShimNode(Node):
    def __init__(self):
        super().__init__('depth_shim_node')
        # --- pool / plant parameters ---
        # These are PLANT parameters and are SUPPOSED to differ from the real robot's.
        # The real pool at 26 C is ~997 kg/m^3 and the real driver is configured for that.
        # This simulator's water is whatever sauvc_qualification.scn DECLARES:
        #     <water density="1000.0" jerlov="0.05" temperature="26.0"/>
        # and Stonefish's gravity is 9.81 (confirmed from the live IMU echo: a floating,
        # level vehicle reports linear_acceleration.z = -9.809994).
        #
        # Stonefish computes P = rho_sim * g_sim * depth. Inverting it with the REAL
        # pool's constants gives a systematic depth scale error of
        #     (1000*9.81)/(997*9.80665) - 1 = +0.335%
        # = +5.0 mm at 1.5 m, i.e. 2.5 sigma against a 2.04 mm sensor. Small, constant,
        # and invisible -- and it propagates depth -> altitude -> flow scale, which is
        # precisely the failure mode flow_scorer_node exists to detect. It would have
        # sat in the scorer's output as a fake ~0.3% calibration error forever.
        #
        # Note the scene is itself slightly inconsistent (fresh water at 26 C is really
        # ~996.8 kg/m^3, not 1000.0) -- but Stonefish uses the DECLARED number, so the
        # declared number is what must be inverted here. Match the scene, not physics.
        #
        # Verify empirically any time: dive to a known depth and check
        #     fluid_pressure / (fluid_density * gravity)  ==  /sauvc_auv/odometry z
        self.declare_parameter('pool_depth', 1.4)
        self.declare_parameter('use_floor_profile', True)
        self.declare_parameter('floor_profile_x', [0.0, 12.5, 25.0])
        self.declare_parameter('floor_profile_depth', [1.2, 1.6, 1.2])
        self.declare_parameter('fluid_density', 1000.0)   # MATCH <water density=...>
        self.declare_parameter('gravity', 9.81)           # MATCH Stonefish, not WGS-84
        # FIX(profile x origin): floor_profile_x is WALL-referenced (0..25) but the
        # odometry x that feeds it is WORLD-referenced (-12.5..+12.5). np.interp
        # silently clamps every negative x to the first endpoint (1.2 m), so the
        # profile was effectively never consulted. profile_x_offset converts world x
        # into profile x: x_profile = x_world + offset. Set 0.0 if you redefine the
        # profile breakpoints in world coordinates instead.
        self.declare_parameter('profile_x_offset', 12.5)
        # FIX(altitude datum): /altitude is consumed by flow as the CAMERA-to-floor
        # range, but it was computed from the PRESSURE SENSOR's depth. In my_auv.scn
        # (FRD, z down): pressure sensor at z=-0.10 (0.10 m ABOVE the body origin),
        # down camera at z=+0.11 (0.11 m BELOW it). So
        #     camera_depth = sensor_depth + sensor_above_origin + camera_below_origin
        # and the published altitude was too LARGE by 0.21 m at the datum, on top of
        # the floor-profile error. Both offsets must match the REAL vehicle's mounts
        # for sim/hardware parity — measure them, don't trust these defaults blindly.
        self.declare_parameter('sensor_above_origin', 0.10)   # m, pressure sensor
        self.declare_parameter('camera_below_origin', 0.11)   # m, down camera
        # --- shim-specific ---
        self.declare_parameter('in_topic', '/sauvc_auv/pressure')
        # FIX(dead x_est): the floor-profile lookup needs an x estimate, but in the
        # flow_eval configuration NOTHING publishes /odometry/filtered — x_est sat at
        # 0.0 forever and the floor was pinned to 1.2 m while the true floor sloped
        # to 1.6 m mid-pool (the measured 0.84 scale ratio). Point this at the
        # ground-truth odometry for diagnostic runs (see flow_eval.launch.py).
        self.declare_parameter('odom_topic', '/odometry/filtered')
        # 0.0 = do not zero. Stonefish already publishes gauge pressure w.r.t. the free
        # surface, and the vehicle spawns 0.3 m deep and floats up -- see the module
        # docstring. Non-zero only if you want to exercise the zeroing code path.
        self.declare_parameter('zero_secs', 0.0)
        self.declare_parameter('settle_tol', 0.005)    # m, spread over the settle window
        self.declare_parameter('settle_timeout', 30.0)  # s, before giving up and warning
        self.declare_parameter('zero_sanity_m', 0.05)   # warn if |zero| exceeds this
        self.declare_parameter('depth_var_override', -1.0)  # <0 = derive from message

        g = lambda n: self.get_parameter(n).value
        self.rho = g('fluid_density')
        self.g = g('gravity')
        self.var_override = g('depth_var_override')
        self.zero_secs = g('zero_secs')
        self.settle_tol = g('settle_tol')
        self.settle_timeout = g('settle_timeout')
        self.zero_sanity = g('zero_sanity_m')

        self.floor = _FloorProfile(
            g('use_floor_profile'), g('pool_depth'),
            g('floor_profile_x'), g('floor_profile_depth'))
        self.get_logger().info(f'altitude source: {self.floor.describe()}')
        self.profile_x_offset = float(g('profile_x_offset'))
        self.sensor_above = float(g('sensor_above_origin'))
        self.cam_below = float(g('camera_below_origin'))
        self.odom_topic = g('odom_topic')

        self.x_est = 0.0
        self.have_odom = False
        self._no_odom_warned = False
        self._n_pressure = 0
        self.window = []
        # zero_secs <= 0 -> no zeroing at all; the sim's gauge pressure is already
        # referenced to the free surface.
        self.zero = 0.0 if self.zero_secs <= 0.0 else None
        self.zero_samples = []
        self.zero_t0 = None
        self.settle_buf = []
        self.first_t = None
        self.settle_warned = False

        self.pub_depth = self.create_publisher(PoseWithCovarianceStamped, '/depth', 10)
        self.pub_alt = self.create_publisher(Float32, '/altitude', 10)
        self.pub_floor = self.create_publisher(Float32, '/floor_depth', 10)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.create_subscription(FluidPressure, g('in_topic'), self.on_pressure, 20)
        self.get_logger().info(
            f'floor-profile x from {self.odom_topic} '
            f'(x_profile = x_odom + {self.profile_x_offset}); altitude datum = down '
            f'camera (sensor +{self.sensor_above} m, camera -{self.cam_below} m '
            'about the body origin)')
        if self.zero_secs <= 0.0:
            zmsg = ('no surface-zeroing (Stonefish gauge pressure is already referenced '
                    'to the free surface)')
        else:
            zmsg = (f'surface-zeroing for {self.zero_secs:.1f}s once depth settles '
                    f'(vehicle spawns 0.3m deep and floats up)')
        self.get_logger().info(
            f"depth_shim: {g('in_topic')} -> /depth + /altitude; {zmsg}. "
            f"Inverting with rho={self.rho} g={self.g} (rho*g={self.rho*self.g:.2f}) "
            f"-- these MUST match <water density=...> in the scene, not the real pool.")

    def on_odom(self, msg):
        # FIX(profile x origin): convert world x -> wall-referenced profile x.
        self.x_est = msg.pose.pose.position.x + self.profile_x_offset
        self.have_odom = True

    def _depth_from_pressure(self, pa):
        return pa / (self.rho * self.g)

    def _settled(self, t, d_raw):
        """True once the depth reading has stopped DRIFTING.

        The scenarios spawn the vehicle 0.3 m deep and it floats up on its net buoyancy.
        The real driver can assume "launched at the surface" because a human put it
        there; here we have to wait for it.

        This tests DRIFT (mean of the recent half minus mean of the older half), not
        SPREAD (max-min). That distinction is not cosmetic: at sigma = 2.05 mm the
        expected max-min of a 20-sample window is ~3.7*sigma ~= 7.6 mm, so any spread
        threshold tight enough to detect a settled vehicle is BELOW the noise floor and
        would never trigger. Differencing two 10-sample means has a standard error of
        only sigma*sqrt(2/10) ~= 0.9 mm, so a 5 mm threshold sits ~5 sigma clear of the
        noise while still catching the tail of the ascent.

        Times out rather than blocking forever.
        """
        self.settle_buf.append(d_raw)
        if len(self.settle_buf) > 20:            # ~1 s at the scene's 20 Hz
            self.settle_buf.pop(0)
        if t - self.first_t > self.settle_timeout:
            if not self.settle_warned:
                self.settle_warned = True
                self.get_logger().warn(
                    f'depth never settled within {self.settle_timeout:.0f}s. '
                    'Zeroing anyway — treat the result with suspicion.')
            return True
        if len(self.settle_buf) < 20:
            return False
        half = len(self.settle_buf) // 2
        drift = abs(float(np.mean(self.settle_buf[half:])) -
                    float(np.mean(self.settle_buf[:half])))
        return drift < self.settle_tol

    def on_pressure(self, msg):
        # Stamps are wall clock from the simulator (ROS2Interface uses
        # get_clock()->now(), not the sample time). Pass through; never synthesise.
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        d_raw = self._depth_from_pressure(msg.fluid_pressure)

        # --- optional surface zero, gated on the vehicle having actually settled ---
        if self.zero is None:
            if self.first_t is None:
                self.first_t = t
            if not self._settled(t, d_raw):
                return
            if self.zero_t0 is None:
                self.zero_t0 = t
            self.zero_samples.append(d_raw)
            if t - self.zero_t0 < self.zero_secs:
                return
            self.zero = float(np.mean(self.zero_samples))
            self.get_logger().info(
                f'surface zero = {self.zero:.4f} m ({len(self.zero_samples)} samples)')
            if abs(self.zero) > self.zero_sanity:
                self.get_logger().error(
                    f'surface zero is {self.zero:+.3f} m, well away from 0. Stonefish '
                    'publishes gauge pressure referenced to the free surface, so a '
                    'correct zero is ~0. This means the vehicle was NOT at the surface '
                    'when zeroing finished, and every depth from now on carries this as '
                    'a constant error. Fix the scenario spawn, or just leave '
                    'zero_secs:=0.0 (the default) -- the sim does not need zeroing.')

        d = d_raw - self.zero

        self.window.append(d)
        if len(self.window) > 5:
            self.window.pop(0)
        d = sorted(self.window)[len(self.window) // 2]

        # --- depth variance: derive from the sim's own declared noise ---
        if self.var_override >= 0.0:
            depth_var = self.var_override
        else:
            sigma_p = float(np.sqrt(max(msg.variance, 0.0)))   # Pa
            sigma_d = sigma_p / (self.rho * self.g)            # m
            depth_var = max(sigma_d * sigma_d, 1e-12)          # never hand the EKF a zero

        m = PoseWithCovarianceStamped()
        m.header.stamp = msg.header.stamp
        m.header.frame_id = 'odom'
        m.pose.pose.position.z = -d
        cov = [0.0] * 36
        cov[14] = depth_var
        m.pose.covariance = cov
        self.pub_depth.publish(m)

        # FIX(dead x_est): if the profile is active but no odometry has arrived after
        # ~5 s of pressure data, say so LOUDLY — the old behaviour silently pinned the
        # floor to the x=0 endpoint for the whole run.
        self._n_pressure += 1
        if (self.floor.use_profile and not self.have_odom
                and not self._no_odom_warned and self._n_pressure > 100):
            self._no_odom_warned = True
            self.get_logger().warn(
                f'floor profile is ON but nothing is publishing {self.odom_topic} — '
                'x_est is stuck at the start-wall endpoint and /altitude will be '
                'wrong everywhere the floor slopes. Remap odom_topic (ground truth '
                'is fine for diagnostics) or run with use_floor_profile:=false.')

        # x_est is already wall-referenced (offset applied in on_odom). Before the
        # first odometry message, assume the start zone at the wall (profile x = 0).
        x = self.x_est if self.have_odom else 0.0
        fd = self.floor.depth_at(x)
        # FIX(altitude datum): altitude the DOWN CAMERA sees, not the pressure sensor:
        #   camera_depth = d (sensor) + sensor_above_origin + camera_below_origin
        cam_depth = d + self.sensor_above + self.cam_below
        self.pub_alt.publish(Float32(data=float(self.floor.altitude(x, cam_depth))))
        self.pub_floor.publish(Float32(data=float(fd)))


def main():
    rclpy.init()
    rclpy.spin(DepthShimNode())


if __name__ == '__main__':
    main()
