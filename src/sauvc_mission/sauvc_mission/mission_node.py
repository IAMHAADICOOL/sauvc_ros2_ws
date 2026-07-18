#!/usr/bin/env python3
"""mission_node — SAUVC mission state machine skeleton (Phase 6+).

Consumes:  /odometry/filtered (nav_msgs/Odometry)  — from the EKF
           /vision/detections (std_msgs/String)     — from gate_detector_node
           /altitude, /heading/pool_relative
Produces:  /cmd/setpoint (geometry_msgs/Twist)      — desired body velocities + yaw rate
           (wire this into YOUR thruster mixer / controller for the vectored config)

Finals sequence per the 2026 rulebook:
  DIVE            submerge fully inside the 140x140 cm start zone before translating
  TRANSIT_GATE    dead-reckon ~16 m along pool axis; watch for ORANGE flare -> SIDESTEP
  SERVO_GATE      center red/green gate in forward cam, hold mid-gate depth, drive through
  RESET_POSE      gate crossed => known position; receive flare order via team comms
  GOTO_DRUMS      dead-reckon to target zone; lawnmower search with DOWN camera
  DROP_BALL       center over chosen drum, actuate dropper
  RECROSS_GATE    required before reacquisition; free pose reset
  FLARES          visit flares in commanded order (color servo when in view)
  SURFACE         end attempt (+5, stops clock for timing bonus)

Every state below is a stub with the intended entry/exit conditions in comments —
fill in as phases come online. Keep states SMALL and testable one at a time in the pool.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32

GATE_WIDTH_M = 1.50   # rulebook: gate is 150cm wide (posts give the pixel width to range off)


class MissionNode(Node):
    STATES = ['IDLE', 'DIVE', 'TRANSIT_GATE', 'SIDESTEP', 'SERVO_GATE',
              'RESET_POSE', 'GOTO_DRUMS', 'DROP_BALL', 'RECROSS_GATE',
              'FLARES', 'SURFACE', 'DONE']

    def __init__(self):
        super().__init__('mission_node')
        self.declare_parameter('cruise_depth', 1.0)      # m; the finals gate sits ON the
        # floor at x~16 m where the V-profile floor is ~1.49 m deep, so the 100 cm gate's
        # midpoint is ~1.0 m below the surface — NOT ~0.5-0.7 m as a flat-floor reading
        # of "100 cm tall gate" would suggest. Recompute if the venue profile differs.
        self.declare_parameter('gate_distance', 16.0)    # m from start zone (rulebook)
        self.declare_parameter('cruise_speed', 0.4)      # m/s
        self.declare_parameter('flare_order', 'R-B-Y')   # set via comms after gate

        self.state = 'IDLE'
        self.odom = None
        self.altitude = None
        self.detections = {}     # label -> (bearing, elev, area, stamp)
        self.landmarks = {}      # label -> (x, y) in odom frame, remembered on first sighting

        self.pub_cmd = self.create_publisher(Twist, '/cmd/setpoint', 10)
        # Anisotropic correction channel: feed robot_localization as an extra pose input
        # (pose1 in ekf.yaml, x+y enabled) rather than /set_pose, so covariance per axis
        # actually controls how much each axis moves — never a hard teleport.
        self.pub_correction = self.create_publisher(PoseWithCovarianceStamped,
                                                     '/pose_correction', 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 10)
        self.create_subscription(String, '/vision/detections', self.on_det, 20)
        self.create_subscription(Float32, '/altitude', self.on_alt, 10)
        self.create_timer(0.1, self.tick)     # 10 Hz state machine

    def on_odom(self, msg):
        self.odom = msg

    def on_alt(self, msg):
        self.altitude = msg.data

    def on_det(self, msg):
        label, bearing, elev, area = msg.data.split(',')
        self.detections[label] = (float(bearing), float(elev), float(area),
                                  self.get_clock().now())

    def fresh(self, label, max_age=0.5):
        d = self.detections.get(label)
        if d is None:
            return None
        if (self.get_clock().now() - d[3]).nanoseconds * 1e-9 > max_age:
            return None
        return d

    # ------------------------------------------------------------------ states
    def tick(self):
        cmd = Twist()
        s = self.state

        if s == 'IDLE':
            # TODO: start trigger (topic/service from your kill-switch/launch logic)
            pass

        elif s == 'DIVE':
            # Descend to cruise_depth using your depth controller; hold position.
            # Exit: |depth - cruise_depth| < 0.1 for 2 s  -> TRANSIT_GATE
            pass

        elif s == 'TRANSIT_GATE':
            # Drive +x at cruise_speed along pool axis (yaw from /heading/pool_relative).
            # If fresh('orange'): -> SIDESTEP.  If fresh('red') and fresh('green')
            # (both gate posts) or odom x > gate_distance - 3: -> SERVO_GATE.
            pass

        elif s == 'SIDESTEP':
            # Strafe (vectored config!) away from the orange flare bearing until it
            # leaves a +/-20 deg cone, then -> TRANSIT_GATE. NEVER touch it: instant abort.
            pass

        elif s == 'SERVO_GATE':
            # P-control yaw rate on gate-center bearing; hold mid-gate depth; drive fwd.
            # Exit: gate posts leave the FOV wide (passed through) -> RESET_POSE.
            pass

        elif s == 'RESET_POSE':
            # Gate's x (~16m) is roughly known, y is NOT (randomized by rulebook) — so
            # correct x only, via publish_gate_x_correction(), and remember_landmark()
            # the gate's (x,y) in OUR odom frame for a cheap return-to-waypoint on the
            # required re-cross before Target Reacquisition. Read flare order param
            # (your comms writes it). -> GOTO_DRUMS
            pass

        elif s == 'GOTO_DRUMS':
            # Waypoint via odom to the drum zone; lawnmower search; DOWN camera looks
            # for blue/red circles (add a drum detector to sauvc_vision when here).
            pass

        elif s == 'DROP_BALL':
            # Center over drum with down-cam visual servo; actuate dropper; -> RECROSS_GATE
            pass

        elif s == 'RECROSS_GATE':
            # Same as SERVO_GATE, opposite direction. Pose reset on crossing. -> FLARES
            pass

        elif s == 'FLARES':
            # For each color in flare_order: goto remembered/guessed position, spiral
            # search, color-servo until proximity/contact drops the golf ball.
            pass

        elif s == 'SURFACE':
            # Zero thrust or gentle ascent; buoyancy does the rest. -> DONE
            pass

        self.pub_cmd.publish(cmd)

    def goto(self, new_state):
        self.get_logger().info(f'{self.state} -> {new_state}')
        self.state = new_state

    # ------------------------------------------------------------------ landmark handling
    def range_from_apparent_width(self, fx, pixel_width_frac, image_width_px):
        """Monocular range from a KNOWN real-world width (rulebook gate = 1.50 m)."""
        pixel_width = pixel_width_frac * image_width_px
        if pixel_width < 5:
            return None
        return fx * GATE_WIDTH_M / pixel_width

    def remember_landmark(self, label, bearing, range_m):
        """First sighting of a landmark: store its (x, y) in OUR odom frame. This is a
        relative note-to-self, not an absolute fix — it lets us return to the same spot
        later (gate re-cross, target reacquisition) without a fresh search, even though
        we still don't know the landmark's true world coordinates."""
        if self.odom is None or label in self.landmarks:
            return
        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        yaw = self._yaw_from_odom()
        lx = px + range_m * math.cos(yaw + bearing)
        ly = py + range_m * math.sin(yaw + bearing)
        self.landmarks[label] = (lx, ly)
        self.get_logger().info(f'remembered {label} at odom ({lx:.2f}, {ly:.2f})')

    def publish_gate_x_correction(self, nominal_x=16.0, x_sigma=0.6, y_sigma=50.0):
        """Called once when the gate is confidently detected/crossed. Tightens ONLY the
        x-axis toward the rulebook's nominal distance (with its placement tolerance as
        sigma). y_sigma is left huge so this measurement contributes ~nothing to y —
        it must NOT collapse the axis we have no ground truth for."""
        if self.odom is None:
            return
        m = PoseWithCovarianceStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.pose.pose.position.x = nominal_x
        m.pose.pose.position.y = self.odom.pose.pose.position.y   # leave y as-is
        cov = [1e6] * 36
        cov[0] = x_sigma ** 2      # x variance: tight
        cov[7] = y_sigma ** 2      # y variance: huge -> effectively a no-op on y
        m.pose.covariance = cov
        self.pub_correction.publish(m)

    def _yaw_from_odom(self):
        if self.odom is None:
            return 0.0
        q = self.odom.pose.pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    rclpy.init()
    rclpy.spin(MissionNode())


if __name__ == '__main__':
    main()
