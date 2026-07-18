#!/usr/bin/env python3
"""flow_scorer_node — grades flow_velocity_node against the simulated DVL.

Sub: /flow/twist       geometry_msgs/TwistWithCovarianceStamped  (body FLU, your estimate)
     /sauvc_auv/dvl    stonefish_ros2/DVL                        (body FRD, reference)
     /sim/rtf          std_msgs/Float32                          (from rtf_monitor)
Pub: nothing. This node is a scorer. It is deliberately a dead end.

THE HARD GUARANTEE THAT THIS NEVER REACHES THE EKF
--------------------------------------------------
The DVL stays on `/sauvc_auv/dvl` -- inside the simulator's namespace -- and NO shim
republishes it. That is a structural guarantee, not a YAML-discipline one:
`sauvc_bringup`'s hardware launch files never create a `/sauvc_auv` namespace, so if
anything in the estimator ever grew a dependency on the DVL, it would fail LOUDLY on the
real robot at the first launch rather than silently degrade in the pool. Compare that to
"we just won't add it to ekf.yaml", which survives exactly until someone edits ekf.yaml.

Your real vehicle has no DVL. The optical flow IS the DVL. This node exists to tell you
how good a DVL it is.

WHAT IT REPORTS
  * scale     -- least-squares slope of flow vs DVL, forced through the origin.
                 THIS IS THE HEADLINE NUMBER. Your Phase 3 spec asks for "% scale error";
                 this is it, computed continuously instead of with a tape measure.
                 A wrong altitude, a wrong fx, or a wrong flat-floor assumption all show
                 up here first.
  * bias      -- mean(flow - DVL). Should be ~0 when stationary and when moving.
  * rmse      -- residual after the scale fit; the noise you cannot tune away.
  * r         -- correlation. If this is low, `scale` is meaningless: you have a
                 tracking failure, not a calibration error.
  * dropout   -- fraction of DVL samples with no flow estimate within `max_dt`
                 (flow returned None: too few features, caustics, bad texture).

THE RTF TRAP -- READ THIS BEFORE BELIEVING `scale`
--------------------------------------------------
flow_velocity_node derives dt from image header stamps, and Stonefish stamps with the
WALL CLOCK (ROS2Interface.cpp: get_clock()->now(), never s.getTimestamp()). The DVL
reports physics velocity directly. So if the simulator's real-time factor is R:

    flow ~= R * v_true,     DVL == v_true,     scale ~= R

At R = 0.6 this node would tell you flow_velocity_node has a 40% scale error when the
algorithm is perfect. That is a simulator timing artifact, not a bug in your code. This
node therefore subscribes to /sim/rtf and REFUSES to report a scale unless R is within
tolerance -- run rtf_monitor_node alongside it. Set `require_rtf: false` only if you know
exactly what you are doing.

FRAMES
  DVL velocity is body FRD (Stonefish is NED throughout; nothing in the wrapper converts).
  /flow/twist is body FLU. The conversion goes through frames.py like everything else --
  never by hand.

SIGN CONVENTIONS
  If `scale` comes out near -1, or if r is strongly negative, your down camera's mounting
  convention in sim does not match `swap_xy`/`sign_x`/`sign_y` in flow.yaml. That is a
  config fix, not a maths fix -- exactly the hand-push test from Phase 3, except the DVL
  does the pushing for you. This node prints the diagnosis when it sees it.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped

from sauvc_sim_bridge.frames import frd_to_flu_vec

try:
    from stonefish_ros2.msg import DVL
except ImportError as e:      # pragma: no cover
    raise ImportError(
        'flow_scorer_node needs stonefish_ros2 messages. Source the sim workspace, '
        'and make sure my_auv.scn actually declares a <sensor type="dvl">.') from e


def _t(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class FlowScorerNode(Node):
    def __init__(self):
        super().__init__('flow_scorer_node')
        self.declare_parameter('dvl_topic', '/sauvc_auv/dvl')
        self.declare_parameter('flow_topic', '/flow/twist')
        self.declare_parameter('max_dt', 0.10)        # s, pairing tolerance
        self.declare_parameter('min_speed', 0.05)     # m/s, ignore near-stationary
        self.declare_parameter('report_period', 10.0)
        self.declare_parameter('require_rtf', True)
        self.declare_parameter('rtf_tolerance', 0.05)
        self.declare_parameter('csv_path', '')        # '' = don't write

        g = lambda n: self.get_parameter(n).value
        self.max_dt = g('max_dt')
        self.min_speed = g('min_speed')
        self.require_rtf = g('require_rtf')
        self.rtf_tol = g('rtf_tolerance')
        self.csv_path = g('csv_path')

        self.flow_buf = []          # (t, vx, vy)
        self.pairs = []             # (flow_vx, flow_vy, dvl_vx, dvl_vy)
        self.n_dvl = 0
        self.n_missed = 0
        self.rtf = None

        self.create_subscription(TwistWithCovarianceStamped, g('flow_topic'),
                                 self.on_flow, 50)
        self.create_subscription(DVL, g('dvl_topic'), self.on_dvl, 50)
        self.create_subscription(Float32, '/sim/rtf', self.on_rtf, 10)
        self.create_timer(g('report_period'), self.report)
        self.get_logger().info(
            f"flow_scorer: grading {g('flow_topic')} against {g('dvl_topic')} "
            '(reference only — never fused)')

    def on_rtf(self, msg):
        self.rtf = msg.data

    def on_flow(self, msg):
        self.flow_buf.append((_t(msg.header.stamp),
                              msg.twist.twist.linear.x,
                              msg.twist.twist.linear.y))
        if len(self.flow_buf) > 400:
            self.flow_buf = self.flow_buf[-400:]

    def on_dvl(self, msg):
        t = _t(msg.header.stamp)
        v_flu = frd_to_flu_vec([msg.velocity.x, msg.velocity.y, msg.velocity.z])
        speed = float(np.hypot(v_flu[0], v_flu[1]))
        if speed < self.min_speed:
            return
        self.n_dvl += 1

        if not self.flow_buf:
            self.n_missed += 1
            return
        # nearest flow sample in time
        idx = int(np.argmin([abs(f[0] - t) for f in self.flow_buf]))
        ft, fvx, fvy = self.flow_buf[idx]
        if abs(ft - t) > self.max_dt:
            self.n_missed += 1
            return
        self.pairs.append((fvx, fvy, float(v_flu[0]), float(v_flu[1])))

    def report(self):
        if len(self.pairs) < 20:
            self.get_logger().info(
                f'flow_scorer: {len(self.pairs)} paired samples so far — drive the vehicle '
                f'above {self.min_speed} m/s to accumulate data.')
            return

        a = np.asarray(self.pairs)
        flow = a[:, 0:2].ravel()
        dvl = a[:, 2:4].ravel()

        # scale through the origin: argmin ||flow - s*dvl||
        denom = float(dvl @ dvl)
        scale = float(flow @ dvl) / denom if denom > 1e-9 else float('nan')
        resid = flow - scale * dvl
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        bias = float(np.mean(flow - dvl))
        r = float(np.corrcoef(flow, dvl)[0, 1]) if len(flow) > 2 else float('nan')
        dropout = self.n_missed / max(self.n_dvl, 1)

        head = (f'flow_scorer [{len(self.pairs)} pairs]  '
                f'scale={scale:.3f}  bias={bias:+.4f} m/s  rmse={rmse:.4f} m/s  '
                f'r={r:.3f}  dropout={dropout*100:.1f}%')

        if self.require_rtf and self.rtf is None:
            self.get_logger().warn(
                head + '\n  scale SUPPRESSED: no /sim/rtf. Run rtf_monitor_node — without '
                'it, a simulator running below real time is indistinguishable from a '
                'genuine scale error. (require_rtf:=false to override.)')
            return
        if self.require_rtf and abs(self.rtf - 1.0) > self.rtf_tol:
            self.get_logger().error(
                head + f'\n  scale IS NOT TRUSTWORTHY: RTF={self.rtf:.3f}. Flow dt comes '
                'from wall-clock stamps but the DVL reports physics velocity, so this '
                f'scale is contaminated by the real-time factor. Expect scale ~= RTF '
                'for a perfectly-working algorithm. Fix the sim speed first.')
            return

        self.get_logger().info(head)

        if not np.isnan(r) and r < 0.5:
            self.get_logger().warn(
                '  r < 0.5 — `scale` is meaningless here. This is a TRACKING failure '
                '(too few features / bad floor texture), not a calibration error.')
        elif scale < 0:
            self.get_logger().warn(
                '  scale is NEGATIVE — the down camera mounting convention in sim does '
                'not match flow.yaml. Fix swap_xy / sign_x / sign_y, NOT the maths. '
                '(This is the Phase 3 hand-push test; the DVL just did the pushing.)')
        elif abs(scale - 1.0) > 0.10:
            self.get_logger().warn(
                f'  scale is {(scale-1)*100:+.1f}% off. With RTF confirmed at 1, suspect '
                'altitude (floor profile / surface zero) or fx/fy in flow.yaml.')

        if self.csv_path:
            np.savetxt(self.csv_path, a, delimiter=',',
                       header='flow_vx,flow_vy,dvl_vx,dvl_vy', comments='')


def main():
    rclpy.init()
    rclpy.spin(FlowScorerNode())


if __name__ == '__main__':
    main()
