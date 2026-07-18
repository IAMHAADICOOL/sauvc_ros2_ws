#!/usr/bin/env python3
"""pressure_check — Pre-Phase sensor bring-up: "am I getting readings from the Bar30
at all?" Does NOT depend on sauvc_drivers, the EKF, or odometry — meant to run BEFORE
depth_altitude_node (Phase 1), to isolate wiring/protocol problems from pipeline
problems.

Two wiring topologies, one `source` parameter:

  source:=i2c (default) — Bar30 wired DIRECTLY to the Jetson's I2C header. Talks to the
    MS5837 over I2C via the ms5837 python library. See SETUP.md Pre-Phase for pin numbers.

  source:=mavlink — Bar30 wired to the PIXHAWK instead (the standard ArduSub/BlueROV
    topology; ArduSub natively supports the MS5837 as its depth source). Reads pressure
    straight over MAVLink via pymavlink — NOT through mavros/ROS topics, so this works
    even before mavros is set up, and it needs EXCLUSIVE access to the serial/USB link:
    stop mavros first if it's already running, use this, then restart mavros.
    `pip3 install --user pymavlink` if not already installed.

    ArduSub can stream up to THREE barometer channels (SCALED_PRESSURE/2/3) — its own
    onboard baro plus whichever external one(s) are configured — and which index is
    "your" Bar30 depends on your ArduSub BARO parameters, not something this script can
    know in advance. It prints every type as it arrives, labeled; identify yours
    empirically (see SETUP.md): press a finger over the Bar30's port, or dunk it
    briefly — the true water-pressure channel visibly jumps, the onboard baro barely
    reacts.

Usage:
  ros2 run sauvc_sensor_check pressure_check --ros-args -p source:=i2c \
      -p i2c_bus:=1 -p sensor_model:=bar30
  ros2 run sauvc_sensor_check pressure_check --ros-args -p source:=mavlink \
      -p mavlink_url:=/dev/pixhawk -p mavlink_baud:=57600
"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class I2CPressureReader:
    """Direct MS5837-over-I2C. read() -> dict(pressure_mbar, temp_c, depth_m) or None."""

    def __init__(self, node, i2c_bus, sensor_model, fluid_density):
        import ms5837
        cls = {'bar02': ms5837.MS5837_02BA,
              'bar30': ms5837.MS5837_30BA}.get(sensor_model.lower())
        if cls is None:
            raise ValueError(f"sensor_model must be 'bar02' or 'bar30', got {sensor_model!r}")
        node.get_logger().info(f'opening {sensor_model} on i2c bus {i2c_bus}...')
        self.sensor = cls(i2c_bus)
        if not self.sensor.init():
            node.get_logger().error(
                'MS5837 init FAILED. Checklist: (1) VCC/GND/SDA/SCL wired per '
                'SETUP.md Pre-Phase, (2) correct i2c_bus param — verify with '
                '`sudo i2cdetect -y -r <bus>` and look for address 0x76, '
                '(3) you are in the i2c group (log out/in after usermod), '
                '(4) try adding external 2.2-4.7 kOhm pull-ups SDA/SCL->3.3V '
                'if the bus shows nothing at all. (Wired to the Pixhawk instead? '
                'Use source:=mavlink.)')
            raise RuntimeError('MS5837 init failed')
        self.sensor.setFluidDensity(fluid_density)
        node.get_logger().info('MS5837 init OK — reading raw pressure/temp/depth '
                               '(no surface-zeroing here; that happens in Phase 1)')

    def read(self):
        if not self.sensor.read():
            return None
        return dict(label='I2C', pressure_mbar=self.sensor.pressure(),
                   temp_c=self.sensor.temperature(), depth_m=self.sensor.depth())


class MavlinkPressureReader:
    """Pixhawk-wired Bar30, read over raw MAVLink (pymavlink, NOT mavros/ROS topics).
    read() polls for the next SCALED_PRESSURE* message (any of the up to 3 channels) and
    returns it labeled — caller decides which channel is the real water sensor."""

    def __init__(self, node, url, baud, timeout_s):
        from pymavlink import mavutil
        self.timeout_s = timeout_s
        node.get_logger().info(f'connecting to MAVLink on {url} (exclusive access — '
                               f'stop mavros first if it is already using this device)...')
        self.conn = mavutil.mavlink_connection(url, baud=baud)
        node.get_logger().info('waiting for heartbeat (up to 10s)...')
        hb = self.conn.wait_heartbeat(timeout=10)
        if hb is None:
            node.get_logger().error(
                'no heartbeat received. Checklist: (1) is something ELSE already '
                'connected to this device (mavros running)? MAVLink serial links are '
                'exclusive — stop it first. (2) is mavlink_url correct (`ls /dev/pixhawk` '
                'or /dev/ttyACM*)? (3) is the Pixhawk powered and running ArduSub? '
                '(4) pymavlink device syntax differs from mavros\' fcu_url — pass just '
                'the device path, baud is a separate parameter, not colon-appended.')
            raise RuntimeError('no MAVLink heartbeat')
        node.get_logger().info(
            f'heartbeat OK from system {self.conn.target_system}, '
            f'component {self.conn.target_component}')

    def read(self):
        msg = self.conn.recv_match(
            type=['SCALED_PRESSURE', 'SCALED_PRESSURE2', 'SCALED_PRESSURE3'],
            blocking=True, timeout=self.timeout_s)
        if msg is None:
            return None
        # press_abs is hPa, numerically == mbar. temperature is centi-degrees C.
        # Rough uncalibrated depth for a ballpark sanity number only (assumes a fixed
        # 1013.25 mbar surface reference, freshwater density — NOT a real zero/calibration).
        depth_m = max((msg.press_abs - 1013.25) * 100.0 / (997.0 * 9.80665), 0.0)
        return dict(label=msg.get_type(), pressure_mbar=msg.press_abs,
                   temp_c=msg.temperature / 100.0, depth_m=depth_m)


class PressureCheckNode(Node):
    def __init__(self):
        super().__init__('pressure_check')
        self.declare_parameter('source', 'i2c')          # 'i2c' or 'mavlink'
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('sensor_model', 'bar30')
        self.declare_parameter('fluid_density', 997.0)
        self.declare_parameter('mavlink_url', '/dev/pixhawk')
        self.declare_parameter('mavlink_baud', 57600)
        self.declare_parameter('rate_hz', 4.0)

        source = self.get_parameter('source').value.lower()
        if source == 'i2c':
            self.reader = I2CPressureReader(
                self, self.get_parameter('i2c_bus').value,
                self.get_parameter('sensor_model').value,
                self.get_parameter('fluid_density').value)
        elif source == 'mavlink':
            self.reader = MavlinkPressureReader(
                self, self.get_parameter('mavlink_url').value,
                self.get_parameter('mavlink_baud').value,
                timeout_s=1.0 / self.get_parameter('rate_hz').value)
        else:
            raise ValueError(f"source must be 'i2c' or 'mavlink', got {source!r}")

        self.pub_p = self.create_publisher(Float32, '/sensor_check/pressure_mbar', 10)
        self.pub_t = self.create_publisher(Float32, '/sensor_check/temp_c', 10)
        self.n_ok = {}     # per-label counts — mavlink can report multiple channels
        self.n_fail = 0
        self.t0 = time.time()
        period = 1.0 / self.get_parameter('rate_hz').value
        self.create_timer(period, self.tick)

    def tick(self):
        r = self.reader.read()
        if r is None:
            self.n_fail += 1
            if self.n_fail % 20 == 0:   # don't spam every failed poll
                self.get_logger().warn(f'no reading (fail count={self.n_fail})')
            return
        label = r['label']
        self.n_ok[label] = self.n_ok.get(label, 0) + 1
        p, t, d = r['pressure_mbar'], r['temp_c'], r['depth_m']
        elapsed = time.time() - self.t0
        hz = self.n_ok[label] / elapsed if elapsed > 0 else 0.0
        self.pub_p.publish(Float32(data=p))
        self.pub_t.publish(Float32(data=t))
        note = ''
        if not (900.0 < p < 1100.0) and d < 0.05:
            note = '  <-- unexpected for in-air pressure (~1013 mbar); check sensor is not submerged/damaged'
        if not (-10.0 < t < 60.0):
            note = '  <-- temperature reading looks implausible, check wiring/power'
        self.get_logger().info(
            f'[{label}] pressure={p:8.2f} mbar  temp={t:6.2f} C  raw_depth={d:6.3f} m  '
            f'(n={self.n_ok[label]}, {hz:.1f} Hz){note}')


def main():
    rclpy.init()
    node = PressureCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

