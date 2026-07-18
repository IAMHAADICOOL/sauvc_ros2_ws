# `sauvc_sensor_check`

Pre-Phase bring-up checks: "is the sensor alive at all?" Deliberately independent of
drivers/EKF/odometry so wiring problems are isolated from pipeline problems. Run each
one standalone before ever touching the phase launches; `pre_phase_sensor_check.launch.py`
(sauvc_bringup) runs them all at once afterwards.

## `pressure_check`

Talks to the Bar30 directly (no ROS pipeline). Two wiring topologies via `source`:

```bash
# Topology A — Bar30 on the Jetson's I2C header:
ros2 run sauvc_sensor_check pressure_check --ros-args -p source:=i2c -p i2c_bus:=1 -p sensor_model:=bar30
# Topology B — Bar30 on the Pixhawk (ArduSub), read over MAVLink (pymavlink):
ros2 run sauvc_sensor_check pressure_check --ros-args -p source:=mavlink -p mavlink_url:=/dev/pixhawk -p mavlink_baud:=57600
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `source` | string | `i2c` | `i2c` (direct) or `mavlink` (via Pixhawk). MAVLink mode needs EXCLUSIVE serial access — stop mavros first. |
| `i2c_bus` | int | `1` | I2C bus (i2c mode). |
| `sensor_model` | string | `bar30` | `bar30`/`bar02` (i2c mode). |
| `fluid_density` | double | `997.0` | kg/m³ for depth conversion. |
| `mavlink_url` | string | `/dev/pixhawk` | Serial device (mavlink mode). |
| `mavlink_baud` | int | `57600` | Baud (mavlink mode). |
| `rate_hz` | double | `4.0` | Print rate. |

**Observe:** periodic pressure/temperature/depth prints. In mavlink mode every
SCALED_PRESSURE/2/3 channel is printed labeled — identify YOUR Bar30 empirically
(finger over the port / brief dunk: the true channel jumps, the onboard baro barely moves).

## `imu_taobotics_check` and `imu_pixhawk_check`

Sanity checks on an already-running IMU driver's topic (they don't start the driver).

```bash
ros2 run sauvc_sensor_check imu_taobotics_check                      # default topic /imu/data
ros2 run sauvc_sensor_check imu_pixhawk_check                        # default /mavros/imu/data
ros2 run sauvc_sensor_check imu_taobotics_check --ros-args -p topic:=/some/other/imu
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `topic` | string | `/imu/data` (taobotics) / `/mavros/imu/data` (pixhawk) | Imu topic to check. |

**Observe:** message rate; roll/pitch/yaw in degrees (tilt the unit, watch them respond
with the right sign); quaternion norm ≈ 1 flag; stationary accel magnitude ≈ 9.8 m/s²
(ok=None means the driver doesn't populate accel — normal for AHRS-only drivers).

## `camera_check_down` / `camera_check_front`

Driver-agnostic Image-topic check (works for v4l2 or RealSense — swap cameras by
remapping the driver, never these).

```bash
ros2 run sauvc_sensor_check camera_check_down                        # /camera_down/image_raw
ros2 run sauvc_sensor_check camera_check_front                       # /camera_front/image_raw
ros2 run sauvc_sensor_check camera_check_down --ros-args -p topic:=/camera/image
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `topic` | string | `/camera_down/image_raw` / `/camera_front/image_raw` | Image topic to check. |

**Observe:** resolution, encoding, measured Hz, mean brightness (≈0 = lens cap / dead
exposure; saturated ≈255 = blown exposure).
