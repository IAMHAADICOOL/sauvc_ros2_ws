# `sauvc_sensor_tests`

One node per sensor. Use these first: if a sensor looks wrong here, nothing built on top of it
can be trusted. All nodes need the simulator running; `pixhawk_imu_test` additionally needs the
bridge + ArduSub SITL.

| Node | What it reads | Needs |
|---|---|---|
| `pressure_test` | Pressure sensor → depth | sim |
| `imu_test` | **Stonefish** IMU (your own sensor) | sim |
| `camera_test` | Front + down cameras | sim, `cv_bridge`, `python3-opencv` |
| `pixhawk_imu_test` | **ArduSub's** EKF attitude + raw IMU | sim + bridge + SITL, `pymavlink` |

---

## `pressure_test`

```bash
ros2 run sauvc_sensor_tests pressure_test [--ros-args -p topic:=/sauvc_auv/pressure -p p_ref:=101325.0]
```

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `topic` | string | `/sauvc_auv/pressure` | Pressure topic to subscribe to. |
| `p_ref` | double | `NaN` | Surface reference pressure [Pa]. `NaN` ⇒ auto-zero on the **first sample** (so start it while the vehicle is at the surface). |

Depth is `(p − p_ref) / (ρ·g)`, ρ = 1000, g = 9.81.

**Observe:**
```
surface pressure reference: 101325.0 Pa
pressure:    111135.00 Pa   depth:  +1.000 m   (variance 20.00)
```
Depth ≈ 0 at the surface; increases as it descends. Cross-check against `/sauvc_auv/odometry`
`position.z` (ground truth) — they should agree within sensor noise.

---

## `imu_test`

```bash
ros2 run sauvc_sensor_tests imu_test [--ros-args -p topic:=/sauvc_auv/imu]
```

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `topic` | string | `/sauvc_auv/imu` | IMU topic. Also works on `/mavros/imu/data` to read the Pixhawk's IMU via MAVROS. |

**Observe:**
```
RPY [deg]:   +0.12   -0.05  +91.20 | gyro [rad/s]: +0.001 -0.002 +0.000 | accel [m/s2]: +0.021 -0.013 -9.807
```
At rest, `accel ≈ (0, 0, −9.81)` — Stonefish outputs **specific force** (gravity included), like a
real accelerometer. RPY should match the odometry attitude.

---

## `camera_test`

```bash
ros2 run sauvc_sensor_tests camera_test [--ros-args -p front_topic:=/sauvc_auv/camera_front -p down_topic:=/sauvc_auv/camera_down]
```

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `front_topic` | string | `/sauvc_auv/camera_front` | Base name for the forward camera. |
| `down_topic` | string | `/sauvc_auv/camera_down` | Base name for the downward camera. |

Subscribes to both `<base>/image_color` and `<base>` (stonefish_ros2 versions differ). Press **q**
in a window to quit.

**Observe:** two OpenCV windows — *front camera* (arena ahead: flares, gate, drums) and
*down camera* (blue mosaic tiles; the texture is what makes optical-flow odometry viable).
No windows ⇒ check `ros2 topic list | grep camera` and `ros2 topic hz <topic>/image_color`.

---

## `pixhawk_imu_test`

```bash
ros2 run sauvc_sensor_tests pixhawk_imu_test [udp:127.0.0.1:14550]
```

CLI argument (not `--ros-args`): MAVLink URL, default `udp:127.0.0.1:14550`.

Requests `ATTITUDE` (10 Hz) and `RAW_IMU` (10 Hz) and prints the **firmware's** view.

**Observe:**
```
EKF RPY [deg]:   +0.30   -0.10  +91.50 | rates [rad/s]: +0.001 ... | raw accel [m/s2]: ... -9.81
```
`(no data for 5 s - is the JSON bridge running?)` ⇒ SITL has nothing to fuse; start the bridge.

**Why both IMU nodes exist:** `imu_test` shows Stonefish's simulated sensor (near ground truth +
noise); `pixhawk_imu_test` shows what ArduSub *believes* after its EKF digests the bridge's state
feed. On the real vehicle these are "my IMU" vs "the Pixhawk via MAVLink". Comparing them side by
side is the fastest check that the bridge is feeding a consistent state — large or growing
disagreement means a frame/timing bug in the bridge, not a sensor problem.

**MAVROS alternative** (no new code needed):
```bash
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14550@
ros2 run sauvc_sensor_tests imu_test --ros-args -p topic:=/mavros/imu/data
```
