# `sauvc_localization`

Dead-reckoning localization for the real vehicle: optical-flow velocity ("DIY DVL"),
lane-line absolute heading, IMU covariance tooling, and an optional GTSAM
preintegration smoother. The math core (`flow_core.py`) is ROS-free and shared with
`sauvc_flow_eval`; it now includes the reference-hold / forward-backward /
predictive-seeding / alias-gate machinery plus optional grid features & CLAHE
(see comments in `flow_core.py`).

## `flow_velocity_node` — Phase 3

Body-frame vx, vy from downward-camera optical flow, scaled metric by `/altitude`.

**Pub:** `/flow/twist` (TwistWithCovarianceStamped, covariance quality-scaled).
**Sub:** `image_topic` (param), `/imu/data` (derotation gyro), `/altitude`.

```bash
ros2 run sauvc_localization flow_velocity_node
ros2 run sauvc_localization flow_velocity_node --ros-args --params-file <ws>/src/sauvc_bringup/config/flow.yaml
ros2 run sauvc_localization flow_velocity_node --ros-args -p fx:=712.3 -p fy:=713.1 -p sign_y:=-1.0
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `fx`,`fy`,`cx`,`cy` | double | `700,700,320,240` | Camera intrinsics — MUST be your UNDERWATER calibration (Phase 0). |
| `swap_xy` | bool | `false` | Camera→body axis swap, set from the hand-push sign test. |
| `sign_x`,`sign_y` | double | `1.0,1.0` | Camera→body sign fixes, same test. |
| `base_var` | double | `0.02` | Base velocity variance; published covariance = base_var × (1+spread) × (100/n_inliers). |
| `image_topic` | string | `/camera_down/image_raw` | Down-camera image topic. |

**Observe:** startup `flow_velocity_node up, waiting for images + altitude`. Stationary:
vx,vy ≈ 0 (mean = bias, std = noise — phase3 logger prints both on Ctrl-C). Hand-push
+x: vx positive and ≈ push speed; if the axis/sign is wrong, fix `swap_xy`/`sign_*`.
No output at all → no `/altitude` (flow can't be scaled) or no images.

## `lane_heading_node` — Phase 4

Absolute heading (mod 90°) from pool floor lines, fused as a slow complementary
correction to gyro yaw. **Pub:** `/heading/pool_relative` (corrected yaw, rad),
`/heading/line_meas` (raw line angle, debug). **Sub:** `/camera_down/image_raw`, `/imu/data`.

```bash
ros2 run sauvc_localization lane_heading_node --ros-args -p pool_axis_offset:=0.35 -p gain:=0.02
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `pool_axis_offset` | double | `0.0` | rad; makes yaw=0 point along your mission axis. Calibrate once at the venue: point at the gate, read `/heading/line_meas`, put that value here. |
| `gain` | double | `0.02` | Complementary correction gain per frame (bigger = trusts lines faster, noisier). |
| `min_lines` | int | `4` | Minimum detected lines before a measurement is accepted. |

**Observe:** `/heading/line_meas` publishes when the floor grid is visible (detection
rate = its row count vs pool_relative's in the phase4 CSVs); `/heading/pool_relative`
stays drift-free over minutes while raw gyro yaw wanders.

## `imu_covariance_check` — Phase 2b

Diagnoses (and optionally patches) covariance on `/imu/data`.

```bash
ros2 run sauvc_localization imu_covariance_check                      # diagnose only
ros2 run sauvc_localization imu_covariance_check --ros-args -p patch:=true \
    -p orientation_var_rpy:="[0.001,0.001,0.0002]" -p gyro_var_xyz:="[1e-4,1e-4,1e-4]"
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `patch` | bool | `false` | `true`: republish `/imu/data` → `/imu/data_corrected` with covariances overwritten from the params below (then point phase2 at the corrected topic). Try madgwick's `orientation_stddev` first — one param, no extra node. |
| `orientation_var_rpy` | double[3] | `[0.001,0.001,0.0002]` | rad² per axis. |
| `gyro_var_xyz` | double[3] | `[1e-4,1e-4,1e-4]` | (rad/s)². |
| `accel_var_xyz` | double[3] | `[0.05,0.05,0.05]` | (m/s²)². |

**Observe:** printed covariance fields. All-zero = EKF treats the IMU as perfect (bad);
−1 = "unknown". Values should come from `scripts/estimate_covariance.py`.

## `preint_smoother_node` — Phase 7 (optional, needs `pip3 install gtsam==4.3a0`)

ISAM2 factor-graph smoother: preintegrated IMU + flow velocity + depth, biases
estimated online → bounded-error coasting through flow dropouts. Run in PARALLEL with
the robot_localization EKF and A/B them (`phase7_preint.launch.py` does this).

**Pub:** `/odometry/preint`. **Sub:** `/imu/data`, `/flow/twist`, `/depth`.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `accel_sigma` | double | `0.15` | m/s² accel noise; INFLATE if thrusters shake the IMU. |
| `gyro_sigma` | double | `0.01` | rad/s gyro noise. |
| `keyframe_dt` | double | `0.2` | s between graph keyframes (~5 Hz). |
| `reset_s` | double | `60.0` | Sliding-window reset period (bounds compute). |

**Observe:** `/odometry/preint` at ~1/keyframe_dt Hz; during an induced flow dropout it
should drift far less than `/odometry/filtered`.

## `scripts/` (plain Python, no ROS)

**`offline_flow_test.py`** — run FlowVelocityEstimator on a recorded video:
```bash
python3 scripts/offline_flow_test.py video.mp4 --fx 700 --fy 700 --altitude 1.0 [--gyro gyro.csv] [--yaw-rate]
```
Prints per-second velocity stats + total integrated distance (uses `dt_eff`, so
reference-hold recoveries integrate correctly); saves `trajectory.png`, `velocity.png`.
The 5 m straight-push test: integrated distance vs tape measure = your scale error.

**`estimate_covariance.py`** — turn a stationary log column into a variance number:
```bash
ros2 topic echo --csv /imu/data/angular_velocity > gyro.csv     # ~60 s, Ctrl-C
python3 scripts/estimate_covariance.py gyro.csv --col 2 --label "gyro wz"
# yaw-drift variant:
python3 scripts/estimate_covariance.py --drift-rate-deg-per-min 0.5 --window-s 10
```
