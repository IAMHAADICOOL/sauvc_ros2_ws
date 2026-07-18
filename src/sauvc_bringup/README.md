# `sauvc_bringup`

Launch-only package: one launch file per bringup phase, plus the shared parameter files
under `config/`. No nodes of its own — it wires nodes from the other packages together
in the order the pipeline is validated.

## Launch files

Run any of them as:
```bash
ros2 launch sauvc_bringup <file>.launch.py
```
None of these declare CLI launch arguments — parameters are set inline (with comments in
each file explaining what to flip) or come from `config/*.yaml`. To change a value
per-run without editing the launch file, run the underlying node standalone with
`--ros-args -p name:=value` (each node's parameters are documented in its own package
README), or edit the yaml and relaunch.

### `pre_phase_sensor_check.launch.py`
Brings up ALL sensor drivers + ALL check nodes at once: `pressure_check` (i2c mode),
the Taobotics IMU driver + `imu_taobotics_check`, mavros + `imu_pixhawk_check`, both
cameras + both camera checks. Use only AFTER each sensor passed its individual check —
debugging five sensors at once is miserable.
**Caveats baked into the file:** Topology B pressure (`source:=mavlink`) cannot run here
(mavros owns the Pixhawk link) — run `pressure_check` standalone for that. Device paths
assume the udev pinning from SETUP.md (`/dev/imu_a9`, `/dev/pixhawk`, `/dev/cam_*`);
if not pinned yet, run the included launches individually with overridden ports.
**Observe:** each check node's periodic report line (rates, values, PASS/FAIL sanity
notes) — details per node in `sauvc_sensor_check/README.md`.

### `cameras.launch.py`
Two `v4l2_camera` nodes namespaced `/camera_down` and `/camera_front`, parameters from
`config/cameras.yaml` (device path via udev symlink, 640x480, YUYV, frame_id).
**Observe:** `ros2 topic hz /camera_down/image_raw` ≈ camera fps; view in rqt_image_view.

### `phase1_depth.launch.py`
`depth_altitude_node` (sauvc_drivers) + csv logger on `/depth`, `/altitude`
(`run_name=phase1_depth`). Inline parameters: `use_floor_profile: False` (flip to True
for the SAUVC V-floor), `pool_depth: 1.4` (set YOUR practice pool's depth),
profile arrays, `i2c_bus: 1`, `sensor_model: bar30`, `rate_hz: 20`, `depth_var: 0.0004`
(PLACEHOLDER — replace with your measured std² from the Phase 1 holds).
**Observe:** "surface pressure reference" style zeroing log, then depth ≈ 0 at surface;
Ctrl-C prints mean/std for the surface / 0.5 m / 1.0 m holds — those numbers ARE the
Phase 1 deliverable.

### `phase2_heading.launch.py`
`imu_filter_madgwick` (`use_mag:=false`, ENU world, `/imu/data_raw` → `/imu/data`) +
csv logger on `/imu/data` (`run_name=phase2_heading`). Skip the filter if your IMU
already outputs a fused quaternion on `/imu/data`.
**Observe:** yaw drift over a 10-min stationary hold (eyeball qz in the CSV); the
Ctrl-C wz std feeds the Phase 2b gyro variance.

### `phase3_flow.launch.py`
Includes cameras + phase1 + phase2, adds `flow_velocity_node` (params from
`config/flow.yaml`) + csv logger on `/flow/twist` (`run_name=phase3_flow`).
**Observe:** stationary vx/vy mean ≈ 0 (bias) and std (noise) from the Ctrl-C summary;
hand-push tests set `swap_xy`/`sign_x`/`sign_y` in `flow.yaml`.

### `phase4_lane_heading.launch.py`
`lane_heading_node` (`pool_axis_offset: 0.0`, `gain: 0.02`) + csv logger on
`/heading/pool_relative`, `/heading/line_meas` (`run_name=phase4_lane_heading`).
**Observe:** detection rate = rows in line_meas vs pool_relative; corrected-yaw drift
over time in the CSV.

### `phase5_ekf.launch.py`
Includes phase3 + phase4, adds `robot_localization` `ekf_node` (params
`config/ekf.yaml`) + csv logger on `/odometry/filtered` (`run_name=phase5_ekf`).
**Observe:** `/odometry/filtered` publishes at `frequency: 30`; square-test closure
error from the logged CSV.

### `phase6_full.launch.py`
Includes phase5, adds `gate_detector_node` + `mission_node` + csv logger on
`/odometry/filtered`, `/vision/detections` (`run_name=phase6_mission`).
**Observe:** mission state-transition log lines; detections stream while props are in view.

### `phase7_preint.launch.py`
Includes phase5, adds `preint_smoother_node` + csv logger on BOTH
`/odometry/filtered` and `/odometry/preint` (`run_name=phase7_preint_ab`) so the EKF vs
preintegration A/B is a straight file diff.
**Observe:** both odometry topics alive; compare drift during induced flow dropouts.

## `config/`

| File | Consumed by | Key contents |
|---|---|---|
| `cameras.yaml` | `cameras.launch.py` | per-camera `video_device` (udev symlink), `image_size [640,480]`, `pixel_format YUYV`, `camera_frame_id` |
| `flow.yaml` | `flow_velocity_node` | `fx/fy/cx/cy` (REPLACE with your underwater calibration), `swap_xy/sign_x/sign_y` (from the hand-push test), `base_var`, `image_topic` |
| `ekf.yaml` | `robot_localization` | 15-bool sensor configs: `imu0=/imu/data` (rpy + rates), `twist0=/flow/twist` (vx,vy), `pose0=/depth` (z), `pose1=/pose_correction` (x only — anisotropic landmark correction; deliberately NOT `/set_pose`), process noise diagonal, `frequency: 30`, `two_d_mode: false` |
