# sauvc_flow_eval

Runs several localization approaches side by side off the same live sensors and plots
each against Stonefish ground truth, so you can see which stays most consistent. No
landmark/gate localization (dead-reckoning-style only), no ArduSub.

## The five estimators

| /eval topic | approach | scale source |
|---|---|---|
| `/eval/ground_truth` | Stonefish odometry — the reference | exact |
| `/eval/flow` | optical flow + altitude, integrated to position | **pressure altitude** |
| `/eval/ekf` | self-contained KF fusing flow velocity + depth | flow (metric) |
| `/eval/pressure` | depth only (world Z; x,y stay 0) | direct |
| `/eval/gtsam` | IMU preintegration, integrated to position | **accel + bias** (optional) |

### On scale ambiguity — the honest framing

Your optical flow is **not** scale-ambiguous: `flow_core` already scales pixel flow by
altitude (`pool_depth − depth` from pressure), so `/eval/flow` is metric without any IMU
help. GTSAM preintegration here is therefore **not** "the thing that makes flow metric" —
it's an *independent* metric estimate whose scale comes from accelerometer integration
with online bias estimation instead of from altitude.

That independence is the point. Flow+altitude and GTSAM share no scale source, so:
- a wrong floor profile corrupts `/eval/flow` but not `/eval/gtsam`
- accel bias / gravity error corrupts `/eval/gtsam` but not `/eval/flow`

When the two diverge, the divergence tells you *which* is at fault. That's a better
experiment than either alone.

`/eval/gtsam` is **optional**: if `gtsam` isn't importable, that estimator is silently
skipped and the other four run normally. One missing dependency never kills the
comparison.

## Frames — the NED question

Everything is published in ONE common frame so PlotJuggler overlays are apples-to-apples.
`compare_frame` defaults to **`ned`** (as requested). Ground truth is native NED and is
left untouched; the ENU-native estimates are converted with the **single tested
conversion** in `sauvc_sim_bridge.frames` (2000-sample verified), applied in one place
(`eval_common`), never as ad-hoc per-estimator sign flips. `compare_frame:=enu` flips
which side converts. Depth sign follows the frame: NED z = +down (a 1.2 m dive reads
+1.2), ENU z = +up (−1.2).

Position tracks for flow and GTSAM are **integrated from velocity and will drift** — that
drift, against the non-drifting ground truth, is exactly what you're here to see.

## Full run sequence

Each in its own terminal. Source the workspace everywhere first.

```bash
# 0. build
cd ~/Robotics_Job/sauvc_ws
colcon build --packages-select sauvc_sim_bridge sauvc_flow_eval sauvc_teleop
source install/setup.bash

# 1. simulator
ros2 launch sauvc_stonefish sauvc_qualification.launch.py

# 2. control (Path A, so you can drive) + shims
ros2 launch sauvc_teleop teleop_direct.launch.py

# 3. the comparison node + its shim drivers + OpenCV windows
#    (sim_drivers is included here too; running it twice is harmless — same nodes,
#     but if you prefer, launch flow_eval with its own drivers and skip teleop's copy)
ros2 launch sauvc_flow_eval flow_eval.launch.py compare_frame:=ned

# 4. teleop keyboard (its own terminal — it owns the TTY)
ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=absolute

# 5. PlotJuggler
ros2 run plotjuggler plotjuggler
#    File -> Layout -> Load -> config/flow_eval_plotjuggler.xml
#    Start a ROS2 Topic Subscriber, select all /eval/* topics.
```

Then drive with `w/a/s/d/q/e` and `r/f` for depth, and watch the three plots (X, Y,
depth). Legend: black = ground truth, blue = flow, green = ekf, orange = pressure,
red = gtsam.

> Note on double drivers: both `teleop_direct.launch.py` and `flow_eval.launch.py`
> include `sim_drivers`. Pick ONE to own the shims — either run `flow_eval.launch.py`
> with `show_windows:=true` and drive via a bare `direct_control_node`, or run teleop's
> stack and launch the eval node alone with
> `ros2 run sauvc_flow_eval flow_eval_node`. Running two copies of the same shim nodes
> will produce duplicate publishers on `/imu/data` etc.

## OpenCV windows

Two windows open from the eval node: **down camera** (raw feed) and **optical flow**
(tracked corners in green, median flow vector in red, inlier count + velocity overlaid).

## Modularity / real vehicle

Each estimator is its own module under `estimators/`; the node only wires them. On the
real AUV the node subscribes to the identical shimmed topics (`/imu/data`, `/depth`,
`/altitude`, `/camera_down/image_raw`); `/sauvc_auv/odometry` (ground truth) simply
doesn't exist there, so `/eval/ground_truth` stays silent and the rest keep working.

## Tests
```bash
PYTHONPATH=. python3 test/test_eval_common.py
```
Frame conversions (involution, magnitude, depth sign), ground-truth vs estimate
conversion direction, and the position integrator (straight line, yaw rotation, bad-dt
rejection).

## Terminal output & toggling the optical-flow window

The node prints all five estimates as a table (throttled to `print_rate`, default 5 Hz):

```
─ estimates [NED frame]  x        y        z ─────────────
  ground_truth    +1.234   -0.567   +0.300
  flow            +1.180   -0.540   +0.298
  ekf             +1.210   -0.555   +0.300
  pressure        +0.000   +0.000   +0.301
  gtsam           +1.050   -0.480   +0.299
```

Read it row-against-row: how far flow/ekf/gtsam have drifted from ground_truth in x/y,
and whether all the z values agree with pressure. `pressure` has x=y=0 by design (depth
only).

### Command-line toggles

`ros2 run` — pass with `--ros-args -p name:=value`:
```bash
# everything on (default)
ros2 run sauvc_flow_eval flow_eval_node

# disable JUST the optical-flow window, keep the raw camera + terminal print
ros2 run sauvc_flow_eval flow_eval_node --ros-args -p show_optical_flow:=false

# headless: no windows at all, terminal print only (good over SSH)
ros2 run sauvc_flow_eval flow_eval_node --ros-args -p show_windows:=false

# windows on, terminal print off
ros2 run sauvc_flow_eval flow_eval_node --ros-args -p print_estimates:=false

# slow the print to 1 Hz
ros2 run sauvc_flow_eval flow_eval_node --ros-args -p print_rate:=1.0
```

`ros2 launch` — same names as launch arguments:
```bash
ros2 launch sauvc_flow_eval flow_eval.launch.py show_optical_flow:=false
ros2 launch sauvc_flow_eval flow_eval.launch.py show_windows:=false print_estimates:=true
```

| param | default | effect |
|---|---|---|
| `show_windows` | true | master switch for ALL OpenCV windows |
| `show_optical_flow` | true | the optical-flow overlay window |
| `show_camera` | true | the raw down-camera window |
| `print_estimates` | true | terminal table of all five estimates |
| `print_rate` | 5.0 | Hz throttle for the terminal table |

## Reading the optical-flow arrows

The overlay draws **two** arrows from the image center:

- **red = raw pixel flow.** Ground features stream *opposite* to your travel (scenery past
  a train window). Driving forward makes this point backward. **This is correct**, not a
  bug — a large arrow opposing your motion is exactly what optical flow should show.
- **cyan = recovered velocity direction** (= −flow), pointing *along* your travel. This is
  what becomes `/eval/flow` vx,vy.

So don't judge the sign by the arrow. Judge it by the **terminal table**: drive steadily
forward and check that `/eval/flow`'s x moves the *same direction* as `/eval/ground_truth`'s
x. If they move opposite ways, the convention is wrong — fix `sign_x`/`sign_y`/`swap_xy`
in `flow_sim.yaml`, not the visualization. If they agree, the backward-pointing red arrow
was right all along.

## Update: GTSAM is now a real iSAM2 factor graph, and the origin offset is fixed

Two corrections from testing against ground truth:

### Origin anchoring (the −12.1 m offset)

`sauvc_qualification.scn` spawns the vehicle at `start_position="-12.1 0.0 0.3"` — the
start-zone wall, not the world origin. The estimators all dead-reckon from zero, so the
plots carried a fixed ~12.1 m x-offset and could never overlay. Every `/eval/*` track is
now **anchored to ground truth's first pose**, so the comparison is displacement-from-start
— which is what dead reckoning actually measures.

### GTSAM: proper iSAM2 graph (config B)

The previous GTSAM estimator misused preintegration — it predicted and reset every frame
with no optimization and no gravity alignment, and diverged to +57 m. That wasn't Forster
et al. failing; it was only half their method. Preintegration is the motion *model*; the
estimate comes from the *graph*.

The estimator is now a real iSAM2 factor graph:
- **CombinedImuFactor** — IMU preintegration + bias evolution between keyframes
- **flow velocity factor** — optical-flow body velocity, rotated to NED
- **pressure depth factor** — GPSFactor constraining Z only (x/y sigma → ∞)

Same three inputs as the eval EKF, so `/eval/gtsam` is now the **fairest possible
competitor** to `/eval/ekf`: identical sensors, smoothing-graph vs recursive-filter.

Validated offline (real accel bias, 6–8 s):
| configuration | x error | why |
|---|---|---|
| IMU + flow + pressure | **~5–8 mm** | flow pins velocity, graph estimates bias |
| IMU + pressure only | ~151 mm | no horizontal measurement — drifts, correctly |
| IMU only | ~907 mm | bounded now gravity is aligned (was +57 m) |

The two bugs fixed: **gravity alignment** (initial attitude seeded from the AHRS so the
9.81 m/s² of specific force isn't integrated as motion) and the **keyframe interval** (set
by the camera rate, ~15 IMU samples — only meaningful *with* the optimization between
keyframes, which the old version lacked).

---

## Complete parameter reference (current node, all upgrades included)

The node has grown a self-sufficient altitude path and several accuracy upgrades; this
table is exhaustive. Pass any of them with
`ros2 run sauvc_flow_eval flow_eval_node --ros-args -p name:=value`.

### Core

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `compare_frame` | string | `ned` | `ned` or `enu` — the ONE frame all `/eval/*` tracks publish in. |
| `robot_name` | string | `sauvc_auv` | Topic namespace for pressure + ground truth. |
| `fx`,`fy`,`cx`,`cy` | double | `381.36, 381.36, 320, 240` | Intrinsics matching the 640×480 scene camera. Old 1280×720 scene → 762.72/762.72/640/360. |
| `gravity` | double | `9.81` | Passed to the GTSAM estimator. |
| `gtsam_keyframe_period` | double | `0.2` | s between full iSAM2 keyframe updates (~5 Hz); IMU preintegrates in between. Lower = smoother graph, more CPU. |

### Altitude (Upgrade A — self-sufficient; the depth_shim odom-feed trap can't recur)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `self_altitude` | bool | `true` | Compute camera-to-floor altitude IN-NODE from pressure depth + ground-truth x + the profile + mount offsets. `false` = legacy: use `/altitude` from depth_shim. |
| `use_floor_profile` | bool | `true` | V-floor profile vs flat `pool_depth`. |
| `floor_profile_x` | double[] | `[0,12.5,25]` | Wall-referenced breakpoints [m]. |
| `floor_profile_depth` | double[] | `[1.2,1.6,1.2]` | Floor depth at breakpoints [m]. |
| `profile_x_offset` | double | `12.5` | world NED x → wall-referenced profile x. |
| `pool_depth` | double | `1.6` | Flat-floor depth when the profile is off / fallback. |
| `sensor_above_origin` | double | `0.10` | Pressure-sensor mount above body origin [m] (my_auv.scn). |
| `camera_below_origin` | double | `0.11` | Down-camera mount below body origin [m]. |
| `alt_mismatch_warn` | double | `0.15` | Warn (10 s throttle) if `/altitude` disagrees with the in-node value by more — diagnoses a misconfigured shim without affecting results. 0 disables. |
| `tilt_compensation` | bool | `true` | Upgrade B: altitude ÷ cos(roll)·cos(pitch) (range along the optical axis), clamped at 60°. |

### Estimation quality (Upgrades C–G)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `r_flow_base` | double | `0.02` | Upgrade E: EKF flow variance = base × (1+spread_px) × (100/n_inliers). |
| `zupt` | bool | `true` | Upgrade F: zero-velocity update — clamp v to exactly 0 when flow AND gyro read ~zero (kills stationary creep). |
| `zupt_vel` | double | `0.03` | m/s flow-speed threshold for ZUPT. |
| `zupt_gyro` | double | `0.02` | rad/s gyro-magnitude threshold for ZUPT. |
| `use_lane_heading` | bool | `false` | Upgrade D: substitute fresh `/heading/pool_relative` yaw for IMU yaw. Enable only when lane_heading_node runs. |
| `lane_heading_topic` | string | `/heading/pool_relative` | Lane yaw topic. |
| `pool_axis_ned_yaw` | double | `0.0` | NED yaw of the pool axis (lane yaw = 0) [rad]. |
| `lane_fresh_s` | double | `1.0` | Max lane-yaw staleness to trust it [s]. |
| `use_clahe` | bool | `false` | Upgrade G: CLAHE contrast equalization. OFF by default: its space-variant mapping breaks held-gap recovery (measured); enable only for genuinely washed-out footage. |
| `feature_grid_rows` / `feature_grid_cols` | int | `3` / `4` | Grid-distributed corner detection (spreads features; 0 disables). |

Gyro derotation timestamp-sync (Upgrade C) is always on — no parameter.

### Display / output

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `show_windows` | bool | `true` | Master switch for all OpenCV windows. |
| `show_camera` | bool | `true` | Raw down-camera window. |
| `show_optical_flow` | bool | `true` | Flow-overlay window (green corners, red raw-flow arrow, cyan velocity arrow, dropout banner). |
| `print_estimates` | bool | `true` | Terminal table of all five estimates. |
| `print_rate` | double | `5.0` | Hz throttle for the table. |

### Launch file `flow_eval.launch.py` — arguments

`ros2 launch sauvc_flow_eval flow_eval.launch.py <arg>:=<value> ...`
(also brings up the sim shim drivers; do NOT run a second copy of sim_drivers).

| Argument | Default | Meaning |
|---|---|---|
| `compare_frame` | `ned` | Passed to the node. |
| `use_floor_profile` | `true` | Passed to the shim drivers (the node has its own parameter of the same name, default true). |
| `alt_odom_topic` | `/sauvc_auv/odometry` | Odometry feeding depth_shim's floor-profile x (ground truth — honest for diagnostics). With `self_altitude` the eval no longer depends on this, but the shim's `/altitude` cross-check does. |
| `show_windows` / `show_optical_flow` / `show_camera` / `print_estimates` | `true` | Passed to the node. |

### What to observe (updated)

- Startup line must read **"ALTITUDE: self-computed in-node (floor profile at GT x,
  camera datum, tilt-compensated)"** — if it says "from /altitude (shim)" you are running
  the old build.
- Flow dropout WARNs name the reason (`lk_none` / `few_tracked` / `few_inliers` /
  `bad_altitude` / `…+track_lost`) and running totals; only `track_lost` gaps are truly
  lost motion (check rtf_monitor for CPU starvation).
- A "/altitude from depth_shim … disagrees" WARN means the shim's odom feed is still
  misconfigured — informational only under `self_altitude`.
- In the table: flow/ekf/gtsam x,y should track ground_truth within a slowly growing
  drift; `pressure` x,y stay at the anchor BY DESIGN (depth-only baseline) while its z
  tracks the dive.
