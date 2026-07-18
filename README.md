# `sauvc_motion_demo`

Direct thruster control — **no ArduSub, no firmware**. The fastest loop for testing control
ideas: your node computes setpoints and publishes them straight to Stonefish.

| Node | Purpose |
|---|---|
| `depth_pid_mission` | Dive to a depth with a PID, hold it, translate on all four axes, surface. |

---

## `depth_pid_mission`

```bash
ros2 run sauvc_motion_demo depth_pid_mission
ros2 run sauvc_motion_demo depth_pid_mission --ros-args -p target_depth:=1.2 -p kp:=1.0 -p leg_time:=4.0
```

**Mission sequence:** `WAIT` (for first pressure sample) → `DESCEND` (PID to `target_depth`,
requires <0.10 m error held 2 s) → `HOLD` → `FORWARD` → `RIGHT` → `LEFT` → `BACKWARD`
(PID keeps correcting depth throughout each leg) → `SURFACE` → `DONE` (zeros thrusters, exits).

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `robot` | string | `sauvc_auv` | Topic namespace; must match `robot_name` in the scenario. |
| `target_depth` | double | `1.0` | Depth to reach and hold [m]. Floor at the start zone is ~1.23 m — keep below it. |
| `hold_time` | double | `5.0` | Seconds to hold depth before the legs [s]. |
| `leg_time` | double | `5.0` | Duration of each translation leg [s]. |
| `surge_cmd` | double | `0.3` | Forward/backward thruster magnitude ∈ [−1, 1]. |
| `sway_cmd` | double | `0.3` | Right/left thruster magnitude ∈ [−1, 1]. |
| `kp` | double | `1.5` | Proportional gain (depth error → vertical thrust). |
| `ki` | double | `0.05` | Integral gain — absorbs the constant buoyancy offset. |
| `kd` | double | `0.8` | Derivative gain (on measurement) — damping. |

PID output is clamped to ±0.6 and the integral to ±2.0. Positive vertical setpoint = descend.

**Observe:**
```
mission started: waiting for pressure...
surface pressure reference: 101325.0 Pa
-> DESCEND (depth 0.08 m)
[DESCEND ] depth +0.55 m  v_cmd +0.42
-> HOLD (depth 0.98 m)
-> FORWARD (depth 1.00 m)
...
mission complete
```
Depth converges to target without oscillation; state transitions in order; vehicle visibly dives,
holds, translates, surfaces.

**Prerequisites:** run the four open-loop group tests from `sauvc_stonefish/README.md` first —
a reversed thruster makes the PID fight itself.

**Start it at the surface** — the first pressure sample becomes the zero reference.

**Killing it mid-run:** the last setpoints keep acting. Zero them:
```bash
ros2 topic pub -1 /sauvc_auv/thruster_setpoints std_msgs/msg/Float64MultiArray "{data: [0,0,0,0,0,0,0,0]}"
```

**Tuning:**

| Symptom | Fix |
|---|---|
| Depth oscillates around target | Lower `kp`, or raise `kd` |
| Settles persistently above/below target | Raise `ki` |
| Dive too sluggish | Raise `kp` |
| Vehicle pitches/rolls while diving | **Not the PID** (all 4 verticals get the same command) — that's CG/CB trim: adjust `FloatVolume`/`BallastVolume` in `my_auv.scn` |

---

# Workspace guide — `sauvc_ws`

Everything below documents the full workspace: every package, every node, every launch
file, every parameter, how to pass arguments, and what to observe when it runs.
Each package also carries its own `README.md` with the same information in depth.

## Package index

| Package | Role | Nodes (executables) | Launch files |
|---|---|---|---|
| `sauvc_bringup` | Phase-by-phase hardware bringup launches + shared configs (`cameras.yaml`, `flow.yaml`, `ekf.yaml`) | — (launch-only) | `pre_phase_sensor_check`, `cameras`, `phase1_depth` … `phase7_preint` |
| `sauvc_sensor_check` | Pre-Phase "is the sensor alive at all" checks, pipeline-independent | `pressure_check`, `imu_taobotics_check`, `imu_pixhawk_check`, `camera_check_down`, `camera_check_front` | — |
| `sauvc_drivers` | Hardware sensor drivers | `depth_altitude_node` (MS5837 Bar30/Bar02) | — |
| `sauvc_localization` | Dead-reckoning localization: optical flow, lane heading, IMU preintegration | `flow_velocity_node`, `lane_heading_node`, `imu_covariance_check`, `preint_smoother_node` (+ offline `scripts/`) | — |
| `sauvc_flow_eval` | Sim-only estimator shoot-out vs Stonefish ground truth | `flow_eval_node` | `flow_eval.launch.py` |
| `sauvc_vision` | Prop detection on the forward camera | `gate_detector_node` | — |
| `sauvc_mission` | Competition state machine (skeleton) | `mission_node` | — |
| `sauvc_logging` | Automatic per-phase CSV logging + Ctrl-C statistics | `csv_logger_node` | — |
| `sauvc_motion_demo` | Direct-thruster control demo (documented at the top of this file) | `depth_pid_mission` | — |

Sim-side packages (`sauvc_stonefish`, `sauvc_sim_bridge`, `sauvc_teleop`) are documented
in their own trees; `sauvc_flow_eval`'s README covers how they combine for an eval run.

Root-level scripts (no package, no build): **`sim_check.py`** — one-shot Stonefish
health check (RTF, /clock, intrinsics, pressure, scene gaps); documented below.

## Build

```bash
cd ~/Robotics_Job/sauvc_ws
colcon build                       # everything
colcon build --packages-select sauvc_flow_eval sauvc_localization   # just some
source install/setup.bash          # EVERY terminal, EVERY rebuild
```

## How to pass arguments — the three templates

**1. Node parameters with `ros2 run`** — `--ros-args` then `-p name:=value` per parameter:

```bash
ros2 run <package> <executable> --ros-args -p <param>:=<value> -p <param2>:=<value2>
# examples
ros2 run sauvc_drivers depth_altitude_node --ros-args -p use_floor_profile:=false -p pool_depth:=2.0
ros2 run sauvc_flow_eval flow_eval_node --ros-args -p show_windows:=false -p print_rate:=1.0
ros2 run sauvc_logging csv_logger_node --ros-args -p topics:="['/depth','/altitude']" -p run_name:=test
```
Types matter: booleans `true/false`, lists in quotes `"[1.0, 2.0]"`, strings plain or quoted.

**2. Launch arguments with `ros2 launch`** — `name:=value` directly, no `--ros-args`:

```bash
ros2 launch <package> <launch_file> <arg>:=<value> <arg2>:=<value2>
# example
ros2 launch sauvc_flow_eval flow_eval.launch.py compare_frame:=ned show_windows:=false
```
List a launch file's arguments with `ros2 launch <package> <file> --show-args`.

**3. Parameter files** — for big sets, YAML instead of the command line:

```bash
ros2 run <package> <executable> --ros-args --params-file path/to/params.yaml
```
`sauvc_bringup/config/*.yaml` are exactly such files (see that package's README).

## Standard workflows

**Hardware pipeline (phases, in order):** Pre-Phase sensor checks →
`phase1_depth` (pressure/altitude + noise stats) → `phase2_heading` (IMU/Madgwick) →
`phase3_flow` (optical flow velocity) → `phase4_lane_heading` →
`phase5_ekf` (robot_localization fusion) → `phase6_full` (vision + mission) →
`phase7_preint` (optional GTSAM A/B). Every phase launch auto-includes `csv_logger_node`
with a phase-appropriate topic list, so each pool session leaves a CSV trail.

**Simulation eval:** Stonefish scenario → control/teleop → `flow_eval.launch.py` →
PlotJuggler. Full sequence in `sauvc_flow_eval/README.md`. Run `sim_check.py` (below)
once at the start of every sim session before trusting any numbers.

## Sim health check — `sim_check.py` (workspace root, no build required)

One-shot read-only diagnostic for the Stonefish sim. It answers, in one go, every
question that can silently corrupt a localization run — most importantly the real-time
factor, because Stonefish stamps every message with the **wall clock** while sensor
rates are declared in **sim time**: at RTF ≠ 1 the flow node reports RTF × true
velocity while gyro/depth are unscaled, i.e. the sim becomes *kinematically
inconsistent*, and `flow_scorer` would report a "scale error" of exactly RTF for a
perfectly working algorithm.

```bash
cd ~/Robotics_Job/sauvc_ws
python3 sim_check.py                 # 10 s measurement window (sim must be running)
python3 sim_check.py --secs 20      # longer window = tighter rate estimates
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `--secs` | float | `10.0` | Measurement window in seconds. |

**What it checks, and what to observe in each section:**

1. **REAL-TIME FACTOR** — observed wall-clock rate ÷ declared `rate="…"` from
   `my_auv.scn`, per sensor (imu 100, pressure 20, odometry 30, both cameras 30), plus
   a kinematic cross-check (|Δpos|/dt ÷ |twist|) when the vehicle is moving. Verdicts:
   `✓ RTF ≈ 1` → proceed; `✗ RTF ≠ 1` → fix before Phase 3 (drop camera resolution to
   640×480 in `my_auv.scn` or lower `rendering_quality`); `⚠ topics DISAGREE`
   (spread > 0.15) → per-sensor throttling (usually GPU-bound cameras), *not* RTF — a
   different problem with a different fix.
2. **`/clock`** — expected **absent** (Stonefish wall-clock stamps), which is why
   `use_sim_time` must stay `false` everywhere. `⚠` if it exists.
3. **CAMERA INTRINSICS** — fx/fy/cx/cy from `camera_info`, ready to paste into
   `flow_sim.yaml`; flags nonzero distortion and fx ≠ fy as unexpected.
4. **PRESSURE** — value, variance (converted to `depth_var` in m² for the EKF), the
   gauge-vs-absolute sign check, and — when the vehicle is submerged > 5 cm — an
   **empirical ρg check** against ground-truth depth (accounts for the sensor sitting
   0.10 m above the body origin). At the surface it tells you to dive and re-run.
5. **WHAT IS MISSING FROM `my_auv.scn`** — DVL presence (flow_scorer needs it), IMU
   yaw σ (near-zero = drift-free sim yaw → lane_heading tests pass vacuously), a
   zero accel covariance (spec-illegal "perfectly known"), and a USBL warning
   (sim-only sensor — never fuse it).

Run it whenever sim behavior looks off: it separates timing artifacts from genuine
algorithm errors before you spend an evening debugging the wrong one.

## Observing any run — generic tools

```bash
ros2 topic list                      # what exists
ros2 topic hz /flow/twist            # is it publishing, at what rate
ros2 topic echo /altitude            # raw values
ros2 run rqt_image_view rqt_image_view   # any Image topic (e.g. /vision/debug_image)
ros2 run plotjuggler plotjuggler     # live plots
```
Per-node "what to observe" is listed in each package README next to each node.
