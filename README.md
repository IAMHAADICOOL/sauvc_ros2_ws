# SAUVC 2026 — AUV Simulation & Control Workspace

A ROS 2 workspace that reproduces the **Singapore AUV Challenge (SAUVC) 2026** arena in the
[Stonefish](https://github.com/patrykcieslak/stonefish) marine simulator, runs your own CAD
vehicle in it, and lets you drive that vehicle **two ways**:

1. **Directly** — publish thruster setpoints from a ROS 2 node (fast iteration, no firmware).
2. **Through ArduSub** — the real Pixhawk firmware in SITL, over MAVLink/MAVROS, exactly the
   control path used on the physical vehicle.

```
sauvc_ws/src/
├── sauvc_stonefish/     arena + vehicle scenarios, launch files, mesh tools, ArduSub bridge
├── sauvc_sensor_tests/  one node per sensor: pressure, IMU, cameras, Pixhawk IMU
├── sauvc_motion_demo/   depth-PID mission driving Stonefish thrusters directly
├── sauvc_ardusub_demo/  same mission via ArduSub SITL + motor-map discovery tool
├── sauvc_bringup/       phase-by-phase HARDWARE bringup launches + shared configs (cameras/flow/ekf yaml)
├── sauvc_sensor_check/  pre-phase "is the sensor alive at all" checks (hardware, pipeline-independent)
├── sauvc_drivers/       hardware sensor drivers (MS5837 Bar30/Bar02 depth + altitude)
├── sauvc_localization/  optical-flow velocity ("DIY DVL"), lane heading, IMU preintegration
├── sauvc_flow_eval/     SIM-only estimator shoot-out vs Stonefish ground truth
├── sauvc_vision/        HSV prop detection on the forward camera
├── sauvc_mission/       SAUVC competition state machine (skeleton)
└── sauvc_logging/       automatic per-phase CSV logging + Ctrl-C statistics
```
plus **`sim_check.py`** at the workspace root — a one-shot, no-build Stonefish health check (§15).

Each package has its own README with node-by-node detail. This document covers the *why*,
the setup, and the cross-package workflow.

---

## 1. Why Stonefish?

| Requirement | Why Stonefish fits |
|---|---|
| **Underwater dynamics that aren't a toy** | Computes buoyancy, added mass, and quadratic + viscous drag **per triangle of the physics mesh**, integrated over the submerged fraction of the hull. Gazebo needs plugins bolted on to approximate this; Stonefish is built around it. |
| **Surface + submerged in one model** | `physics="floating"` handles a vehicle that is half out of the water (start box, surfacing) — not just a fully-submerged approximation. |
| **Thrusters as first-class actuators** | Real propeller model: rotor dynamics (first-order lag), handedness, quadratic thrust curve, and **zero thrust in air** — so bugs behave like real bugs. |
| **Sensor simulation with rendering** | Cameras with underwater light attenuation (Jerlov water types), pressure, IMU (proper specific force, not fake accel), odometry, USBL. Needed for vision + localization work. |
| **Scriptable scenarios** | XML scenes with includes: one arena file, swappable vehicle files, and programmatic randomization for competition-realistic runs. |
| **ROS 2 native** | `stonefish_ros2` publishes standard `sensor_msgs`/`nav_msgs` and subscribes thruster setpoints — no custom middleware. |
| **Fast enough to iterate** | 300 Hz physics with a GUI on a laptop GPU; headless is faster. |

**Why not the alternatives:** Gazebo's underwater plugins are less faithful for buoyancy and
drag; UWSim is effectively unmaintained; HoloOcean/UUV are heavier or Unreal-bound. For an AUV
where *hydrodynamics + thruster mixing + depth control* are the hard parts, Stonefish models
exactly those.

---

## 2. Prerequisites

* Ubuntu 24.04 + **ROS 2 Jazzy** (Humble works with path changes)
* An **NVIDIA GPU** with proprietary drivers (see §7 — this is the #1 cause of a blank window)
* ~4 GB free disk (meshes are large), 8 GB+ RAM

```bash
sudo apt update && sudo apt install -y \
  git cmake build-essential libglm-dev libsdl2-dev libfreetype6-dev \
  ros-jazzy-desktop python3-colcon-common-extensions \
  ros-jazzy-cv-bridge python3-opencv
pip install pymavlink numpy scipy trimesh   # trimesh/scipy only for the mesh tools
```

---

## 3. Install the Stonefish library

Stonefish is a **standalone C++ library** — build and install it *before* the ROS 2 wrapper.

```bash
cd ~/  # anywhere OUTSIDE the ROS workspace src/
git clone https://github.com/patrykcieslak/stonefish.git
cd stonefish && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)          # 5-15 min
sudo make install
sudo ldconfig            # so the ROS node finds libStonefish.so
```

Verify: `ls /usr/local/lib/libStonefish.so && ls /usr/local/include/Stonefish/`

---

## 4. Set up the workspace

```bash
mkdir -p ~/Robotics_Job/sauvc_ws/src && cd ~/Robotics_Job/sauvc_ws/src

# ROS 2 wrapper for Stonefish
git clone https://github.com/patrykcieslak/stonefish_ros2.git

# this workspace's four packages (unzip the delivery here)
unzip ~/Downloads/sauvc_ws.zip -d .

cd ~/Robotics_Job/sauvc_ws
colcon build --symlink-install
source install/setup.bash
echo "source ~/Robotics_Job/sauvc_ws/install/setup.bash" >> ~/.bashrc
```

**Gotchas that cost real debugging time:**

* **Keep the ArduPilot checkout OUT of `src/`.** If it must live there, add an empty
  `COLCON_IGNORE` file inside it, or colcon crawls the whole tree every build.
* **Scripts need the executable bit.** Zip transfers drop it. `CMakeLists.txt` sets explicit
  install permissions, but if `ros2 run sauvc_stonefish ardusub_json_bridge.py` says
  *"No executable found"*: `chmod +x src/sauvc_stonefish/scripts/*.py` and rebuild.
* **After changing scenarios or meshes**, a clean reinstall removes all doubt:
  `rm -rf build/sauvc_stonefish install/sauvc_stonefish && colcon build --packages-select sauvc_stonefish`

---

## 5. Set up ArduSub SITL (only for the ArduSub path)

```bash
cd ~/   # OUTSIDE sauvc_ws/src
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot && git submodule update --init --recursive
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
```

**Create the frame parameter file — this is not optional.** ArduSub boots as a **6-motor**
Vectored frame; this vehicle has **8 thrusters** and needs Vectored-6DOF (`FRAME_CONFIG = 2`):

```bash
echo "FRAME_CONFIG 2" > ~/sub_6dof.parm
```

`FRAME_CONFIG` is read **at boot** to build the motor mixer, so it must be passed at startup
(§9 explains how to detect and fix a wrong frame).

---

## 6. Start the simulation

Order matters: **sim → bridge → SITL**. SITL's scheduler is slaved to physics data from the
bridge; if the bridge isn't feeding, MAVProxy commands (`param set`, mode changes) time out.

```bash
# Terminal 1 — simulator
ros2 launch sauvc_stonefish sauvc_finals.launch.py

# Terminal 2 — bridge (only for the ArduSub path)
ros2 run sauvc_stonefish ardusub_json_bridge.py

# Terminal 3 — ArduSub SITL (only for the ArduSub path)
cd ~/ardupilot
Tools/autotest/sim_vehicle.py -v ArduSub -f json:127.0.0.1 \
    --add-param-file=$HOME/sub_6dof.parm -l 18.25,109.5,0,0 --console
```

`-l 18.25,109.5,0,0` puts home at **sea level**. Without it, SITL's default home (CMAC) sits
at ~584 m AMSL and every absolute-altitude number is nonsense.

**Direct control needs only Terminal 1.**

---

## 7. Blank window / nothing renders → GPU

If the Stonefish window opens but the pool and objects don't appear, it's almost always the
renderer running on integrated graphics instead of NVIDIA.

1. Install the proprietary NVIDIA drivers properly (`ubuntu-drivers devices`, then
   `sudo ubuntu-drivers autoinstall`, reboot). Verify with `nvidia-smi`.
2. If it still won't render, force PRIME offload **in the terminal you launch from**:

```bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
ros2 launch sauvc_stonefish sauvc_finals.launch.py
```

Make it permanent by adding both lines to `~/.bashrc`. Check which GPU is actually in use:
`__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia glxinfo | grep "OpenGL renderer"`

**Other launch symptoms:**

| Symptom | Cause |
|---|---|
| Exit code **-6** (SIGABRT) right after "Building scenario" | A referenced mesh file is missing — Stonefish `abort()`s. Compare the scenario's `data/...obj` references against the installed `share/sauvc_stonefish/data/`. Usually a stale/partial deploy: clean rebuild. |
| Load pauses ~1-5 s on `my_auv_vis_black.obj` | Normal. 921k faces; the "Loaded mesh..." line prints *after* the load. |
| Vehicle uniformly grey instead of CAD colors | Installed `sauvc_pool.scn` is stale (missing the `auv_*` `<look>` definitions). Unknown look names fall back to grey **silently**. Check: `grep -c auv_cream install/sauvc_stonefish/share/sauvc_stonefish/scenarios/sauvc_pool.scn` — `0` means stale. |

---

## 8. Arena variants: fixed vs randomized

| | Fixed | Randomized |
|---|---|---|
| Launch | `sauvc_finals.launch.py` | `sauvc_finals_random.launch.py seed:=N` |
| Prop positions | Hand-placed, identical every run | Drawn from the rulebook zones |
| Use for | Debugging, regression, tuning | Competition realism, robustness testing |

Randomization mirrors the official rules: **orange flare** anywhere in the 4–8 m band
(x ∈ [−8.5, −4.5]), **R/Y/B flares** anywhere between that band and the gate line
(x ∈ [−4.4, 3.9], ≥1.5 m apart), **gate** anywhere along its line (x = 4.4). Floor heights are
recomputed from the V-shaped floor profile, `d(x) = 1.6 − 0.032·|x|`.

**Seeding:** same seed ⇒ byte-identical arena, every launch, every machine (verified). The
chosen layout is printed at launch — log it with your results. The installed scenario is never
modified; the randomized scene is generated into `/tmp`.

`seed:=0` is **not** "no randomization" — it's just the draw for seed 0. Use the fixed launch
for hand-placed positions.

---

## 9. Fixing the frame: only 6 PWMs in the bridge log

The bridge prints its PWM input once per second. **This line is the ground truth for the frame:**

```
pwm[1-8]=[1500, 1500, 1500, 1500, 1500, 1500, 0, 0]   ← BROKEN: 6-motor frame
pwm[1-8]=[1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500]   ← CORRECT: 8-motor Vectored-6DOF
```

Channels 7–8 reading `0` while armed means ArduSub is running the **6-motor Vectored** frame.
Its mixing, applied to 8 thrusters, produces the wrong geometry — the vehicle topples as soon as
a stabilized mode (ALT_HOLD/STABILIZE) starts correcting attitude.

**Fix — boot with the frame parameter (recommended):**

```bash
echo "FRAME_CONFIG 2" > ~/sub_6dof.parm
# restart SITL with:
Tools/autotest/sim_vehicle.py -v ArduSub -f json:127.0.0.1 \
    --add-param-file=$HOME/sub_6dof.parm -l 18.25,109.5,0,0 --console
```

**Fix — interactively (needs sim + bridge already streaming):**

```
param set FRAME_CONFIG 2
param show FRAME_CONFIG      # must print 2
```
then **restart SITL** — the mixer is built at boot, so a live `param set` alone changes nothing.

> If `param set` replies *"Failed to set FRAME_CONFIG (no PARAM_VALUE received)"*, the sim or
> bridge isn't running. SITL's clock is driven by the bridge's physics packets; with no data
> the loop stalls and all MAVLink requests time out.

**Verify:** arm in MANUAL and watch for `pwm[1-8]=[1500 ×8]`. Then run
`ros2 run sauvc_ardusub_demo motor_map_check` (§10) to confirm which motor drives which thruster.

---

## 10. Recommended bring-up order

1. **Sim only** — `ros2 launch sauvc_stonefish sauvc_finals.launch.py`; vehicle floats in the start box.
2. **Sensors** — `pressure_test`, `imu_test`, `camera_test` (see `sauvc_sensor_tests/README.md`).
3. **Open-loop thruster checks** — the four group tests in `sauvc_stonefish/README.md`. Do this
   *before* any closed-loop work; a reversed thruster makes every controller fight itself.
4. **Direct control** — `ros2 run sauvc_motion_demo depth_pid_mission`.
5. **ArduSub** — start the bridge + SITL, confirm 8 PWMs (§9), run `motor_map_check`, paste the
   printed `MOTOR_MAP`/`MOTOR_SIGN` into the bridge, then `ros2 run sauvc_ardusub_demo ardusub_mission`.

---

## 11. Argument-passing templates

```bash
# Launch file arguments — space-separated name:=value AFTER the launch file
ros2 launch <package> <launch_file>.py <arg>:=<value> <arg2>:=<value2>

# Node parameters — after --ros-args, each with its own -p
ros2 run <package> <node> --ros-args -p <param>:=<value> -p <param2>:=<value2>

# Standalone scripts (ros2 run, plain CLI args — NOT --ros-args)
ros2 run sauvc_stonefish randomize_arena.py --seed 42 --in-place

# Root-level scripts (plain python, plain CLI args)
python3 sim_check.py --secs 20

# Big parameter sets — a YAML file instead of many -p flags
ros2 run <package> <node> --ros-args --params-file src/sauvc_bringup/config/flow.yaml
```

**Every launch argument in this workspace:**

| Launch file | Argument | Values | Default | Effect |
|---|---|---|---|---|
| `sauvc_finals_random.launch.py` | `seed` | any integer | `0` | Arena layout seed. Same seed ⇒ identical arena. |
| | `vehicle` | `colored` \| `grey` | `colored` | As above. |
| `sauvc_qualification.launch.py` | — | — | — | No arguments; fixed qualification arena. |
| `flow_eval.launch.py` (sauvc_flow_eval) | `compare_frame` | `ned` \| `enu` | `ned` | The one common frame every `/eval/*` track publishes in. |
| | `use_floor_profile` | bool | `true` | V-floor altitude profile vs flat pool (passed to the shim drivers + node). |
| | `alt_odom_topic` | topic | `/sauvc_auv/odometry` | Odometry feeding depth_shim's floor-profile x (ground truth — honest for diagnostics; the node also self-computes altitude, so this only affects the shim cross-check). |
| | `show_windows` / `show_camera` / `show_optical_flow` | bool | `true` | OpenCV windows: master / raw camera / flow overlay. |
| | `print_estimates` | bool | `true` | Terminal table of all five estimates. |
| `sauvc_bringup` phase launches | — | — | — | No CLI arguments by design: parameters are inline (with comments saying what to flip) or from `config/*.yaml`. Override per-run by running the underlying node standalone with `-p`, or edit the yaml. |

```bash
ros2 launch sauvc_stonefish sauvc_finals.launch.py 
ros2 launch sauvc_stonefish sauvc_finals_random.launch.py seed:=42 
```

**Every node parameter is listed in the per-package READMEs.** Quick index:

| Node | Package | Key parameters |
|---|---|---|
| `pressure_test` | sauvc_sensor_tests | `topic`, `p_ref` |
| `imu_test` | sauvc_sensor_tests | `topic` |
| `camera_test` | sauvc_sensor_tests | `front_topic`, `down_topic` |
| `pixhawk_imu_test` | sauvc_sensor_tests | *(CLI arg: MAVLink URL)* |
| `depth_pid_mission` | sauvc_motion_demo | `robot`, `target_depth`, `hold_time`, `leg_time`, `surge_cmd`, `sway_cmd`, `kp`, `ki`, `kd` |
| `ardusub_mission` | sauvc_ardusub_demo | *(CLI arg: MAVLink URL; tuning via file constants)* |
| `motor_map_check` | sauvc_ardusub_demo | `robot` |
| `ardusub_json_bridge.py` | sauvc_stonefish | `debug`; `MOTOR_MAP`/`MOTOR_SIGN` are file constants |
| `depth_altitude_node` | sauvc_drivers | `use_floor_profile`, `pool_depth`, `floor_profile_x/depth`, `fluid_density`, `i2c_bus`, `sensor_model`, `rate_hz`, `depth_var` |
| `pressure_check` | sauvc_sensor_check | `source` (i2c\|mavlink), `i2c_bus`, `sensor_model`, `fluid_density`, `mavlink_url`, `mavlink_baud`, `rate_hz` |
| `imu_taobotics_check` / `imu_pixhawk_check` | sauvc_sensor_check | `topic` |
| `camera_check_down` / `camera_check_front` | sauvc_sensor_check | `topic` |
| `flow_velocity_node` | sauvc_localization | `fx`,`fy`,`cx`,`cy`, `swap_xy`, `sign_x`, `sign_y`, `base_var`, `image_topic` |
| `lane_heading_node` | sauvc_localization | `pool_axis_offset`, `gain`, `min_lines` |
| `imu_covariance_check` | sauvc_localization | `patch`, `orientation_var_rpy`, `gyro_var_xyz`, `accel_var_xyz` |
| `preint_smoother_node` | sauvc_localization | `accel_sigma`, `gyro_sigma`, `keyframe_dt`, `reset_s` |
| `flow_eval_node` | sauvc_flow_eval | `compare_frame`, intrinsics, `self_altitude` + profile/mount params, `tilt_compensation`, `gtsam_keyframe_period`, `zupt*`, `r_flow_base`, `use_lane_heading*`, `use_clahe`, `feature_grid_*`, `show_*`, `print_*` (full table in the package README) |
| `gate_detector_node` | sauvc_vision | `hfov_deg`, `vfov_deg` |
| `mission_node` | sauvc_mission | `cruise_depth`, `gate_distance`, `cruise_speed`, `flare_order` |
| `csv_logger_node` | sauvc_logging | `topics`, `out_dir`, `run_name` |

---

## 12. Topic map (`robot_name = sauvc_auv`)

| Topic | Type | Direction | Meaning |
|---|---|---|---|
| `/sauvc_auv/thruster_setpoints` | `std_msgs/Float64MultiArray` | **in** | 8 values ∈ [−1, 1]: `HFP HFS HAP HAS VFP VFS VAP VAS` |
| `/sauvc_auv/thruster_state` | `stonefish_ros2` msg | out | Per-thruster setpoint / rpm / thrust |
| `/sauvc_auv/imu` | `sensor_msgs/Imu` | out | Body-frame specific force + rates |
| `/sauvc_auv/pressure` | `sensor_msgs/FluidPressure` | out | Absolute pressure → depth |
| `/sauvc_auv/camera_front`, `/sauvc_auv/camera_down` | `sensor_msgs/Image` (+`camera_info`) | out | 1280×720, 80° FOV |
| `/sauvc_auv/odometry` | `nav_msgs/Odometry` | out | **Ground truth** — debug only, never as a sensor |
| `/sauvc_auv/usbl` | `stonefish_ros2` msg | out | Range/bearing to the pinger drum |
| `/mavros/*` | various | both | The **Pixhawk's** view — only with SITL + bridge + MAVROS running |

There is no "Pixhawk topic" from Stonefish: the Pixhawk is emulated by ArduSub SITL, fed by the
bridge. Its IMU/attitude/depth arrive over MAVLink (or MAVROS), *in parallel* with the Stonefish
sensor topics — mirroring the real vehicle, where you read both your own sensors and the
autopilot.

---

## 13. Vehicle & physics reference

* **Frame:** 8 thrusters — 4 horizontal vectored at ±45°, 4 vertical. NED, +Z down.
* **Thrust model (verified against a headless Stonefish build):** `T = Kt · ω · |ω|`, ω in rad/s.
  **No density or diameter in the formula.** Shipped `Kt = 0.0005`, `max_setpoint = 314 rad/s`
  → ≈49 N per thruster (T200-class). For your thruster: `Kt = T_max / ω_max²`.
* **Left-handed propellers** (`right="false"`) produce **negative** thrust for a positive
  setpoint; every LH thruster carries `inverted_setpoint="true"` to compensate. Reaction torques
  still cancel across the RH/LH diagonal pairs.
* **Vertical thrusters:** positive setpoint = **down** (descend).
* **Physics mode:** `physics="floating"` (4 tagged switches in `my_auv.scn`). Accurate at the
  surface *and* underwater. `submerged` is only a speed shortcut for fully-submerged headless runs.
* **Mass/buoyancy:** ~24 kg, ≈ +1.2 kgf net, CG 3 cm below CB — set by the internal
  `FloatVolume`/`BallastVolume` boxes in `my_auv.scn`, not by the hull mesh.

**Plant parameters live in the scene file; controller gains live in the controller.** Keeping the
simulated plant physically truthful is what makes controllers transfer to the real vehicle.

---

## 14. Per-package documentation

* [`sauvc_stonefish/README.md`](src/sauvc_stonefish/README.md) — scenarios, launch files, the
  ArduSub bridge, mesh tools, thruster conventions, and the open-loop direction-debug procedure.
* [`sauvc_sensor_tests/README.md`](src/sauvc_sensor_tests/README.md) — the four sensor nodes.
* [`sauvc_motion_demo/README.md`](src/sauvc_motion_demo/README.md) — the depth-PID mission.
* [`sauvc_ardusub_demo/README.md`](src/sauvc_ardusub_demo/README.md) — the ArduSub mission and
  `motor_map_check`.
* [`sauvc_bringup/README.md`](src/sauvc_bringup/README.md) — the nine phase launch files, what
  each starts and logs, and the shared `cameras.yaml` / `flow.yaml` / `ekf.yaml` configs.
* [`sauvc_sensor_check/README.md`](src/sauvc_sensor_check/README.md) — the five pre-phase
  hardware checks (pressure i2c/mavlink, both IMUs, both cameras).
* [`sauvc_drivers/README.md`](src/sauvc_drivers/README.md) — `depth_altitude_node` (MS5837),
  surface-zeroing, the V-floor profile, and the measured-not-datasheet `depth_var` rule.
* [`sauvc_localization/README.md`](src/sauvc_localization/README.md) — `flow_velocity_node`,
  `lane_heading_node`, `imu_covariance_check`, `preint_smoother_node`, plus the offline
  `scripts/` (video flow test, covariance estimator).
* [`sauvc_flow_eval/README.md`](src/sauvc_flow_eval/README.md) — the five-estimator sim
  comparison, full run sequence, and the exhaustive parameter reference of the upgraded node.
* [`sauvc_vision/README.md`](src/sauvc_vision/README.md) — HSV prop detector, tuning workflow.
* [`sauvc_mission/README.md`](src/sauvc_mission/README.md) — the state machine, its topics,
  and the anisotropic `/pose_correction` design.
* [`sauvc_logging/README.md`](src/sauvc_logging/README.md) — automatic CSV logging, the topic
  registry, and the Ctrl-C statistics.

---

## 15. Sim health check — `sim_check.py` (workspace root, no build required)

One-shot, read-only diagnostic for the running Stonefish sim. Run it at the start of every
sim session, **before** trusting any localization numbers.

```bash
cd ~/Robotics_Job/sauvc_ws
python3 sim_check.py                 # 10 s measurement window
python3 sim_check.py --secs 20      # longer window = tighter rate estimates
```

| Argument | Type | Default | Effect |
|---|---|---|---|
| `--secs` | float | `10.0` | Measurement window in seconds. |

**What it checks, and what to observe per section:**

1. **REAL-TIME FACTOR** — the big one. Sensor `rate="…"` in `my_auv.scn` is *simulation*
   time, but `stonefish_ros2` stamps every message with the **wall clock**, so
   observed wall rate ÷ declared rate = RTF. At RTF ≠ 1 the flow node reports
   RTF × true velocity while gyro/depth are unscaled — the sim is **kinematically
   inconsistent**, not merely slow, and `flow_scorer` would report a "scale error" of
   exactly RTF for a perfectly working algorithm. Verdicts: `✓ RTF ≈ 1` proceed;
   `✗ RTF ≠ 1` fix before Phase 3 (drop cameras to 640×480 in `my_auv.scn` or lower
   `rendering_quality`); `⚠ topics DISAGREE` (spread > 0.15) = **per-sensor throttling**
   (usually GPU-bound cameras), a different problem with a different fix. A kinematic
   cross-check (|Δpos|/dt ÷ |twist|) corroborates whenever the vehicle is moving.
2. **`/clock`** — expected **absent** (wall-clock stamps), which is why `use_sim_time`
   must stay `false` everywhere. `⚠` if present.
3. **CAMERA INTRINSICS** — fx/fy/cx/cy from `camera_info`, ready to paste into
   `flow_sim.yaml`; flags nonzero distortion and fx ≠ fy as unexpected.
4. **PRESSURE** — value + variance (converted to the `depth_var` [m²] the EKF wants),
   the gauge-vs-absolute sign check, and — with the vehicle submerged > 5 cm — an
   **empirical ρg check** against ground-truth depth (accounting for the sensor sitting
   0.10 m above the body origin). At the surface it tells you to dive and re-run.
5. **WHAT IS MISSING FROM `my_auv.scn`** — DVL presence (`flow_scorer_node` needs it),
   IMU yaw σ (near-zero = sim yaw cannot drift → lane-heading tests pass vacuously),
   zero accel covariance (spec-illegal "perfectly known"), and a USBL note (sim-only
   sensor — never fuse it).

It separates timing artifacts from genuine algorithm errors before you spend an evening
debugging the wrong one.

---

## 16. The localization & hardware-pipeline packages

The eight packages added alongside the sim stack split into a **hardware pipeline**
(bring the real vehicle's localization up phase by phase) and a **sim evaluation loop**
(prove the estimators against Stonefish ground truth). Same rule as everywhere else in
this workspace: every node's full parameter table, run command, and what-to-observe
lives in its package README (§14); this section is the map.

| Package | Role | Nodes (executables) | Launch files |
|---|---|---|---|
| `sauvc_bringup` | Phase-by-phase hardware launches + shared configs | — (launch-only) | `pre_phase_sensor_check`, `cameras`, `phase1_depth` … `phase7_preint` |
| `sauvc_sensor_check` | Pre-phase "is the sensor alive at all", pipeline-independent | `pressure_check`, `imu_taobotics_check`, `imu_pixhawk_check`, `camera_check_down`, `camera_check_front` | — |
| `sauvc_drivers` | Hardware sensor drivers | `depth_altitude_node` (MS5837 Bar30/Bar02) | — |
| `sauvc_localization` | Dead-reckoning localization | `flow_velocity_node`, `lane_heading_node`, `imu_covariance_check`, `preint_smoother_node` (+ offline `scripts/`) | — |
| `sauvc_flow_eval` | Sim-only estimator shoot-out vs ground truth | `flow_eval_node` | `flow_eval.launch.py` |
| `sauvc_vision` | Prop detection, forward camera | `gate_detector_node` | — |
| `sauvc_mission` | Competition state machine (skeleton) | `mission_node` | — |
| `sauvc_logging` | Automatic per-phase CSV logging | `csv_logger_node` | — |

**Hardware pipeline (phases, in order).** Pre-phase sensor checks (`sauvc_sensor_check`,
one at a time, then the combined `pre_phase_sensor_check.launch.py`) →
`phase1_depth` (pressure/altitude + measured noise) → `phase2_heading` (IMU/Madgwick) →
`phase3_flow` (optical-flow velocity) → `phase4_lane_heading` →
`phase5_ekf` (robot_localization fusion) → `phase6_full` (vision + mission) →
`phase7_preint` (optional GTSAM A/B). Every phase launch auto-includes `csv_logger_node`
with a phase-appropriate topic list and `run_name`, so each pool session leaves a CSV
trail and a Ctrl-C mean/std summary — the numbers the phase procedures ask for.

```bash
ros2 launch sauvc_bringup phase1_depth.launch.py        # …through phase7_preint
```

**Sim evaluation loop.** Stonefish scenario (§6) → control/teleop → the eval:

```bash
ros2 launch sauvc_flow_eval flow_eval.launch.py compare_frame:=ned
# headless variant of the node alone:
ros2 run sauvc_flow_eval flow_eval_node --ros-args -p show_windows:=false
```

Run `sim_check.py` (§15) first; then observe the five-row `/eval/*` terminal table —
flow/ekf/gtsam tracking ground truth within slowly-growing drift, `pressure` x/y pinned
by design (depth-only baseline) while its z tracks the dive. Full run sequence,
PlotJuggler layout, and the exhaustive `flow_eval_node` parameter reference:
`sauvc_flow_eval/README.md`.

**Shared-core note.** `sauvc_localization/flow_core.py` is the ROS-free optical-flow
math used by BOTH the hardware `flow_velocity_node` and the sim `flow_eval_node` —
fixes and upgrades land once and serve both sides, which is the whole point of keeping
the plant/algorithm split honest (§13's closing rule, applied to software).
