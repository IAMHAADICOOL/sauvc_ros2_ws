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
└── sauvc_ardusub_demo/  same mission via ArduSub SITL + motor-map discovery tool
```

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
```

**Every launch argument in this workspace:**

| Launch file | Argument | Values | Default | Effect |
|---|---|---|---|---|
| `sauvc_finals.launch.py` | `vehicle` | `colored` \| `grey` | `colored` | `colored` = CAD-colored meshes; `grey` = uniform grey (`my_auv_grey.scn`). Physics identical. |
| `sauvc_finals_random.launch.py` | `seed` | any integer | `0` | Arena layout seed. Same seed ⇒ identical arena. |
| | `vehicle` | `colored` \| `grey` | `colored` | As above. |
| `sauvc_qualification.launch.py` | — | — | — | No arguments; fixed qualification arena. |

```bash
ros2 launch sauvc_stonefish sauvc_finals.launch.py vehicle:=grey
ros2 launch sauvc_stonefish sauvc_finals_random.launch.py seed:=42 vehicle:=colored
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
