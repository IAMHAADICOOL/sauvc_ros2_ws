# sauvc_stonefish

A replica of the **SAUVC 2026 arena** (finals + qualification) for the [Stonefish](https://stonefish.readthedocs.io) marine robotics simulator, packaged for **ROS2 Jazzy** via [stonefish_ros2](https://github.com/patrykcieslak/stonefish_ros2).

## What's inside

```
sauvc_stonefish/
├── scenarios/
│   ├── sauvc_finals.scn         # main 25x16x2 m pool with all 4 task props
│   ├── sauvc_qualification.scn  # qualification arena (gate hanging from surface)
│   ├── sauvc_pool.scn           # shared pool shell + starting zone (included by both)
│   └── vehicle_template.scn     # 6-thruster template AUV (replace with yours)
├── data/
│   ├── drum.obj                 # hollow 60 cm drum (open top, CLOSED bottom)
│   ├── cup.obj                  # golf-ball cup on flare tops
│   ├── propeller.obj            # placeholder propeller
│   └── pool_tiles.png           # procedural blue mosaic floor texture
├── launch/
│   ├── sauvc_finals.launch.py
│   └── sauvc_qualification.launch.py
└── scripts/generate_meshes.py   # regenerates the OBJ files
```

### About the "pool of water"

Stonefish **cannot** simulate a bounded water volume — enabling the `<ocean>` creates an
infinite flat ocean. The standard workaround, used here, is to enable a calm, clear ocean
(waves = 0, jerlov = 0.05, fresh water 1000 kg/m³) and build the **pool floor and walls as
static bodies inside it**. The vehicle is physically confined to the pool; the water beyond
the walls is unreachable. Underwater rendering, buoyancy and hydrodynamics all behave exactly
as inside a real pool.

### Frames & layout

NED convention: **Z points DOWN**, water surface at `z = 0`, origin at pool center.
Pool interior: `x ∈ [-12.5, 12.5]` (starting wall at -12.5), `y ∈ [-8, 8]`.

**Sloped floor** (per the official side view): 1.2 m deep at both end walls, 1.6 m at
the pool center — a V-shaped floor built from two inclined slabs. Floor depth:

```
d(x) = 1.6 - 0.032 * |x|       (slope ≈ 1.83°)
```

Use this when placing anything on the floor. If the slope renders inverted on your
Stonefish version, flip the pitch sign in `sauvc_pool.scn`.

Zones per the official top view (positions inside zones are randomized per attempt —
these are one valid draw; edit the `<world_transform>` values to re-randomize):

| Prop | Zone (official) | Placed at | Notes |
|---|---|---|---|
| Starting zone 1.4×1.4 m | near starting wall | (-11.6, -2.0), surface | white floating frame |
| Orange flare Ø15 cm | 4–8 m from starting wall → x∈[-8.5,-4.5], full width | (-6.5, -1.0) | full depth — AVOID |
| Colored flares R/Y/B, 80 cm | band between orange zone and gate line → x∈[-4.5,4.4] | B(-1.0,3.0) R(1.5,4.5) Y(0.0,-4.0) | golf ball in cup on top (dynamic) |
| Navigation gate 150×100 cm | anywhere along the line ~16 m from start → x=4.4 | (4.4, -1.5) | red = port, green = starboard |
| Drums Ø60×30 cm ×4 | target zone: last ~2 m → x∈[10.5,12.5] | x=11.4, y = 4.5(blue)/1.5(red+pinger)/-1.5/-4.5 | blue topmost per figure |
| Qual. gate (qual scenario) | 10 m from starting line | (-2.5, ±0.75), surface→floor | orange posts |

The floor uses a **blue mosaic tile texture** (`data/pool_tiles.png`, tiled by face
dimensions via `uv_mode="2"`) so a down-looking camera sees a repeating grid — useful
for visual odometry. Regenerate with a different `seed`/`tiles` count in
`scripts/generate_meshes.py` if you want a different pattern density. A **green carpet**
(thin slab matched to the floor slope) lies under the four drums in the finals arena,
as in the real target zone.

Known simplifications: gate posts are solid red/green instead of striped; flares release
the golf ball on physical contact (bump the flare/cup and the ball drops and sinks).
The golf balls spawn a few mm above their cups and settle in during the first second.

### Simulated pinger (Task 2)

Stonefish does not simulate raw 45 kHz acoustics/TDOA hydrophone arrays, but it DOES
simulate acoustic devices with spherical propagation at the speed of sound and optional
line-of-sight occlusion. So the pinger is approximated the way it is used:

- an **`acoustic_modem` (device_id 1) fixed to the world frame inside the pinger drum**
  (`sauvc_finals.scn`), and
- a **`usbl` (device_id 2) on the vehicle** (`vehicle_template.scn`), auto-pinging at
  1 Hz and publishing **range + bearing to the drum** on `/sauvc_auv/usbl`.

That is functionally what your real hydrophone array estimates from the ULB-362B. Noise
std-devs on range/angles are set in the `<noise>` tag — widen them to make it realistic
for a noisy pool. Move the modem's `<world_transform>` together with the drum when you
re-randomize the pinger location. (The qualification scenario includes a "TestBeacon"
modem at the gate purely so the USBL has a peer; delete it if unwanted.)

---

## 1. Installation (Ubuntu 24.04 + ROS2 Jazzy)

### 1.1 Install the Stonefish library

Requires an OpenGL 4.3+ capable GPU.

```bash
sudo apt update
sudo apt install libglm-dev libsdl2-dev libfreetype6-dev build-essential cmake git

cd ~
git clone https://github.com/patrykcieslak/stonefish.git
cd stonefish
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

> The Stonefish library version **must match** the stonefish_ros2 version (e.g. both 1.6).

### 1.2 Build the ROS2 workspace

```bash
mkdir -p ~/sauvc_ws/src && cd ~/sauvc_ws/src
git clone https://github.com/patrykcieslak/stonefish_ros2.git
# copy/clone THIS package (sauvc_stonefish) into src/ as well
cd ~/sauvc_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 2. Running the arena

```bash
source ~/sauvc_ws/install/setup.bash
ros2 launch sauvc_stonefish sauvc_finals.launch.py          # finals arena
ros2 launch sauvc_stonefish sauvc_qualification.launch.py   # qualification arena
```

A Stonefish window opens with the pool, props and the template AUV in the starting zone.
Useful GUI keys: drag to orbit the camera; the side panel lets you toggle displays
(hydrodynamics, sensors) and pause/step the simulation.

### Quick sanity checks

```bash
ros2 topic list
ros2 topic echo /sauvc_auv/odometry
ros2 run rqt_image_view rqt_image_view   # view /sauvc_auv/camera_front
```

Drive it open-loop (6 thrusters, normalized setpoints in [-1, 1] — order = definition
order in `vehicle_template.scn`: FP, FS, AP, AS, HeaveP, HeaveS):

```bash
# gentle forward surge
ros2 topic pub -r 10 /sauvc_auv/thruster_setpoints std_msgs/msg/Float64MultiArray \
  "{data: [0.3, 0.3, 0.3, 0.3, 0.0, 0.0]}"

# dive
ros2 topic pub -r 10 /sauvc_auv/thruster_setpoints std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.4, 0.4]}"
```

Headless (no GPU / CI): use `stonefish_simulator_nogpu.launch.py` from stonefish_ros2
instead — note vision sensors (cameras) are unavailable in that mode.

## 3. Importing your own vehicle

The template in `scenarios/vehicle_template.scn` is the pattern to follow. Steps:

1. **Prepare meshes.** Export your hull (and any appendages) from CAD, clean it in Blender:
   closed/watertight mesh, low triangle count, **NED orientation (Z down)**, origin at your
   chosen body origin. Export ASCII `.obj`. Keep a fine *visual* mesh and a coarse *physical*
   mesh if you want accurate hydrodynamics + fast collisions. See "Preparing geometry files"
   in the Stonefish docs.

2. **Create `my_vehicle.scn`** (copy the template). Key blocks:
   - `<base_link type="compound" physics="submerged">` with one `<external_part>` per hull
     piece (`type="model"` + your meshes, or primitives). `buoyant="true"` computes buoyancy
     from the actual geometry — pick a material density that makes the vehicle slightly
     positive (the template defines `Vehicle` at 980 kg/m³).
   - Internal parts (batteries, electronics) can be added as `internal_part` elements to
     place mass correctly and trim pitch/roll.
   - One `<actuator type="thruster">` per thruster: set `<origin>` (thrust acts along the
     actuator's local **X** axis), propeller diameter/handedness, and tune the
     `thrust_model` coefficient until max thrust matches your real thruster (e.g. a T200 at
     16 V ≈ 50 N).
   - Sensors (`imu`, `pressure`, `dvl`, `odometry`, `camera`, `fls`, multibeam, USBL...) each
     with a `<ros_publisher topic="..."/>`.
   - ROS wiring at robot level:
     `<ros_subscriber thrusters="/my_auv/thruster_setpoints"/>` — expects
     `std_msgs/Float64MultiArray`, one value per thruster in [-1, 1], in definition order.

3. **Swap it into the arena.** In `sauvc_finals.scn`, replace the include:

   ```xml
   <include file="$(find my_vehicle_pkg)/scenarios/my_vehicle.scn">
       <arg name="robot_name" value="my_auv"/>
       <arg name="start_position" value="-11.4 0.0 0.3"/>
       <arg name="start_yaw" value="0.0"/>
   </include>
   ```

   `$(find pkg)` works anywhere in the file thanks to the stonefish_ros2 parser, so your
   vehicle can live in its own package.

4. **Rebuild & run:** `colcon build --symlink-install && ros2 launch sauvc_stonefish sauvc_finals.launch.py`.
   Verify buoyancy/trim first (vehicle should float still or rise slowly with zero
   setpoints), then check `ros2 topic list` for all your sensor topics, then connect your
   control stack to the thruster setpoint topic.

If the parser rejects a tag, cross-check the exact syntax against the docs for your
installed version (stonefish.readthedocs.io + stonefish-ros2.readthedocs.io) — the thruster
and sensor definitions changed between 1.4 and 1.5+. This package targets **1.6**.

## Regenerating meshes

```bash
python3 scripts/generate_meshes.py
```


---

## Using your real vehicle (my_auv.scn)

`scenarios/my_auv.scn` is your CAD vehicle (AUV_CAD_5_v8), already wired into both
arenas. Generated meshes: `data/my_auv_vis.obj` (decimated CAD, ~130k faces, visual)
and `data/my_auv_phy.obj` (convex hull, collision + drag). Regenerate after CAD
changes with:

```bash
python3 scripts/convert_vehicle_mesh.py AUV_CAD_5_v8.obj data/
```

Thruster layout was measured from the CAD: 4 vertical at (±0.165, ±0.120) m and
4 vectored horizontal at (±0.287, ±0.242) m with 45° X-configuration. Setpoint topic:
`/sauvc_auv/thruster_setpoints` (`Float64MultiArray`, 8 values in [-1,1], order
HFP HFS HAP HAS VFP VFS VAP VAS — documented in the file).

**Mass & buoyancy:** the open frame means the hull mesh must not provide buoyancy
(its solid volume is ~86 L). Instead `my_auv.scn` uses two internal boxes —
`FloatVolume` (Foam) and `BallastVolume` (Ballast) — giving ~24 kg, slight positive
buoyancy, CG ~3 cm below CB. **Tune the box sizes and the `Foam`/`Ballast` densities
(in the arena file's `<materials>`) to your real mass and trim.**

**Cameras:** front + down cameras publish `/sauvc_auv/camera_front` and
`/sauvc_auv/camera_down`. All the parameters to match your real cameras (rate,
resolution, horizontal FoV, mount pose) are in clearly-commented blocks in
`my_auv.scn`. Stonefish renders an ideal pinhole (no lens distortion); rendering
fidelity also depends on `rendering_quality` in the launch file.

## ArduSub / Pixhawk in the loop

Straight answer first: classic **hardware-in-the-loop with a physical Pixhawk is not
practically supported by ArduPilot anymore** (the old HIL modes were removed). The
supported, standard way to get "everything working like the real world" — ArduSub
firmware logic, MAVROS, pymavlink, QGroundControl — is **ArduSub SITL**, which runs
the *identical* firmware code on your PC. If you later truly need real hardware in
the loop, look at ArduPilot's "SITL on hardware" option, but start with SITL.

The glue is ArduPilot's **JSON physics backend**: SITL sends 16 servo PWMs over UDP
and expects the external simulator (Stonefish) to reply with vehicle state.
`scripts/ardusub_json_bridge.py` implements this:

```
Stonefish odom+IMU ─► bridge ─► JSON state ─► ArduSub SITL (udp:9002)
Stonefish thrusters ◄─ bridge ◄─ servo PWM ◄─┘
MAVROS / pymavlink / QGC ◄──── MAVLink (udp:14550) ────► SITL
```

Steps:

```bash
# 1. ArduPilot (once)
git clone https://github.com/ArduPilot/ardupilot.git && cd ardupilot
git submodule update --init --recursive
Tools/environment_install/install-prereqs-ubuntu.sh -y

# 2. Run the sim + SITL + bridge (3 terminals)
ros2 launch sauvc_stonefish sauvc_finals.launch.py
Tools/autotest/sim_vehicle.py -v ArduSub -f json:127.0.0.1 --console --map
ros2 run sauvc_stonefish ardusub_json_bridge.py

# 3. MAVROS (ROS2) and/or pymavlink
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14550@
python3 -c "from pymavlink import mavutil; m=mavutil.mavlink_connection('udp:127.0.0.1:14550'); m.wait_heartbeat(); print('ArduSub alive')"
```

In SITL set `FRAME_CONFIG` to the vectored-6DOF (8-motor) frame to match this
vehicle. **Verify the motor mapping**: `MOTOR_MAP`/`MOTOR_SIGN` at the top of the
bridge map SERVO1..8 to the Stonefish thruster order — check against the ArduSub
motor layout documentation and flip as needed (test with QGC motor test sliders).
Depth-hold etc. work because SITL synthesizes its baro/EKF from the state the bridge
sends; your *Stonefish* pressure/IMU topics remain available in parallel for your
own autonomy nodes, exactly like reading the Pixhawk sensors via MAVROS in reality.


## ROS2 topic map (my_auv, robot_name=sauvc_auv)

| Topic | Type | Direction | What |
|---|---|---|---|
| `/sauvc_auv/thruster_setpoints` | `std_msgs/Float64MultiArray` | you -> sim | 8 values in [-1,1]: HFP HFS HAP HAS VFP VFS VAP VAS |
| `/sauvc_auv/thruster_state` | stonefish_ros2 msg | sim -> you | thruster feedback |
| `/sauvc_auv/imu` | `sensor_msgs/Imu` | sim -> you | body-frame gyro + accel |
| `/sauvc_auv/pressure` | `sensor_msgs/FluidPressure` | sim -> you | absolute pressure (depth) |
| `/sauvc_auv/camera_front` | `sensor_msgs/Image` (+camera_info) | sim -> you | forward camera |
| `/sauvc_auv/camera_down` | `sensor_msgs/Image` (+camera_info) | sim -> you | downward camera |
| `/sauvc_auv/odometry` | `nav_msgs/Odometry` | sim -> you | ground truth (debug only) |
| `/sauvc_auv/usbl` | stonefish_ros2 msg | sim -> you | range+bearing to pinger drum |

There is no "Pixhawk topic" from Stonefish itself: the Pixhawk is emulated by ArduSub
SITL. Its "sensor readings" (attitude, depth from its synthesized baro/EKF) and its
control come over MAVLink -> MAVROS topics (`/mavros/imu/data`, `/mavros/...`) once the
bridge is running. Your Stonefish sensor topics above remain available in parallel.

## Moving the vehicle via ArduSub (example)

`scripts/ardusub_move_example.py` - connects with pymavlink, arms in MANUAL, dives,
drives forward, yaws, disarms. Run it after the sim + SITL + bridge are up (see the
ArduSub section above). MANUAL_CONTROL semantics for Sub: x/y/r in -1000..1000,
z 0..1000 with 500 = neutral throttle.


## Thruster direction conventions & debugging (READ BEFORE TRUSTING MOTION)

How direction works in this package, per thruster in `my_auv.scn`:

- **Thrust direction** = the actuator's local **+X axis**, set by `<origin ... rpy>`.
  A POSITIVE setpoint pushes along +X. The rpy values implement the standard
  BlueROV2-Heavy-style vectored-6DOF geometry:
  - HFP yaw +45deg -> thrust (fwd, stbd);  HFS yaw -45deg -> (fwd, port)
  - HAP yaw -45deg -> (fwd, port);          HAS yaw +45deg -> (fwd, stbd)
  - V* pitch -90deg -> thrust +Z (DOWN in NED): positive setpoint = descend
- **`inverted_setpoint="false"`** — flip to `"true"` to reverse an individual
  thruster's response (equivalent to swapping motor wires on the real vehicle).
- **`right="true|false"`** — propeller handedness. IMPORTANT (from the Stonefish
  source): a left-handed prop produces NEGATIVE thrust for the same setpoint, so
  every `right="false"` thruster carries `inverted_setpoint="true"` to compensate.
  Net effect: positive setpoint = thrust along +X on ALL thrusters, and reaction
  torques still cancel across the RH/LH diagonal pairs. If you change a `right=`
  value, change that thruster's `inverted_setpoint` with it.
- **`thrust_coeff` (quadratic model)** — EMPIRICALLY VERIFIED against a headless
  Stonefish 1.6 build: the model is simply **T = Kt * w * |w|** with w the prop
  speed in rad/s — density and diameter are NOT in the formula. So
  Kt = T_max / w_max^2. Shipped: 0.0005 with max_setpoint 314 rad/s -> ~49 N max
  per thruster (T200@16V-class). For your thruster: put its bench max thrust and
  max speed into that formula.

### Open-loop verification (no ArduSub — do this FIRST)

Command one group at a time and check the motion (order: HFP HFS HAP HAS VFP VFS VAP VAS):

| Test | Command `data:` | Expected motion (no other motion) |
|---|---|---|
| Surge fwd | `[.3,.3,.3,.3,0,0,0,0]` | forward, no yaw, no sway |
| Sway stbd | `[.3,-.3,-.3,.3,0,0,0,0]` | right-strafe, no yaw |
| Yaw right | `[.3,-.3,.3,-.3,0,0,0,0]` | clockwise from top |
| Descend | `[0,0,0,0,.3,.3,.3,.3]` | straight down, level |

```bash
ros2 topic pub -r 10 /sauvc_auv/thruster_setpoints std_msgs/msg/Float64MultiArray "{data: [0.3,0.3,0.3,0.3,0,0,0,0]}"
```

Debugging rules (fix in `my_auv.scn`):
- **One test moves the OPPOSITE way as a whole** (e.g. all-descend rises): flip
  `inverted_setpoint` to `"true"` on that whole group of 4.
- **Motion is contaminated** (surge also yaws, descend also rolls): one thruster in
  the group is reversed. Isolate it by commanding one index at a time
  (`[.3,0,0,0,...]`, then `[0,.3,0,0,...]`, ...) and check against the geometry
  above (e.g. HFP alone: pushes fwd+stbd applied at front-port -> vehicle yaws LEFT
  while translating fwd-right). Flip `inverted_setpoint` on the culprit line:
  `<specs max_setpoint="314.0" inverted_setpoint="true" .../>` in that thruster's block.
- **Vehicle spins slowly with all thrusters equal**: handedness mismatch — swap a
  `right=` pair so diagonals alternate.

### With ArduSub

After open-loop passes, the remaining unknown is only the SERVO->thruster mapping.
Use QGroundControl's Motor Test sliders (or `motortest` in the SITL console): actuate
ArduSub motors 1..8 one by one, note which Stonefish thruster responds and in which
direction, then edit `MOTOR_MAP` (reorder) and `MOTOR_SIGN` (flip) at the top of
`scripts/ardusub_json_bridge.py`. Set `FRAME_CONFIG` to vectored-6DOF first. Do NOT
fix ArduSub-side reversals in my_auv.scn — keep the sim ground truth correct and do
the mapping in the bridge, exactly like Motor Direction setup on the real vehicle.


## Companion packages (drop into the same src/)

- **sauvc_sensor_tests** — sensor sanity nodes:
  `ros2 run sauvc_sensor_tests pressure_test` (pressure + depth in terminal),
  `ros2 run sauvc_sensor_tests imu_test` (RPY + gyro + accel in terminal),
  `ros2 run sauvc_sensor_tests camera_test` (two OpenCV windows: front + down;
  needs `ros-jazzy-cv-bridge`, `python3-opencv`),
  `ros2 run sauvc_sensor_tests pixhawk_imu_test` (the PIXHAWK's IMU via MAVLink:
  ArduSub EKF attitude + raw IMU; needs sim + SITL + bridge up, `pip install pymavlink`.
  MAVROS route: `ros2 run sauvc_sensor_tests imu_test --ros-args -p topic:=/mavros/imu/data`).
- **sauvc_motion_demo** — `ros2 run sauvc_motion_demo depth_pid_mission`:
  dives to `target_depth` with a depth PID on the vertical thrusters, holds,
  then forward / right / left / backward (PID still holding depth), then surfaces.
  All gains and timings are ROS params.
- **sauvc_ardusub_demo** — `ros2 run sauvc_ardusub_demo ardusub_mission`:
  the same mission through ArduSub: MANUAL dive, ALT_HOLD (the firmware holds
  depth using the baro synthesized from the simulated pressure via the JSON
  bridge), MANUAL_CONTROL x/y for the legs, surface, disarm. Requires the sim +
  SITL + bridge running first (see the ArduSub section). `pip install pymavlink`.

## Randomized arena (competition-realistic)

Per the rulebook top view: the orange flare randomizes in the 4-8 m band, the three
colored flares in the band between it and the gate line (>=1.5 m apart), and the
gate anywhere along its line. Floor heights are recomputed from the slope.

```bash
# one-off file:
python3 scripts/randomize_arena.py --seed 42            # -> scenarios/sauvc_finals_seed42.scn
# or generate + launch in one go (installed scenario is never modified):
ros2 launch sauvc_stonefish sauvc_finals_random.launch.py seed:=42
```

Same seed = byte-identical arena (reproducible across machines and launches);
change the seed for a new draw. The chosen layout is printed at launch.
