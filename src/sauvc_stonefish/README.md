# `sauvc_stonefish`

The simulation package: **arena + vehicle scenarios**, **launch files**, the **ArduSub bridge**,
and the **mesh/arena generation tools**. Everything else in the workspace talks to what this
package launches.

```
sauvc_stonefish/
├── launch/      sauvc_finals.launch.py, sauvc_finals_random.launch.py, sauvc_qualification.launch.py
├── scenarios/   sauvc_finals.scn, sauvc_qualification.scn, sauvc_pool.scn,
│                my_auv.scn, my_auv_grey.scn, vehicle_template.scn
├── scripts/     ardusub_json_bridge.py, motor/mesh/arena tools (see §4)
└── data/        meshes + textures (drum, cup, propeller, pool tiles, vehicle meshes)
```

---

## 1. Scenarios

| File | Role |
|---|---|
| `sauvc_finals.scn` | **Top-level finals arena.** Includes `sauvc_pool.scn` + `my_auv.scn`, places the orange flare, R/Y/B flares with golf balls, gate, 4 drums, start frame. |
| `sauvc_qualification.scn` | Top-level qualification arena (gate only). |
| `sauvc_pool.scn` | **Shared include:** pool geometry, water, materials, and **all `<look>` definitions** (arena + the seven `auv_*` vehicle colors). |
| `my_auv.scn` | **The vehicle**, CAD-colored. Thrusters, sensors, buoyancy volumes, physics mode. |
| `my_auv_grey.scn` | Identical vehicle, uniform grey looks. Same meshes, same physics. |
| `vehicle_template.scn` | Minimal 6-thruster example to build a new vehicle from. |

**Arena geometry:** 25 × 16 m pool, NED, origin at pool centre, surface `z = 0`, start wall at
`x = −12.5`. V-shaped floor: `d(x) = 1.6 − 0.032·|x|` (1.2 m at the walls, 1.6 m at the centre).
Floor carries a procedural blue mosaic tile texture for downward-camera odometry.

**Editing the vehicle** — the blocks you'll actually touch in `my_auv.scn`:

| Block | Controls |
|---|---|
| `<thrust_model><thrust_coeff>` | `Kt` in `T = Kt·ω·|ω|` (ω in rad/s). `Kt = T_max / ω_max²`. |
| `<specs max_setpoint inverted_setpoint>` | Max ω [rad/s]; per-thruster reversal. |
| `<propeller right="true|false">` | Propeller handedness (torque sign + spin direction). |
| `<origin xyz rpy>` on each thruster | Thruster position and thrust axis (**+X local**). |
| `FloatVolume` / `BallastVolume` | Mass, buoyancy, CG–CB separation. |
| `physics="floating"` (4 × `PHYSICS_MODE` tags) | Floating vs submerged approximation. |
| `<sensor>` blocks | Rates, noise, resolution, FOV. |

---

## 2. Launch files

### `sauvc_finals.launch.py` — fixed arena

```bash
ros2 launch sauvc_stonefish sauvc_finals.launch.py [vehicle:=colored|grey]
```

| Argument | Values | Default | Effect |
|---|---|---|---|
| `vehicle` | `colored` \| `grey` | `colored` | Selects `my_auv.scn` or `my_auv_grey.scn`. Physics/sensors/thrusters identical; looks only. |

**Observe:** window opens → `Loading scenario...`, `Including ... sauvc_pool.scn`, `... my_auv.scn`
→ mesh loads (the 921k-face black mesh takes ~1–5 s; the "Loaded mesh" line prints *after*)
→ pool + vehicle floating in the start box. Fixed prop positions every run.

### `sauvc_finals_random.launch.py` — randomized arena

```bash
ros2 launch sauvc_stonefish sauvc_finals_random.launch.py seed:=42 [vehicle:=colored|grey]
```

| Argument | Values | Default | Effect |
|---|---|---|---|
| `seed` | any integer | `0` | Layout seed. **Same seed ⇒ byte-identical arena.** `seed:=0` is a draw, not "off". |
| `vehicle` | `colored` \| `grey` | `colored` | As above. |

**Observe:** before the sim starts, a line like
`[randomize_arena] {'seed': 42, 'orange_flare': (-5.94, -6.65), 'blue_flare': (-2.12, -3.88), ..., 'gate_center': (4.4, -0.94)}`
— log it with your results. The generated scene goes to `/tmp/sauvc_finals_seed<N>.scn`; the
installed scenario is untouched.

### `sauvc_qualification.launch.py` — qualification arena

```bash
ros2 launch sauvc_stonefish sauvc_qualification.launch.py
```
No arguments. Gate-only arena.

---

## 3. `ardusub_json_bridge.py` — the ArduSub ⇄ Stonefish bridge

```bash
ros2 run sauvc_stonefish ardusub_json_bridge.py [--ros-args -p debug:=true|false]
```

Connects the simulator to ArduSub SITL's JSON physics backend:

```
Stonefish /odometry + /imu  ──► JSON state ──► ArduSub SITL (UDP :9002)
ArduSub SITL servo PWM      ──► [-1,1] setpoints ──► /sauvc_auv/thruster_setpoints
```

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `debug` | bool | `true` | 1 Hz `pwm → sp | state` lines + thruster active/neutral transitions. |

**File constants (edit the top of the script):**

| Constant | Shipped | Meaning |
|---|---|---|
| `MOTOR_MAP` | `[1,0,3,2,5,4,7,6]` | ArduSub servo *i* → Stonefish thruster index. Fix **placement** errors here. |
| `MOTOR_SIGN` | `[-1,-1,1,1,1,1,1,1]` | Per-servo thrust sign. Fix **direction** errors here. Derived empirically: the two front horizontals are reversed vs ArduSub's frame. |
| `PWM_VALID_MIN/MAX` | `800 / 2200` | PWM outside this window → neutral. ArduPilot emits `0` while disarmed; without this guard every thruster went full reverse before arming. |
| `ROBOT` | `sauvc_auv` | Must match `robot_name` in the scenario. |
| `SITL_ADDR` | `127.0.0.1:9002` | ArduPilot JSON backend endpoint. |

**Observe (this log is the main diagnostic in the whole workspace):**

```
pwm[1-8]=[0,0,0,0,0,0,0,0] -> sp=[0.0,...] | state: depth=0.08 rpy=(0.0, 0.0, -12.2)   ← disarmed: zeros→neutral, truth steady
pwm[1-8]=[1500 ×8]        -> sp=[0.0,...] | state: depth=0.08 ...                      ← armed neutral, 8-motor frame OK
pwm[1-8]=[1500 ×6, 0, 0]  ...                                                          ← WRONG FRAME → see root README §9
```

* `state:` is **ground truth** from Stonefish. Compare it with the EKF depth/attitude printed by
  `ardusub_mission`: **agree** ⇒ any misbehaviour is control/mapping; **diverge** ⇒ state-feed bug.
* Conventions in the state feed are source-verified: IMU passed through unchanged (Stonefish
  already outputs specific force in body NED); odometry twist is **body-frame**, so it is rotated
  to world before being sent as `velocity`; timestamps are wall-clock (odometry-stamp bumping made
  SITL's clock crawl and the EKF drift).

---

## 4. Tools in `scripts/`

| Script | Run | Purpose |
|---|---|---|
| `randomize_arena.py` | `ros2 run sauvc_stonefish randomize_arena.py --seed 42 [-i in.scn] [-o out.scn] [--in-place]` | Writes a randomized copy of the finals arena. Prints the layout dict. Used internally by the random launch. |
| `ardusub_move_example.py` | `ros2 run sauvc_stonefish ardusub_move_example.py` | Minimal pymavlink demo: arm in MANUAL, dive, forward, yaw, disarm. |
| `convert_vehicle_mesh.py` | `python3 scripts/convert_vehicle_mesh.py <cad.obj> data/` | CAD → `my_auv_vis.obj` (single grey visual) + `my_auv_phy.obj` (convex physics hull). Culls hardware <1.5 cm, **no decimation**, double-sided, flat-shaded, expanded vertices. |
| `generate_colored_meshes.py` | `python3 scripts/generate_colored_meshes.py <cad.obj> data/` | Splits the CAD by `usemtl` into `my_auv_vis_<color>.obj` groups + prints the `<look>` mapping. |
| `generate_meshes.py` | `python3 scripts/generate_meshes.py` | Generates arena props: `drum.obj`, `cup.obj`, `propeller.obj`, `pool_tiles.png`. |

**Mesh-pipeline rules learned the hard way** (all baked into the scripts):
decimation tears sliver holes in thin plates → **off**; smoothed vertex normals make CAD look
like a cushion → **flat per-face normals**; per-face normals on shared vertices blow up
Stonefish's loader dedup (RAM spike → load stall) → **pre-expanded vertices**; CAD normals are
inconsistent → **double-side everything**.

---

## 5. Thruster conventions & the direction-debug procedure

**Order everywhere:** `[0] HFP  [1] HFS  [2] HAP  [3] HAS  [4] VFP  [5] VFS  [6] VAP  [7] VAS`

* Thrust acts along each actuator's **local +X**, set by its `<origin rpy>`.
* Horizontals: HFP/HAS yaw +45° (thrust fwd+stbd), HFS/HAP yaw −45° (fwd+port).
* Verticals: pitch −90° ⇒ **positive setpoint = descend**.
* `right="false"` (LH prop) ⇒ negative thrust ⇒ compensated by `inverted_setpoint="true"`.

**Run these four tests before any closed-loop work.** A reversed thruster makes every controller
fight itself, and this takes two minutes:

```bash
ros2 topic pub -r 10 /sauvc_auv/thruster_setpoints std_msgs/msg/Float64MultiArray "{data: [0.3,0.3,0.3,0.3,0,0,0,0]}"
```

| Test | `data:` | Expected — and nothing else |
|---|---|---|
| Surge fwd | `[.3,.3,.3,.3,0,0,0,0]` | forward; no yaw, no sway |
| Sway stbd | `[.3,-.3,-.3,.3,0,0,0,0]` | right strafe; no yaw |
| Yaw right | `[.3,-.3,.3,-.3,0,0,0,0]` | clockwise from above |
| Descend | `[0,0,0,0,.3,.3,.3,.3]` | straight down, level |

**Reading the failures:**

| Symptom | Fix (in `my_auv.scn`) |
|---|---|
| A whole group goes the wrong way (e.g. "descend" rises) | Set `inverted_setpoint="true"` on all 4 thrusters of that group. |
| Motion contaminated (surge also yaws; descend also rolls) | One thruster reversed. Isolate with single-index commands (`[.3,0,0,...]`, `[0,.3,0,...]`, …) — e.g. HFP alone should translate fwd-right *and* yaw the vehicle **left** — then flip `inverted_setpoint` on the culprit only. |
| Slow parasitic spin with all thrusters equal | Handedness pairing wrong: swap a `right=` value so diagonals alternate. |

**Do ArduSub-side corrections in the bridge** (`MOTOR_MAP`/`MOTOR_SIGN`), *not* in the scenario —
keep the simulated plant physically truthful, exactly like motor-direction setup on the real vehicle.
