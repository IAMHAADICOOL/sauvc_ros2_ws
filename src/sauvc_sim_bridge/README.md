# sauvc_sim_bridge

Makes the Stonefish simulation indistinguishable from the real vehicle's drivers, so
`sauvc_localization`, `sauvc_vision` and `sauvc_mission` run **byte-identical** in sim
and on hardware.

## The architecture, and why it isn't a fork

"Translate each package" has two readings. Forking `sauvc_localization` into
`sauvc_localization_sim` would give you two copies of `flow_core.py` drifting apart, and
would defeat the purpose: you'd never learn whether a sim result transfers, because the
sim would be running different code.

Instead the sim/real boundary sits at the **plant**, exactly where you already put
`thrust_coeff`. One shim package converts Stonefish's topics into the topics your drivers
publish. Everything above that line is untouched.

| Real package | Status in sim |
|---|---|
| `sauvc_drivers` | **Replaced** by `imu_shim_node` + `depth_shim_node` |
| `sauvc_localization` | **Unchanged code.** New config: `config/flow_sim.yaml` |
| `sauvc_vision` | **Unchanged code.** Colour thresholds need retuning vs the renderer |
| `sauvc_mission` | **Unchanged code** — drive through ArduSub SITL, same MAVLink as hardware |
| `sauvc_logging` | **Unchanged** |
| `sauvc_sensor_check` | Superseded by the existing `sauvc_sensor_tests` |
| `sauvc_bringup` | **New launch tree** here (`sim_*.launch.py`) |

## The two facts that drove every design decision

Both were read out of `stonefish_ros2/src/stonefish_ros2/ROS2Interface.cpp` in your own
workspace, not from the docs — and one of them differs from the ROS1 package, so do not
reason from `stonefish_ros`'s source.

**1. Stonefish is NED/FRD. Nothing converts it.** `PublishOdometry` hard-codes
`frame_id = "world_ned"`. Your stack is REP-103 ENU/FLU. All conversion happens in
`frames.py` and nowhere else. A sign flip anywhere else in the workspace is a bug.

Two things `frames.py`'s test suite pinned down that are easy to get wrong by hand:

- **Roll is sign-invariant; pitch and yaw are not.** `D·Rx(θ)·D = Rx(+θ)` for
  `D = diag(1,−1,−1)` — negating both y and z preserves rotations about x. FRD roll +30
  (right side down) and FLU roll +30 (left side up) are the same physical attitude. The
  natural guess — flip all three — is wrong.
- **A diagonal covariance cannot reveal a sign error.** Under `Σ' = RΣRᵀ` with signed
  axis permutations the signs square away, so `cov3_frd_to_flu` leaves a diagonal matrix
  *completely untouched*. A covariance sign bug won't blow up; it'll silently attach to
  the wrong axis. Off-diagonals do flip, hence full conjugation rather than hand-permuting.

**2. Every message is stamped with the WALL CLOCK at publish time.** `header.stamp =
nh_->get_clock()->now()`, in all 20 publishers; `s.getTimestamp()` is used **zero** times.
There is no `/clock`. Therefore:

- `use_sim_time` must be **false** everywhere. Setting it true starves every node.
- **Optical-flow velocity is scaled by the real-time factor.** `flow_velocity_node`
  computes `dt` from image stamps (wall clock), but the pixels come from physics that
  advanced in *sim* time. At real-time factor `R`, flow reports `R × v_true`. Depth,
  gyro rates and the DVL are **not** scaled. So at `R ≠ 1` the sim isn't just slow — it's
  **kinematically inconsistent**, and the EKF fuses true angular rates against
  R-scaled linear velocity.

This is why `rtf_monitor_node` is in `sim_drivers.launch.py` and is not optional, and why
`flow_scorer_node` **refuses to report a scale** unless `R ≈ 1`. With two 1280×720
cameras at 30 Hz plus underwater rendering, `R < 1` is the default expectation, not an
edge case.

**Check it first, before anything else:**
```bash
ros2 topic hz /sauvc_auv/imu        # scene says rate="100.0" → expect ~100
ros2 topic hz /sauvc_auv/odometry   # scene says rate="30.0"  → expect ~30
```
Anything materially below the declared rate *is* your RTF. Fix it by dropping camera
resolution/rate in `my_auv.scn` or `rendering_quality` in the launch file.

## Plant parameters that MUST differ from the real robot

These are the values the shim inverts the simulator with. Getting them from the real
robot's config is silently wrong — same argument as `thrust_coeff`, and it bit me once
already.

| parameter | sim | real | source of truth |
|---|---|---|---|
| `fluid_density` | **1000.0** | 997.0 | `<water density="1000.0"/>` in the scenario |
| `gravity` | **9.81** | 9.80665 | Stonefish default; IMU reads `az = -9.809994` |
| `depth_var` | **4.157e-06** | 4.0e-04 | derived from `FluidPressure.variance` (= 20², the scene's `<noise pressure="20.0"/>`) |
| `zero_secs` | **0.0** | 3.0 | Stonefish publishes gauge pressure (see below) |

Inverting with the real pool's 997.0/9.80665 gives a **+0.335%** depth scale error —
+5.0 mm at 1.5 m, 2.5σ against a 2.04 mm sensor. Constant, invisible, and it propagates
depth → altitude → flow scale, i.e. straight into the number `flow_scorer_node` exists to
measure. (The scene is itself slightly unphysical — fresh water at 26 °C is ~996.8 kg/m³,
not 1000.0 — but Stonefish uses the *declared* number, so the declared number is what must
be inverted. Match the scene, not physics.)

**Surface-zeroing is off by default, and that IS parity.** `fluid_pressure: -16.72` is
negative: Stonefish publishes gauge pressure referenced to the free surface, with no
atmospheric term and no clamp above the waterline. The real zeroing exists to remove the
atmospheric-constant mismatch and the sensor's calibration offset — *neither exists here*.
Parity means matching the post-zero semantics ("metres below the free surface"), not the
pre-zero mechanism.

Zeroing wouldn't *break* anything as things stand — you launch this node against an
already-running sim, where the vehicle has settled and a window would capture ≈0. The
settle-detector is insurance for one specific future: a combined launch file that starts
Stonefish and the shim together. The scenarios spawn 0.3 m deep (`start_position="... 0.3"`)
and the vehicle floats up, so a naive window from t=0 *would* average the ascent and bake in
~11 cm, silently.

## Topic names, verified against the parser

`stonefish_ros2` publishes a ColorCamera on **`<topic>/image_color`** plus
`<topic>/camera_info` — verified at `ROS2ScenarioParser.cpp:859`. There is **no
`/image_raw`** for a ColorCamera (only DepthCamera and Thermal publish that), and nothing
publishes on the bare `<topic>` name. Encoding is `rgb8`; `flow_velocity_node` asks
cv_bridge for `mono8`, which converts cleanly.

## What the sim cannot test — read before trusting a green result

- **No yaw drift** unless you add `<noise ... yaw_drift="..."/>` to the scene
  (see `SCENE_CHANGES.md`). Until then `lane_heading_node` has nothing to correct and
  every test of it passes vacuously.
- **No lens distortion, no flat-port refraction.** `GenerateCameraMsgPrototypes` does
  `D.resize(5, 0.0)` and derives `fy` from `fx`, so `fx == fy` by construction. On
  hardware, water behind a flat port shifts the effective focal length by ~1.33× and an
  in-air calibration gives ~33% velocity scale error — the single most likely thing to
  bite you in the pool. **That failure mode does not exist here.** The sim validates the
  flow *algorithm*; it cannot validate your *calibration*. A clean sim score is not an
  argument for skipping Phase 0.
- **No IMU bias walk or init transient.** Orientation is ground truth + white noise.

## Usage

```bash
# Terminal 1 — simulator
ros2 launch sauvc_stonefish sauvc_qualification.launch.py

# Terminal 2 — Phase 3: flow, scored against the DVL
ros2 launch sauvc_sim_bridge sim_phase3_flow.launch.py

# Just the shim layer (drop-in for phase1_depth + cameras)
ros2 launch sauvc_sim_bridge sim_drivers.launch.py use_floor_profile:=true
```

`use_floor_profile` must match the scene: `sauvc_pool.scn` is flat, the competition
arenas are V-shaped.

**Launch the depth shim with the vehicle at the surface** — it surface-zeroes over the
first 3 s exactly like the Bar30 driver, which also sidesteps whether Stonefish's
pressure includes the atmospheric offset (whatever the constant is, zeroing eats it).

## Nodes

| Node | Role |
|---|---|
| `imu_shim_node` | `/sauvc_auv/imu` (NED/FRD) → `/imu/data` (ENU/FLU). Warns if the sim AHRS is too perfect to test `lane_heading_node`. Rewrites zero accel covariance to `-1`. |
| `depth_shim_node` | `/sauvc_auv/pressure` → `/depth` + `/altitude`. Surface-zeroes, median-of-5, floor profile — same logic as the real driver. **Derives `depth_var` from the message** rather than inheriting the Bar30's placeholder. |
| `rtf_monitor_node` | Measures the real-time factor two ways and shouts if it isn't 1. Publishes `/sim/rtf`. |
| `flow_scorer_node` | Grades `/flow/twist` against the sim DVL: scale, bias, RMSE, correlation, dropout. Suppresses the scale when RTF is off. |

## Why `floor_profile.py` moved

The V-profile interpolation used to live inside `depth_altitude_node.py`, next to the
MS5837 I2C reads. The shape of the pool floor is a property of the **environment**, not
of the pressure sensor. Leaving it there meant the sim — which has no MS5837 and never
will — would have to copy-paste it, where it would drift out of sync. It now lives in
`sauvc_localization/floor_profile.py`, pure and unit-testable like `flow_core.py`, and
both the real driver and the shim import it. `pool_depth` describes the pool; `i2c_bus`
describes the sensor; they no longer share a class.

## Tests

```bash
python3 -m pytest test/test_frames.py -v     # or: PYTHONPATH=. python3 test/test_frames.py
```
Known-answer cases plus a 2000-sample structural check that converting-then-rotating
equals rotating-then-converting.
