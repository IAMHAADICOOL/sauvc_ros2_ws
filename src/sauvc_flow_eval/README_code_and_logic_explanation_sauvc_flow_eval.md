# `sauvc_flow_eval` — the estimator comparison harness (flow vs EKF vs GTSAM vs pressure)

This package answers one question: **"of all the ways we could estimate where the vehicle is,
which one is best, and by how much?"** It runs several estimators off the **same live sensor
stream** at the same time, converts them all into **one common frame**, and lines them up
next to the simulator's **ground truth** so you can literally watch them drift on a plot.

It is a **dead-reckoning bake-off**. The estimators are, on `/eval/*`:

| Topic | What it is |
|---|---|
| `/eval/ground_truth` | Stonefish's true pose (the reference everyone is scored against) |
| `/eval/flow` | optical-flow body velocity, integrated to a position |
| `/eval/ekf` | a small self-contained EKF fusing flow + depth |
| `/eval/pressure` | depth only (x, y stay 0) — the "do nothing horizontal" baseline |
| `/eval/gtsam` | an iSAM2 factor graph: IMU preintegration + flow + depth |
| `/eval/dvl` | (sim-only reference) the simulated DVL's true velocity — **never fused** |

The **optical-flow maths itself lives in `sauvc_localization`** (see
`README_sauvc_localization.md §1`); this package **reuses that exact core unchanged**. The
key equations are repeated below where they are used, and linked back. The **frame-conversion
maths lives in `sauvc_sim_bridge`** (`README_sauvc_sim_bridge.md §1.2–1.4`), also repeated
where used.

---

## Part 1 — The maths, estimator by estimator

### 1.0 The shared front-end: getting a metric body velocity from flow

Every horizontal estimator here starts from the **same** optical-flow body velocity produced
by `sauvc_localization`'s `FlowVelocityEstimator`. In one paragraph (full derivation in
`README_sauvc_localization.md §1.1–1.5`): track floor corners with Lucas–Kanade, take the
**median** pixel displacement and reject outliers with **MAD**, **derotate** by subtracting
`(−fx·wy·dt, +fy·wx·dt)` so turning isn't mistaken for translation, then **scale by altitude**
to get metres:

```
Vx_cam = −(du_t/dt_eff)·h/fx ,   Vy_cam = −(dv_t/dt_eff)·h/fy      → body FLU (bx, by)
```

That FLU velocity is the raw material for §1.2 (integrator), §1.3 (EKF) and §1.4 (GTSAM).

### 1.1 One common frame, converted in exactly one place

To overlay estimates on a plot they must share a frame. The default is **NED** (as requested).
The seam is a single parameter, `compare_frame` (`'ned'` or `'enu'`).

The deliberate design choice: **convert the ground truth, not each estimate.** Estimates
arrive in several frames (flow = body, EKF = world, pressure = a scalar depth). Converting
*N* estimators into NED is *N* chances for a silent sign error — the exact failure mode that
has bitten this project. Ground truth is **one** source in NED. So:

- `compare_frame='ned'` → leave Stonefish ground truth alone, convert the ENU-native
  estimates once (via `sauvc_sim_bridge.frames`).
- `compare_frame='enu'` → convert ground truth NED→ENU once, leave estimates alone.

Either way the **only** conversion used is the tested involution
`(x,y,z)↔(y,x,−z)` from `README_sauvc_sim_bridge.md §1.2` — never an ad-hoc per-estimator sign
flip.

### 1.2 Flow → position by honest dead reckoning (trapezoidal integration)

To turn body velocity into a position track, rotate it into the world by the current yaw and
integrate. This package does it **honestly** — no ZUPT-in-the-integrator, no loop closure, no
fusion — because the *whole point* is to **see the drift accumulate**:

```
world velocity:   vxw = cos(yaw)·vx_body − sin(yaw)·vy_body
                  vyw = sin(yaw)·vx_body + cos(yaw)·vy_body

trapezoidal step: x += ½·(vxw + vxw_prev)·dt
                  y += ½·(vyw + vyw_prev)·dt      (ignore absurd dt ≥ 1 s)
```

Depth (`z`) comes from pressure, never from integrating vertical velocity.

**The flip that must happen first (`README_sauvc_localization.md §1.9`):** flow_core outputs
body **FLU**, but a **NED** yaw expects body **FRD**, so the planar conversion `vy → −vy` must
be applied **before** rotating. Skip it and every sideways leg integrates **mirrored** — the
real bug that showed up as ground truth "+2.28 m East" vs flow "−0.55 m". In `'enu'` mode the
FLU vector already matches the ENU yaw, so no flip.

### 1.3 The self-contained EKF

A compact **constant-velocity Kalman filter** — deliberately *not* a copy of the running
`robot_localization` node, so the comparison stands alone. State:

```
x = [px, py, pz, vx, vy]           (world = compare frame)
```

**Predict (constant velocity):**

```
px += vx·dt ,  py += vy·dt         (F is identity with F[0,3]=F[1,4]=dt)
P  ← F·P·Fᵀ + Q ,  Q = diag(q_pos,q_pos,q_pos,q_vel,q_vel)·dt
```

**Update — three measurement types**, each the standard Kalman correction
`y = z − H·x ; S = H·P·Hᵀ + R ; K = P·Hᵀ·S⁻¹ ; x += K·y ; P = (I − K·H)·P`:

- **Flow velocity.** Rotate the body velocity into the world by yaw to get a `(vx, vy)`
  measurement; `H` picks the two velocity states. `R` is the **quality-scaled** variance from
  `README_sauvc_localization` (`flow_velocity_node`'s model): worse spread / fewer inliers →
  bigger `R` → less trust.
- **Depth.** `H` picks `pz`; very tight `R` (the pressure sensor is near-perfect in sim).
- **Landmark position (§1.5).** `H` picks `px, py` with a **per-axis (anisotropic)** `R` — a
  gate observation whose x is known from the rulebook but whose y is randomized passes
  `var_y = 1e12` (a no-op on y) so it constrains x tightly and leaves y untouched.

Yaw is **not** a state — it comes from the IMU orientation and is used only to rotate the flow
measurement into the world.

### 1.4 The GTSAM factor graph (IMU preintegration done properly)

This is the most involved estimator, and it exists to be the **fairest competitor to the
EKF** — same three inputs (flow + depth + IMU), but fused as a **factor graph** with online
bias estimation instead of a filter.

**Why a graph at all.** Forster et al. (RSS 2015) **preintegrate** all the high-rate IMU
samples between two camera keyframes into a *single* relative-motion factor (with bias
Jacobians), then **optimize** position / velocity / bias **at** each keyframe against the
other factors. Preintegration is the *motion model*; the *estimate* comes from the
optimization. An earlier version did only the first half (predict-then-reset, no optimization)
and diverged to **+57 m** — that is what happens when you drop the graph.

**The graph per keyframe (config B):**

- **`CombinedImuFactor`** — the preintegrated IMU between keyframes, including bias evolution.
- **`PriorFactorVector` on velocity** — the optical-flow body velocity, rotated into NED. It
  is **horizontal only**: the z-sigma is huge (`1e6`) because the down-camera flow says
  *nothing* about vertical velocity, and constraining world-`vz` to 0 would fight the depth
  factor during any dive and inject pitch error.
- **`GPSFactor` on position (z only)** — pressure depth, with x/y sigmas `~∞` so **only depth**
  is constrained.

**Two bugs that had to be fixed, both pure geometry:**

1. **Gravity alignment.** `MakeSharedD(g)` sets `n_gravity = (0, 0, +g)` in NED (z-down). The
   initial attitude **must** be gravity-aligned or the ~9.81 m/s² specific force the
   accelerometer always reads gets integrated as if it were **motion**. We seed the attitude
   from the IMU's fused orientation (the AHRS gives it directly). This is what turned the "+57
   m" divergence into a bounded ~0.9 m IMU-only drift.
2. **Keyframe interval.** ~5 Hz keyframes (~15 IMU samples). Too frequent and bias becomes
   unobservable; too sparse and the first-order bias-update linearization stops being valid.
   Only works *with* the between-keyframe optimization.

**Static bias initialization (Forster TRO16).** Before `initialize()`, the vehicle floats
motionless at spawn, so buffer a few hundred stationary IMU samples and **measure** the
biases instead of assuming zero:

```
static specific force in body = Rᵀ·(0, 0, −g)          (what a level, still IMU should read)
accel_bias = mean(measured accel) − that expected value
gyro_bias  = mean(measured gyro)                        (should be 0 if truly still)
```

The bias **prior** is then centered on the measured value with an honest sigma, instead of
pinned at zero with a tiny sigma that *fights* any real bias exactly like a residual-gravity
error.

**The attitude anchor (fixing "gtsam yaw goes wild").** Here is a subtle, important failure:
at (near-)zero velocity — the mission's startup hold, and to a lesser degree any straight
line — **yaw is unobservable** from IMU + a velocity prior + depth. A zero (or forward-only)
velocity vector rotated by *any* heading is equally consistent, so the yaw direction is
rank-deficient; verified offline, the yaw marginal balloons to ~160° and iSAM2's Gauss-Newton
steps **teleport** the heading by tens of degrees. The fix is a **loose absolute attitude
prior** from the IMU's own orientation: roll/pitch are gravity-referenced so they get a tight
sigma; yaw gets a loose sigma (~11°) that only removes the rank deficiency and prevents the
blow-up, while still letting real motion refine heading.

**The lateral-drift refinement (why the anchor sometimes needs to be *tight*).** There is a
second twist. The graph's flow-velocity prior is rotated into the world by the **graph's own
yaw**, which makes that measurement **yaw-blind** (self-referential) — so the graph's heading
is set **entirely** by the attitude anchor. If the anchor uses the **raw published IMU quat**,
and the scene injects a slow `yaw_drift` ramp into the *published orientation*, then the
graph's yaw tracks that ramp, and rotating the flow velocity by the wrong yaw produces a
lateral velocity bias `≈ v·sin(yaw_err)` that integrates into **~5–7 cm of sideways drift per
metre**. The fix: when a **drift-free** absolute yaw is available (lane-heading fusion active),
substitute *that* yaw into the anchor quaternion (keeping the IMU's trustworthy roll/pitch)
and **tighten** the yaw sigma (~3°) so the anchor actually *corrects* heading rather than
merely bounding it.

**A key insight about the raw gyro (why the graph can beat the EKF on heading).** Confirmed by
reading Stonefish's `IMU.cpp`: the scene's `yaw_drift` is added **only to the published
orientation's yaw channel** as a post-hoc ramp. The **raw angular-velocity** channel that the
graph's `add_imu()` consumes is computed from the **true** angular velocity, entirely
*upstream* of that injection. So the graph's own preintegrated attitude **never sees** the
drift — it is mathematically clean. The EKF and the plain integrator, which derive yaw from the
*published* orientation, **do** carry the drift (corrected only by the external EWMA below).
That is why the running log verifies `gtsam_yaw` tracks ground truth better than `imu_yaw`
does.

### 1.5 Landmark localization (bounding the drift with known features)

Dead reckoning always drifts; landmarks pin it. Two modes:

- **`gate`** — the SAUVC gate's **x** is known from the rulebook. When the gate detector
  reports the gate's position relative to the vehicle `(bx, by)`, rotate it into the world by
  yaw and form a **position pseudo-measurement**: `p_meas_x = gate_x_known − rel_world_x`.
  Apply it as an **x-only** anisotropic correction (y is randomized, so `sigma_y` is huge).
  This is the biggest single win and costs almost nothing.
- **`map`** — additionally builds a tiny semantic map: each uniquely-named feature's world
  position is **frozen** after a few quality-gated near sightings
  (`mean(vehicle_pos + rel_obs)`), and later re-observations correct **both** axes with
  `var = stored_var + observation_var`. It is deliberately **not** SLAM state augmentation —
  the state stays `[x y z vx vy]` and the stored variance is inflated to pay the "decoupling
  tax". Observation variance scales with range: `obs_var = (lm_obs_sigma · range)²`. An
  innovation gate (`lm_innov_gate`) rejects corrections that jump too far.

Both feed the **EKF** (`update_position_xy`) and the **graph** (`add_landmark_xy`, an
anisotropic `GPSFactor` on the latest pose).

### 1.6 The flow → IMU yaw alignment (the extrinsic every DVL/flow system needs)

A lateral leak that grows with **forward distance** (not with time) can only be a **fixed yaw
misalignment** between the flow's body frame and the yaw used to rotate it into the world —
i.e. an IMU-shim yaw bias or the down-camera/IMU mount yaw. This is the standard extrinsic
every DVL is calibrated for. The correction is a single angle `flow_yaw_offset` added to the
yaw used by **all three** flow-based estimators.

**In the sim it is auto-calibrated from ground truth on straight legs**, and — importantly —
with a **never-freezing circular EWMA**, because the scene's `yaw_drift` is a *growing random
walk*, not a fixed extrinsic. A freeze-once estimate is exact only at the instant it froze and
grows stale forever after; the EWMA on the unit circle tracks it with a small **bounded** lag:

```
per straight-leg sample:  d = atan2(gt_vy, gt_vx) − atan2(flow_vy_world, flow_vx_world)
EWMA on the circle:       s ← (1−α)·s + α·sin(d) ,   c ← (1−α)·c + α·cos(d)
offset:                   flow_yaw_offset = atan2(s, c)
steady-state lag ≈ drift_per_sample / α   (~0.03° at the scene's drift rate, α = 0.02)
```

On hardware you measure it once and set `flow_yaw_offset` with `flow_yaw_autocal:=false`.

### 1.7 ZUPT — killing stationary creep

When the flow **and** the gyro both read ~0, the vehicle is genuinely stationary, so integrator
noise would otherwise creep the position. **Zero-velocity update:** clamp velocity to exactly 0
and mark the measurement as highly confident (tiny `R`), which pins the EKF's and graph's
velocity-bias estimation. Gate: `|v_flow| < zupt_vel` **and** `|gyro| < zupt_gyro`.

### 1.8 Anchoring — so every track starts where ground truth starts

Every dead-reckoned track starts from **zero**, but ground truth starts at the spawn (e.g.
`x = −12.1 m`). So each estimate is shifted by ground truth's **first** pose (or, before the
first GT message / on hardware, the start pose parsed from the scene file). Without this the
plots carry a fixed ~12 m offset and can never overlay. This makes the comparison
**displacement-from-start**, which is exactly what dead reckoning measures.

### 1.9 Self-sufficient altitude (with tilt compensation)

Because the depth-shim's altitude feed proved fragile (its odometry input was left unconnected
three runs running, freezing the floor lookup at a stale x — see
`README_sauvc_sim_bridge`), the eval node computes altitude **itself** from data it already
has: floor-profile depth at the ground-truth x, minus (sensor depth + mount offsets), and
uses the shim's `/altitude` only as a **cross-check** (warning loudly on a persistent
mismatch). **Tilt compensation:** the range along the (tilted) optical axis is longer than the
straight-down altitude, so:

```
altitude_along_axis = altitude / (cos(roll)·cos(pitch))     (clamped so >60° tilt can't blow up scale)
```

### 1.10 Timestamped gyro sync

For derotation (`README_sauvc_localization §1.3`) the *right* gyro sample is the one at the
**inter-frame midpoint**, not the latest arrival. The node keeps an IMU **ring buffer** and
linearly interpolates the rates at `t − dt/2`, removing lag-induced derotation error during
turns.

---

## Part 2 — The code, file by file

### `eval_common.py` — frame conversion + honest integration (no ROS)

- **World-vector converters** `enu_to_ned_vec` / `ned_to_enu_world` (both the `(x,y,z)→(y,x,−z)`
  involution from `README_sauvc_sim_bridge §1.2`) and body `flu_to_frd_vec` (`(x,y,z)→(x,−y,−z)`).
- `depth_to_world_z(depth, frame)` — a positive depth becomes `+depth` in NED (z-down) or
  `−depth` in ENU (z-up).
- `to_compare_frame_world` / `gt_world_to_compare` — the §1.1 "convert one side only" seam.
- `PositionIntegrator` — §1.2 exactly: rotate body velocity into world by yaw, **trapezoidal**
  accumulate, ignore `dt ≥ 1 s`. Its docstring is blunt that it does nothing to suppress drift
  — that is the point.

### `estimators/flow_estimator.py` — thin adapter around the localization core

- Wraps `sauvc_localization.flow_core.FlowVelocityEstimator` **unchanged** and exposes
  `estimate()` (feed gray + gyro + altitude → body `(vx, vy)`) and `overlay()` (the OpenCV
  visualization).
- **Dropout bookkeeping:** counts `dropouts` and the current `dropout_streak`, and surfaces
  `last_failure` / `fail_counts` from the core — because every `None` is displacement
  permanently lost from a dead-reckoned track (`README_sauvc_localization §1.6`), and this is
  what makes a 48 % freeze rate impossible to hide.
- **The two-arrow overlay** teaches the sign convention: **red** = raw pixel flow (points
  *opposite* travel, like scenery past a train window — this is correct and expected);
  **cyan** = recovered velocity direction (`−flow`, points *along* travel — this is what
  becomes `/eval/flow`). If cyan disagrees with actual motion, *then* the sign convention is
  wrong.

### `estimators/ekf_estimator.py` — the self-contained EKF (§1.3)

- State `[px, py, pz, vx, vy]`; `_predict` is the constant-velocity model; `_kalman` is the
  standard correction. `update_flow` (rotates body → world by yaw, optional per-measurement
  `r_var`), `update_depth`, and `update_position_xy` (the anisotropic landmark update from
  §1.5, with per-axis variances so a known-x/unknown-y gate constrains only x). Frame-agnostic:
  the caller hands it measurements already in the compare frame.

### `estimators/gtsam_estimator.py` — the iSAM2 factor graph (§1.4)

- `__init__` builds `PreintegrationCombinedParams.MakeSharedD(g)` (**gravity alignment**,
  §1.4 bug 1), the horizontal-only flow noise (z-sigma `1e6`), the depth-only noise, and the
  attitude-anchor noise (tight roll/pitch, loose-or-tight yaw).
- `add_imu` preintegrates at IMU rate, and **before initialization** buffers stationary
  samples for **static bias estimation**.
- `initialize` seeds a gravity-aligned `Pose3` from the IMU quaternion, measures the biases
  from the buffer (§1.4), and adds the priors on `X(0)/V(0)/B(0)`.
- `add_keyframe` adds the `CombinedImuFactor` + flow `PriorFactorVector` (per-keyframe
  `flow_sigma` for EKF parity + ZUPT) + depth `GPSFactor` + the **attitude anchor** (with the
  optional tight `att_yaw_sigma` when lane-fused yaw is available — the lateral-drift fix),
  runs one `isam.update`, and reads back pose/vel/bias.
- `current_ned_yaw` exposes the graph's **own** drift-free yaw (§1.4, the `IMU.cpp` insight);
  `add_landmark_xy` is the anisotropic landmark `GPSFactor` (§1.5). iSAM2 runs with **default**
  relinearization on purpose — an earlier tight setting made it relinearize the IMU factors far
  from their linearization point and diverge; the right lever for frame-drop spikes is
  throttling keyframes upstream, not hammering relinearization.
- Degrades gracefully: if `gtsam` isn't installed, `available=False` and the eval node just
  skips this estimator.

### `flow_eval_node.py` — the orchestrator that wires it all together

This is the big node; it owns §1.1 and §1.6–§1.10 and drives the three estimators.

- **Parameters** cover everything: `compare_frame`, intrinsics (defaulted to the 640×480 scene
  — `fx = fy = (640/2)/tan(40°) = 381.36`), `gtsam_keyframe_period`, the floor-profile /
  altitude / tilt settings, lane-heading fusion, quality-scaled `r_flow_base`, the
  flow-yaw-autocal EWMA (`flow_yaw_cal_alpha/min_n/min_speed`), ZUPT thresholds, and the
  landmark mode.
- **`on_imu`** caches body gyro/accel, converts the ENU/FLU orientation quaternion to NED/FRD
  via `sauvc_sim_bridge.frames` for the graph's gravity-aligned init, extracts roll/pitch (for
  tilt comp) and yaw (ENU→NED via `yaw_ned = π/2 − yaw_enu`), **ring-buffers** the gyro (§1.10),
  and feeds `gtsam.add_imu` at IMU rate.
- **`on_pressure`** computes depth `= P/(ρg)` (`README_sauvc_sim_bridge §1.5`), updates the
  EKF depth, and publishes `/eval/pressure`.
- **`on_ground_truth`** converts GT to the compare frame, sets the **anchor** (§1.8), derives
  the GT planar velocity for the **yaw autocal** (§1.6), and caches the GT rotation for the DVL
  reference row.
- **`on_image`** is the main pipeline per frame: compute `dt`; get the **midpoint-interpolated
  gyro** (§1.10) mapped to camera axes `(−wy, −wx)`; get the **self-computed, tilt-compensated
  altitude** (§1.9); call `flow.estimate`; on `None`, **count + warn** the dropout with its
  reason; on success apply the **FLU→FRD flip** for NED (§1.2), **ZUPT** (§1.7), the **fused
  yaw** (lane-heading, §1.4/`README_sauvc_localization §1.10`) plus the **flow-yaw offset**
  (§1.6), the **quality-scaled `r_var`**, then drive the EKF, the `PositionIntegrator`, and the
  GTSAM keyframe (rotating flow into NED by the **graph's own yaw**, §1.4), and publish each on
  `/eval/*`.
- **`_fused_yaws`** implements the lane-heading substitution with the correct **ENU→NED
  reflection** `yaw_ned = π/2 − yaw_enu` (a reflection, **not** a constant offset — the old
  additive formula was wrong everywhere except the one heading it was checked at, which
  silently disabled lane fusion on every turn). It also records whether fusion was active and
  whether it is actually *helping* versus raw IMU, so the logs can answer "is lane_heading
  doing anything?"
- **`_update_yaw_autocal`** is the circular-EWMA extrinsic estimator (§1.6).
- **`on_feature`** is the landmark localization (§1.5): the rulebook gate x-correction and the
  small-map freeze-then-correct.
- **`_publish` / `_maybe_print`** emit the `/eval/*` odometry and the throttled terminal table,
  including a **yaw verification line** (does `gtsam_yaw` track truth better than `imu_yaw`?)
  and a **lane-heading effectiveness line**.

### `landmark_truth_node.py` — sim-only ground-truth relative landmark poses

- Parses the scene for every gate/flare/drum/tub world position, subscribes to ground-truth
  odometry, and rotates `(landmark − vehicle)` into the vehicle's **body FRD** frame — giving
  the **exact** relative pose of every prop. Publishes `/truth/rel/<name>` and a table of
  body x/y/z, range, bearing, elevation.
- This is the **scoring stick for any detector**: your detector's reported relative pose minus
  these numbers **is** the detection error, with no other error mixed in. (Bearing =
  `atan2(y_frd, x_frd)`, positive to starboard; elevation positive down.)
- Honest caveat: flare/ball positions are the **spawn** poses; after the vehicle bumps one, its
  true pose changes and Stonefish doesn't republish it, so that row goes stale — fine for
  pre-contact detector scoring, which is when it matters.

### `yaw_attribution.py` — *which* thing is causing the yaw offset?

- A standalone diagnostic for the §1.6 offset. It compares, on genuine straight legs, the
  IMU-derived NED yaw against the ground-truth NED yaw, and reports a **speed-gated circular
  mean** ("settled") — the only number that means anything, because at rest or during
  spin/turn transients the raw difference is several degrees of real dynamics, not sensor bias.
- Its verdict tree: **settled ≈ 0** → IMU yaw is correct, so the offset lives in the
  **down-camera mount yaw** (check the camera rpy in the `.scn`); **settled ≈ the flow-autocal
  offset** → IMU yaw is wrong (the IMU shim conversion or the IMU mount yaw). A **staleness
  guard** warns if odometry stamps stop advancing, so you never trust frozen numbers.

### `flow_eval_launch.py` / `test_eval_common.py`

- The launch file brings up the four `sauvc_sim_bridge` shims (`imu_shim`, `depth_shim`,
  `image_relay`, `rtf_monitor`) plus this eval node — it deliberately does **not** start
  Stonefish or control (run those separately). `test_eval_common.py` unit-tests the pure-maths
  helpers in `eval_common.py`.

---

## How this package answers "which estimator wins" (the one-paragraph summary)

`flow_eval_node` takes the one live sensor stream and runs four horizontal estimators off it —
raw flow integration, a self-contained EKF, an iSAM2 factor graph, and depth-only — plus the
ground-truth and DVL reference rows, all converted into one frame with a single tested
conversion so they overlay on a plot. Along the way it fixes every subtle error that separates
a toy dead-reckoner from a real one: the FLU→FRD flip before rotating flow, a never-freezing
yaw-extrinsic auto-calibration, gravity-aligned graph initialization with measured biases, an
attitude anchor that switches from "loose, just don't blow up" to "tight, actually correct
heading" when a drift-free lane yaw is available, ZUPT to kill stationary creep, self-computed
tilt-compensated altitude, midpoint gyro sync for derotation, and optional landmark corrections
to bound the drift. The flow maths it consumes is the master copy in
`README_sauvc_localization.md`; the frame conversions it relies on are the master copy in
`README_sauvc_sim_bridge.md`; both are repeated above where used.
