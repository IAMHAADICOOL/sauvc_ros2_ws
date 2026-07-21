# `sauvc_sim_bridge` — the translation layer between the simulator and your real robot

This package is the **plumbing**. It does not estimate odometry itself. Its whole job is to
make the Stonefish simulator produce sensor messages that are *byte-for-byte identical* to
what the real AUV's drivers produce, so that every node further up the stack
(`sauvc_localization`, `sauvc_flow_eval`, the EKF) **cannot tell whether it is talking to
the simulator or the real vehicle.**

If that illusion is perfect, then anything you tune in simulation carries straight over to
the pool with no code changes. Most of the subtle work in this package is about keeping the
illusion honest — not faking numbers, but converting them correctly and shouting loudly when
something is off.

There is one hard fact that shapes everything here:

> **Stonefish speaks NED/FRD. Your stack speaks ENU/FLU. Somebody has to convert, exactly
> once, in exactly one place.** That place is `frames.py`.

---

## Part 1 — The maths this package is responsible for

There are four pieces of maths in this package, and they are all "boring but deadly": each
one is simple, and each one silently corrupts everything downstream if you get a sign wrong.

### 1.1 The two coordinate conventions

A "frame" is just a choice of which way the x, y, z axes point. Two different communities
made two different choices, and both are used in this project.

**Stonefish (the simulator) uses NED / FRD:**

- **World = NED**: x points **N**orth, y points **E**ast, z points **D**own (into the water).
- **Body = FRD**: x points **F**orward, y points **R**ight (starboard), z points **D**own.

**Your ROS stack uses ENU / FLU** (this is the ROS standard, "REP-103"):

- **World = ENU**: x points **E**ast, y points **N**orth, z points **U**p.
- **Body = FLU**: x points **F**orward, y points **L**eft (port), z points **U**p.

Notice the world frames swap x and y and flip z; the body frames keep forward the same but
flip left/right and up/down.

### 1.2 The conversion is two fixed 180° rotations

You never need a general rotation to go between these — each conversion is a single fixed
half-turn, and (this is the nice part) **each is its own inverse**. So the *same* function
converts both directions.

**World, NED ↔ ENU** — swap x and y, flip z:

```
(x, y, z)  ->  (y, x, -z)
```

This is a 180° rotation about the diagonal axis `(√2/2, √2/2, 0)`.

**Body, FRD ↔ FLU** — keep x, flip y and z:

```
(x, y, z)  ->  (x, -y, -z)
```

This is a 180° rotation about the x-axis.

Because each map equals its own inverse (an *involution*), there is deliberately **no**
`enu_to_ned` twin function — you call the same one both ways. That is a feature: it removes
a whole class of "I called the wrong-direction converter" bugs.

### 1.3 Rotating an orientation (quaternion) between frames

A vehicle's *attitude* (which way it is pointing) is stored as a quaternion. To move an
attitude from (NED world, FRD body) into (ENU world, FLU body) you sandwich it between the
two fixed rotations above:

```
q_enu_flu  =  Q_NED_ENU  ·  q_ned_frd  ·  Q_FRD_FLU
```

where `Q_NED_ENU` is the world half-turn and `Q_FRD_FLU` is the body half-turn, and `·` is
quaternion multiplication (the "Hamilton product"). Read it right-to-left: first re-label
the body axes, then re-label the world axes.

Two traps this guards against:

- **Quaternion order.** ROS stores quaternions as `(x, y, z, w)`. Some libraries (scipy,
  Eigen constructors) expect `(w, x, y, z)`. Getting this wrong does **not** crash — it
  produces a plausible-looking but wrong attitude. So the whole file sticks to one order,
  `(x, y, z, w)`, and says so everywhere.
- **Normalization.** Quaternions must have length 1. Every conversion re-normalizes and
  guards against the all-zero quaternion.

### 1.4 Rotating a *covariance* between frames

A covariance is an uncertainty ellipsoid (how unsure we are about x vs y vs z). To move it
into another frame you conjugate it with the rotation matrix `R`:

```
Σ'  =  R · Σ · Rᵀ
```

Here is the subtle, important warning baked into this package: because our two rotations are
just **signed axis swaps**, when the covariance is *diagonal* the sign flips square away
(`(-1)² = 1`) and the diagonal is merely permuted. That means **a sign error in a diagonal
covariance is invisible** — it will not blow up, it will just silently attach your x-variance
to the y-axis. So you must always use the full `R·Σ·Rᵀ` conjugation and never hand-permute
the diagonal, because the off-diagonal terms *do* change sign and those are what catch you.

### 1.5 Depth from pressure

The pressure sensor reads pressure `P` (in Pascals). Depth below the surface is:

```
depth  =  P / (ρ · g)
```

where `ρ` is water density and `g` is gravity. The single most important rule here:

> **`ρ` and `g` must match the numbers the *simulator's scene file* declares, not the real
> pool's.** The scene says `density = 1000.0` and `g = 9.81`. Real pool water at 26 °C is
> ~997 kg/m³ at g = 9.80665. If you invert Stonefish's pressure with the real pool's
> constants you get a constant **+0.335 % depth scale error** (~5 mm at 1.5 m). That error
> then flows depth → altitude → flow-scale and would masquerade as a fake calibration error
> forever. Match the scene, not physics.

Stonefish also publishes **gauge** pressure (already referenced to the free surface), so
`P/(ρg)` is *already* true depth — no atmospheric term, no surface-zeroing needed. The real
Bar30 driver needs surface-zeroing to remove its atmospheric offset; the simulator does not,
and adding zeroing here would only inject noise.

### 1.6 Altitude from a floor profile (and why depth alone isn't enough)

The optical-flow velocity (over in `sauvc_localization`) needs to know the camera's height
above the floor — the **altitude** — to turn pixel motion into metres. Altitude is:

```
altitude(x)  =  floor_depth(x)  −  vehicle_depth
```

The floor is not flat: a competition pool has a **V-shaped** deep section. So
`floor_depth(x)` is a piecewise-linear lookup ("floor profile") along the pool's long axis:
you give it breakpoints like `x = [0, 12.5, 25] m` → `depth = [1.2, 1.6, 1.2] m`, and it
linearly interpolates between them, clamping flat beyond the ends.

There is a **mild feedback loop** here worth understanding: altitude depends on `x`, `x` comes
from the flow estimate, the flow estimate depends on altitude. It is benign because the floor
slope is gentle (~3 %), so 1 m of x-error costs only ~3 cm of altitude error.

### 1.7 The Real-Time-Factor trap (the sneakiest bug in the whole project)

This one is not a conversion — it is a timing landmine, and this package exists partly to
catch it.

Stonefish stamps **every** message with the **wall clock at publish time**
(`get_clock()->now()`), throwing away the sample's true simulation time. There is no `/clock`
topic. Now follow the consequence through the flow node, which computes its time step from
those stamps:

- The **pixels** in two frames were rendered from physics that advanced by **simulation
  time**: a "30 Hz" sensor means 30 samples per *simulated* second.
- The **dt** the flow node computes is **wall-clock** elapsed time.
- If the simulator runs at real-time factor `R` (e.g. R = 0.6 means it runs at 60 % speed
  because the GPU can't keep up), then:

```
true physical displacement between frames = v_true · (1/30)     [sim seconds]
wall-clock dt the node sees                = (1/30) / R          [wall seconds]
flow's reported velocity = displacement/dt = v_true · R
```

**So optical-flow velocity is scaled by exactly the real-time factor `R`.** Depth, gyro
rates, and the DVL are *not* (they're direct physics quantities). This means at `R ≠ 1` the
simulator is not merely slow — it is **kinematically inconsistent**: the EKF would fuse true
angular rates against R-scaled linear velocity, and any scale-check would report a fake
error of exactly `R`. **`R = 1` is a hard prerequisite**, not a nice-to-have, and
`rtf_monitor_node` exists to enforce it.

---

## Part 2 — The code, file by file

### `frames.py` — the single source of truth for all frame conversions

This is the only file in the entire workspace allowed to write a sign flip between NED/FRD
and ENU/FLU. Everything in §1.2–1.4 lives here.

- `ned_to_enu_vec(v)` and `frd_to_flu_vec(v)` — the two involutions from §1.2, as plain
  vector maps `(x,y,z)->(y,x,-z)` and `(x,y,z)->(x,-y,-z)`. Used for anything measured as a
  vector: acceleration, angular velocity, DVL velocity, optical-flow velocity.
- `quat_mul(q1, q2)` — the Hamilton product in `(x,y,z,w)` order. `quat_normalize(q)` guards
  the zero quaternion.
- `ned_frd_quat_to_enu_flu(q)` — the attitude sandwich from §1.3,
  `Q_NED_ENU · q · Q_FRD_FLU`, with re-normalization. `enu_flu_quat_to_ned_frd(q)` is
  *structurally identical* (same code) precisely because both fixed rotations are their own
  inverse — the docstring proves it algebraically.
- `flu_frd_to_ned_wxyz(x,y,z,w)` — convenience wrapper that also **reorders** the output to
  `(w,x,y,z)`, because GTSAM's `Rot3.Quaternion` constructor wants that order. This is the
  order-trap from §1.3 handled once, centrally.
- `rot_matrix_ned_enu()`, `rot_matrix_frd_flu()`, `rotate_cov3(cov, R)`, `cov3_ned_to_enu`,
  `cov3_frd_to_flu` — the covariance conjugation from §1.4. `rotate_cov3` accepts either a
  3×3 matrix or ROS's flat 9-vector and returns the same shape. Its docstring carries the
  "diagonal sign errors are invisible" warning — that comment *is* the reason the function
  exists instead of hand-permutation.

**Rule of thumb enforced by this file:** if a sign flip ever appears *anywhere else* in the
workspace, that is a bug — fix it in `frames.py`, not in the offending file.

### `imu_shim_node.py` — makes Stonefish's IMU look like the real HFI-A9

- **Subscribes** `/sauvc_auv/imu` (NED/FRD, simulator) → **publishes** `/imu/data` (ENU/FLU,
  your stack).
- It converts orientation via `ned_frd_quat_to_enu_flu` (§1.3), angular velocity and linear
  acceleration via `frd_to_flu_vec` (§1.2), orientation covariance via `cov3_ned_to_enu` and
  the body covariances via `cov3_frd_to_flu` (§1.4). **No sign flip is written by hand here** —
  it all comes from `frames.py`.
- **The zero-covariance landmine.** When the scene declares no acceleration noise, Stonefish
  fills the acceleration covariance with **all zeros**. Per the ROS spec, zero means
  "perfectly known" — a *singular* covariance that would blow up any filter that tried to
  invert it. The shim rewrites it to `[-1, 0, …]`, the ROS sentinel for "unknown", so anything
  that later tries to fuse acceleration fails **loudly** instead of silently inverting a
  singular matrix.
- **The stamp is passed through untouched.** It is already wall-clock; a past bug came from
  *synthesizing* timestamps, so the rule is now "never invent a stamp". (This is what makes
  the RTF trap in §1.7 possible, and why `rtf_monitor` is needed.)
- **The "too-good AHRS" warning.** The real HFI-A9's yaw slowly drifts; Stonefish's does not,
  *unless* the scene sets `<noise yaw_drift=...>`. If the yaw variance arriving is tiny, the
  shim warns once that the sim IMU is "too perfect" — meaning `lane_heading_node` (§ in the
  localization README) would have nothing to correct and its tests would pass *vacuously*.
  This is the shim honestly telling you your test is meaningless, rather than letting you
  believe a fixed problem.

### `depth_shim_node.py` — makes Stonefish's pressure sensor look like the Bar30 driver

- **Subscribes** `/sauvc_auv/pressure` (+ `/odometry/filtered` for the x-lookup) →
  **publishes** `/depth` (z as negative depth, REP-103 z-up), `/altitude`, and `/floor_depth`
  (debug).
- **Depth** is §1.5: `depth = P / (ρg)` with `ρg` forced to match the scene's declared water
  (`1000 × 9.81`), *not* the real pool. The docstring works the +0.335 % error all the way
  through so you know exactly why "match the scene, not physics".
- **Depth variance is derived from the message**, not hand-set: `σ_depth = σ_P / (ρg)`. The
  scene's `<noise pressure="20"/>` gives `σ_depth ≈ 2.04 mm`, i.e. `var ≈ 4.16e-6 m²` — 96×
  tighter than the real Bar30's placeholder. Feeding the real sensor's pessimism to the sim
  EKF would make it distrust a near-perfect sensor and you'd tune against a lie.
- **Altitude** is §1.6: the `_FloorProfile` class does the piecewise-linear
  `floor_depth(x)` lookup (with `np.interp` flat-clamping past the ends), then
  `altitude = floor_depth(x) − depth`. Before the first odometry message it assumes `x = 0`
  (the start zone is at the wall).
- **Surface-zeroing is OFF by default** and the docstring is emphatic that this is the
  *parity-preserving* choice, not laziness: Stonefish's gauge pressure is already
  surface-referenced. The `_settled()` detector (which waits for the vehicle to stop
  drifting upward before zeroing) is **insurance** for the one dangerous case — a launch file
  that starts the sim and this node together, where a naive zero-window from t=0 would average
  the vehicle's buoyant ascent and bake in ~11 cm of error. It tests **drift** (difference of
  two half-window means, std-error ~0.9 mm) rather than **spread** (max−min, ~7.6 mm at this
  noise) because a spread threshold tight enough to detect "settled" would sit *below the
  noise floor* and never trigger.
- **Known constant bias, flagged honestly:** the sensor is mounted 10 cm above the body
  origin, so `/depth` is really the *sensor's* depth. This is a constant ~10 cm offset that
  the real driver shares (so it cancels between sim and hardware), and depth-hold absorbs it
  into the setpoint — it only matters if you compare z against ground truth.

> **Note (relevant to `sauvc_flow_eval`):** the eval node found this altitude path fragile —
> `depth_shim`'s odometry feed was left unconnected at launch three runs in a row, freezing
> the floor-profile lookup at a stale x. So the eval node computes its *own* altitude instead
> and uses `/altitude` only as a cross-check. That is a launch-wiring fragility, not a maths
> error in this node.

### `image_relay_node.py` — renames the camera topics

- **Subscribes** `/sauvc_auv/camera_*/image_color` → **publishes** `/camera_*/image_raw`.
- Pure passthrough: same message, same header (same wall-clock stamp), only the topic name
  changes. No re-encoding, no re-throttling.
- **Why a relay instead of launch-file remaps:** three real nodes consume camera images and
  two of them *hardcode* the topic name (`lane_heading_node`, `gate_detector_node`), so a
  per-node remap would fix only the one node that uses a parameter. One relay that makes
  `/camera_*/image_raw` exist fixes all three at once and leaves the hardware nodes untouched.
  It is the camera version of what the IMU/depth shims do.
- It explicitly does **not** fix the low camera frame rate — that is a rendering-cost problem
  solved in the scene (drop resolution) or launch (rendering quality), not by renaming.

### `rtf_monitor_node.py` — enforces the "R = 1" prerequisite (§1.7)

- **Publishes** `/sim/rtf` and warns whenever the real-time factor strays from 1.
- Uses **two independent estimators** because each fails in a different place:
  1. **Rate method** (always available): a sensor declared at rate `f` in sim-time publishes
     at `f · R` in wall-time, so `R = observed_rate / declared_rate`. Works even while
     stationary; depends on you telling it the declared rate.
  2. **Kinematic method** (needs motion): differentiate ground-truth *position* over
     wall-clock stamps and divide by the ground-truth *twist* magnitude (reported directly in
     physics units). The ratio is `R`. It uses magnitudes only, so it needs no frame
     conversion and doesn't care whether the twist is body or world. Uses the **median** of
     the ratios to shrug off render hitches.
- If the two methods disagree, it tells you your `declared_odom_rate` parameter is wrong.
- It reads only ground truth, so it can never contaminate the estimator — it is a pure
  diagnostic. Its message when `R ≠ 1` spells out the exact fix (drop resolution/rate, lower
  rendering quality, or go headless).

### `flow_scorer_node.py` — grades the optical flow against the simulated DVL

- **Subscribes** `/flow/twist` (your estimate, body FLU), `/sauvc_auv/dvl` (reference, body
  FRD), `/sim/rtf`. **Publishes nothing** — it is deliberately a dead end.
- **The structural guarantee that the DVL never reaches the EKF:** the DVL stays inside the
  `/sauvc_auv` namespace and *no shim republishes it*. Because the hardware launch files never
  create that namespace, any accidental dependency on the DVL would fail **loudly** on the
  real robot instead of silently degrading in the pool. "The optical flow *is* the DVL" — this
  node just tells you how good a DVL it is.
- **What it reports:** `scale` (least-squares slope of flow vs DVL through the origin — this
  is the headline "% scale error" number), `bias`, `rmse`, `r` (correlation — if this is low,
  `scale` is meaningless because you have a tracking failure not a calibration error), and
  `dropout` (fraction of DVL samples with no matching flow).
- **It refuses to report a scale unless RTF ≈ 1** (§1.7). At R = 0.6 it would otherwise tell
  you the flow has a 40 % scale error when the algorithm is perfect — a simulator artifact,
  not a bug. It converts the DVL from FRD to FLU through `frames.frd_to_flu_vec` — never by
  hand.
- **Sign diagnosis:** if the fitted scale comes out **negative**, that's the down-camera
  mounting convention not matching `swap_xy`/`sign_x`/`sign_y` — a config fix, exactly the
  hand-push calibration from Phase 3, except the DVL does the pushing for you.

### The rest of the package (not part of odometry)

For completeness: this package also carries the **ArduSub bridge** (`ardusub_json_bridge.py`,
`ardusub_setpoint_node.py`, `ardusub_move_example.py`), **scene/mesh tooling**
(`convert_vehicle_mesh.py`, `generate_meshes.py`, `randomize_arena.py`), the **scene files**
(`*.scn`), and the **launch files** (`sim_drivers.launch.py` brings up the four shims above;
`sim_*_launch.py`). These are about *driving* and *building* the simulated vehicle, not about
*estimating* where it is, so they are outside the scope of this odometry write-up — but they
live in the same package because they share the "make sim look like hardware" mission.

---

## How this package feeds the odometry stack (the one-paragraph summary)

`sauvc_sim_bridge` turns the simulator into a fake robot: `imu_shim` gives you `/imu/data`
(ENU/FLU orientation, gyro, accel), `depth_shim` gives you `/depth` and `/altitude`,
`image_relay` gives you `/camera_down/image_raw`, and `frames.py` guarantees every one of
those conversions is done correctly, in one place. `rtf_monitor` guarantees the timing is
consistent so the flow velocity isn't secretly scaled, and `flow_scorer` grades the result
against a reference DVL that is structurally prevented from ever leaking into the estimate.
Everything the localization and eval packages consume on the un-namespaced topics
(`/imu/data`, `/depth`, `/altitude`, `/camera_down/image_raw`) is produced here. The
conversion maths (§1.2–1.4) is repeated in the other two READMEs wherever they rotate a
vector or attitude, and always links back here.
