# `sauvc_localization` — the real-vehicle localization stack ("DIY DVL" from optical flow)

This is where the vehicle actually figures out how fast it is moving and which way it is
pointing, using only the sensors a cheap AUV really has: a **downward camera**, a **pressure
sensor**, and an **IMU**. There is no real DVL (Doppler velocity log) and no GPS underwater.
The trick of this whole package is:

> **Point a camera at the pool floor, watch the floor slide past, and turn that sliding
> motion into a metric velocity.** The camera plus the pressure sensor together become a
> home-made DVL.

**This README holds the master copy of the optical-flow maths.** The evaluation package
(`sauvc_flow_eval`) reuses this exact core, so its README links back here and repeats the key
equations. The frame-conversion maths belongs to `sauvc_sim_bridge` — it is summarized here
where used and linked back to `README_sauvc_sim_bridge.md`.

---

## Part 1 — The maths, in the exact order it happens to one camera frame

Everything below is what happens between two consecutive camera frames. Follow it top to
bottom and you have the entire pipeline.

### 1.0 The picture to hold in your head

The camera looks straight down at the floor from a height `h` (the **altitude**). When the
vehicle moves forward, every feature on the floor appears to stream **backward** through the
image — exactly like scenery sliding past a train window. Measure how many pixels the floor
slid, know how high up you are, and you can work out how far you actually moved in metres.

### 1.1 Track the floor: sparse Lucas–Kanade

We do **not** track every pixel. We pick a few dozen good corners (`goodFeaturesToTrack`) and
follow *those* from the previous frame to the current one using **pyramidal Lucas–Kanade
(LK)** optical flow. Two reasons this choice matters:

- It is cheap (a few dozen points, not a whole image).
- It has **no loop closure and no global map**, so the fact that pool tiles all look
  identical is **not a problem** — we only ever compare one frame to the very next one, never
  to a tile seen a metre ago.

Each tracked corner gives a **pixel displacement** `d = (du, dv)` — how far that corner moved
in the image, in pixels.

### 1.2 Be robust: median + MAD outlier gate

Some of those corners will be garbage: a caustic (dancing light pattern), a fish, a floating
rope, a reflection. If we averaged all displacements, one bad point would poison the answer.
So instead:

- Take the **median** displacement `med = median(d)`. The median ignores a handful of wild
  outliers by construction.
- Measure the spread with the **MAD** (median absolute deviation): `mad = median(|d − med|)`.
- **Keep** only corners within `4·mad + 1` pixels of the median; throw the rest away.

```
inlier  ⟺  |d − med|  <  4·mad + 1        (per axis)
du, dv  =  median of the surviving displacements
```

This is classic robust statistics: the median gives a central value, the MAD gives a
scale, and anything too far from center in units of MAD is rejected.

### 1.3 Derotate: remove the flow caused by *turning* (the rotational correction)

Here is the key confusion this step fixes: **if the vehicle just tilts or turns without
moving, the floor still slides across the image** — and we must not mistake that for
translation.

When the camera rotates at angular rates `(wx, wy)` (about the camera's own x and y axes),
even a perfectly stationary scene produces optical flow. The **full** rotational-flow field
at an image point `(x, y)` (measured from the image center, focal length `f`) is:

```
u_dot_rot  =  −f·wy  +  wz·y  +  (wx·x·y − wy·x²)/f
v_dot_rot  =  +f·wx  −  wz·x  +  (wx·y² − wy·x·y)/f
```

The last group of terms — the ones divided by `f` — are the **second-order** terms
(quadratic in image position). **Near the image center** (`x, y ≈ 0`) they vanish, and the
yaw term `wz` also contributes almost nothing to the *median* flow of a centered feature
cloud. So we keep just the dominant first-order center terms:

```
u_dot_rot  ≈  −f·wy
v_dot_rot  ≈  +f·wx
```

and **subtract** the predicted rotational displacement over the interval `dt` from the
measured flow, leaving only the **translational** part:

```
du_t  =  du  −  (−fx · wy · dt)
dv_t  =  dv  −  (+fy · wx · dt)
```

So this is a **first-order rotational correction**, exact frame-to-frame, and the dropped
second-order terms are why you should keep features away from the extreme image corners.
(You can *verify* it works with a stationary pitch/roll wiggle: after derotation the flow
should read ~0.)

### 1.4 Scale to metres: altitude resolves the ambiguity

Now the geometry. A ground point at metric offset `X` from the optical axis, seen from height
`h`, projects to a pixel offset `u = fx · X / h`. Differentiate in time (h roughly constant
across one frame):

```
du/dt  =  (fx / h) · dX/dt
```

But the ground point's motion `dX/dt` is the **opposite** of the camera's own translation
(train-window effect): `dX/dt = −Vx_cam`. So:

```
du/dt  =  −(fx / h) · Vx_cam
```

Invert to recover the **metric camera-frame velocity**:

```
Vx_cam  =  −(du_t / dt) · h / fx
Vy_cam  =  −(dv_t / dt) · h / fy
```

**This is where altitude earns its keep.** Pixel flow alone (`du/dt`) only tells you
`−(fx/h)·Vx` — you cannot separate "moved a lot, high up" from "moved a little, low down".
That is the famous **monocular scale ambiguity**. The pressure sensor supplies `h` (altitude
= floor depth − vehicle depth), so **there is no scale ambiguity to resolve** — the range
comes for free. `fx, fy` must be from an **underwater** calibration (water bends light and
changes the effective focal length).

### 1.5 Rotate camera-axes into body-axes

The velocity above is in *camera* axes (image-right, image-down). We want it in *vehicle
body* axes. For the standard down-camera mount:

```
bx  =  −Vy_cam      (image "up" is vehicle forward)
by  =  −Vx_cam      (image "right" is vehicle right)
```

then apply the mounting fix-ups from the calibration hand-push test: an optional `swap_xy`,
and per-axis sign flips `sign_x`, `sign_y`. These three parameters exist because you cannot
know the exact camera mounting orientation from first principles — you push the vehicle a
known way once and set them so the reported velocity points the right way.

The output `(bx, by)` is a **body-frame FLU** velocity (x forward, y left). Remember that —
it matters a lot the moment anyone rotates it into a world frame (§1.9).

### 1.6 Handling dropouts: hold the reference instead of losing the gap

This is the single most impactful fix in the file, so it gets its own section.

**The old bug:** whenever tracking failed for a frame, the code advanced its reference to the
current frame *before* giving up — which **permanently threw away** the vehicle's motion
across that gap. In dead reckoning, thrown-away motion subtracts straight off your total
distance. Measured effect: the flow track came out at **0.44×** the true distance because
**48 % of moving intervals were frozen**.

**The fix — reference hold:** on a *tracking* failure, **keep the last good reference frame**.
Then the next successful LK solve spans the **whole gap**, and we divide the recovered
displacement by the **whole spanned time**:

```
dt_eff  =  dt  +  hold_dt         (current gap + all held frames' time)
Vx_cam  =  −(du_t / dt_eff) · h / fx
```

If we divided by the single-frame `dt` instead of `dt_eff`, a recovered track would
**overestimate** speed by `(1 + hold_dt/dt)`. The hold is **bounded**: after
`max_hold_frames` (5) or `max_hold_dt` (0.5 s) the track is declared genuinely lost and the
reference hard-resets — accepting that one gap's loss, but now **counted and visible** via
`last_failure` / `fail_counts` (reasons: `lk_none`, `few_tracked`, `few_inliers`,
`bad_altitude`, `bad_dt`, `+track_lost`). `max_hold_dt` is deliberately kept below the
integrator's 1.0 s gap guard.

### 1.7 Rejecting false matches: the forward–backward check

LK can "succeed" onto the **wrong** texture — especially across a held gap or on repetitive
tiles — giving a confident but wrong (often near-zero) flow, which is *worse* than a dropout
because it gets integrated. Defense: after tracking forward (prev → current), track **back**
(current → prev) and keep only points whose round-trip lands within `fb_max_err` (1 px) of
where they started:

```
keep point  ⟺  ‖ prev_pt − back_tracked_pt ‖  <  fb_max_err
```

On ≤150 points this is nearly free next to detecting the corners in the first place, so it
runs on every frame.

### 1.8 Beating tile aliasing: predictive seeding + a velocity plausibility gate

Pool tiles repeat, so across a held gap LK can lock onto the *wrong grid line, one period
over* — and the forward–backward check **passes**, because the false lock is self-consistent.
Two defenses:

1. **Predictive seeding.** Seed LK's search at the displacement predicted from the last good
   pixel rate over the spanned interval (`OPTFLOW_USE_INITIAL_FLOW`):
   `guess = prev_pts + px_rate · dt_span`. Frame-to-frame this is a harmless refinement;
   across a held gap it steers convergence to the **true** lock instead of a tile-period
   alias.
2. **Velocity plausibility gate.** A real AUV cannot change speed by more than
   `a_max · gap`. So a velocity recovered across a hold that jumps more than `v_jump_max`
   (0.8 m/s) from the pre-dropout velocity is an alias, not motion — reject it and keep
   holding (the next seeded frame usually resolves it).

### 1.9 The frame the velocity comes out in (and the flip everyone forgets)

flow_core outputs **body FLU** `(bx, by)`. If a downstream node wants to integrate the
velocity in a **NED** world (as the eval package does), it must first convert body **FLU →
FRD**, which for the planar part is simply:

```
vy_frd  =  −vy_flu        (FRD = FLU with y flipped; see README_sauvc_sim_bridge §1.2)
```

**and only then** rotate by the NED yaw. Skipping this flip mirrors every sideways
(sway/east) leg. This is `frames.frd_to_flu_vec` again — the same involution — and it is
called out here because it is the exact bug that made the eval package's lateral legs come
out mirrored until it was fixed.

### 1.10 Absolute heading without a compass: lane-line fusion (mod 90°)

The gyro gives you *change* in heading but slowly **drifts**; underwater you have no
magnetometer to pin absolute heading. The trick: a competition pool's **lane lines and tile
grout are aligned with the pool axes**. The dominant straight-line direction in the down
camera therefore gives the vehicle's yaw **relative to the pool**, but only **modulo 90°**
(a grid looks the same rotated by 90°). That ambiguity is fine, because the gyro never drifts
anywhere near 45° between corrections, so you always know which 90° branch you're in.

The maths:

- Run Canny edge detection, then a Hough transform to find straight segments.
- Fold every segment angle into mod-90° space by **multiplying the angle by 4** (period π/2 →
  period 2π), then take a **length-weighted circular mean**:

```
s = Σ Lᵢ · sin(4·angleᵢ),   c = Σ Lᵢ · cos(4·angleᵢ)
mean_angle = atan2(s, c) / 4
```

- Reject the frame if the directions disagree too much — the **concentration**
  `R = √(s²+c²) / Σ Lᵢ` must exceed 0.6 (all lines roughly parallel).
- Fuse the accepted line angle as a **slow complementary correction** to the fast
  gyro-integrated yaw: `offset ← offset + gain · error`, with a sanity gate rejecting
  corrections that disagree with the current yaw by more than 20°.

This cancels gyro drift with **no magnetometer**.

### 1.11 Optional: the preintegration smoother (coasting through dropouts)

The flow gives velocity only when it can see texture. During a dropout (caustics, a
featureless patch, pitching up at a flare) the plain EKF coasts *blind*. The optional
factor-graph smoother instead **preintegrates the IMU** between camera keyframes and
estimates the accel/gyro **biases online** while flow is healthy, so during a dropout it
coasts on **bias-corrected** IMU with bounded error. The full factor-graph maths is explained
in the `sauvc_flow_eval` README (the eval package's GTSAM estimator is the same idea) — here
it is a parallel `/odometry/preint` output you A/B against the EKF.

---

## Part 2 — The code, file by file

### `flow_core.py` — the pure-maths heart (no ROS, unit-testable)

Class `FlowVelocityEstimator`. This file *is* §1.1–1.9. Walking `process(gray, dt,
gyro_xy_cam, altitude)` in order:

1. **Bootstrap / guards.** If there is no reference yet, or too few reference points, detect
   fresh corners and return `None` (`'bootstrap'`). If `dt ≤ 0` (bad stamps), return `None`
   (`'bad_dt'`) **keeping** the reference — no provable time passed, so don't accrue hold
   time.
2. **Predictive seeding (§1.8).** `dt_span = dt + hold_dt`; if a `px_rate` exists, build a
   `guess` and set `OPTFLOW_USE_INITIAL_FLOW`.
3. **Forward LK** (`calcOpticalFlowPyrLK`). On `None`, `_fail('lk_none', hold=True)` — the
   critical §1.6 fix: **hold**, do not advance.
4. **Forward–backward check (§1.7).** Track back, compute round-trip error, keep points with
   `status==1 & bstat==1 & fb_err < fb_max_err`.
5. **`few_tracked` gate.** If fewer than `min_features` survive, `_fail('few_tracked',
   hold=True)`.
6. **Altitude gate.** If altitude is missing or `< 0.1 m`, `_fail('bad_altitude',
   hold=False)` — tracking was fine but the displacement can't be scaled to metres, so
   holding buys nothing; advance so the pixel gap doesn't grow.
7. **`dt_eff = dt + hold_dt` (§1.6).**
8. **Median + MAD (§1.2).** `med`, `mad`, inlier mask `|d−med| < 4·mad + 1`; if fewer than
   `min_features` inliers, `_fail('few_inliers', hold=True)`. `du, dv = median of inliers`.
9. **Derotation (§1.3).** `du_t = du − (−fx·wy·dt_eff)`, `dv_t = dv − (+fy·wx·dt_eff)`.
10. **Scale to metres (§1.4).** `vx_cam = −(du_t/dt_eff)·alt/fx`, likewise `vy_cam`.
11. **Camera → body (§1.5).** `bx = −vy_cam`, `by = −vx_cam`, then `swap_xy`/`sign_x`/
    `sign_y`.
12. **Aliasing velocity gate (§1.8).** If this estimate spans a hold and jumps more than
    `v_jump_max` from `last_v`, `_fail('alias_reject', hold=True)`.
13. **Commit success.** Update `px_rate`, `last_v`; **now** advance the reference
    (`_advance_ref`); return a dict with `vx, vy, n_tracked, n_inliers, spread_px, flow_px,
    dt_eff, recovered_gap_s`.

Helper methods: `_detect` (corner detection), `_advance_ref` (move reference + reset hold
counters — called **only** on success or on non-tracking failures), and `_fail` (record the
reason, optionally hold, and trip `+track_lost` when the hold limits are exceeded). Key
parameters: `max_corners`, `quality`, `min_distance`, `min_features`, `swap_xy`, `sign_x`,
`sign_y`, `max_hold_frames`, `max_hold_dt`, `fb_max_err`, `v_jump_max`. The LK settings
(`winSize 21×21`, `maxLevel 3`) give a pyramid deep enough to catch fast motion.

> `flow_core_original.py` is the **pre-fix** version kept for reference — it has none of the
> reference-hold, forward–backward, seeding, or aliasing logic. Diffing the two is the
> clearest way to see exactly what §1.6–1.8 added.

### `flow_velocity_node.py` — the thin ROS wrapper (Phase 3)

- **Subscribes** the down-camera image, `/imu/data` (for gyro), `/altitude`. **Publishes**
  `/flow/twist` (`TwistWithCovarianceStamped`, body FLU).
- **Gyro body → camera axes (§1.3).** For the standard mount:
  `gx_cam = −gyro_y`, `gy_cam = −gyro_x`. This is the hardware-validated mapping (the
  down-camera's x = image-right = vehicle starboard = −body-y; its y = image-down =
  vehicle-aft = −body-x). It feeds `gyro_xy_cam` into `process`.
- **`dt` from image header stamps** — which is exactly where the RTF trap
  (`README_sauvc_sim_bridge §1.7`) bites in simulation; run `rtf_monitor` alongside.
- **Quality-scaled covariance.** The published variance is
  `var = base_var · (1 + spread_px) · (100 / n_inliers)`: worse spread or fewer inlier
  features → larger variance → the EKF trusts this measurement less. This exact model is
  reused by both estimators in `sauvc_flow_eval`.

### `lane_heading_node.py` — absolute heading from floor lines (Phase 4, §1.10)

- **Subscribes** `/camera_down/image_raw` and `/imu/data`. **Publishes**
  `/heading/pool_relative` (corrected yaw) and `/heading/line_meas` (raw line angle, debug).
- `detect_line_angle` is §1.10 in code: Gaussian blur → Canny → `HoughLinesP` → fold to
  mod-90° via the `×4` circular mean → concentration check `R ≥ 0.6`. It records **why** a
  frame failed (`too_few_lines` vs `low_concentration`) so you can tell "no lines at all" from
  "lines that disagreed".
- `on_imu` provides the fast, drifting yaw; `on_image` provides the slow correction. It picks
  the mod-90 branch nearest the current yaw, applies a **20° sanity gate**, and nudges the
  offset by `gain · error` (a complementary filter).
- **A QoS bug fix worth noting:** Stonefish's camera publishes **BEST_EFFORT**; a default
  **RELIABLE** subscriber literally cannot receive from it (a hard incompatibility, not a soft
  mismatch). It subscribes with `qos_profile_sensor_data` to match.
- **Heavy self-diagnosis:** a 10 s heartbeat reports accepted / gate-rejected / too-few-lines
  / low-concentration counts, and a debug window (`show_detections`) draws **blue** = every
  Canny edge, **green** = surviving Hough segments — so you can distinguish "edges exist but
  are too short/broken (lower `hough_min_line_frac`, raise `hough_max_gap`)" from "Canny finds
  almost nothing (lower the Canny thresholds, or the floor texture is too fine)". Tunable
  params: `canny_low/high`, `hough_threshold`, `hough_min_line_frac`, `hough_max_gap`,
  `min_lines`, `gain`, `pool_axis_offset`.

> **Interface note for `sauvc_flow_eval`:** the eval node consumes this node's
> `/heading/pool_relative`, but it expects the node run with `pool_axis_offset := 0.0` (its
> default) — the eval node does its own ENU→NED conversion, which is a **reflection**, not a
> constant offset. See the eval README.

### `preint_smoother_node.py` — optional GTSAM smoother (Phase 7, §1.11)

- `PreintFusionCore` (pure Python) + a ROS wrapper. **Subscribes** `/imu/data`,
  `/flow/twist`, `/depth`; **publishes** `/odometry/preint` to A/B against the EKF's
  `/odometry/filtered`.
- Per keyframe (default 5 Hz): an **`ImuFactor`** (preintegrated IMU between keyframes), a
  **`BetweenFactor`** on the bias (bias random walk), a **`PriorFactorVector`** on velocity
  from the flow (when fresh, `< 0.3 s` old), and a **`GPSFactor`** on position with **huge
  x/y sigmas** so it constrains **only depth** (the same anisotropic-covariance trick the eval
  package uses for the gate x-correction). Solved incrementally with iSAM2; the graph resets
  every `reset_s` to bound compute over a 15-minute mission.
- The detailed preintegration/gravity maths is in the `sauvc_flow_eval` README, because the
  eval package's `gtsam_estimator.py` is the more complete version of the same technique.

### `offline_flow_test.py` — validate the flow on a recorded video, no ROS

- Runs `FlowVelocityEstimator` on an `.mp4` (+ optional gyro CSV), integrates a top-down path,
  and prints velocity stats + integrated distance. **This is the Phase 3 "5-metre push
  test": integrated distance vs tape measure = your scale error.**
- It correctly uses **`dt_eff`** (`out['dt_eff']`) as the integration step, not the raw frame
  `dt` — because after a reference-hold the recovered velocity spans the *whole* gap
  (§1.6); integrating it over a single-frame `dt` would silently shrink the path. It falls
  back to `dt` for older cores that don't report `dt_eff`. (`offline_flow_test_original.py` is
  the pre-`dt_eff` version — a clean illustration of why that one fix matters.)

### `estimate_covariance.py` — turn logged samples into a variance number

- Feed it a column of stationary sensor samples and it prints the **variance** to paste into
  a covariance field. This is how you honestly populate the noise numbers the EKF and GTSAM
  need, instead of guessing.
- It also handles the **yaw-drift** case, where what you have is a *drift rate* (deg/min)
  rather than a stationary sample: it converts drift over a realistic correction window into
  an equivalent variance (`σ = drift_rate · window`, `var = σ²`), so the number means "how
  much yaw uncertainty accumulates in one filter cycle without a correction". It warns if the
  sample mean is more than 3σ from zero (an un-zeroed bias that must be tared first).

### `imu_covariance_check.py` — diagnose (and optionally patch) the IMU covariance

- Prints the covariance fields actually arriving on `/imu/data` and flags the two failure
  modes: **all-zero** (EKF treats the IMU as perfect and lets it dominate) and **−1 /
  unknown** (EKF falls back to an internal default unrelated to your hardware).
- Optionally **re-publishes** `/imu/data_corrected` with covariance overwritten from the
  numbers you measured with `estimate_covariance.py` — a stopgap when the upstream driver
  can't be configured to emit real covariance.

---

## How this package produces odometry (the one-paragraph summary)

`flow_core` watches the floor, robustly measures how many pixels it slid (median + MAD),
subtracts the sliding caused by turning (derotation), and multiplies by altitude-over-focal
to get a **metric body velocity** — with careful handling so that dropouts, false matches,
and repeating tiles don't corrupt the answer. `flow_velocity_node` wraps that into
`/flow/twist` with a quality-scaled covariance. `lane_heading_node` adds a drift-free absolute
heading from the pool's own lane lines. The optional `preint_smoother_node` fuses all three
(IMU + flow + depth) into a bias-aware estimate that survives dropouts. Everything here runs
identically on the real robot and on the simulator, because the simulator is dressed up to
look like the real robot by `sauvc_sim_bridge`. To *measure how good* all of this is against
ground truth, see `sauvc_flow_eval`.
