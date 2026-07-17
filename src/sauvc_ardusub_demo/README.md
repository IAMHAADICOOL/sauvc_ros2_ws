# `sauvc_ardusub_demo`

The **real firmware path**: ArduSub SITL flies the vehicle over MAVLink, exactly as the Pixhawk
will on the physical AUV. Requires **sim + bridge + SITL** running (root README §6).

| Node | Purpose |
|---|---|
| `motor_map_check` | Discovers which ArduSub motor drives which Stonefish thruster, and in which direction. **Run this first.** |
| `ardusub_mission` | Dive to a depth setpoint, hold, translate on four axes, surface — all through ArduSub. |

---

## `motor_map_check`

```bash
ros2 run sauvc_ardusub_demo motor_map_check [--ros-args -p robot:=sauvc_auv]
```

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `robot` | string | `sauvc_auv` | Topic namespace for `/thruster_state`. |

**What it does:** reads `FRAME_CONFIG` (must be `2`), temporarily sets `ARMING_CHECK=0` (restored
at the end), then runs ArduSub's own `MOTOR_TEST` on motors 1–8 **while disarmed** (ArduPilot
rejects motor test when already armed), watching `/sauvc_auv/thruster_state` to see which thruster
responds and with what thrust sign. Prints paste-ready constants. Run with the vehicle floating
free and the sim window visible.

**Observe (success):**
```
FRAME_CONFIG = 2.0
ARMING_CHECK = 3.0 -> setting 0 for the mapping session
ArduSub motor 1: -> Stonefish [1] HFS response +12.3
...
================ RESULT ================
MOTOR_MAP  = [1, 0, 3, 2, 5, 4, 7, 6]
MOTOR_SIGN = [-1, -1, 1, 1, 1, 1, 1, 1]
```
Paste both lines into the top of `sauvc_stonefish/scripts/ardusub_json_bridge.py`, restart the bridge.

**Observe (diagnostics):**

| Output | Meaning |
|---|---|
| `FRAME_CONFIG` ≠ 2 warning | 6-motor frame — root README §9. |
| `MOTOR_TEST FAILED` on every motor | Arming checks refused the internal arm. The tool sets `ARMING_CHECK=0` itself; if it also prints *"could not set ARMING_CHECK"*, SITL isn't receiving physics from the bridge. |
| `ACK ACCEPTED but NO thruster responded` | That motor channel is unused by the current frame (wrong `FRAME_CONFIG`) or the bridge isn't running. |
| `duplicate assignments` | Two motors drove the same thruster — check the frame and rerun. |
| **Fallback: ARMED GROUP TEST** | Motor test never ran, so it arms in MANUAL and commands one pilot axis at a time, printing which thrusters respond. Doesn't isolate single motors but validates the map at group level: `forward` → only HFP/HFS/HAP/HAS same-sign; `throttle` → only VFP/VFS/VAP/VAS same-sign; `yaw` → 0–3 split diagonally. |

---

## `ardusub_mission`

```bash
ros2 run sauvc_ardusub_demo ardusub_mission [tcp:127.0.0.1:5762]
```

CLI argument (not `--ros-args`): MAVLink URL. **Default `tcp:127.0.0.1:5762`** — SITL's spare
serial port, a private loss-free link. The shared UDP `:14550` silently loses telemetry streams
when a second client binds it (that caused `EKF depth = nan` with attitude frozen at 0).

**Sequence:** connect → raise `SR2_*` stream rates (SERIAL2 defaults to 0) → arm → confirmed mode
switch to `ALT_HOLD` → settle → send `SET_POSITION_TARGET_GLOBAL_INT` depth setpoint at 10 Hz
until reached → hold → forward/right/left/backward via `MANUAL_CONTROL` → surface (target 0.1 m)
→ disarm. Sends GCS heartbeats at 1 Hz throughout (failsafe requirement).

**Tuning constants (top of the file):**

| Constant | Default | Effect |
|---|---|---|
| `TARGET_DEPTH` | `1.0` | Depth to reach and hold [m]. |
| `DEPTH_TOL` | `0.10` | "Reached" tolerance [m]. |
| `MAX_DEPTH` | `1.15` | Abort guard [m] — floor at the start zone is ~1.23 m. |
| `DIVE_TIMEOUT` | `30.0` | Give up if the depth target isn't reached [s]. |
| `NEUTRAL_Z` | `500` | Neutral throttle (Sub convention: 0–1000, 500 = neutral). |
| `FWD` | `400` | Leg intensity for `MANUAL_CONTROL` x/y ∈ [−1000, 1000]. |
| `LEG_TIME` / `HOLD_TIME` | `5.0` | Leg duration / hold duration [s]. |
| `TOPPLE_DEG` | `60.0` | Abort if |roll| or |pitch| exceeds this [deg]. |

**Firmware-side depth tuning** (`param set` in the SITL console — these are the knobs that matter):

| Parameter | Effect |
|---|---|
| `PSC_VELZ_I` | **Raise if depth slowly sags/rises in hold** — the integrator absorbs net buoyancy. |
| `PSC_POSZ_P` | Depth error → climb rate. Lower if depth oscillates. |
| `PSC_VELZ_P` | Climb-rate loop gain. Lower if oscillating. |
| `PSC_ACCZ_P` / `PSC_ACCZ_I` | Acceleration → throttle. |
| `PILOT_SPEED_UP` / `PILOT_SPEED_DN` | Max pilot climb/descend rate [cm/s]. |

**Observe:**
```
connected (sys 1)
ARMED
mode -> ALT_HOLD (confirmed)
  surface pressure reference: 101325 Pa
  [settle   ] EKF depth=+0.08 m  rpy=( +0.0, +0.0,  -7.5) cmd(...)
  telemetry types seen: ['ATTITUDE', 'GLOBAL_POSITION_INT', 'SCALED_PRESSURE2', ...]
  [dive     ] EKF depth=+0.62 m  target=1.0
  reached 1.00 m
  [hold     ] EKF depth=+1.00 m  ...
```

| Output | Meaning |
|---|---|
| `EKF depth` sane and stable at target | Working. |
| `no telemetry streams on this link` + the seen-types list | Streams aren't arriving; the list tells you what *is*. |
| `EKF depth` disagrees with the bridge's `state: depth=` | **State-feed bug** — the bridge's frames/timing. |
| `EKF depth` agrees but behaviour is wrong | Control/mapping — rerun `motor_map_check`, then tune `PSC_*`. |
| `MISSION ABORTED: TOPPLE ...` / `DEPTH GUARD ...` | Watchdog fired; compare the bridge log at that moment. |

**Depth semantics — the trap that cost the most time:** depth is read from `SCALED_PRESSURE2`
(ArduSub's water-pressure sensor), auto-zeroed on the first sample. `VFR_HUD.alt` is **absolute
AMSL** and unusable as depth. Likewise the setpoint uses `MAV_FRAME_GLOBAL_RELATIVE_ALT_INT`:
with `MAV_FRAME_GLOBAL_INT`, `alt = −1.0` means 1 m below **sea level**, so at SITL's default home
(~584 m AMSL) it commanded a 585 m dive and drove the vehicle into the floor. Launch SITL with
`-l 18.25,109.5,0,0` to keep home at sea level.
