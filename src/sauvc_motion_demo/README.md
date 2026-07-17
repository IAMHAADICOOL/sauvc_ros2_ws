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
