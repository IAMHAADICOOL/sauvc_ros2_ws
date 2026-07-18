# `sauvc_mission`

## `mission_node` — Phase 6+ state machine (skeleton)

SAUVC finals sequence with every state stubbed and its entry/exit conditions in
comments — fill in as phases come online; keep states small and pool-testable.

**States:** IDLE → DIVE → TRANSIT_GATE (→ SIDESTEP on orange flare) → SERVO_GATE →
RESET_POSE → GOTO_DRUMS → DROP_BALL → RECROSS_GATE → FLARES → SURFACE → DONE.
10 Hz tick timer.

**Sub:** `/odometry/filtered`, `/vision/detections`, `/altitude`, `/heading/pool_relative`.
**Pub:** `/cmd/setpoint` (geometry_msgs/Twist — body velocities + yaw rate; wire into
YOUR thruster mixer), `/pose_correction` (PoseWithCovarianceStamped — anisotropic
landmark correction fed to robot_localization as `pose1`; per-axis covariance controls
how much each axis moves, never a hard teleport like `/set_pose`).

```bash
ros2 run sauvc_mission mission_node
ros2 run sauvc_mission mission_node --ros-args -p cruise_depth:=1.0 -p gate_distance:=16.0 \
    -p cruise_speed:=0.4 -p flare_order:=R-B-Y
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `cruise_depth` | double | `1.0` | m. The finals gate sits ON the V-floor at x≈16 m (floor ≈1.49 m deep), so the 100 cm gate's midpoint is ≈1.0 m below the surface — NOT the ~0.5–0.7 m a flat-floor reading suggests. Recompute if the venue profile differs. |
| `gate_distance` | double | `16.0` | m from start zone along the pool axis (rulebook). |
| `cruise_speed` | double | `0.4` | m/s dead-reckon transit speed. |
| `flare_order` | string | `R-B-Y` | Flare visit order, set via team comms after the gate. |

**Observe:** `STATE -> NEWSTATE` log lines in sequence; `remembered <label> at odom (x, y)`
when a prop is first sighted; `/cmd/setpoint` publishing during motion states.
