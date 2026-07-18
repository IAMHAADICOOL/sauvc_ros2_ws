# `sauvc_drivers`

Hardware sensor drivers. Currently one node.

## `depth_altitude_node`

Reads a Bluerobotics MS5837 (Bar30/Bar02) over I2C, publishes depth and altitude.
Phase 1 of the pipeline. Requires `pip install ms5837`.

**Publishes:** `/depth` (geometry_msgs/PoseWithCovarianceStamped, `pose.position.z = -depth`),
`/altitude` (std_msgs/Float32, camera... vehicle height above the floor).
**Subscribes:** `/odometry/filtered` (nav_msgs/Odometry) — supplies the along-pool x for
the V-floor profile lookup.

### Run

```bash
ros2 run sauvc_drivers depth_altitude_node
ros2 run sauvc_drivers depth_altitude_node --ros-args \
    -p use_floor_profile:=false -p pool_depth:=2.0 -p i2c_bus:=1 -p sensor_model:=bar30
```
**Start it with the vehicle floating at the surface** — the first 3 s of readings are
averaged as the surface-zero reference.

### Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `use_floor_profile` | bool | `true` | `true` = SAUVC mode: altitude from the V-floor profile evaluated at the EKF's x. `false` = flat practice-pool mode: altitude = `pool_depth − depth`. |
| `pool_depth` | double | `1.4` | Constant floor depth [m] used whenever the profile is OFF (and as fallback if profile arrays are empty). Set to YOUR practice pool's measured depth. |
| `floor_profile_x` | double[] | `[0.0, 12.5, 25.0]` | Along-pool breakpoints [m] (wall-referenced; rulebook side view — confirm at venue, ±5% tolerance). |
| `floor_profile_depth` | double[] | `[1.2, 1.6, 1.2]` | Floor depth at each breakpoint [m] (2026 qualification V-profile). Must be same length as `floor_profile_x`. |
| `fluid_density` | double | `997.0` | kg/m³ — 997 freshwater (SAUVC), 1029 salt. |
| `i2c_bus` | int | `1` | Jetson I2C bus (`i2cdetect -l`; often 1 or 7/8). |
| `sensor_model` | string | `bar30` | `bar30` or `bar02` — MUST match the actual board. |
| `rate_hz` | double | `20.0` | Publish rate. |
| `depth_var` | double | `0.0004` | Depth measurement variance [m²] fed to the EKF. **PLACEHOLDER** — replace with (your Phase-1 stationary-hold std)². Datasheet "Resolution" is quantization, "Absolute Accuracy" is offset (mostly cancelled by surface zeroing) — neither is this number. |

### Observe

- Startup: `altitude source: PROFILE mode (V-floor, 3 breakpoints)` (or FLAT mode) and
  the surface-zero completing.
- `ros2 topic echo /depth` ≈ 0 at surface, ≈ −(actual depth) submerged; `/altitude`
  ≈ pool depth at surface, shrinking as you dive.
- A `use_floor_profile=true but the profile is empty` warning means bad profile params.
- If init throws `MS5837 init failed` — wrong `i2c_bus` or wiring; verify with the
  pipeline-independent `sauvc_sensor_check pressure_check` first.
