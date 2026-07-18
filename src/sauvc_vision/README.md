# `sauvc_vision`

## `gate_detector_node` — Phase 6 starter

Deliberately simple HSV color-blob detection of SAUVC props on the FORWARD camera,
publishing bearing + rough size for visual servoing and landmark resets. Upgrade to a
small CNN (YOLO on the Jetson) later if pool lighting defeats the thresholds.

**Sub:** `/camera_front/image_raw`.
**Pub:** `/vision/detections` (std_msgs/String, one message per blob:
`"label,bearing_rad,elev_rad,area_frac"`), `/vision/debug_image` (thresholded overlay).

Targets: `red` (gate side / flare), `green` (gate side), `orange` (AVOID flare),
`yellow`, `blue` (comms flares). HSV ranges live in the `COLORS` dict at the top of the
file — TUNE THEM IN YOUR POOL (Phase 6 test 1); red uses two ranges (hue wraps at 180).
Blobs under 0.15% of the image are ignored (`MIN_AREA_FRAC`).

```bash
ros2 run sauvc_vision gate_detector_node
ros2 run sauvc_vision gate_detector_node --ros-args -p hfov_deg:=80.0 -p vfov_deg:=60.0
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `hfov_deg` | double | `80.0` | Horizontal FOV underwater — converts pixel offset → bearing [rad]. |
| `vfov_deg` | double | `60.0` | Vertical FOV underwater — pixel offset → elevation [rad]. |

**Observe:** `ros2 topic echo /vision/detections` while holding a colored prop in view —
bearing ≈ 0 when centered, sign flips left/right of center, `area_frac` grows as you
approach (the gate's known 1.5 m width gives range from pixel width). View
`/vision/debug_image` in rqt_image_view to tune the HSV thresholds.
