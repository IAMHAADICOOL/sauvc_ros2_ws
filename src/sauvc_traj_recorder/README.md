# sauvc_traj_recorder

Records camera, IMU, pressure/depth, and (in sim) ground-truth odometry for one
teleoperated trajectory at a time — a replayable rosbag2 bag plus plain images, an
`.mp4` preview per camera, and CSVs, for building a dataset to test VIO / visual-SLAM
pipelines against.

Works on **both** Stonefish and the real robot: the recorder node itself doesn't know or
care which topics it's reading (they're all parameters), and the two launch files below
supply the right names for each platform.

## What it records

| Stream | Sim topic | Hardware topic | File(s) |
|---|---|---|---|
| Front camera | `/<robot>/camera_front/image_color` | `/camera_front/image_raw` | `camera_front/*.png`, `camera_front.mp4`, `camera_front_index.csv`, `camera_front_info.yaml` |
| Down camera | `/<robot>/camera_down/image_color` | `/camera_down/image_raw` | same, `camera_down_*` |
| IMU | `/<robot>/imu` | `/imu/data` | `imu.csv` |
| Pressure | `/<robot>/pressure` | — (not published on hardware) | `pressure.csv` |
| Depth | — | `/depth` | `depth.csv` |
| Altitude | — | `/altitude` | `altitude.csv` |
| Ground truth | `/<robot>/odometry` | — (doesn't exist on hardware) | `odometry.csv` |

## Output layout

```
<base_dir>/trajectory_01/
    bag/                       # rosbag2 — ros2 bag play <dir>/bag
    camera_front/              # 000000_<t>.png, 000001_<t>.png, ...
    camera_down/
    camera_front.mp4           # preview only — see "On the video files" below
    camera_down.mp4
    camera_front_index.csv     # idx, sec, nanosec, t, filename
    camera_down_index.csv
    camera_front_info.yaml     # intrinsics snapshot (captured once)
    camera_down_info.yaml
    imu.csv                    # idx, sec, nanosec, t, qx,qy,qz,qw, wx,wy,wz, ax,ay,az
    pressure.csv                                                            [sim]
    depth.csv / altitude.csv                                                [hardware]
    odometry.csv               # idx, sec, nanosec, t, x,y,z, qx,qy,qz,qw, vx,vy,vz,wx,wy,wz  [sim]
    odometry_anchored.csv      # same, x/y/z shifted to start at (0,0,0)               [sim]
    meta.yaml                  # counts, rates, and a per-stream sync/gap report
```

## Ground truth vs. a downstream VIO/SLAM estimate — the spawn-offset problem

`odometry.csv` is **absolute world position**: it starts wherever the vehicle spawned
(`start_position` in the `.scn`, e.g. `-12.1 0 0.3`), not at the origin. Whatever you run
downstream — this project's own EKF/GTSAM estimators, or a real VIO/SLAM pipeline —
starts at *its own* origin. Compare the two raw and you get a large constant "error"
that's just a frame mismatch, not estimator drift.

`odometry.csv` is deliberately left untouched — that's what ground truth should mean.
Instead:

- **`odometry_anchored.csv`** — a convenience copy with x/y/z shifted so the trajectory
  starts at `(0,0,0)` (orientation/velocity untouched). This is a plain *translation*,
  matching exactly what `flow_eval_node._anchor` does for this project's own AHRS-
  referenced EKF/GTSAM estimators, whose axes already match ground truth and only the
  origin differs. **Not sufficient for a general VIO/SLAM pipeline** — its frame is
  usually rotated relative to this one too (IMU-aided pipelines align gravity but not
  heading; monocular aligns nothing), and monocular pipelines have no absolute scale
  either. A first-pose-only offset can't fix rotation or scale, and is fragile to a
  single noisy first pose besides.

- **`trajectory_tools/`** — for that general case, the standard practice (what `evo`, the
  TUM RGB-D benchmark, and the KITTI devkit all do) is a rigid or similarity transform
  fit by **least squares over the whole trajectory**:

  ```bash
  # convert this package's ground truth to TUM format (most VIO/SLAM tools export
  # TUM directly for their own side — ORB-SLAM3, VINS-Fusion, OpenVINS, ...)
  ros2 run sauvc_traj_recorder csv_to_tum odometry.csv gt.tum

  # SE(3) alignment (rotation + translation, scale=1) — stereo/RGB-D/VIO-with-IMU
  ros2 run sauvc_traj_recorder align_and_evaluate gt.tum est.tum --align

  # Sim(3) alignment (+ scale) — monocular-only VO/SLAM, which has no absolute scale
  ros2 run sauvc_traj_recorder align_and_evaluate gt.tum est.tum \
      --align-scale --plot ate.png
  ```

  Both also run standalone without a build (pure numpy, plus matplotlib only if you
  pass `--plot` — no ROS or GTSAM needed), by calling the script path directly:
  `python3 sauvc_traj_recorder/trajectory_tools/csv_to_tum.py odometry.csv gt.tum`.

  For anything more rigorous than a quick per-trajectory sanity check (large datasets,
  numbers you're going to publish), reach for
  [`evo`](https://github.com/MichaelGrupp/evo) (`pip install evo`) instead —
  `evo_ape tum gt.tum est.tum -a` / `-as`. `align_and_evaluate.py` implements the same
  Umeyama alignment `evo` uses, kept minimal on purpose so it's easy to read and trust,
  not to replace `evo`'s more careful association and richer metrics (RPE, plots, etc.).

## On synchronization

Every row is timestamped from the message's own `header.stamp`, never local arrival
time. Stonefish (and, as long as nothing sets `use_sim_time`, the real driver stack too)
stamps every sensor with the *same wall clock*, so every stream here already shares one
time base — that's what makes offline association possible.

Streams are **not** forced into lockstep (no `ApproximateTimeSynchronizer`). Cameras,
IMU, and pressure/depth arrive at different, non-integer rates on purpose — a VIO
pipeline wants that raw per-sensor timing, not something resampled and quietly missing
frames that didn't line up. Pair frames with the nearest IMU/odometry sample offline:

```python
import pandas as pd
imu = pd.read_csv('imu.csv').sort_values('t')
frames = pd.read_csv('camera_down_index.csv').sort_values('t')
merged = pd.merge_asof(frames, imu, on='t', direction='nearest')
```

To catch the things that WOULD break a VIO run — dropped messages, out-of-order
delivery, a sensor that started late or stopped early — every stream is tracked live
(count, time span, average rate, largest gap, out-of-order count) and reported in
`meta.yaml`, including the **overlap window**: the time range where every active stream
actually has data. Trim to that window before feeding a SLAM pipeline. Anything that
looks off (a big gap, an out-of-order timestamp) is also logged as a warning the moment
it's recorded, not just buried in the final file.

## On the video files

`camera_front.mp4` / `camera_down.mp4` are a **quick-look preview only**. A video
container needs one fixed fps, but the real capture rate is irregular (rendering-cost
dependent in sim, whatever the driver delivers on hardware), so playback speed drifts
from real time. Every frame in the video is the same frame saved to `camera_*/`, in the
same order — good for scrubbing through a take to see what happened, not for anything
quantitative. For VIO/SLAM, use the per-frame images and their index CSV.

## Usage

Build and source as usual (`colcon build --packages-select sauvc_traj_recorder`, then
`source install/setup.bash`). Bring up the scenario/hardware drivers and your teleop
separately, then run the recorder once per take.

### Simulation

```bash
ros2 launch sauvc_traj_recorder record_trajectory_sim.launch.py
```

Works standalone against any `sauvc_stonefish` scenario — no `sim_drivers` or shims
needed, since it reads Stonefish's own topics directly. Includes ground-truth odometry.

### Real robot

```bash
ros2 launch sauvc_traj_recorder record_trajectory_hw.launch.py
```

Reads the topics the real driver stack publishes (`/camera_front/image_raw`,
`/imu/data`, `/depth`, `/altitude`). No ground truth — none exists on hardware.

### Either way

Drive the vehicle around, then `Ctrl+C` when the take is done. Relaunch for the next
one — the trajectory number auto-increments (`trajectory_01`, `trajectory_02`, ...) by
scanning `base_dir`, so you never have to pass anything for a normal ~10-trajectory
session.

### Arguments (both launch files)

| Arg | Default | Meaning |
|---|---|---|
| `base_dir` | `~/sauvc_data/trajectories` | parent folder for `trajectory_NN/` |
| `trajectory_id` | *(auto)* | force a specific number, e.g. `trajectory_id:=5` |
| `image_format` | `png` | `png` (lossless) or `jpg` (smaller, lossy) |
| `video_fps` | `10.0` | nominal fps for the preview `.mp4` (see above) |
| `record_bag` | `true` | set `false` to skip the rosbag2 copy |

`record_trajectory_sim.launch.py` additionally takes `robot_name` (default `sauvc_auv`,
must match the `.scn`).

## Notes

- If `trajectory_id` points at a folder that already exists: `ros2 bag record` refuses
  to write into an existing bag directory and exits immediately (rosbag2 doesn't
  append), while the recorder node overwrites the CSVs/images there and restarts
  numbering from `000000`. Treat an explicit `trajectory_id` as "start fresh at this
  number," not "resume." Auto-increment (the default) always picks an unused number.
- Hardware assumes your camera driver publishes `camera_info` alongside `image_raw`
  under the standard `<ns>/camera_info` convention. If it doesn't, you'll just get
  images without an intrinsics snapshot — nothing else is affected.
- No cam-IMU extrinsics or hand-eye calibration are recorded here — that's a separate
  step (e.g. Kalibr) outside this package's scope.
