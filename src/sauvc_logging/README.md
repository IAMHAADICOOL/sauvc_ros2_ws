# `sauvc_logging`

## `csv_logger_node`

Automatic CSV logging so every pool session leaves a data trail — no manual
`ros2 topic echo --csv` + timed Ctrl-C. Subscribes to a configurable topic list, writes
ONE CSV per topic under a timestamped run directory
(`<out_dir>/<run_name>_<YYYYmmdd_HHMMSS>/<topic>.csv`), and on Ctrl-C prints per-column
mean/std/variance for the whole run — the numbers the phase procedures ask for.

Normally you don't run it standalone: every `phaseN` launch in `sauvc_bringup` already
includes it with a phase-appropriate topic list and `run_name`.

```bash
ros2 run sauvc_logging csv_logger_node --ros-args \
    -p topics:="['/depth','/altitude']" -p run_name:=phase1_depth -p out_dir:=~/sauvc_logs
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `topics` | string[] | every registry topic | Which topics to log. Must be in the TOPIC_REGISTRY (message type + column extractor): `/imu/data`, `/imu/data_corrected`, `/depth`, `/altitude`, `/flow/twist`, `/heading/pool_relative`, `/heading/line_meas`, `/odometry/filtered`, `/odometry/preint`, `/vision/detections`. Add new topics by adding a registry entry. |
| `out_dir` | string | `~/sauvc_logs` | Root output directory. |
| `run_name` | string | `run` | Prefix of the timestamped run directory. |

**Observe:** startup log names the run directory; one row lands per message (check file
growth); Ctrl-C prints the mean/std/variance table per numeric column. Use
`estimate_covariance.py` (sauvc_localization) for combining multiple saved CSVs later.
