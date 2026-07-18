# sauvc_teleop

Keyboard teleop for the 4 DOF the vehicle actually has: surge, sway, yaw, and depth.
Publishes `/cmd/setpoint` — the exact topic `mission_node` publishes — so it drives
whichever controller is running underneath without either side knowing about the other.

## Only 4 DOF — not 6

The vehicle's mixer (`control_core.ThrusterMixer`) actively controls **Fx, Fy, Fz, Mz**
(surge, sway, heave, yaw) and deliberately leaves roll and pitch to passive stability —
8 thrusters can't independently command all 6 DOF without coupling axes nothing here has
clean feedback on. There is no roll/pitch key because there is nothing for one to
command; a "6-DOF teleop" for this vehicle would be four working sticks and two that
silently do nothing, which is worse than not having them.

## Depth hold — the actual feature

Two `depth_mode`s, one per control path:

- **`absolute`** (Path A, `direct_control_node`): `r`/`f` nudge a *stored* depth target.
  It's republished unchanged every tick — `direct_control_node`'s own depth PID
  (`cmd_z_is_depth:=true`) does the holding. This is the literal "stay where I leave the
  depth key": the target doesn't move until you press `r`/`f` again.
- **`pulse`** (Path B, `ardusub_setpoint_node` + ArduSub SITL in `ALT_HOLD`): `r`/`f`
  send a brief deflection, then auto-return to neutral. **This node does not hold depth
  in pulse mode** — ArduSub's own autopilot does, once the signal is back at neutral.
  That's the same thing a safety pilot does: tap, let go, the autopilot holds.

## Run it — in its own terminal, separately from the launch file

Raw-terminal keystroke reading only works cleanly when the node owns the TTY. Under
`ros2 launch` sharing a console with other nodes, stdin isn't reliably forwarded and
other nodes' log lines will scribble over the status line. So each path is two terminals:

```bash
# Path A — direct PID, no ArduSub
# Terminal 1:
ros2 launch sauvc_teleop teleop_direct.launch.py
# Terminal 2:
ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=absolute
```

```bash
# Path B — through ArduSub SITL
# (start ArduSub SITL + ardusub_json_bridge.py first, per the sim README)
# Terminal 1:
ros2 launch sauvc_teleop teleop_ardusub.launch.py
# Terminal 2:
ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=pulse
```

## Keys

```
  MOVEMENT (persists until changed)        DEPTH
    w/s : surge +/-                          r : shallower (up)
    a/d : sway  left/right                   f : deeper    (down)
    q/e : yaw   left/right (CCW/CW)          0 : surface [absolute mode only]

  space : zero surge/sway/yaw (depth hold is unaffected)
  x     : FULL STOP — zero surge/sway/yaw; depth hold continues unchanged
  +/-   : bigger/smaller surge & sway step    [ / ] : bigger/smaller depth step
  CTRL-C: quit (publishes neutral on the way out)
```

`a`/`d` and `q`/`e` signs come from REP-103 body FLU (`x` forward, `y` **left**, `z` up),
matching `/cmd/setpoint` throughout the stack: strafing right is *negative* y, turning
right (CW) is *negative* yaw rate. Verified in `test/test_teleop_core.py` before writing
the node, not after.

## Why the seed-then-hold logic on startup

Before the first `/depth` message arrives, `depth_target` is `None` and (in `absolute`
mode) nothing is published at all — publishing `z=0.0` before knowing the real depth
would command an immediate lurch toward the surface. Once seeded, later `/depth`
messages never overwrite the target again — only `r`/`f`/`0` change it — so the EKF's own
depth noise can't fight your manual setting.

## Architecture

`teleop_core.py` is pure Python (no ROS, no termios): `TeleopState`, `apply_key()`,
`command_twist()`. `keyboard_teleop_node.py` is a thin ROS + termios wrapper around it.
Same split as `flow_core.py`/`flow_velocity_node.py` and `control_core.py` — the logic is
unit-testable without a real TTY or a running sim.

```bash
python3 -m pytest test/test_teleop_core.py -v     # or: PYTHONPATH=. python3 test/test_teleop_core.py
```
28 checks: sign conventions, clamping, seed-once semantics, both depth modes, pulse
auto-return-to-neutral, `x`/space behavior, step-size scaling.
