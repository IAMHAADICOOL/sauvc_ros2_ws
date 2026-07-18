# sauvc_gym

A Gymnasium interface to the SAUVC 2026 Stonefish arena.

It sits on top of `sauvc_stonefish` and reuses its scenario, its topics and its
thruster wiring unchanged. Nothing in your existing workspace needs to move.

```
sauvc_gym/
├── sauvc_gym/
│   ├── scn_parse.py       reads thruster geometry out of your .scn
│   ├── allocation.py      wrench command -> 8 setpoints
│   ├── ros_link.py        the ROS 2 node and state snapshot
│   ├── backends.py        the three ways to fake a reset
│   ├── wrappers.py        slew limiting, safety shield, episode stats
│   ├── envs/              base env + station keeping + qualification gate
│   └── scripts/           verify, identify, random agent, PPO baseline
├── docs/STEPPED_BACKEND.md   the fast path, and what I couldn't verify
├── config/station_keeping.yaml
└── test/                  15 tests, no ROS needed
```

---

## Read this first: the throughput ceiling

This env talks to a free-running, real-time simulator over ROS topics. At the
default 10 Hz that is **~36,000 steps per hour per instance**, and no amount of
tuning in this package changes it — the bound is Stonefish's clock.

PPO on station keeping wants roughly 1–5M steps. Single instance: **30–140
hours**. That is the number to plan around.

The mitigations, cheapest first:

1. **Run 8 headless instances on separate `ROS_DOMAIN_ID`s** with
   `scripts/make_vec_env.py`. ~290k steps/h. This is the practical answer and
   why your existing headless build matters.
2. **Strip cameras from the scenario** for control tasks. They dominate the
   per-step cost and a station-keeping policy never reads them.
3. **Build the stepped backend** (`docs/STEPPED_BACKEND.md`) and break the
   real-time coupling entirely.

Worth knowing before you commit: the Stonefish authors hit this exact wall. Their
ICRA 2025 paper reports that connecting Gym to Stonefish *through ROS* was
measured to slow training, and their answer was to bypass ROS with direct Python
bindings plus console mode. This package takes the ROS route deliberately —
it reuses your working arena, bridge and sensor wiring, and it runs today — but
it is the slow road. Go in knowing that.

---

## The two design decisions worth arguing about

### 1. The action is a wrench, not eight thruster setpoints

`action = [surge, sway, heave, yaw]`, each in `[-1, 1]`. The allocator turns it
into 8 setpoints.

Raw thruster control would be over-parameterised (8 thrusters spanning a 4-D
useful subspace), would make the policy rediscover geometry that is already
written down in your `.scn`, would let it silently learn around a sign error, and
would produce a policy with no clean mapping onto ArduSub — which takes a
wrench-like command and does its own allocation. The wrench-level action keeps
the policy in a low-dimensional, physically meaningful, hardware-portable space.

Two details a naive allocator gets wrong, both handled here and both unit-tested:

- **The setpoint law is quadratic.** `T = Kt·ω·|ω|` and `ω = u·ω_max`, so
  `T = T_max·u·|u|`. Half thrust needs `u = 0.707`, not `0.5`.
- **Saturation scales, it does not clip.** Clipping one thruster rotates the
  delivered wrench — ask for hard surge, get surge plus a yaw you never
  commanded. The whole group is scaled by a common factor instead.

The allocation matrix is **derived from your scene file**, not hardcoded. Move a
thruster in the `.scn` and the allocation follows. Plant parameters stay in the
plant.

### 2. Reward reads ground truth. Observation does not.

The sim publishes perfect pose on `/sauvc_auv/odometry`. Feeding it to the policy
would train an agent that cannot exist: the real vehicle has a Bar30, an HFI-A9
and two cameras, and its best estimate is whatever the EKF produces — drifting
in x/y, decent in depth and heading.

So `_observation()` may use only what the hardware can produce (depth, attitude,
rates, body velocities, previous action). `_reward()` and `_terminated()` may use
anything. Ground truth is in `info`, for logging and analysis, never for the
policy.

The gate env is the honest exception — it hands the policy the gate's bearing,
which the real vehicle gets from a detector you have not built. Its docstring
says so, and `gate_observation_source="noisy"` exists so you can degrade it with
your detector's real error statistics rather than train against perfect knowledge.

---

## Setup

```bash
cd ~/Robotics_Job/sauvc_ws/src
# drop sauvc_gym/ here
python3 -m pip install "gymnasium>=1.0" --user      # not a rosdep key
cd ~/Robotics_Job/sauvc_ws
colcon build --symlink-install --packages-select sauvc_gym
source install/setup.bash
```

`--symlink-install` matters: you will be editing reward weights constantly and
you do not want to rebuild each time.

## Verify before you train

Three checks, in order. Each catches a class of bug the next one cannot.

```bash
# 1. Offline. No sim, no ROS. Is the geometry self-consistent?
python3 -m sauvc_gym.scripts.verify_allocation \
    --scn ~/Robotics_Job/sauvc_ws/src/sauvc_stonefish/scenarios/my_auv.scn

# 2. Against the running sim. Do the thrusters actually push that way?
ros2 launch sauvc_stonefish sauvc_qualification.launch.py
python3 -m sauvc_gym.scripts.identify_allocation --scn .../my_auv.scn

# 3. End to end. Does the loop close, and does the sim keep up?
python3 -m sauvc_gym.scripts.random_agent --scn .../my_auv.scn --zero
```

Step 2 is not optional. `verify_allocation` reads the `.scn` and applies a
*documented* rule for how `right` and `inverted_setpoint` interact — that rule is
an assumption until measured. `identify_allocation` pulses each thruster alone
and checks what the vehicle actually does. This project has already been bitten
once by a left-handed-propeller sign, and by a thrust coefficient wrong by three
orders of magnitude. Both were found by running an experiment.

Run `--zero` first: with zero setpoints a correctly trimmed vehicle should hang
still or rise very slowly. If it accelerates anywhere, fix buoyancy before RL.

## Use

```python
import gymnasium as gym
import sauvc_gym

env = gym.make("SauvcStationKeeping-v0", vehicle_scn=".../my_auv.scn")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

| id | action | what it is |
|---|---|---|
| `SauvcDepthHold-v0` | `[heave]` | Start here. Compare against your existing depth PID. |
| `SauvcStationKeeping-v0` | `[surge, sway, heave, yaw]` | The workhorse. |
| `SauvcQualGate-v0` | `[surge, sway, heave, yaw]` | Gate out and back. Read the docstring. |

Do not touch the gate task until depth hold beats the PID baseline in
`sauvc_motion_demo`. If a learned policy cannot match a PID on a problem a PID
solves, the bug is in this env, and you do not want to be debugging two things.

## Three things to confirm against your build

These are the places where I made a defensible choice that your workspace can
overrule. All three are config, not hardcoded.

1. **`odom_twist_frame`** (default `"world"`). `nav_msgs/Odometry` nominally
   reports twist in `child_frame_id`, but simulators vary, and this project has
   already resolved this once — the answer is in your working
   `ardusub_json_bridge.py`. Copy it from there. If it is wrong the body
   velocities in the observation are silently rotated garbage whenever yaw ≠ 0,
   which trains fine at yaw 0 and fails everywhere else.
2. **Thrust-model tag names.** `scn_parse.py` accepts several spellings for the
   coefficient (`thrust_model/@thrust_coeff`, `@coeff`, `@kt`, a child element).
   If your `.scn` uses something else it raises rather than defaulting. Fix the
   parser, not the scene.
3. **`max_omega`** (default 314 rad/s). If your scenario declares it, it is read;
   otherwise the default gives ~49 N at `Kt = 0.0005`, matching your empirical
   result. `verify_allocation` flags anything outside 20–120 N.

## Known limits

- **`soft` reset is approximate.** It flies the vehicle home with a PD
  controller. Residual water motion carries between episodes and knocked-over
  props stay knocked over. Fine for control tasks; not fine for Task 4. Use
  `reset_mode="relaunch"` for anything where props move, and periodically anyway
  to confirm `soft` is not corrupting your episodes.
- **The gate env's observation is not deployable.** By design, and documented.
- **No domain randomisation of the plant.** Buoyancy, drag and thrust
  coefficients live in the `.scn` and this package will not reach in and rewrite
  them — that would put plant parameters in the controller, which is exactly the
  split you established. Randomise by generating scenario variants and passing
  different files per worker.
- **USBL is untouched.** It segfaults in your build; nothing here uses it.
- **No stepped backend.** See `docs/STEPPED_BACKEND.md` for why, and for what I
  need from your headers to write it.

## Tests

```bash
python3 -m pytest test/ -q      # 15 tests, no rclpy, no simulator
```

They cover the allocation maths, the quadratic setpoint law, the left-handed
propeller sign logic, axis decoupling, direction-preserving saturation, and the
V-floor profile. `test/fixtures/vehicle_example.scn` is a **fixture**, not a
delivery — its thruster angles are this package's own assumption. Point the tests
at your real vehicle scenario once you are happy with it.
