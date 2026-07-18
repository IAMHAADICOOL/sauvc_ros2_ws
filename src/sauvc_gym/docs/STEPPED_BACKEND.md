# The stepped backend: what it would take, and what I could not verify

This document exists because the honest answer to "can this env be made fast and
deterministic?" is "yes, but not without code I cannot write blind."

## The problem

`stonefish_ros2`'s standard simulator node parses a scenario, starts free-running
in real time, and talks over topics. That gives the Gym env two structural
defects:

1. **Wall-clock coupling.** `step()` cannot advance the simulator; it can only
   wait for it. Throughput is pinned at real time forever.
2. **No true reset.** There is no teleport, no pause, no rewind. Every reset
   strategy in `backends.py` is a workaround.

Both are fixable, and upstream has already laid the groundwork.

## What upstream provides

Stonefish **v1.5** (the ICRA 2025 release) changelog includes:

- *"Extended application classes to enable manual stepping of simulation"*
- *"Added a test application that shows how to use the library in a
  reinforcement learning setting"*

and the release notes state the change was made "to allow for manual stepping of
the simulation, facilitating integration of Stonefish in the reinforcement
learning research."

So the capability exists. **Read that test application first** — it is in the
`Tests/` directory of the Stonefish source tree, and it is the authoritative
answer to every question below.

The accompanying paper (Grimaldi, Cieślak et al., *Stonefish: Supporting Machine
Learning Research in Marine Robotics*, ICRA 2025, arXiv:2502.11887) is worth
reading in full, because it also reports that going through ROS to reach Gym was
measured to slow training down, and that their fix was direct Python bindings
plus console mode. That is a load-bearing finding for this package — see the
README's *Throughput* section.

## What the node would need to do

Subclass `stonefish_ros2`'s `ROS2SimulationManager` (which already builds the
scenario, publishes sensors and subscribes to setpoints), and add:

| Service | Type | Behaviour |
|---|---|---|
| `/sauvc_sim/reset` | custom | Restore the scenario to its initial state, optionally at a given start pose; return once the world is settled. |
| `/sauvc_sim/step` | custom | Advance exactly N physics ticks, publish the resulting sensor readings, return. |

The Gym env then becomes fully synchronous: `send setpoints -> call step(N) ->
read state`. No sleeps, no real-time factor, no `wait_for_new_state`, and a
`reset` that is exact rather than approximate. `ServiceResetBackend` in
`backends.py` is the client half, stubbed and raising `NotImplementedError`.

## What I could not verify, and will not guess

I do not have your Stonefish build, and I could not confirm these against the
1.6 headers:

1. **The manual-stepping entry point.** The changelog says the *application*
   classes were extended, so the method is likely on `SimulationApp` /
   `GraphicalSimulationApp` / `ConsoleSimulationApp` rather than on
   `SimulationManager`. I do not know its name or signature.
2. **Whether a reset primitive exists at all,** or whether reset must be
   implemented as "tear down the scenario and re-parse it in-process."
3. **How `ROS2SimulationManager` is exposed** — whether its headers are
   installed for downstream subclassing, or whether you would need to vendor the
   node source into your own package.
4. **Threading.** Stonefish owns the render/physics loop; a rclpy service
   callback firing on another thread and touching simulation state is a data
   race unless the manual-stepping API is designed for exactly that. The RL test
   application will show the intended pattern.

Note also that 1.6 is the documented version but **v1.5 is the newest tagged
release** — if you built from `master`, confirm which of these you actually have.

## How to find out

```bash
# Where are the headers?
dpkg -L stonefish 2>/dev/null | grep -i include | head
ls /usr/local/include/Stonefish/core/

# The stepping API — grep the app classes, not the manager
grep -rn "Step\|step" /usr/local/include/Stonefish/core/SimulationApp.h
grep -rn "class.*SimulationApp" /usr/local/include/Stonefish/core/*.h

# Reset / respawn primitives
grep -rn "Restart\|Reset\|Respawn\|Teleport" /usr/local/include/Stonefish/core/*.h \
                                             /usr/local/include/Stonefish/entities/*.h

# The RL test application: the actual answer
find / -path /proc -prune -o -iname "*.cpp" -print 2>/dev/null | xargs grep -ln "SimulationApp" 2>/dev/null | grep -i test

# Is ROS2SimulationManager subclassable?
find / -name "ROS2SimulationManager.h" 2>/dev/null
```

Send me the output of those and I will write the node against your actual API
rather than against a plausible guess at it.

## Whether it is worth it

Only if you are actually going to train. If the Gym wrapper is for evaluating
hand-written controllers, or for a small amount of fine-tuning, the `soft`
backend with 8 parallel headless instances is enough and costs you nothing. The
stepped backend is a day or two of C++ against an API you would have to learn,
and it buys you roughly an order of magnitude. With SAUVC 2026 on the calendar,
that trade is a scheduling question, not a technical one.
