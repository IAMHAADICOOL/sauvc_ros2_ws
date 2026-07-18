import sys
sys.path.insert(0, '.')
from sauvc_teleop.teleop_core import TeleopState, TeleopLimits, apply_key, seed_depth, command_twist

ok = True
def check(name, cond):
    global ok; ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

lim = TeleopLimits(max_surge=0.6, max_sway=0.6, max_yaw=1.2, min_depth=0.0, max_depth=1.5)

print("Surge/sway/yaw signs and clamping")
s = TeleopState()
apply_key(s, 'w', lim, 'absolute', 0.0, 0.6)
check("w increases surge", s.surge > 0)
for _ in range(20): apply_key(s, 'w', lim, 'absolute', 0.0, 0.6)
check("surge clamps at max_surge", abs(s.surge - lim.max_surge) < 1e-9)

s = TeleopState()
apply_key(s, 'a', lim, 'absolute', 0.0, 0.6)
check("a (strafe LEFT) is POSITIVE sway (FLU: +y=left)", s.sway > 0)
s = TeleopState()
apply_key(s, 'd', lim, 'absolute', 0.0, 0.6)
check("d (strafe RIGHT) is NEGATIVE sway", s.sway < 0)

s = TeleopState()
apply_key(s, 'q', lim, 'absolute', 0.0, 0.6)
check("q (turn LEFT/CCW) is POSITIVE yaw", s.yaw > 0)
s = TeleopState()
apply_key(s, 'e', lim, 'absolute', 0.0, 0.6)
check("e (turn RIGHT/CW) is NEGATIVE yaw", s.yaw < 0)

print("\nspace zeroes translation/rotation, depth untouched")
s = TeleopState(surge=0.3, sway=0.2, yaw=0.5, depth_target=0.8)
apply_key(s, ' ', lim, 'absolute', 0.0, 0.6)
check("surge/sway/yaw zeroed", (s.surge, s.sway, s.yaw) == (0.0, 0.0, 0.0))
check("depth_target untouched", s.depth_target == 0.8)

print("\nseed_depth seeds ONCE, never overwrites afterward")
s = TeleopState()
seed_depth(s, 0.42)
check("first seed sets target", s.depth_target == 0.42)
apply_key(s, 'f', lim, 'absolute', 0.0, 0.6)     # user nudges deeper
target_after_nudge = s.depth_target
seed_depth(s, 0.99)                              # a later /depth message arrives
check("later seed does NOT override the user's nudge",
      s.depth_target == target_after_nudge and s.depth_target != 0.99)

print("\nabsolute mode: r/f adjust and clamp depth_target")
s = TeleopState(depth_target=0.5, depth_step=0.1)
apply_key(s, 'f', lim, 'absolute', 0.0, 0.6)     # deeper
check("f increases depth_target", s.depth_target > 0.5)
apply_key(s, 'r', lim, 'absolute', 0.0, 0.6)
apply_key(s, 'r', lim, 'absolute', 0.0, 0.6)
check("r decreases depth_target", s.depth_target < 0.6)
s = TeleopState(depth_target=1.49, depth_step=0.5)
apply_key(s, 'f', lim, 'absolute', 0.0, 0.6)
check("depth_target clamps at max_depth", s.depth_target == lim.max_depth)
s = TeleopState(depth_target=0.05, depth_step=0.5)
apply_key(s, 'r', lim, 'absolute', 0.0, 0.6)
check("depth_target clamps at min_depth (surface)", s.depth_target == lim.min_depth)

print("\nabsolute mode: depth key before seeding is a safe no-op")
s = TeleopState(depth_target=None)
apply_key(s, 'f', lim, 'absolute', 0.0, 0.6)
check("depth_target stays None (not seeded yet)", s.depth_target is None)
vx, vy, vw, z = command_twist(s, 'absolute', 0.0, 0.3)
check("command_twist emits z=0.0 while unseeded (no lurch)", z == 0.0)

print("\n'0' surfaces (absolute mode only)")
s = TeleopState(depth_target=1.2)
apply_key(s, '0', lim, 'absolute', 0.0, 0.6)
check("0 sets depth_target to 0.0", s.depth_target == 0.0)
s = TeleopState(depth_target=1.2)
apply_key(s, '0', lim, 'pulse', 0.0, 0.6)
check("'0' is a no-op in pulse mode", s.depth_target == 1.2)

print("\npulse mode: r/f arm a timed pulse, auto-return to neutral")
s = TeleopState()
apply_key(s, 'f', lim, 'pulse', now=10.0, pulse_duration=0.6)
check("pulse armed with correct sign (+1 for deeper)", s.pulse_sign == 1.0)
check("pulse_until = now + duration", s.pulse_until == 10.6)
_,_,_,z = command_twist(s, 'pulse', now=10.3, pulse_rate=0.3)
check("mid-pulse: z is nonzero and positive", z == 0.3)
_,_,_,z = command_twist(s, 'pulse', now=10.6, pulse_rate=0.3)
check("exactly at pulse end: z is neutral", z == 0.0)
_,_,_,z = command_twist(s, 'pulse', now=11.0, pulse_rate=0.3)
check("after pulse: z is neutral (ArduSub holds from here)", z == 0.0)

s = TeleopState()
apply_key(s, 'r', lim, 'pulse', now=0.0, pulse_duration=0.6)
_,_,_,z = command_twist(s, 'pulse', now=0.1, pulse_rate=0.3)
check("r pulse is negative (shallower)", z == -0.3)

print("\n'x' cancels an in-flight pulse (pulse mode) and zeroes translation always")
s = TeleopState(surge=0.3, sway=0.2, yaw=0.4)
apply_key(s, 'f', lim, 'pulse', now=0.0, pulse_duration=0.6)
apply_key(s, 'x', lim, 'pulse', now=0.1, pulse_duration=0.6)
check("x zeroes surge/sway/yaw", (s.surge, s.sway, s.yaw) == (0.0, 0.0, 0.0))
check("x cancels the in-flight pulse", s.pulse_until == 0.0)
_,_,_,z = command_twist(s, 'pulse', now=0.2, pulse_rate=0.3)
check("post-x: z reads neutral immediately", z == 0.0)

print("\nstep-size adjustment: +/-/[/] scale and clamp")
s = TeleopState(surge_step=0.15, sway_step=0.15, depth_step=0.1)
apply_key(s, '+', lim, 'absolute', 0.0, 0.6)
check("+ grows surge_step and sway_step together",
      s.surge_step > 0.15 and s.sway_step > 0.15)
apply_key(s, '-', lim, 'absolute', 0.0, 0.6)
apply_key(s, '-', lim, 'absolute', 0.0, 0.6)
check("- shrinks surge_step", s.surge_step < 0.15)
for _ in range(50): apply_key(s, '-', lim, 'absolute', 0.0, 0.6)
check("surge_step floors at STEP_SCALE_MIN", s.surge_step >= 0.02 - 1e-9)
for _ in range(50): apply_key(s, '+', lim, 'absolute', 0.0, 0.6)
check("surge_step ceilings at STEP_SCALE_MAX", s.surge_step <= 2.0 + 1e-9)
apply_key(s, ']', lim, 'absolute', 0.0, 0.6)
check("] grows depth_step only", s.depth_step > 0.1)

print("\ncommand_twist output shape")
s = TeleopState(surge=0.2, sway=-0.1, yaw=0.3, depth_target=0.9)
out = command_twist(s, 'absolute', 0.0, 0.3)
check("returns (surge, sway, yaw, z) matching state", out == (0.2, -0.1, 0.3, 0.9))

print("\nunknown key is a safe no-op")
s = TeleopState(surge=0.1)
apply_key(s, 'z', lim, 'absolute', 0.0, 0.6)   # not bound to anything
check("unbound key leaves state unchanged", s.surge == 0.1)

print("\n" + ("ALL TELEOP CORE TESTS PASSED" if ok else "SOMETHING FAILED"))
import sys as _s; _s.exit(0 if ok else 1)
