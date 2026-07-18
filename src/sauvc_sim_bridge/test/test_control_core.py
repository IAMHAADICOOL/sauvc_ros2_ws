import sys, numpy as np
sys.path.insert(0, '.')
from sauvc_sim_bridge.control_core import ThrusterMixer, PID, build_allocation

ok = True
def check(name, cond):
    global ok; ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("Allocation matrix")
A = build_allocation()
check("shape 6x8", A.shape == (6, 8))
# surge: all 4 horizontals push +x equally
check("Fx row all-positive horizontals", np.allclose(A[0,:4], 0.7071, atol=1e-3) and np.allclose(A[0,4:], 0))
# heave: all 4 verticals, +z
check("Fz row = verticals", np.allclose(A[2,4:], 1.0) and np.allclose(A[2,:4], 0))

print("\nMixer: each DOF produces the physically sensible pattern")
m = ThrusterMixer()
check(f"condition number {m.cond:.2f} < 5", m.cond < 5)

u = m.wrench_to_thrust(1, 0, 0, 0)          # pure surge
check("surge: 4 horizontals equal & positive, verticals zero",
      np.allclose(u[:4], u[0]) and u[0] > 0 and np.allclose(u[4:], 0))

u = m.wrench_to_thrust(0, 0, 1, 0)          # pure heave
check("heave: 4 verticals equal & positive, horizontals zero",
      np.allclose(u[4:], u[4]) and u[4] > 0 and np.allclose(u[:4], 0))

u = m.wrench_to_thrust(0, 0, 0, 1)          # pure yaw
check("yaw: horizontals differential (sum ~0), verticals zero",
      abs(np.sum(u[:4])) < 1e-9 and np.allclose(u[4:], 0))

print("\nMixer saturation preserves direction")
u = m.wrench_to_thrust(10, 0, 0, 0)         # huge surge -> must clamp
check("peak clamped to 1.0", np.isclose(np.max(np.abs(u)), 1.0))
check("still pure surge after clamp (verticals zero)", np.allclose(u[4:], 0))
check("still symmetric surge", np.allclose(u[:4], u[0]))

print("\nPID: derivative-on-measurement, no setpoint-step kick")
pid = PID(kp=1.0, ki=0.0, kd=1.0, out_limit=100.0)
# step the setpoint with meas constant: derivative term must NOT react (d on meas)
o1 = pid.update(setpoint=0.0, meas=0.0, dt=0.1)
o2 = pid.update(setpoint=5.0, meas=0.0, dt=0.1)   # big setpoint jump, meas still 0
check("no derivative kick on setpoint step (out = kp*err = 5)", np.isclose(o2, 5.0))

print("\nPID: anti-windup halts integral under saturation")
pid = PID(kp=0.0, ki=1.0, kd=0.0, out_limit=1.0, i_limit=1.0)
for _ in range(100):
    pid.update(setpoint=10.0, meas=0.0, dt=0.1)   # drive hard into +saturation
check("integral clamped at i_limit", abs(pid.integral) <= 1.0 + 1e-9)
check("output clamped at out_limit", abs(pid.update(10.0, 0.0, 0.1)) <= 1.0 + 1e-9)

print("\nPID: converges on a first-order plant")
pid = PID(kp=2.0, ki=0.5, kd=0.1, out_limit=1.0)
x = 0.0
for _ in range(400):
    u = pid.update(setpoint=1.0, meas=x, dt=0.05)
    x += (u - 0.2 * x) * 0.05        # toy plant: velocity ~ thrust, light drag
check(f"reaches setpoint (x={x:.3f} in [0.9,1.1])", 0.9 < x < 1.1)

print("\nPID: dt<=0 guarded", )
check("dt=0 returns 0", PID(1,1,1).update(1, 0, 0.0) == 0.0)

print("\n" + ("ALL CONTROL TESTS PASSED" if ok else "SOMETHING FAILED"))
sys.exit(0 if ok else 1)
