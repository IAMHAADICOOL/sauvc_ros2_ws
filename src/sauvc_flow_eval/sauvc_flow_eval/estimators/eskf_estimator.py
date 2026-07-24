#!/usr/bin/env python3
"""estimators/eskf_estimator.py — 15-state ERROR-STATE Kalman filter (config D).

WHY A THIRD ARCHITECTURE (2026-07-24). The eval now compares three genuinely
different estimator designs on identical inputs:
  * ekf   : KINEMATIC total-state EKF — no accel integration; velocity is a
            random-walk state measured by flow; yaw propagated by published-yaw
            increments.
  * gtsam : IMU PREINTEGRATION + SMOOTHING (iSAM2 factor graph).
  * eskf  : IMU STRAPDOWN + RECURSIVE FILTERING — this file. The classic INS
            architecture (Sola, "Quaternion kinematics for the error-state KF"):
            a NOMINAL state integrated nonlinearly at IMU rate, plus a small
            linearized ERROR state that measurements correct and that is
            injected + reset after every update. Errors stay small, so the
            linearization is honest even during fast turns.

STATE. Nominal: p(3, NED), v(3, NED), q(4, body FRD -> NED, wxyz), b_a(3),
b_g(3). Error: dx = [dp dv dtheta db_a db_g] (15). The filter runs INTERNALLY
IN NED like the graph does; the node converts at the seam.

FRAMES — every convention here is a fact verified this session, none assumed:
  * /imu/data body vectors are ENU/FLU (proven by the +9.81 static accel-z and
    the mirrored-turn episode); the node feeds THIS filter the same FLU->FRD
    flip (x, -y, -z) it feeds the graph.
  * Gravity in NED is (0, 0, +g); a level, static FRD accel reads (0, 0, -g).
  * Lane model (corrected, run 225022 + offline replica):
        ang == gamma - psi_ned  (mod 90),   gamma = pool_axis_ned_yaw
  * The published IMU yaw is NEVER fused: it is the integral of the same gyro
    this filter integrates (modified IMU.cpp) — fusing it would count the gyro
    bias twice. Yaw's absolute reference is the lane, exactly as in the graph.
  * Roll/pitch ARE softly anchored to the AHRS quaternion (gravity-referenced
    and trustworthy per this project's record; the same choice as the graph's
    attitude prior rp channel). Yaw is excluded from that update.

NOISE UNITS. accel_sigma / gyro_sigma are PER-SAMPLE stddevs at imu_rate_hz
(what the static buffer measures, what run 231739 showed: 0.02 / 0.0017 @
100 Hz); they are converted internally to continuous densities. This mirrors
the unit fix that resolved the graph's bias-invisibility — do not pass
densities here.

STATIC INIT (same pattern as the graph, same reasons): predict() defers until
`init_min_samples` post-settle-skip stationary samples are buffered, then
measures b_g (gyro mean) and b_a (accel mean minus R^T(0,0,-g)) with a 4-MAD
trim, and starts from them. Unlike the graph, the ESKF also converges its
biases ONLINE without any of this (recursive filters don't have the iSAM2
consolidation problem), so the wait is a head start, not a requirement —
init_min_samples=0 starts instantly from zero biases.

MEASUREMENTS.
  update_flow(vx_frd, vy_frd, t, r_var)  planar body velocity, h = (R^T v)_xy;
                                          couples dv and dtheta (incl. roll/
                                          pitch through gravity leakage, the
                                          standard ESKF tilt-observability
                                          path).
  update_depth(depth_m, t)               NED z = +depth.
  update_lane(ang, sigma, t)             mod-90 fold on dtheta_z, chi2(1) gate.
  update_rp(roll, pitch, sigma, t)       NED/FRD roll+pitch from the AHRS quat.

INJECTION + RESET: after each update, dx folds into the nominal (q <- q *
Exp(dtheta)) and dx resets to zero. The reset Jacobian's second-order dtheta
term (Sola eq. 285) is omitted — legitimate below ~5 deg error, which the
lane + rp anchors guarantee here; noted rather than silently ignored.

Pure numpy, no ROS. Outputs NED; the caller converts to the compare frame.
"""

import numpy as np

NX = 15
IDP, IDV, ITH, IBA, IBG = 0, 3, 6, 9, 12
CHI2_1DOF_99 = 6.63
CHI2_2DOF_99 = 9.21


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _fold90(a):
    return (a + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _q_exp(dtheta):
    """Rotation-vector -> quaternion (wxyz)."""
    a = float(np.linalg.norm(dtheta))
    if a < 1e-12:
        return np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])
    ax = dtheta / a
    return np.concatenate(([np.cos(a / 2.0)], np.sin(a / 2.0) * ax))


def _q_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _q_normalize(q):
    return q / max(np.linalg.norm(q), 1e-12)


def _rpy_from_R(R):
    """ZYX roll/pitch/yaw of a body FRD -> NED rotation matrix."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


class EskfEstimator:
    def __init__(self, gravity=9.81, imu_rate_hz=100.0,
                 accel_sigma=0.02, gyro_sigma=0.0017,       # PER-SAMPLE stddevs
                 accel_bias_rw=1e-4, gyro_bias_rw=1e-5,     # rw densities:
                 # tightened vs the graph's values — the plant biases are
                 # CONSTANT (modified IMU.cpp); 1e-4 gyro rw licensed +/-2.7
                 # deg/min of 1-sigma wander per minute and made the online
                 # estimate oscillate enough to lose the lane-dropout coast
                 # advantage in validation (T4).
                 r_flow=0.04, r_depth=4e-6,
                 lane_grid=0.0, chi2_lane=CHI2_1DOF_99,
                 chi2_flow=25.0,
                 p0_att=np.radians(2.0), p0_ba=0.05, p0_bg=1e-3,
                 init_min_samples=200, init_settle_skip_s=1.0,
                 init_max_wait_s=6.0,
                 flow_gate_relock_n=40, flow_relock_r_inflate=25.0):
        self.g_ned = np.array([0.0, 0.0, float(gravity)])
        dt_nom = 1.0 / float(imu_rate_hz)
        # per-sample -> continuous density (the unit fix, see module docstring)
        self.qc_acc = (float(accel_sigma) ** 2) * dt_nom     # (m/s^2)^2 * s
        self.qc_gyr = (float(gyro_sigma) ** 2) * dt_nom      # (rad/s)^2 * s
        self.qc_ba = float(accel_bias_rw) ** 2
        self.qc_bg = float(gyro_bias_rw) ** 2
        self.R_flow = float(r_flow)
        self.R_depth = float(r_depth)
        self.lane_grid = float(lane_grid)
        self.chi2_lane = float(chi2_lane)
        self.chi2_flow = float(chi2_flow)
        # nominal
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        # error covariance
        self.P = np.zeros((NX, NX))
        self.P[IDP:IDP + 3, IDP:IDP + 3] = np.eye(3) * 1.0
        self.P[IDV:IDV + 3, IDV:IDV + 3] = np.eye(3) * 0.1
        self.P[ITH:ITH + 3, ITH:ITH + 3] = np.eye(3) * float(p0_att) ** 2
        self.P[IBA:IBA + 3, IBA:IBA + 3] = np.eye(3) * float(p0_ba) ** 2
        self.P[IBG:IBG + 3, IBG:IBG + 3] = np.eye(3) * float(p0_bg) ** 2
        self.t_prev = None
        self.initialized = False
        # static init (same pattern + reasons as the graph's; see docstring)
        self.init_min_samples = int(init_min_samples)
        self.init_settle_skip_s = float(init_settle_skip_s)
        self.init_max_wait_s = float(init_max_wait_s)
        self._init_buf = []
        self._init_t0 = None
        self._init_first_wait_t = None
        self._init_logged = False
        # bookkeeping for the terminal report
        self.lane_ok_n = 0
        self.lane_gate_n = 0
        self.flow_gate_n = 0
        # GATE-BREAKDOWN RECOVERY (run 011645): once the state is confidently
        # WRONG (corrupted init bias), the chi2 flow gate rejects every
        # correct measurement and the filter free-runs on the bad IMU — the
        # 3 m phantom excursion. After flow_gate_relock_n CONSECUTIVE
        # rejections, the next flow measurement is accepted with its R
        # inflated flow_relock_r_inflate-fold: bounded influence, breaks the
        # spiral. Same philosophy as lane_heading_node's stuck-lock relock.
        self.flow_gate_relock_n = int(flow_gate_relock_n)
        self.flow_relock_r_inflate = float(flow_relock_r_inflate)
        self._flow_reject_streak = 0
        self.flow_relock_n = 0

    # ------------------------------------------------------------ init path
    def try_initialize(self, quat_wxyz_ned, depth, t):
        """Called repeatedly by the node until it returns True. Defers while
        the static buffer fills (or until init_max_wait_s), then measures the
        biases and seeds attitude/position. init_min_samples=0 -> instant
        start with zero biases (the ESKF converges them online regardless)."""
        if self.initialized:
            return True
        n = len(self._init_buf)
        if n < self.init_min_samples:
            if self._init_first_wait_t is None:
                self._init_first_wait_t = t
            if t - self._init_first_wait_t < self.init_max_wait_s:
                if not self._init_logged:
                    self._init_logged = True
                    print(f'[eskf] init deferred: {n}/{self.init_min_samples} '
                          f'static samples (settle skip '
                          f'{self.init_settle_skip_s:.1f} s).')
                return False
            print(f'[eskf] init wait expired with {n} samples — proceeding.')
        self.q = _q_normalize(np.asarray(quat_wxyz_ned, float))
        self.p = np.array([0.0, 0.0, float(depth)])
        self.v = np.zeros(3)
        if n >= 20:
            A = np.array([a for a, _ in self._init_buf])
            W = np.array([w for _, w in self._init_buf])
            med = np.median(W, axis=0)
            mad = np.median(np.abs(W - med), axis=0) * 1.4826 + 1e-12
            keep = np.all(np.abs(W - med) < 4.0 * mad, axis=1)
            Wk = W[keep] if keep.sum() >= 20 else W
            Ak = A[keep] if keep.sum() >= 20 else A
            # STILLNESS VALIDATION (run 011645: init ran while the vehicle was
            # being SUBMERGED; the maneuver's acceleration was measured into
            # b_a as +0.0115 m/s^2 and, with the tight bias RW, poisoned the
            # whole run). A constant bias agrees between the two halves of the
            # window; smooth real motion does not. Reject the ACCEL
            # measurement if the halves disagree beyond 5x the noise-limited
            # expectation; the gyro test guards b_g the same way. On rejection
            # fall back to zero bias with the ctor's WIDE prior covariance —
            # an honest "don't know" beats a confident wrong number, and the
            # filter converges the biases online regardless.
            h = len(Ak) // 2
            am1, am2 = Ak[:h].mean(axis=0), Ak[h:].mean(axis=0)
            wm1, wm2 = Wk[:h].mean(axis=0), Wk[h:].mean(axis=0)
            a_tol = 5.0 * Ak.std(axis=0, ddof=1) / max(np.sqrt(h), 1.0) + 1e-6
            w_tol = 5.0 * Wk.std(axis=0, ddof=1) / max(np.sqrt(h), 1.0) + 1e-9
            acc_still = bool(np.all(np.abs(am1 - am2) < a_tol))
            gyr_still = bool(np.all(np.abs(wm1 - wm2) < w_tol))
            f_expected = _q_to_R(self.q).T @ (-self.g_ned)
            if gyr_still:
                self.bg = Wk.mean(axis=0)
            else:
                print(f'[eskf] init window NOT still (gyro split-half '
                      f'{np.round(np.abs(wm1 - wm2), 5)} > tol '
                      f'{np.round(w_tol, 5)}) — b_g starts at ZERO, wide P.')
            if acc_still and gyr_still:
                self.ba = Ak.mean(axis=0) - f_expected
            else:
                self.ba = np.zeros(3)
                self.P[IBA:IBA + 3, IBA:IBA + 3] = np.eye(3) * 0.05 ** 2
                if not acc_still:
                    print(f'[eskf] init window NOT still (accel split-half '
                          f'{np.round(np.abs(am1 - am2), 4)} > tol '
                          f'{np.round(a_tol, 4)}) — vehicle was maneuvering; '
                          f'b_a starts at ZERO with wide P instead of a '
                          f'motion-corrupted value.')
            sig = Wk.std(axis=0, ddof=1) / max(np.sqrt(len(Wk)), 1.0)
            print(f'[eskf] static init from {len(Wk)}/{len(W)} samples: '
                  f'b_g={np.round(self.bg, 5)} +/- {np.round(sig, 5)} '
                  f'(bgz {np.degrees(self.bg[2])*60:+.2f} deg/min), '
                  f'b_a={np.round(self.ba, 4)}')
        else:
            print(f'[eskf] static init skipped ({n} < 20 samples) — biases '
                  f'start at zero and converge online.')
        self._init_buf = []
        self.initialized = True
        return True

    # ---------------------------------------------------------- propagation
    def predict(self, accel_frd, gyro_frd, t):
        """IMU-rate strapdown propagation of nominal + covariance. Before
        initialization this only buffers samples for the static init."""
        f = np.asarray(accel_frd, float)
        w = np.asarray(gyro_frd, float)
        if not self.initialized:
            if self._init_t0 is None:
                self._init_t0 = t
            if t - self._init_t0 >= self.init_settle_skip_s:
                self._init_buf.append((f, w))
            return
        if self.t_prev is None:
            self.t_prev = t
            return
        dt = t - self.t_prev
        if dt <= 0.0:
            return                      # out-of-order guard (no rewind)
        self.t_prev = t
        if dt > 0.1:
            return                      # gap: skip covariance blow-up
        fb = f - self.ba
        wb = w - self.bg
        R = _q_to_R(self.q)
        # nominal
        acc_w = R @ fb + self.g_ned
        self.p = self.p + self.v * dt + 0.5 * acc_w * dt * dt
        self.v = self.v + acc_w * dt
        self.q = _q_normalize(_q_mul(self.q, _q_exp(wb * dt)))
        # error covariance: F = I + Fc*dt (first order; dt = 10 ms)
        F = np.eye(NX)
        F[IDP:IDP + 3, IDV:IDV + 3] = np.eye(3) * dt
        F[IDV:IDV + 3, ITH:ITH + 3] = -R @ _skew(fb) * dt
        F[IDV:IDV + 3, IBA:IBA + 3] = -R * dt
        F[ITH:ITH + 3, ITH:ITH + 3] = np.eye(3) - _skew(wb) * dt
        F[ITH:ITH + 3, IBG:IBG + 3] = -np.eye(3) * dt
        Q = np.zeros((NX, NX))
        Q[IDV:IDV + 3, IDV:IDV + 3] = np.eye(3) * self.qc_acc * dt
        Q[ITH:ITH + 3, ITH:ITH + 3] = np.eye(3) * self.qc_gyr * dt
        Q[IBA:IBA + 3, IBA:IBA + 3] = np.eye(3) * self.qc_ba * dt
        Q[IBG:IBG + 3, IBG:IBG + 3] = np.eye(3) * self.qc_bg * dt
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------- updates
    def _correct(self, nu, H, Rm):
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ nu
        I = np.eye(NX)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T
        # inject + reset (second-order reset Jacobian omitted; see docstring)
        self.p = self.p + dx[IDP:IDP + 3]
        self.v = self.v + dx[IDV:IDV + 3]
        self.q = _q_normalize(_q_mul(self.q, _q_exp(dx[ITH:ITH + 3])))
        self.ba = self.ba + dx[IBA:IBA + 3]
        self.bg = self.bg + dx[IBG:IBG + 3]

    def update_position_xy(self, px, py, t, var_x, var_y, chi2_gate=CHI2_2DOF_99):
        """ABSOLUTE horizontal position fix (NED world x, y). Added because this
        filter has NO other observation of p_x/p_y: update_flow only measures
        VELOCITY (indirectly, coupled with attitude — see the module docstring's
        gravity-leakage note), so any residual velocity/attitude error integrates
        into position unbounded. This is the standard "velocity-aided strapdown
        INS also needs a position aid" fix. Mirrors EkfEstimator.update_position_xy
        (anisotropic var_x/var_y so a rulebook-known axis and an unknown one can
        be fused in one call), plus a chi2 innovation gate (this filter has no
        cheap way to sanity-check a position jump otherwise, unlike flow/lane's
        physically-bounded innovations).
        """
        if not self.initialized:
            return 'uninit'
        H = np.zeros((2, NX))
        H[0, IDP + 0] = 1.0
        H[1, IDP + 1] = 1.0
        Rm = np.diag([max(float(var_x), 1e-6), max(float(var_y), 1e-6)])
        nu = np.array([float(px) - self.p[0], float(py) - self.p[1]])
        S = H @ self.P @ H.T + Rm
        if float(nu @ np.linalg.solve(S, nu)) > chi2_gate:
            return 'gated'
        self._correct(nu, H, Rm)
        return 'ok'

    def update_flow(self, vx_frd, vy_frd, t, r_var=None):
        """Planar body-FRD velocity from optical flow. h = (R^T v_world)_xy.
        d h/d dtheta = ([R^T v]_x)_xy rows — includes the gravity-free tilt
        coupling; d h/d dv = (R^T)_xy rows.

        JACOBIAN FIX (2026-07): the attitude block was (R^T [v]_x)_xy, which is
        WRONG under this filter's LOCAL/body error convention (R_true = R * Exp(dtheta),
        the same convention predict() encodes via q<-q*Exp(w dt), F[th,th]=I-[w_b]_x dt,
        F[v,th]=-R[f]_x). For h = R^T v the local-error derivative is
            d h/d dtheta = [R^T v]_x = R^T [v]_x R   (NOT R^T [v]_x).
        Verified by finite difference against R*Exp(dtheta): the old form carried a
        ~0.04 error concentrated in the dtheta_x/dtheta_y (tilt) columns — exactly the
        gravity-leakage path — so every flow update was mis-correcting tilt and
        mis-splitting the innovation between dv and dtheta. Corrected below."""
        if not self.initialized:
            return 'uninit'
        R = _q_to_R(self.q)
        h = (R.T @ self.v)[:2]
        z = np.array([float(vx_frd), float(vy_frd)])
        Hth = _skew(R.T @ self.v)[:2, :]        # [R^T v]_x  (was R^T [v]_x; see docstring)
        H = np.zeros((2, NX))
        H[:, IDV:IDV + 3] = R.T[:2, :]
        H[:, ITH:ITH + 3] = Hth
        rv = self.R_flow if r_var is None else max(float(r_var), 1e-6)
        Rm = np.eye(2) * rv
        nu = z - h
        S = H @ self.P @ H.T + Rm
        if float(nu @ np.linalg.solve(S, nu)) > self.chi2_flow:
            self.flow_gate_n += 1
            self._flow_reject_streak += 1
            if self._flow_reject_streak >= self.flow_gate_relock_n:
                # gate breakdown: re-admit with inflated R (see __init__)
                self._flow_reject_streak = 0
                self.flow_relock_n += 1
                self._correct(nu, H, Rm * self.flow_relock_r_inflate)
                return 'relock'
            return 'gated'
        self._flow_reject_streak = 0
        self._correct(nu, H, Rm)
        return 'ok'

    def update_depth(self, depth_m, t):
        if not self.initialized:
            return 'uninit'
        H = np.zeros((1, NX))
        H[0, IDP + 2] = 1.0
        nu = np.array([float(depth_m) - self.p[2]])
        self._correct(nu, H, np.array([[self.R_depth]]))
        return 'ok'

    def update_lane(self, ang, sigma, t):
        """Corrected lane model: ang == lane_grid - psi_ned (mod 90).
        h = fold90(grid - psi); d h/d dtheta_z = -1 (dpsi/ddtheta_z = +1 at
        near-level attitude)."""
        if not self.initialized:
            return 'uninit'
        _, _, psi = _rpy_from_R(_q_to_R(self.q))
        h = _fold90(self.lane_grid - psi)
        nu = np.array([_fold90(float(ang) - h)])
        H = np.zeros((1, NX))
        H[0, ITH + 2] = -1.0
        Rm = np.array([[max(float(sigma), 1e-4) ** 2]])
        S = float((H @ self.P @ H.T + Rm)[0, 0])
        if nu[0] * nu[0] / S > self.chi2_lane:
            self.lane_gate_n += 1
            return 'gated'
        self._correct(nu, H, Rm)
        self.lane_ok_n += 1
        return 'ok'

    def update_rp(self, roll_meas, pitch_meas, sigma, t):
        """Soft roll/pitch anchor from the AHRS quaternion (NED/FRD angles —
        use _rp_from_quat_wxyz on the converted quat, NOT the raw ENU r/p).
        Small-angle: droll ~ dtheta_x, dpitch ~ dtheta_y at near-level trim."""
        if not self.initialized:
            return 'uninit'
        r, p, _ = _rpy_from_R(_q_to_R(self.q))
        nu = np.array([_wrap(float(roll_meas) - r),
                       _wrap(float(pitch_meas) - p)])
        H = np.zeros((2, NX))
        H[0, ITH + 0] = 1.0
        H[1, ITH + 1] = 1.0
        self._correct(nu, H, np.eye(2) * max(float(sigma), 1e-4) ** 2)
        return 'ok'

    # -------------------------------------------------------------- readout
    @property
    def position_ned(self):
        return self.p.copy()

    @property
    def velocity_ned(self):
        return self.v.copy()

    def ned_yaw(self):
        return float(_rpy_from_R(_q_to_R(self.q))[2])

    def yaw_sigma(self):
        return float(np.sqrt(max(self.P[ITH + 2, ITH + 2], 0.0)))

    def gyro_bias(self):
        return tuple(float(b) for b in self.bg)
