#!/usr/bin/env python3
"""estimators/ekf_estimator.py — EKF with yaw IN THE STATE, three landmark modes.

STATE (v2, yaw upgrade):
    x = [ px py pz vx vy psi b_psi | f1x f1y | f2x f2y | ... ]     BASE = 7
        psi   : vehicle yaw in the COMPARE frame [rad]
        b_psi : gyro z-bias expressed as a yaw-rate bias [rad/s]

WHY YAW MOVED INTO THE STATE (2026-07-23). The sim's IMU.cpp was locally modified
so yaw_drift is a CONSTANT GYRO Z-BIAS added to both the reported z-rate and the
integrated (published) yaw — a physically-consistent magnetometer-less AHRS,
matching real HFI-A9 behavior. Under that model the externally-supplied yaw the
old 5-state filter consumed drifts without bound, and the old remedy (a
never-freezing EWMA offset in the node) was only defensible against the OLD
random-walk injection. A constant bias driving an integrator is textbook material
for a bias state: while the lane-line compass is visible, b_psi is observable and
converges; during lane dropouts psi coasts on the bias-CORRECTED gyro instead of
accumulating error at the full bias rate.

PROPAGATION — via PUBLISHED-YAW INCREMENTS, not raw gyro rates. The modified
IMU.cpp guarantees published_yaw(t) = integral of the reported (biased) z-rate,
so wrap(yaw_pub[k] - yaw_pub[k-1]) IS the biased-gyro increment over that dt,
in the node's ALREADY-VALIDATED yaw convention. Integrating the raw
angular_velocity.z instead would require asserting the shim's body-axis (FLU vs
FRD) sign — precisely the class of unverified sign assumption that has produced
every silent bug in this project. The increment form has zero sign ambiguity,
and white per-sample yaw noise telescopes instead of accumulating.
    predict:  psi   += dpsi_published - b_psi * dt        (dpsi is a known input)
              b_psi += 0                                  (constant + tiny RW)

LANE MEASUREMENT (mod-90 line compass, from /heading/line_meas2):
    h(x)   = fold90( lane_sign * psi - lane_grid )        fold90 -> (-pi/4, pi/4]
    nu     = fold90( ang_meas - h )                       branch resolved by fold
    H_psi  = lane_sign
  SIGN CORRECTED (2026-07-23, run 225022): the empirical model is
      ang == gamma - psi_ned (mod pi/2),      gamma = pool_axis_ned_yaw
  so the caller passes lane_sign = -1 / lane_grid = -gamma for compare 'ned'
  and lane_sign = +1 / lane_grid = -gamma for 'enu' (psi_ned = pi/2 - psi_enu;
  the pi/2 offset vanishes mod 90). Established empirically: the 225022 circle
  showed this filter's lane column erring by exactly -2*psi (mod 90) under the
  old (opposite) sign, and an offline iSAM2 pipeline replica reproduced both
  observed GTSAM failure modes with the old sign and clean tracking with this
  one. The earlier "spin-validated" chain had validated only the ENU/NED
  reflection of the published yaw (the spin ran above the lane node's
  freeze_rate, so the line-angle sign was never exercised, and every other run
  sat at psi ~ 0 mod 90 where the signs are indistinguishable).
  Gated with a chi-square(1) test. NOTE: psi is observable only mod pi/2 from
  this sensor — the fold resolves the branch against the CURRENT psi, valid
  while psi error < 45 deg (same argument as lane_heading_node's own unwrap).

WHAT THE PUBLISHED IMU YAW IS *NOT* USED FOR: it is never fused as an absolute
measurement. Under the new IMU model it contains exactly the information already
entering through the increments (it IS their integral), so fusing it would count
the biased gyro twice and inject the bias as pseudo-truth.

SLAM-MODE CHANGE: yaw uncertainty used to be folded into R via sigma_yaw
(yaw was not a state). Now the SAME Jacobian vectors become real H/G columns on
psi, so feature updates legitimately correct heading through the map and the
robot-yaw/feature cross-covariance is carried exactly:
  update:  H[:, psi] = d(R^T (f - p))/dpsi = [[-s, c], [-c, -s]] (f - p)
  augment: Gx[:, psi] = d(p + R z)/dpsi    = [[-s,-c], [ c, -s]] z
           Rw = R Rbody R^T                 (yaw term now flows through P)
The `yaw` argument of update_feature/... is RETAINED for caller compatibility
but IGNORED — the state's psi is used. sigma_yaw is likewise retained but
unused (documented deprecation, not silent removal).

mode 'none' / 'gate' / 'slam' semantics are unchanged; _predict/_kalman remain
dimension-generic. Pure numpy, no ROS; caller supplies measurements in the
compare frame.
"""

import numpy as np

BASE = 7                       # px py pz vx vy psi b_psi
IPX, IPY, IPZ, IVX, IVY, IPSI, IBPSI = range(BASE)
CHI2_1DOF_99 = 6.63            # lane gate, 1 dof, 99 %
CHI2_2DOF_99 = 9.21            # Mahalanobis gate, 2 dof, 99 %


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _fold90(a):
    """Fold an angle into the mod-90 line-compass space (-pi/4, pi/4]."""
    return (a + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0


class EkfEstimator:
    def __init__(self, q_pos=0.01, q_vel=0.5, r_flow=0.04, r_depth=4e-6,
                 sigma_yaw=0.02, chi2_gate=CHI2_2DOF_99,
                 init_sightings=3, init_max_spread=1.0,
                 min_range=0.3, max_range=25.0,
                 # --- yaw-state parameters (v2) ---
                 q_yaw=1e-5, q_bias=1e-10,
                 psi0_var=1.2e-3, bias0_var=1e-6,
                 lane_sign=1.0, lane_grid=0.0,
                 chi2_lane=CHI2_1DOF_99):
        """q_yaw  [rad^2/s]  : psi process noise density (gyro white noise +
                               published-yaw quantization; gtsam gyro_sigma
                               0.0017 -> ~3e-6, default padded to 1e-5).
        q_bias [rad^2/s^3-ish, applied as var/s] : b_psi random walk. The sim
                               bias is CONSTANT, so this is tiny by design.
        psi0_var / bias0_var : initial variances ((~2 deg)^2 and (1e-3 rad/s)^2).
                               bias0_var was 1e-4 (0.01 rad/s ~ 34 deg/min 1-sigma);
                               run 224010 showed b_psi chasing the published-yaw
                               noise to -18 deg/min in the first ~90 s before
                               converging — a 1-sigma of 34 deg/min licenses that
                               excursion for a bias that is physically ~1-3
                               deg/min. 1e-6 (3.4 deg/min 1-sigma) still covers
                               any plausible .scn yaw_drift while killing the
                               transient chase.
        lane_sign, lane_grid : measurement model constants, see module docstring.
        sigma_yaw            : DEPRECATED (yaw is a state now). Accepted so
                               existing constructions don't break; unused."""
        self.x = np.zeros(BASE)
        self.P = np.eye(BASE) * 1.0
        self.P[IPSI, IPSI] = psi0_var
        self.P[IBPSI, IBPSI] = bias0_var
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_yaw = q_yaw
        self.q_bias = q_bias
        self.R_flow = np.eye(2) * r_flow
        self.R_depth = r_depth
        self.sigma_yaw = sigma_yaw          # DEPRECATED, unused (see docstring)
        self.chi2_gate = chi2_gate
        self.chi2_lane = chi2_lane
        self.lane_sign = float(lane_sign)
        self.lane_grid = float(lane_grid)
        self.init_sightings = init_sightings
        self.init_max_spread = init_max_spread
        self.min_range = min_range
        self.max_range = max_range
        self.t_prev = None
        self.initialized = False
        self._yaw_pub_prev = None           # last published-yaw input (increments)
        self._yaw_seeded = False
        # lane bookkeeping (for the terminal report)
        self.lane_ok_n = 0
        self.lane_gate_n = 0
        # slam bookkeeping
        self.features = {}                  # name -> slot j
        self._pending = {}                  # name -> list of candidate world (fx, fy)
        self.rejected = {}                  # name -> count of chi2-gated updates

    # ------------------------------------------------------------------ core
    def _predict(self, dt):
        n = self.x.size
        F = np.eye(n)
        F[IPX, IVX] = dt                    # px += vx dt
        F[IPY, IVY] = dt                    # py += vy dt
        F[IPSI, IBPSI] = -dt                # psi += -b_psi dt  (bias correction;
                                            # the dpsi input is added by
                                            # propagate_yaw, outside F)
        self.x = F @ self.x
        self.x[IPSI] = _wrap(self.x[IPSI])
        Q = np.zeros((n, n))
        Q[IPX, IPX] = Q[IPY, IPY] = Q[IPZ, IPZ] = self.q_pos * dt
        Q[IVX, IVX] = Q[IVY, IVY] = self.q_vel * dt
        Q[IPSI, IPSI] = self.q_yaw * dt
        Q[IBPSI, IBPSI] = self.q_bias * dt
        # features get ZERO process noise: SAUVC props are static. The map may
        # only move through correlation with the robot, never on its own.
        self.P = F @ self.P @ F.T + Q

    def _step_time(self, t):
        if self.t_prev is None:
            self.t_prev = t
            return 0.0
        dt = t - self.t_prev
        # FIX(out-of-order stamps): with FOUR async sources now feeding the
        # filter (IMU 100 Hz, camera ~20 Hz, pressure, lane ~5 Hz), a stamp
        # slightly older than t_prev is routine pipeline latency. The old code
        # REWOUND t_prev on dt<=0, so the next in-order measurement predicted
        # over an inflated dt. Now: no predict, no rewind — the measurement is
        # applied at the current state (error bounded by the few-ms skew).
        if dt <= 0.0:
            return 0.0
        self.t_prev = t
        if dt < 1.0:
            self._predict(dt)
        return dt

    def _kalman_nu(self, nu, H, R):
        """Innovation-form update: needed now that h(x) is nonlinear in psi
        (flow, lane, features). _kalman keeps the linear z - Hx path."""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[IPSI] = _wrap(self.x[IPSI])
        I = np.eye(self.x.size)
        # Joseph form: keeps P symmetric positive-definite once it grows large
        # with features (the plain (I-KH)P form was fine at 5x5, it is not once
        # cross-covariance blocks matter).
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    def _kalman(self, z, H, R):
        self._kalman_nu(z - H @ self.x, H, R)

    # --------------------------------------------------------- yaw channel
    def propagate_yaw(self, yaw_pub, t):
        """Feed the PUBLISHED compare-frame yaw at IMU rate. First call seeds
        psi (the sim seeds the published yaw from truth; on hardware the AHRS
        start value is the shared reference everything is relative to anyway).
        Subsequent calls add wrap(yaw_pub - prev) as the gyro increment; the
        -b_psi*dt correction is applied inside _predict for EVERY dt, whichever
        callback triggered it, so bias correction never depends on call order."""
        if not self._yaw_seeded:
            self.x[IPSI] = _wrap(float(yaw_pub))
            self._yaw_pub_prev = float(yaw_pub)
            self._yaw_seeded = True
            self._step_time(t)
            return
        self._step_time(t)
        dpsi = _wrap(float(yaw_pub) - self._yaw_pub_prev)
        self._yaw_pub_prev = float(yaw_pub)
        self.x[IPSI] = _wrap(self.x[IPSI] + dpsi)

    def update_lane(self, ang_meas, sigma, t):
        """One raw line-compass measurement: the image-frame line angle from
        /heading/line_meas2, in (-pi/4, pi/4], with its stddev [rad] (caller
        scales by the concentration R). Returns 'ok' | 'gated' | 'unseeded'.
        This is what makes psi AND b_psi observable — see module docstring."""
        if not self._yaw_seeded:
            return 'unseeded'
        self._step_time(t)
        h = _fold90(self.lane_sign * self.x[IPSI] - self.lane_grid)
        nu = _fold90(float(ang_meas) - h)
        n = self.x.size
        H = np.zeros((1, n))
        H[0, IPSI] = self.lane_sign
        R = np.array([[max(float(sigma), 1e-4) ** 2]])
        S = float((H @ self.P @ H.T + R)[0, 0])
        if nu * nu / S > self.chi2_lane:
            self.lane_gate_n += 1
            return 'gated'
        self._kalman_nu(np.array([nu]), H, R)
        self.lane_ok_n += 1
        return 'ok'

    # ------------------------------------------------- original measurements
    def update_flow(self, vx_body, vy_body, t, r_var=None, yaw=None):
        """Body-frame planar velocity from optical flow. The rotation now uses
        the STATE's psi (the external-yaw path is gone with the EWMA it served):
            h(x) = R(psi)^T [vx_w, vy_w],   z = [vx_body, vy_body]
        The d/dpsi columns are what couple heading to velocity — yaw becomes
        weakly observable from trajectory shape while moving, and a ZUPT zero
        (v_w ~ 0) correctly contributes ~nothing to psi since dh/dpsi ~ 0.
        r_var: optional per-measurement variance (m/s)^2 — quality-scaled
        noise seam (spread_px / n_inliers), as before.
        yaw: DEPRECATED and ignored (kept so stale call sites fail loudly in
        review, not silently at runtime)."""
        self._step_time(t)
        c, s = np.cos(self.x[IPSI]), np.sin(self.x[IPSI])
        vwx, vwy = self.x[IVX], self.x[IVY]
        h = np.array([c * vwx + s * vwy,
                      -s * vwx + c * vwy])          # R^T v_world
        n = self.x.size
        H = np.zeros((2, n))
        H[0, IVX] = c;  H[0, IVY] = s
        H[1, IVX] = -s; H[1, IVY] = c
        # d(R^T v_w)/dpsi = [[-s, c], [-c, -s]] v_w
        H[0, IPSI] = -s * vwx + c * vwy
        H[1, IPSI] = -c * vwx - s * vwy
        z = np.array([float(vx_body), float(vy_body)])
        R = self.R_flow if r_var is None else np.eye(2) * max(float(r_var), 1e-6)
        self._kalman_nu(z - h, H, R)
        self.initialized = True

    def update_depth(self, pz, t):
        self._step_time(t)
        H = np.zeros((1, self.x.size)); H[0, IPZ] = 1.0
        self._kalman(np.array([pz]), H, np.array([[self.R_depth]]))
        self.initialized = True

    def update_position_xy(self, px, py, t, var_x, var_y):
        """GATE MODE (unchanged semantics). World-frame position pseudo-measurement
        p_meas = landmark_world - rel_obs_world, anisotropic per axis. Gate x known
        from the rulebook -> finite var_x; gate y randomized -> var_y = 1e12 no-op.
        State is NOT augmented; this is localization against a known map."""
        self._step_time(t)
        H = np.zeros((2, self.x.size)); H[0, IPX] = 1.0; H[1, IPY] = 1.0
        R = np.diag([max(float(var_x), 1e-6), max(float(var_y), 1e-6)])
        self._kalman(np.array([px, py]), H, R)

    @property
    def position(self):
        return self.x[IPX], self.x[IPY], self.x[IPZ]

    @property
    def velocity(self):
        return self.x[IVX], self.x[IVY]

    @property
    def yaw_est(self):
        """(psi [rad], 1-sigma [rad]) — compare-frame yaw estimate."""
        return float(self.x[IPSI]), float(np.sqrt(max(self.P[IPSI, IPSI], 0.0)))

    @property
    def yaw_bias(self):
        """(b_psi [rad/s], 1-sigma [rad/s]) — should converge to the .scn
        yaw_drift magnitude (in this filter's compare-frame yaw-rate sense)
        once enough lane-visible time has accumulated. That convergence is the
        falsifiable check that the bias is being ESTIMATED, not suffered."""
        return float(self.x[IBPSI]), float(np.sqrt(max(self.P[IBPSI, IBPSI], 0.0)))

    # --------------------------------------------------------- slam helpers
    @staticmethod
    def polar_to_cart_cov(rng, brg, sigma_r, sigma_b):
        """Detector noise is naturally polar (range from apparent size, bearing
        from centroid column). Convert diag(sigma_r^2, sigma_b^2) to the body
        xy covariance via J = d(x,y)/d(r,b)."""
        c, s = np.cos(brg), np.sin(brg)
        J = np.array([[c, -rng * s],
                      [s,  rng * c]])
        return J @ np.diag([sigma_r ** 2, sigma_b ** 2]) @ J.T

    def _slot(self, name):
        j = self.features[name]
        i = BASE + 2 * j
        return i, i + 2

    def feature_estimate(self, name):
        """(fx, fy, var_x, var_y) of a mapped feature, or None."""
        if name not in self.features:
            return None
        i0, i1 = self._slot(name)
        return (self.x[i0], self.x[i1 - 1],
                self.P[i0, i0], self.P[i1 - 1, i1 - 1])

    def feature_names(self):
        return sorted(self.features, key=self.features.get)

    # ------------------------------------------------------ slam: main entry
    def update_feature(self, name, zx, zy, yaw, t, R_body, known=None):
        """SLAM MODE. One body-frame xy observation of the named feature.

        R_body : 2x2 observation covariance in body xy (use polar_to_cart_cov).
        known  : optional {axis_index: (value, sigma)} of world coordinates known
                 a priori (gate x from the rulebook). Applied ONCE, right after
                 augmentation, as a direct measurement of the feature coordinate —
                 through the fresh cross-covariance it also pulls the robot.
        yaw    : DEPRECATED and ignored — the STATE's psi is used, which is the
                 entire point of the yaw upgrade (a feature update may now
                 legitimately correct heading through the H psi column).
        Returns 'init-pending' | 'init' | 'ok' | 'gated' | 'range'.
        """
        rng = float(np.hypot(zx, zy))
        if not (self.min_range <= rng <= self.max_range):
            return 'range'                      # GUARD 1: implausible range
        self._step_time(t)
        c, s = np.cos(self.x[IPSI]), np.sin(self.x[IPSI])
        Ry = np.array([[c, -s], [s, c]])
        z = np.array([zx, zy], float)

        if name not in self.features:
            return self._try_init(name, z, Ry, s, c, R_body, known)

        # ---- update an existing feature -------------------------------
        i0, i1 = self._slot(name)
        f = self.x[i0:i1]
        p = self.x[IPX:IPY + 1]
        d = f - p
        h = Ry.T @ d                            # expected body observation
        n = self.x.size
        H = np.zeros((2, n))
        H[:, IPX:IPY + 1] = -Ry.T               # robot block  (J_ominus path)
        H[:, i0:i1] = Ry.T                      # feature block (J_2boxplus path)
        # YAW-UPGRADE: dh/dpsi = d(R^T)/dpsi d = [[-s, c], [-c, -s]] d used to be
        # folded into R via sigma_yaw (yaw was not a state); it is now a REAL
        # H column, so the update corrects psi through the map and R_eff is
        # just the detector covariance.
        H[0, IPSI] = -s * d[0] + c * d[1]
        H[1, IPSI] = -c * d[0] - s * d[1]
        R_eff = np.asarray(R_body, float)

        nu = z - h
        S = H @ self.P @ H.T + R_eff
        m2 = float(nu @ np.linalg.solve(S, nu))
        if m2 > self.chi2_gate:                 # GUARD 4: Mahalanobis gate
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return 'gated'
        self._kalman_nu(nu, H, R_eff)
        return 'ok'

    def _try_init(self, name, z, Ry, s, c, R_body, known):
        """GUARD 2+3: a feature only enters the state after `init_sightings`
        consistent world-frame candidates (spread below init_max_spread). The
        first frame that ever sees a prop is usually the worst one (partial blob,
        grazing angle) and in SLAM the first fix is what the map — and every
        later relocalization — inherits."""
        cand = self.x[IPX:IPY + 1] + Ry @ z
        lst = self._pending.setdefault(name, [])
        lst.append(cand)
        if len(lst) < self.init_sightings:
            return 'init-pending'
        arr = np.array(lst[-self.init_sightings:])
        if np.max(np.std(arr, axis=0)) > self.init_max_spread:
            del lst[0]                          # keep sliding, wait for agreement
            return 'init-pending'

        # ---- augment: x <- [x; f],  P gets true cross-covariance blocks ----
        n = self.x.size
        f = self.x[IPX:IPY + 1] + Ry @ z        # linearize at the CURRENT obs
        Gx = np.zeros((2, n)); Gx[0, IPX] = 1.0; Gx[1, IPY] = 1.0
        # YAW-UPGRADE: dg/dpsi = [[-s,-c],[c,-s]] z is now a Gx COLUMN instead of
        # a sigma_yaw term in Rw — psi's CURRENT variance flows into the feature
        # through Gx P Gx^T, and the robot-yaw/feature cross-covariance is born
        # exact (this is what lets a later sighting correct heading).
        Gx[0, IPSI] = -s * z[0] - c * z[1]
        Gx[1, IPSI] = c * z[0] - s * z[1]
        Rw = Ry @ np.asarray(R_body, float) @ Ry.T

        PGxT = self.P @ Gx.T
        Pff = Gx @ PGxT + Rw
        Pnew = np.zeros((n + 2, n + 2))
        Pnew[:n, :n] = self.P
        Pnew[:n, n:] = PGxT
        Pnew[n:, :n] = PGxT.T
        Pnew[n:, n:] = Pff
        self.P = Pnew
        self.x = np.concatenate([self.x, f])
        self.features[name] = (n - BASE) // 2
        self._pending.pop(name, None)

        # rulebook knowledge as a one-shot measurement of the feature coordinate
        if known:
            i0, _ = self._slot(name)
            for axis, (val, sig) in known.items():
                H = np.zeros((1, self.x.size)); H[0, i0 + axis] = 1.0
                self._kalman(np.array([float(val)]), H,
                             np.array([[max(float(sig), 1e-3) ** 2]]))
        return 'init'
