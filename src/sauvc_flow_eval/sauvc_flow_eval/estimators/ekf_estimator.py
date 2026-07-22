#!/usr/bin/env python3
"""estimators/ekf_estimator.py — EKF with THREE landmark modes, one math core.

  mode 'none'  : the original 5-state constant-velocity filter (flow + depth).
                 NOTHING in this file changes its behavior — same state, same
                 update equations, same defaults.
  mode 'gate'  : the original MAP-BASED localization (FEKFMBL-style). The state
                 stays [px py pz vx vy]; a gate observation becomes a world-frame
                 position pseudo-measurement via update_position_xy(), anisotropic
                 (x tight because the rulebook fixes gate x, y = 1e12 no-op because
                 the arena randomizes y). This was and remains CORRECT for what it
                 knows: with only gate-x known, only robot-x may be corrected.
  mode 'slam'  : FEKFSLAM-style STATE AUGMENTATION. Each named feature appends
                 [fx fy] to the state; P grows with the proper cross-covariance
                 blocks so a later observation of the gate legitimately corrects
                 robot y THROUGH the correlation created at first sighting — this
                 is the mathematically sound version of "learn gate y, then use it".

The mode is chosen by the CALLER (flow_eval_node's `landmark_mode` parameter):
'none' calls neither method, 'gate' calls update_position_xy, 'slam' calls
update_feature. All three share _predict/_kalman, which are dimension-generic, so
switching modes cannot corrupt the base filter.

STATE LAYOUT (slam mode):
    x = [ px py pz vx vy | f1x f1y | f2x f2y | ... ]      features are STATIC
    feature j occupies columns BASE+2j : BASE+2j+2, BASE = 5.

FEATURE AUGMENTATION (first sighting), mirroring FEKFSLAM AddNewFeatures:
    z    = relative position of the feature in body xy (from the detector)
    f    = p_xy + R(yaw) z                       (inverse observation g(x, z))
    Gx   = dg/dx  = [I2  0 ...]                  (2 x n, identity on px,py)
    Gz   = dg/dz  = R(yaw)
    Jyaw = dg/dyaw = [[-s,-c],[c,-s]] z          (yaw is NOT in the state: the AHRS
                                                  supplies it; its sigma_yaw is
                                                  injected here instead)
    Rw   = Gz R_body Gz^T + Jyaw sigma_yaw^2 Jyaw^T
    x <- [x; f]
    P <- [[P,      P Gx^T          ],
          [Gx P,   Gx P Gx^T + Rw ]]             <- THE cross-covariance blocks
                                                     that were missing before.

FEATURE UPDATE (later sightings), mirroring FEKFSLAMFeature.hfj/Jhfjx:
    h(x)  = R(yaw)^T (f_j - p_xy)                (expected body-frame observation)
    H     = [ -R^T  0 | 0 ... R^T ... 0 ]        (robot block, feature block)
    R_eff = R_body + Jh_yaw sigma_yaw^2 Jh_yaw^T
    gated with a chi-square(2) Mahalanobis test before it may touch the state.

Pure numpy, no ROS. Frame-agnostic as before: the caller supplies measurements
already in the compare frame.
"""

import numpy as np

BASE = 5                       # px py pz vx vy
CHI2_2DOF_99 = 9.21            # Mahalanobis gate, 2 dof, 99 %


class EkfEstimator:
    def __init__(self, q_pos=0.01, q_vel=0.5, r_flow=0.04, r_depth=4e-6,
                 sigma_yaw=0.02, chi2_gate=CHI2_2DOF_99,
                 init_sightings=3, init_max_spread=1.0,
                 min_range=0.3, max_range=25.0):
        # state: px py pz vx vy (+ 2 per feature in slam mode)
        self.x = np.zeros(BASE)
        self.P = np.eye(BASE) * 1.0
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.R_flow = np.eye(2) * r_flow
        self.R_depth = r_depth
        self.sigma_yaw = sigma_yaw          # AHRS yaw 1-sigma [rad] (~1 deg)
        self.chi2_gate = chi2_gate
        self.init_sightings = init_sightings
        self.init_max_spread = init_max_spread
        self.min_range = min_range
        self.max_range = max_range
        self.t_prev = None
        self.initialized = False
        # slam bookkeeping
        self.features = {}                  # name -> slot j
        self._pending = {}                  # name -> list of candidate world (fx, fy)
        self.rejected = {}                  # name -> count of chi2-gated updates

    # ------------------------------------------------------------------ core
    def _predict(self, dt):
        n = self.x.size
        F = np.eye(n)
        F[0, 3] = dt                        # px += vx dt
        F[1, 4] = dt                        # py += vy dt
        self.x = F @ self.x
        Q = np.zeros((n, n))
        Q[0, 0] = Q[1, 1] = Q[2, 2] = self.q_pos * dt
        Q[3, 3] = Q[4, 4] = self.q_vel * dt
        # features get ZERO process noise: SAUVC props are static. The map may
        # only move through correlation with the robot, never on its own.
        self.P = F @ self.P @ F.T + Q

    def _step_time(self, t):
        if self.t_prev is None:
            self.t_prev = t
            return 0.0
        dt = t - self.t_prev
        self.t_prev = t
        if 0.0 < dt < 1.0:
            self._predict(dt)
        return dt

    def _kalman(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.x.size)
        # Joseph form: keeps P symmetric positive-definite once it grows large
        # with features (the plain (I-KH)P form was fine at 5x5, it is not once
        # cross-covariance blocks matter).
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

    # ------------------------------------------------- original measurements
    def update_flow(self, vx_body, vy_body, yaw, t, r_var=None):
        """r_var: optional per-measurement variance (m/s)^2 — quality-scaled
        noise seam (spread_px / n_inliers), as before."""
        self._step_time(t)
        c, s = np.cos(yaw), np.sin(yaw)
        z = np.array([c * vx_body - s * vy_body,
                      s * vx_body + c * vy_body])
        H = np.zeros((2, self.x.size)); H[0, 3] = 1.0; H[1, 4] = 1.0
        R = self.R_flow if r_var is None else np.eye(2) * max(float(r_var), 1e-6)
        self._kalman(z, H, R)
        self.initialized = True

    def update_depth(self, pz, t):
        self._step_time(t)
        H = np.zeros((1, self.x.size)); H[0, 2] = 1.0
        self._kalman(np.array([pz]), H, np.array([[self.R_depth]]))
        self.initialized = True

    def update_position_xy(self, px, py, t, var_x, var_y):
        """GATE MODE (unchanged semantics). World-frame position pseudo-measurement
        p_meas = landmark_world - rel_obs_world, anisotropic per axis. Gate x known
        from the rulebook -> finite var_x; gate y randomized -> var_y = 1e12 no-op.
        State is NOT augmented; this is localization against a known map."""
        self._step_time(t)
        H = np.zeros((2, self.x.size)); H[0, 0] = 1.0; H[1, 1] = 1.0
        R = np.diag([max(float(var_x), 1e-6), max(float(var_y), 1e-6)])
        self._kalman(np.array([px, py]), H, R)

    @property
    def position(self):
        return self.x[0], self.x[1], self.x[2]

    @property
    def velocity(self):
        return self.x[3], self.x[4]

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
        Returns 'init-pending' | 'init' | 'ok' | 'gated' | 'range'.
        """
        rng = float(np.hypot(zx, zy))
        if not (self.min_range <= rng <= self.max_range):
            return 'range'                      # GUARD 1: implausible range
        self._step_time(t)
        c, s = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[c, -s], [s, c]])
        z = np.array([zx, zy], float)

        if name not in self.features:
            return self._try_init(name, z, Ry, s, c, R_body, known)

        # ---- update an existing feature -------------------------------
        i0, i1 = self._slot(name)
        f = self.x[i0:i1]
        p = self.x[0:2]
        d = f - p
        h = Ry.T @ d                            # expected body observation
        n = self.x.size
        H = np.zeros((2, n))
        H[:, 0:2] = -Ry.T                       # robot block  (J_ominus path)
        H[:, i0:i1] = Ry.T                      # feature block (J_2boxplus path)
        # yaw is not a state: fold its uncertainty into R.
        # dh/dyaw = d(R^T)/dyaw d = [[-s, c], [-c, -s]] d
        Jy = np.array([-s * d[0] + c * d[1],
                       -c * d[0] - s * d[1]])
        R_eff = R_body + np.outer(Jy, Jy) * self.sigma_yaw ** 2

        nu = z - h
        S = H @ self.P @ H.T + R_eff
        m2 = float(nu @ np.linalg.solve(S, nu))
        if m2 > self.chi2_gate:                 # GUARD 4: Mahalanobis gate
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return 'gated'
        self._kalman(z, H, R_eff)
        return 'ok'

    def _try_init(self, name, z, Ry, s, c, R_body, known):
        """GUARD 2+3: a feature only enters the state after `init_sightings`
        consistent world-frame candidates (spread below init_max_spread). The
        first frame that ever sees a prop is usually the worst one (partial blob,
        grazing angle) and in SLAM the first fix is what the map — and every
        later relocalization — inherits."""
        cand = self.x[0:2] + Ry @ z
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
        f = self.x[0:2] + Ry @ z                # linearize at the CURRENT obs
        Gx = np.zeros((2, n)); Gx[0, 0] = 1.0; Gx[1, 1] = 1.0
        Jyaw = np.array([-s * z[0] - c * z[1],
                          c * z[0] - s * z[1]])
        Rw = Ry @ R_body @ Ry.T + np.outer(Jyaw, Jyaw) * self.sigma_yaw ** 2

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
