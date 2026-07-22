#!/usr/bin/env python3
"""
flow_core.py — Downward-camera optical-flow velocity ("DIY DVL") for SAUVC.

Method:
  1. Track sparse corners frame-to-frame with pyramidal Lucas-Kanade (no loop closure,
     so repetitive pool tiles are NOT a problem).
  2. Take the MEDIAN pixel displacement (robust to caustics / moving highlights) and
     gate outliers with MAD (median absolute deviation).
  3. Derotate: subtract the apparent flow caused by pitch/roll rates (gyro), so tilting
     the vehicle is not mistaken for translation.
  4. Scale by altitude above the pool floor (from pressure sensor: pool_depth - depth)
     to get metric velocity, and rotate into the body frame.

Publishes: geometry_msgs/TwistWithCovarianceStamped on /flow/twist
Subscribes: sensor_msgs/Image on /camera_down/image_raw
            sensor_msgs/Imu   on /imu/data        (angular velocity, body frame)
            std_msgs/Float32  on /altitude        (meters above pool floor)

The math lives in FlowVelocityEstimator (no ROS imports) so you can unit-test it and run
it offline with offline_flow_test.py.
"""

import math
import numpy as np
import cv2


class FlowVelocityEstimator:
    """Pure-python core. Feed grayscale frames + gyro rates + altitude, get body velocity."""

    def __init__(self, fx, fy, cx, cy,
                 max_corners=150, quality=0.01, min_distance=12,
                 swap_xy=False, sign_x=1.0, sign_y=1.0,
                 min_features=25,
                 max_hold_frames=5, max_hold_dt=0.5, fb_max_err=1.0,
                 use_clahe=False, grid_rows=0, grid_cols=0,
                 compensate_vz=False):
        # Camera intrinsics — MUST be from an UNDERWATER calibration.
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.min_features = min_features
        # Mounting convention fix-ups (set from the hand-push test, see README Phase 3).
        self.swap_xy = swap_xy
        self.sign_x = sign_x
        self.sign_y = sign_y

        # FIX(dropout = lost displacement): on a tracking failure the old code advanced
        # prev_gray/prev_pts to the CURRENT frame before returning None, so the motion
        # across every failed interval was permanently discarded — in dead reckoning
        # that subtracts straight off total displacement (measured: flow track = 0.44x
        # ground truth, 48% of moving intervals frozen). Now a failure HOLDS the last
        # good reference frame so the next successful LK solve spans the whole gap, and
        # the accumulated gap time (hold_dt) is used as the velocity denominator.
        # Bounded: after max_hold_frames or max_hold_dt the track is declared lost and
        # the reference hard-resets (accepting that one gap's loss, now VISIBLE via
        # last_failure/fail_counts instead of silent). max_hold_dt should stay below
        # the integrator's gap guard (PositionIntegrator ignores dt >= 1.0 s).
        self.max_hold_frames = max_hold_frames
        self.max_hold_dt = max_hold_dt
        self.fb_max_err = fb_max_err  # px, forward-backward round-trip gate
        # FIX(periodic-texture aliasing): pool tiles repeat, so across a held gap LK
        # can lock onto the WRONG grid line one period over — and the FB check passes,
        # because the false lock is self-consistent. Two defences:
        #   1. Seed LK with the displacement PREDICTED from the last good pixel rate
        #      (OPTFLOW_USE_INITIAL_FLOW), so convergence starts near the true lock.
        #   2. Gate recovered estimates on velocity plausibility: an AUV cannot jump
        #      more than ~a_max * gap in speed, so a recovered velocity far from the
        #      pre-dropout one is an alias, not motion.
        self.px_rate = None           # (du/s, dv/s) from the last good estimate
        self.last_v = None            # (vx, vy) body, last good estimate
        self.v_jump_max = 0.8         # m/s, max plausible speed change across a hold
        # IMPROVEMENT(texture robustness):
        #   * CLAHE equalizes contrast so washed-out / caustic-lit floor patches still
        #     yield corners. CAVEAT (measured): CLAHE's per-tile mapping is SPACE-
        #     VARIANT — across a held-reference gap the same physical texture lands in
        #     different tiles and gets different equalization, which breaks the LK
        #     forward-backward check and kills gap recovery. Enable it only for
        #     genuinely low-contrast footage where corners are otherwise starved;
        #     default OFF.
        #   * Grid-distributed detection forces goodFeaturesToTrack to spread corners
        #     across an R x C grid instead of clustering on the strongest texture,
        #     which is what starves 'few_tracked' on half-featureless frames.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        # OPTIONAL(expansion/Vz compensation): the translational-flow derivation drops
        # the term fx*X*(dZ/dt)/Z^2, which is only exactly zero if Vz=0 or the tracked
        # feature sits at the image center. Moving toward/away from the floor while
        # ALSO translating leaks a position-dependent "zoom" pattern into the median
        # flow whenever the inlier features aren't perfectly symmetric about the
        # principal point -- grid-distributed detection above makes that assumption
        # much safer, but doesn't guarantee it every frame. This removes the leftover
        # leakage using the same altitude reading this class already receives every
        # frame -- no new sensor input required. Off by default: validate with the
        # hand-push test (see offline_flow_test.py) before trusting it.
        self.compensate_vz = bool(compensate_vz)
        self.ref_altitude = None      # altitude recorded when prev_gray/prev_pts was set
        self.hold_frames = 0          # consecutive failed frames on the held reference
        self.hold_dt = 0.0            # seconds accumulated across held frames
        self.last_failure = None      # why the most recent process() returned None
        self.fail_counts = {}         # reason -> count, for diagnostics

        self.prev_gray = None
        self.prev_pts = None
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                        30, 0.01))

    def _detect(self, gray):
        if self.grid_rows > 0 and self.grid_cols > 0:
            # IMPROVEMENT: per-cell quota keeps features spread over the whole image.
            h, w = gray.shape[:2]
            quota = max(self.max_corners // (self.grid_rows * self.grid_cols), 3)
            found = []
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    y0, y1 = h * r // self.grid_rows, h * (r + 1) // self.grid_rows
                    x0, x1 = w * c // self.grid_cols, w * (c + 1) // self.grid_cols
                    pts = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1],
                                                  maxCorners=quota,
                                                  qualityLevel=self.quality,
                                                  minDistance=self.min_distance)
                    if pts is not None:
                        pts[:, 0, 0] += x0
                        pts[:, 0, 1] += y0
                        found.append(pts)
            if not found:
                return None
            return np.vstack(found).astype(np.float32)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners,
                                      qualityLevel=self.quality,
                                      minDistance=self.min_distance)
        return pts  # shape (N,1,2) float32 or None

    # --- reference-frame management (FIX: dropout = lost displacement) -----------------
    def _advance_ref(self, gray, altitude=None):
        """Move the LK reference to the current frame. Call ONLY on a successful
        estimate, or on non-tracking failures (bad altitude/dt) where the pixel
        displacement could not have been used anyway.

        altitude: the altitude reading AT this reference frame, stored so the next
        successful process() can compute how much altitude changed across the
        interval it spans (dt_eff) for the optional Vz compensation. None (bootstrap,
        or an invalid reading) disables compensation until a good reading returns."""
        self.prev_gray = gray
        self.prev_pts = self._detect(gray)
        self.hold_frames = 0
        self.hold_dt = 0.0
        self.ref_altitude = altitude

    def _fail(self, gray, dt, reason, hold, altitude=None):
        """Record a failure. hold=True keeps the previous reference so the next
        successful track spans the gap; the gap time accrues in hold_dt. Once the
        hold limits are exceeded the track is declared lost and the reference
        resets to the current frame (the loss is accepted — but counted)."""
        self.last_failure = reason
        self.fail_counts[reason] = self.fail_counts.get(reason, 0) + 1
        if not hold:
            self._advance_ref(gray, altitude)
            return None
        self.hold_frames += 1
        self.hold_dt += max(dt, 0.0)
        if self.hold_frames > self.max_hold_frames or self.hold_dt > self.max_hold_dt:
            self.last_failure = reason + '+track_lost'
            self.fail_counts['track_lost'] = self.fail_counts.get('track_lost', 0) + 1
            self._advance_ref(gray, altitude)
        return None

    def process(self, gray, dt, gyro_xy_cam, altitude):
        """
        gray        : uint8 grayscale frame (undistorted, or low-distortion lens)
        dt          : seconds since previous frame
        gyro_xy_cam : (wx, wy) angular rates in the CAMERA frame [rad/s]
                      (camera x = image u/right, camera y = image v/down)
        altitude    : meters between camera and pool floor
        Returns dict with vx, vy (body frame, m/s), quality info; or None if no estimate.
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if self.clahe is not None:
            gray = self.clahe.apply(gray)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.last_failure = 'bootstrap'
            self._advance_ref(gray, altitude)
            return None

        if dt <= 0:
            # Bad stamps: nothing sane can be integrated. Keep the reference (no time
            # has provably passed) and don't accrue hold_dt.
            self.last_failure = 'bad_dt'
            self.fail_counts['bad_dt'] = self.fail_counts.get('bad_dt', 0) + 1
            return None

        # FIX(aliasing defence 1): predictive seeding. Start LK's search at the
        # displacement the last good pixel rate predicts over the spanned interval.
        # Frame-to-frame this is a no-op refinement; across a held gap it is what
        # steers convergence to the TRUE lock instead of a tile-period alias.
        dt_span = dt + self.hold_dt
        lk_kwargs = dict(self.lk_params)
        guess = None
        if self.px_rate is not None:
            pred = np.array(self.px_rate, dtype=np.float32) * dt_span
            guess = (self.prev_pts + pred.reshape(1, 1, 2)).astype(np.float32)
            lk_kwargs['flags'] = cv2.OPTFLOW_USE_INITIAL_FLOW
        nxt, status, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                                     self.prev_pts, guess, **lk_kwargs)
        if nxt is None:
            # FIX: was `self.prev_gray, self.prev_pts = gray, self._detect(gray)` —
            # discarding the gap. HOLD instead.
            return self._fail(gray, dt, 'lk_none', hold=True, altitude=altitude)

        # FIX(false matches): forward-backward consistency check. LK can "succeed"
        # onto the wrong texture — especially across a held gap or on repetitive
        # tiles — yielding a confident, WRONG (often near-zero) flow that is worse
        # than a dropout because it gets integrated. Track back to the reference and
        # keep only points whose round trip lands within fb_max_err px. Sparse LK on
        # <=150 points is ~free next to goodFeaturesToTrack, so this runs always.
        back, bstat, _berr = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray,
                                                      nxt, None, **self.lk_params)
        if back is None:
            return self._fail(gray, dt, 'lk_none', hold=True, altitude=altitude)
        fb_err = np.linalg.norm(
            self.prev_pts.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
        good = ((status.reshape(-1) == 1) & (bstat.reshape(-1) == 1)
                & (fb_err < self.fb_max_err))
        p0 = self.prev_pts.reshape(-1, 2)[good]
        p1 = nxt.reshape(-1, 2)[good]
        n_tracked = len(p0)

        # FIX: the old code re-seeded the reference HERE, "regardless of what happens
        # below" — which meant every failed check below silently deleted this
        # interval's displacement. The reference now advances only on SUCCESS (end of
        # this function) or on non-tracking failures where the displacement is
        # unusable anyway (bad altitude).

        if n_tracked < self.min_features:
            return self._fail(gray, dt, 'few_tracked', hold=True, altitude=altitude)

        if altitude is None or altitude < 0.1:
            # Tracking is fine but the displacement cannot be scaled to metres, so
            # holding buys nothing — advance so the pixel gap doesn't grow. altitude
            # is invalid here, so don't record it as the new ref_altitude (None keeps
            # Vz compensation disabled until a good reading re-establishes it).
            return self._fail(gray, dt, 'bad_altitude', hold=False, altitude=None)

        # Effective interval: the current frame gap PLUS any frames held across
        # failures. Velocity must be displacement over the WHOLE spanned time or a
        # recovered track would overestimate speed by (1 + hold_dt/dt).
        dt_eff = dt + self.hold_dt

        d = p1 - p0                                   # per-feature pixel displacement
        med = np.median(d, axis=0)                    # robust central flow (du, dv)
        mad = np.median(np.abs(d - med), axis=0) + 1e-6
        # Keep features within 4 MAD of the median (rejects caustic sparkles, fish, ropes).
        inlier = np.all(np.abs(d - med) < 4.0 * mad + 1.0, axis=1)
        n_inliers = int(inlier.sum())
        if n_inliers < self.min_features:
            return self._fail(gray, dt, 'few_inliers', hold=True, altitude=altitude)
        du, dv = np.median(d[inlier], axis=0)         # pixels per SPANNED interval

        # --- Derotation ---------------------------------------------------------------
        # Near the image center, rotational flow:  u_dot ~= -f * wy ,  v_dot ~= +f * wx
        # Subtract it to isolate translational flow. Uses the CURRENT rates over
        # dt_eff — exact for frame-to-frame, first-order for a recovered hold (rates
        # over a <=0.5 s gap are treated as constant; acceptable for the transients
        # this term exists to reject).
        wx, wy = gyro_xy_cam
        du_t = du - (-self.fx * wy * dt_eff)
        dv_t = dv - (+self.fy * wx * dt_eff)

        # --- OPTIONAL: expansion/Vz compensation ---------------------------------------
        # Full translational term: u_dot = -fx*Vx/Z + (x_img/Z)*Vz -- moving toward/away
        # from the floor (Vz != 0) adds a "zoom" pattern growing with distance from the
        # principal point. Antisymmetric in x_img, so it cancels in the median IF inlier
        # features are symmetric about the image center (grid-distributed detection
        # above makes that likelier, not guaranteed). This removes the leftover leakage
        # directly: Vz*dt_eff = ref_altitude - altitude (altitude change across the SAME
        # interval this flow spans), so the correction needs no separate dt term.
        if self.compensate_vz and self.ref_altitude is not None:
            delta_alt = self.ref_altitude - altitude   # >0 while descending (closer to floor)
            x_bar = float(np.mean(p0[inlier, 0])) - self.cx
            y_bar = float(np.mean(p0[inlier, 1])) - self.cy
            du_t -= x_bar * delta_alt / altitude
            dv_t -= y_bar * delta_alt / altitude

        # --- Scale to metric camera-frame velocity --------------------------------------
        # Ground point image motion is opposite camera translation: u_dot = -f * Vx_c / h
        vx_cam = -(du_t / dt_eff) * altitude / self.fx    # camera x = image right
        vy_cam = -(dv_t / dt_eff) * altitude / self.fy    # camera y = image down

        # --- Camera -> body mapping (fix with hand-push test) ---------------------------
        # Default assumption: image "up" (-v) is vehicle forward (+x body),
        # image "right" (+u) is vehicle right (-y body in ROS, since y is left).
        bx, by = -vy_cam, -vx_cam
        if self.swap_xy:
            bx, by = by, bx
        bx *= self.sign_x
        by *= self.sign_y

        recovered = self.hold_dt          # >0 means this estimate spans a held gap

        # FIX(aliasing defence 2): a velocity recovered across a held gap that jumps
        # implausibly from the pre-dropout velocity is a periodic-texture alias, not
        # motion. Keep holding (the next frame, with seeding, usually resolves it);
        # the hold limits still bound the worst case.
        if recovered > 0.0 and self.last_v is not None:
            jump = math.hypot(bx - self.last_v[0], by - self.last_v[1])
            if jump > self.v_jump_max:
                return self._fail(gray, dt, 'alias_reject', hold=True, altitude=altitude)

        self.px_rate = (du / dt_eff, dv / dt_eff)
        self.last_v = (float(bx), float(by))
        self.last_failure = None
        self._advance_ref(gray, altitude)  # success: NOW the reference moves forward

        # Quality: spread of inlier flow (pixels) — big spread = unreliable frame.
        spread = float(np.mean(np.std(d[inlier], axis=0)))
        return dict(vx=float(bx), vy=float(by),
                    n_tracked=n_tracked, n_inliers=n_inliers,
                    spread_px=spread, flow_px=(float(du), float(dv)),
                    dt_eff=float(dt_eff), recovered_gap_s=float(recovered))


