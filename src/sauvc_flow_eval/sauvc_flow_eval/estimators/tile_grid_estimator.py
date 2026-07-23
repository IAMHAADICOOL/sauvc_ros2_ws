#!/usr/bin/env python3
"""estimators/tile_grid_estimator.py — grid-phase odometry on the pool floor tiles.

IDEA (why this can beat optical flow): the pool floor is a KNOWN PERIODIC RULER.
sauvc_pool.scn puts pool_tiles.png on the floor boxes with uv_mode="2" (texture tiled
once per meter of face) and the texture holds 20 tiles -> grout pitch P = 0.05 m, grid
axes fixed to the WORLD x/y axes. So instead of integrating velocity (flow: every
frame's error accumulates forever), we measure the PHASE of the grid every frame:
where the camera sits *within* a tile, along both grid axes. That is an ABSOLUTE
measurement mod P — sub-tile error can never accumulate. Only whole-tile miscounts
can, and only if the vehicle moves > P/2 between measurements untracked (a velocity
hint from the flow estimator covers that, see below).

PIPELINE per frame (gray image, altitude, yaw hint):
  1. SCALE     px-per-meter = fx / altitude  ->  expected grid period p_px = P*fx/alt.
  2. ANGLE     grout lines are strong edges; a gradient-orientation histogram folded
               mod 90 deg gives the grid angle alpha in the image. The absolute grid
               direction in the world is a multiple of 90 deg, so alpha + the IMU yaw
               hint picks the multiple k — the hint may drift up to +-45 deg and k is
               still right. Bonus output: a DRIFT-FREE heading fix (grid_yaw).
  3. PHASE     rotate a central patch by -alpha (grid-aligned), average rows/columns
               into two 1-D profiles, and lock-in (single-bin Fourier transform) at
               the known period p_px. arg -> camera position mod P along each grid
               axis; normalized magnitude -> quality gate.
  4. WORLD     grid axes ARE world axes (up to the snapped k*90), so the two phases
               map straight to "position mod P along world-x / world-y". Heading drift
               therefore NEVER corrupts position — unlike flow dead reckoning, where
               yaw error rotates the whole integrated track.
  5. UNWRAP    frame-to-frame phase deltas wrapped to +-P/2 accumulate into a
               continuous world x,y. An optional world-velocity hint (from the flow
               estimator) predicts the coarse displacement so speeds beyond the
               "half-a-tile-per-frame" Nyquist limit (P/2 * fps = 0.75 m/s at 30 Hz
               for P=5 cm) still unwrap onto the correct tile, and the same
               prediction coasts across quality dropouts (flare crossings, blur).

CONVENTIONS (self-contained, verified by tile_grid_offline_test.py against synthetic
renders of the real pool_tiles.png):
  * Image frame: x right, y down; angles measured from image-x toward image-y.
  * World frame: NED horizontal (x north, y east); yaw from x toward y. The horizontal
    2-D math is a proper rotation between the two.
  * `cam_axis_yaw` = world angle of the IMAGE X AXIS. For my_auv.scn's down camera
    (mount rpy z=+90 deg: image right = body right, image up = body forward) this is
    vehicle_yaw_ned + pi/2; the eval node adds a tunable offset parameter for the same
    reason flow has sign_x/sign_y — verify against ground truth, then trust it.
  * Output x,y are DISPLACEMENT FROM START in NED world (like the other dead-reckoning
    estimators; the eval node anchors all tracks to ground truth's first pose).

No ROS imports; mirrors the FlowEstimator module pattern (estimate() + overlay()).
"""

import numpy as np

try:
    import cv2
    _HAVE_CV = True
except Exception:  # pragma: no cover
    _HAVE_CV = False

TWO_PI = 2.0 * np.pi


def _wrap(v, period):
    """Wrap v into (-period/2, +period/2]."""
    return v - period * np.round(v / period)


class TileGridEstimator:
    """Feed grayscale down-camera frames + altitude + a coarse yaw; get world x,y."""

    def __init__(self, fx, tile_pitch=0.05, patch=256,
                 min_quality=1.5, min_period_px=6.0, max_speed=2.0,
                 innov_gate_m=0.015, innov_relock_n=8):
        self.fx = float(fx)
        self.pitch = float(tile_pitch)
        self.patch = int(patch)
        self.min_quality = float(min_quality)
        self.min_period_px = float(min_period_px)
        self.max_speed = float(max_speed)   # sanity clamp on per-frame displacement
        # INNOVATION GATE (added after the 2026-07-23 run): a +50 mm x-jump happened
        # WHILE STATIONARY at the spawn — the patch there sees a mix of tiles and the
        # starting-zone marker, and the fold peak jumped one full period while still
        # passing the quality gate. Because the integer tile count is dead-reckoned,
        # that slip then persisted for the whole run as a fixed offset. With a velocity
        # hint available, the wrapped innovation vs the prediction should be a few mm
        # per frame; anything above innov_gate_m is physically impossible motion ->
        # reject the measurement and coast on the hint instead. innov_relock_n guards
        # the opposite failure (hint broken, measurement right): after that many
        # CONSECUTIVE gated frames the measurement is accepted again (re-lock).
        self.innov_gate = float(innov_gate_m)
        self.innov_relock_n = int(innov_relock_n)
        self._gated_streak = 0

        self.available = _HAVE_CV
        # continuous world position (displacement from first lock), NED x/y
        self.x = 0.0
        self.y = 0.0
        self.locked = False
        self.prev_m = None      # previous (mx, my): position mod pitch per world axis
        self.prev_t = None
        self.last = None        # last result dict (for overlay / debugging)

    # ------------------------------------------------------------------ internals --
    def _grid_angle(self, patch):
        """Grid angle mod 90 deg in image coords, from the gradient orientation
        histogram folded onto 4*theta (grout edges dominate). Returns (alpha, coherence)."""
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag2 = gx * gx + gy * gy
        theta = np.arctan2(gy, gx)
        z = np.sum(mag2 * np.exp(4j * theta))
        denom = np.sum(mag2) + 1e-9
        coherence = np.abs(z) / denom
        alpha = 0.25 * np.angle(z)          # in (-pi/8, pi/8]
        return alpha, coherence

    def _fold_peak(self, profile, period_px, oversample=64):
        """Grout-line position mod period, by FOLDING the profile modulo the known
        period and taking the sub-pixel (parabolic) peak.

        Why not a single-bin Fourier ("lock-in")? Tried first; the random tile colors
        put broadband power right next to the grout frequency, and the leakage skews
        the measured phase RATE by ~8-10% (spectral-centroid bias). Folding is the
        matched filter for the full grout comb (all harmonics at once): the random
        background folds to roughly flat, the grout folds to one sharp peak, and the
        peak POSITION is a direct spatial measurement — no rate bias. Grout is BRIGHT
        (look grout rgb ~0.9 vs blue tiles), so median-subtract + clamp >= 0 isolates
        it. Returns (peak_position_px, quality_zscore)."""
        p = profile - np.median(profile)
        p = np.clip(p, 0.0, None)
        n = len(p)
        S = int(oversample)
        step = period_px / S
        ss = np.arange(S) * step
        # positions of all fold samples: ss[:,None] + j*period
        nper = int((n - 1 - ss[-1]) // period_px) + 1
        pos = ss[:, None] + np.arange(nper)[None, :] * period_px
        i0 = pos.astype(np.int64)
        f = (pos - i0).astype(np.float32)
        score = np.mean(p[i0] * (1.0 - f) + p[i0 + 1] * f, axis=1)
        k = int(np.argmax(score))
        a, b, c = score[(k - 1) % S], score[k], score[(k + 1) % S]
        denom = a - 2.0 * b + c
        d = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
        d = float(np.clip(d, -0.5, 0.5))
        peak = ((k + d) % S) * step
        q = float((b - score.mean()) / (score.std() + 1e-9))
        return peak, q

    # ------------------------------------------------------------------ public API --
    def estimate(self, gray, t, altitude, cam_axis_yaw, vel_world_hint=None):
        """One frame.

        gray          uint8 grayscale image
        t             timestamp [s]
        altitude      camera height above the floor [m]
        cam_axis_yaw  world (NED) angle of the image x axis [rad]
                      (vehicle_yaw_ned + pi/2 + mount offset for this vehicle)
        vel_world_hint  optional (vx, vy) world NED [m/s] coarse velocity (e.g. from
                      the flow estimator) used to unwrap fast motion / coast dropouts

        Returns dict(x, y, mx, my, alpha, grid_yaw, quality, p_px, locked) or None
        if the frame is unusable. x,y are cumulative world displacement from start.
        """
        if not self.available or gray is None or altitude is None:
            return None
        dt = 0.0 if self.prev_t is None else max(0.0, t - self.prev_t)

        # predicted displacement since last measurement (also used to coast dropouts)
        if vel_world_hint is not None and dt > 0.0:
            pdx = float(np.clip(vel_world_hint[0] * dt, -self.max_speed * dt,
                                self.max_speed * dt))
            pdy = float(np.clip(vel_world_hint[1] * dt, -self.max_speed * dt,
                                self.max_speed * dt))
        else:
            pdx = pdy = 0.0

        p_px = self.pitch * self.fx / max(float(altitude), 0.05)
        h, w = gray.shape[:2]
        half = min(self.patch, min(h, w) - 2) // 2
        if p_px < self.min_period_px or half * 2 < 4 * p_px:
            return self._miss(t, pdx, pdy)   # too high / patch sees <4 periods

        cxp, cyp = w // 2, h // 2
        patch = np.asarray(
            gray[cyp - half:cyp + half, cxp - half:cxp + half], dtype=np.float32)

        # ---- grid orientation in the image (mod 90 deg) ----
        alpha4, coh = self._grid_angle(patch)
        if coh < 0.05:
            return self._miss(t, pdx, pdy)
        # image angle of the grid axis expected nearest the world-K axis pointed at by
        # the image x axis: a priori that image angle is -cam_axis_yaw (mod 90).
        m_u = alpha4 + (np.pi / 2) * np.round((-cam_axis_yaw - alpha4) / (np.pi / 2))
        # exact world angle of that grid axis, then snap to the nearest multiple of 90:
        w_u = cam_axis_yaw + m_u
        k_u = (np.pi / 2) * np.round(w_u / (np.pi / 2))
        # drift-free heading CORRECTION: true cam_axis_yaw = supplied hint + grid_yaw
        # (the grid axis truly lies at k_u*90; any residual is the hint's error).
        grid_yaw_corr = _wrap(k_u - w_u, TWO_PI)

        # ---- grid-align the patch and project to two 1-D profiles ----
        c = half  # patch center
        deg = np.degrees(m_u)
        # rotate the SAMPLING GRID so the output x axis lies along the grid u axis.
        # warpAffine WITHOUT WARP_INVERSE_MAP inverts M internally (M is src->dst), so
        # passing +deg makes dst-x sample along image angle m_u (verified empirically
        # in tile_grid_offline_test.py across cam yaws 0/90/180/-45).
        M = cv2.getRotationMatrix2D((c, c), deg, 1.0)
        aligned = cv2.warpAffine(patch, M, (2 * half, 2 * half),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # trim the rotation-invalid corners
        q = int(half * (np.sqrt(2) - 1) / np.sqrt(2)) + 2
        core = aligned[q:2 * half - q, q:2 * half - q]
        prof_u = core.mean(axis=0)   # varies along aligned x  -> grid u axis
        prof_v = core.mean(axis=1)   # varies along aligned y  -> grid v axis

        pk_u, amp_u = self._fold_peak(prof_u, p_px)
        pk_v, amp_v = self._fold_peak(prof_v, p_px)
        quality = min(amp_u, amp_v)
        if quality < self.min_quality:
            return self._miss(t, pdx, pdy)

        # peak -> CAMERA position mod pitch along the aligned axes. A grout line sits
        # at world multiples of P; its pixel position x_g (measured FROM THE PRINCIPAL
        # POINT — critical: altitude changes scale the image about the principal point,
        # so a corner-referenced peak drifts ~O_px * d(mpp) per frame, ~1% of distance
        # on an altitude ramp before this fix) obeys c + x_g*mpp = n*P + g0, so x_g
        # moves OPPOSITE to the camera: c mod P = -peak_px * mpp (+ const, cancels in
        # diffs). Profile index 0 sits at aligned-patch coord q, i.e. (q - half) px
        # from the patch center = principal point.
        mpp = float(altitude) / self.fx
        off = float(q - half)
        s_u = -((pk_u + off) % p_px) * mpp
        s_v = -((pk_v + off) % p_px) * mpp

        # ---- map (s_u, s_v) onto WORLD axes via the snapped k*90 ----
        cu, su_ = np.cos(k_u), np.sin(k_u)          # world dir of the u grid axis
        if abs(cu) > abs(su_):                       # u axis ~ world x
            mx = _wrap(np.sign(cu) * s_u, self.pitch)
            my = _wrap(np.sign(cu) * s_v, self.pitch)   # v = u rotated +90 -> sign follows
        else:                                        # u axis ~ world y
            my = _wrap(np.sign(su_) * s_u, self.pitch)
            mx = _wrap(-np.sign(su_) * s_v, self.pitch)

        # ---- unwrap into continuous world displacement ----
        if not self.locked:
            self.locked = True
            self.x = self.y = 0.0
        else:
            pm = self.prev_m
            inn_x = _wrap((mx - pm[0]) - pdx, self.pitch)
            inn_y = _wrap((my - pm[1]) - pdy, self.pitch)
            if (vel_world_hint is not None and 0.0 < dt < 0.2
                    and max(abs(inn_x), abs(inn_y)) > self.innov_gate
                    and self._gated_streak < self.innov_relock_n):
                self._gated_streak += 1
                return self._miss(t, pdx, pdy)   # coast; do NOT ingest the outlier
            self._gated_streak = 0
            dx = pdx + inn_x
            dy = pdy + inn_y
            lim = self.max_speed * max(dt, 1e-3)
            self.x += float(np.clip(dx, -lim, lim))
            self.y += float(np.clip(dy, -lim, lim))
        self.prev_m = (mx, my)
        self.prev_t = t

        self.last = dict(x=self.x, y=self.y, mx=mx, my=my,
                         alpha=float(m_u), grid_yaw=float(grid_yaw_corr),
                         quality=float(quality), coherence=float(coh),
                         p_px=float(p_px), locked=True, coasting=False)
        return self.last

    def _miss(self, t, pdx, pdy):
        """Unusable frame: coast on the velocity hint so the next phase measurement
        unwraps onto the right tile, but report coasting so the caller can tell."""
        if self.locked:
            self.x += pdx
            self.y += pdy
            if self.prev_m is not None:
                self.prev_m = (_wrap(self.prev_m[0] + pdx, self.pitch),
                               _wrap(self.prev_m[1] + pdy, self.pitch))
        self.prev_t = t
        if self.last is not None:
            self.last = dict(self.last, x=self.x, y=self.y, coasting=True)
        return None

    # ------------------------------------------------------------------ overlay ----
    def overlay(self, frame_bgr):
        """Draw the detected grid orientation + phase state onto a BGR frame."""
        img = frame_bgr.copy()
        if self.last is None:
            return img
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        a = self.last['alpha']
        L = 60
        for ang, col in ((a, (0, 255, 255)), (a + np.pi / 2, (255, 200, 0))):
            dx, dy = int(L * np.cos(ang)), int(L * np.sin(ang))
            cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), col, 2)
        txt = (f"tile: x={self.last['x']:+.3f} y={self.last['y']:+.3f} m  "
               f"q={self.last['quality']:.2f}  p={self.last['p_px']:.0f}px"
               + ("  COAST" if self.last.get('coasting') else ""))
        cv2.putText(img, txt, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        return img
