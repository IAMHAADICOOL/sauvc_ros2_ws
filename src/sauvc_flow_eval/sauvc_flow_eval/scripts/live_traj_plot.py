#!/usr/bin/env python3
"""live_traj_plot — live x-y trajectory animation for flow_eval_node.

A self-contained, OpenCV-only (no matplotlib) top-down plotter that overlays the
ground-truth track and every estimator track in ONE window while the run is going,
then saves the exact current frame as a PNG when the node shuts down (Ctrl+C).

DESIGN NOTES (why it works the way it does):

  * DATA-DRIVEN AUTOSCALE, GROW-ONLY. The view is never hardcoded to the world
    origin. The first point that arrives (usually the ground-truth anchor, which
    can be anywhere in the world frame, e.g. (9, -12) in the finals scene) defines
    the initial view center, with a small minimum span (`min_span_m`) so the plot
    starts ZOOMED-IN instead of wasting the window on empty pool. As the tracks
    spread out, the data bounding box grows and the view re-fits every frame —
    and because a bounding box over an append-only point set can only grow, the
    scale change is monotonic and smooth ("grows interactively"), never jumpy.
    Offsets therefore need no special handling at all: whatever the spawn pose,
    the view is centred on the data, not on (0, 0).

  * EQUAL ASPECT. Horizontal and vertical metres-per-pixel are forced equal
    (the smaller data span is padded out), so the trajectory shape is never
    stretched — a square in the pool looks square on screen.

  * FRAME-AWARE AXES. compare_frame='ned' (the node default) is drawn map-style:
    North (x) UP, East (y) RIGHT. 'enu' is drawn math-style: x RIGHT, y UP.
    The axis labels say which is which, so there is no ambiguity in screenshots.

  * THREAD-SAFE FEEDING. add() is called from the node's subscription callbacks
    via _publish(); render()/save() may run from a ROS timer or from main()'s
    shutdown path. Everything mutable sits behind one lock. Points are decimated
    on ingest (min_step_m) so hour-long runs stay light on memory and draw time.

  * NO-OVERWRITE SAVING. save() timestamps the filename to the second AND, if a
    file with that name somehow already exists (two nodes killed in the same
    second, clock reset, ...), appends _1, _2, ... until the name is free.

  * SAFE WHEN HEADLESS. render() returns an image; the caller decides whether to
    cv2.imshow it. save() works even if no window was ever shown, so the PNG is
    still produced on a machine with no display (imshow failure is the caller's
    concern, and flow_eval_node guards it the same way it guards its camera
    windows).

Only dependency: numpy + cv2, both already hard requirements of flow_eval_node's
visual features.
"""
import os
import threading
import time as _time

import numpy as np

try:
    import cv2
    _HAVE_CV = True
except Exception:            # pragma: no cover - matches flow_eval_node's guard
    _HAVE_CV = False


# BGR colors per track — chosen to match the mental model from the terminal
# table: truth is white, flow warm, ekf green, gtsam magenta, dvl cyan.
_DEFAULT_COLORS = {
    'ground_truth': (255, 255, 255),
    'flow':         (60, 160, 255),    # orange-ish
    'ekf':          (80, 220, 80),     # green
    'gtsam':        (230, 80, 230),    # magenta
    'dvl':          (230, 220, 60),    # cyan-ish
    'tile_grid':    (0, 215, 255),     # yellow
    'pressure':     (140, 140, 140),   # grey (normally excluded anyway)
}
_FALLBACK_COLOR = (0, 215, 255)        # yellow, for any unknown key


class TrajectoryPlotter:
    """Accumulates (x, y) points per named track and renders a top-down view."""

    def __init__(self, frame='ned', tracks=('ground_truth', 'flow', 'ekf',
                                            'gtsam', 'dvl'),
                 size_px=760, margin_px=64, min_span_m=2.0, pad_frac=0.08,
                 min_step_m=0.02, max_points_per_track=20000, colors=None):
        """
        frame       : 'ned' (north-up plot) or 'enu' (y-up plot). Labeling only —
                      the points are stored exactly as given (compare-frame x, y).
        tracks      : keys accepted by add(); anything else is ignored. Excluding
                      'pressure' by default matters: its x/y are fixed at 0 and
                      would drag the autoscaled view out to the world origin.
        min_span_m  : view is never tighter than this many metres across, so the
                      start doesn't render as a giant dot at max zoom.
        pad_frac    : padding added around the data bbox (fraction of span).
        min_step_m  : ingest decimation — a point is stored only if it moved at
                      least this far from the previously stored point.
        max_points_per_track : hard cap; beyond it every 2nd point is dropped
                      (self-halving), preserving overall shape indefinitely.
        """
        self.frame = str(frame).lower()
        self.include = set(tracks)
        self.size = int(size_px)
        self.margin = int(margin_px)
        self.min_span = float(min_span_m)
        self.pad = float(pad_frac)
        self.min_step = float(min_step_m)
        self.max_pts = int(max_points_per_track)
        self.colors = dict(_DEFAULT_COLORS)
        if colors:
            self.colors.update(colors)
        self._lock = threading.Lock()
        self._pts = {}            # key -> list[(x, y)]
        self._bbox = None         # [min_x, min_y, max_x, max_y] over ALL tracks
        self._t0 = _time.time()

    # ------------------------------------------------------------------ feed
    def add(self, key, x, y):
        """Record one estimate. Cheap, lock-guarded, decimated. Call freely at
        publish rate; NaN/inf points are dropped (a NaN would poison the bbox
        and blank the whole plot forever)."""
        if key not in self.include:
            return
        x = float(x)
        y = float(y)
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        with self._lock:
            pts = self._pts.setdefault(key, [])
            if pts:
                lx, ly = pts[-1]
                if (x - lx) ** 2 + (y - ly) ** 2 < self.min_step ** 2:
                    return
            pts.append((x, y))
            if len(pts) > self.max_pts:                 # self-halving history
                del pts[::2]
            if self._bbox is None:
                self._bbox = [x, y, x, y]
            else:
                b = self._bbox
                if x < b[0]: b[0] = x
                if y < b[1]: b[1] = y
                if x > b[2]: b[2] = x
                if y > b[3]: b[3] = y

    # ------------------------------------------------------------- projection
    def _view(self):
        """Current world window [cx, cy, half_span] — equal aspect, padded,
        clamped to min_span, and monotonic because the bbox only ever grows."""
        b = self._bbox
        cx, cy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
        span = max(b[2] - b[0], b[3] - b[1])
        span = max(span * (1.0 + 2.0 * self.pad), self.min_span)
        return cx, cy, span * 0.5

    def _to_px(self, x, y, cx, cy, half, plot_px):
        """World -> pixel. NED: y(East) right, x(North) up. ENU: x right, y up."""
        if self.frame == 'ned':
            h, v = y - cy, x - cx        # horizontal=East, vertical=North
        else:
            h, v = x - cx, y - cy
        s = plot_px / (2.0 * half)
        px = self.margin + (h + half) * s
        py = self.margin + (half - v) * s          # +v is UP on screen
        return int(round(px)), int(round(py))

    # ---------------------------------------------------------------- render
    def render(self):
        """Draw and return the current frame (BGR uint8). Never raises on empty
        data — shows a 'waiting for data' placeholder instead."""
        W = self.size + 2 * self.margin
        img = np.full((W, W, 3), 24, np.uint8)          # near-black background
        plot_px = self.size
        # plot area
        cv2.rectangle(img, (self.margin, self.margin),
                      (self.margin + plot_px, self.margin + plot_px),
                      (55, 55, 55), 1)
        with self._lock:
            snapshot = {k: list(v) for k, v in self._pts.items()}
            bbox = None if self._bbox is None else list(self._bbox)
        if bbox is None:
            cv2.putText(img, 'trajectory: waiting for data...',
                        (self.margin + 20, W // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (160, 160, 160), 1, cv2.LINE_AA)
            return img
        self._bbox_snapshot = bbox
        cx, cy, half = self._view()
        self._draw_grid(img, cx, cy, half, plot_px)
        # tracks: ground truth last so it stays on top of the estimates
        order = sorted(snapshot, key=lambda k: k == 'ground_truth')
        for key in order:
            pts = snapshot[key]
            col = self.colors.get(key, _FALLBACK_COLOR)
            if len(pts) >= 2:
                arr = np.array([self._to_px(x, y, cx, cy, half, plot_px)
                                for x, y in pts], np.int32)
                cv2.polylines(img, [arr], False, col, 2, cv2.LINE_AA)
            # start marker (hollow) + current position (filled)
            sx, sy = self._to_px(*pts[0], cx, cy, half, plot_px)
            ex, ey = self._to_px(*pts[-1], cx, cy, half, plot_px)
            cv2.circle(img, (sx, sy), 5, col, 1, cv2.LINE_AA)
            cv2.circle(img, (ex, ey), 4, col, -1, cv2.LINE_AA)
        self._draw_legend(img, snapshot)
        self._draw_titles(img, half)
        return img

    # ------------------------------------------------------------ decorations
    def _nice_tick(self, span):
        """Pick a 1/2/5-style grid pitch giving ~4-8 lines across the view."""
        raw = span / 6.0
        mag = 10.0 ** np.floor(np.log10(max(raw, 1e-9)))
        for m in (1.0, 2.0, 5.0, 10.0):
            if raw <= m * mag:
                return m * mag
        return 10.0 * mag

    def _draw_grid(self, img, cx, cy, half, plot_px):
        tick = self._nice_tick(2.0 * half)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # world-axis ranges shown horizontally/vertically depend on the frame
        if self.frame == 'ned':
            h0, v0 = cy, cx          # horizontal axis = y(E), vertical = x(N)
        else:
            h0, v0 = cx, cy
        lo_h, hi_h = h0 - half, h0 + half
        lo_v, hi_v = v0 - half, v0 + half
        m, sz = self.margin, plot_px
        for val in np.arange(np.ceil(lo_h / tick) * tick, hi_h + 1e-9, tick):
            px = int(round(m + (val - lo_h) / (2 * half) * sz))
            cv2.line(img, (px, m), (px, m + sz), (45, 45, 45), 1)
            cv2.putText(img, f'{val:g}', (px - 14, m + sz + 22), font, 0.42,
                        (150, 150, 150), 1, cv2.LINE_AA)
        for val in np.arange(np.ceil(lo_v / tick) * tick, hi_v + 1e-9, tick):
            py = int(round(m + (hi_v - val) / (2 * half) * sz))
            cv2.line(img, (m, py), (m + sz, py), (45, 45, 45), 1)
            cv2.putText(img, f'{val:g}', (6, py + 4), font, 0.42,
                        (150, 150, 150), 1, cv2.LINE_AA)

    def _draw_titles(self, img, half):
        font = cv2.FONT_HERSHEY_SIMPLEX
        W = img.shape[1]
        if self.frame == 'ned':
            hlab, vlab = 'y East [m]', 'x North [m]'
        else:
            hlab, vlab = 'x [m]', 'y [m]'
        cv2.putText(img, hlab, (W // 2 - 50, W - 10), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, vlab, (8, self.margin - 12), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        el = _time.time() - self._t0
        cv2.putText(img,
                    f'{self.frame.upper()} top-down   span {2 * half:.1f} m   '
                    f't+{el:5.1f} s',
                    (self.margin, 20), font, 0.5, (200, 200, 200), 1,
                    cv2.LINE_AA)

    def _draw_legend(self, img, snapshot):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x0 = self.margin + 12
        y = self.margin + 22
        keys = [k for k in ('ground_truth', 'dvl', 'flow', 'ekf', 'gtsam',
                            'tile_grid', 'pressure') if k in snapshot]
        keys += [k for k in sorted(snapshot) if k not in keys]  # unknown tracks too
        for key in keys:
            col = self.colors.get(key, _FALLBACK_COLOR)
            cv2.line(img, (x0, y - 4), (x0 + 24, y - 4), col, 2, cv2.LINE_AA)
            lx, ly = snapshot[key][-1]
            cv2.putText(img, f'{key}  ({lx:+.2f}, {ly:+.2f})',
                        (x0 + 32, y), font, 0.45, col, 1, cv2.LINE_AA)
            y += 20

    # ------------------------------------------------------------------ save
    def save(self, directory, basename='flow_eval_traj'):
        """Render the CURRENT frame and write it as a PNG into `directory`
        (created if needed). Timestamped name + _N suffix on collision, so an
        existing file is never overwritten. Returns the path, or None if there
        was never any data (nothing worth saving) or cv2 is unavailable."""
        if not _HAVE_CV:
            return None
        with self._lock:
            empty = self._bbox is None
        if empty:
            return None
        directory = os.path.expanduser(directory)
        os.makedirs(directory, exist_ok=True)
        stamp = _time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(directory, f'{basename}_{stamp}.png')
        n = 0
        while os.path.exists(path):
            n += 1
            path = os.path.join(directory, f'{basename}_{stamp}_{n}.png')
        cv2.imwrite(path, self.render())
        return path
