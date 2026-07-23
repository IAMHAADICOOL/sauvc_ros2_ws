# `flow_core.py` — Exhaustive Walkthrough

A complete, line-by-line explanation of `flow_core.py`: what each method does, the
purpose and reasoning behind every parameter, and the math behind the core equations.

---

## What this whole file is for

Your AUV has a camera pointing straight down at the pool floor. As the vehicle moves, the floor texture "slides" across the image. This file watches that sliding and converts it into "the vehicle is moving at this many meters per second, in this direction." It's basically building a cheap, camera-based DVL (Doppler Velocity Log — the real, expensive sensor that measures speed off the seafloor).

The whole thing is one class, `FlowVelocityEstimator`, with no ROS code in it at all — it just takes images/numbers in and gives velocity numbers out, so it can be tested by itself.

---

## The big-picture method (from the docstring, lines 1–22)

1. **Track corners frame to frame** using Lucas-Kanade (LK) optical flow — pick out distinctive little spots on the floor (grout lines, tile edges) and follow where each one moves between two frames.
2. **Take the median** of all those little movements, not the average — median ignores oddball trackers (sun-glint sparkles, a passing fish, a rope) much better than an average would.
3. **Derotate** — if the vehicle rotates in place (yaws, pitches, rolls) the image appears to "move" even with zero translation. Subtract that fake motion using the gyroscope.
4. **Scale to real-world meters** using altitude (how far the camera is from the floor) — the same pixel movement means a bigger real movement when you're farther from the floor.

---

## Imports (lines 24–26)

```python
import math
import numpy as np
import cv2
```
- `math` — plain Python math functions (used once, for `math.hypot`, i.e. straight-line distance between two numbers).
- `numpy` (`np`) — array math, used everywhere for handling lists of tracked points efficiently.
- `cv2` — OpenCV, the computer vision library. Does the actual corner-detection and optical-flow tracking.

---

## `__init__` — the constructor (lines 32–110)

This is where the object is created and all its settings (parameters) get stored. Let's go setting by setting.

### Camera intrinsics: `fx, fy, cx, cy`
```python
self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
```
These describe the camera lens itself:
- `fx, fy` — the "focal length" in pixels (how zoomed-in the lens is, essentially). Bigger number = more pixels per meter of real-world size at a given distance.
- `cx, cy` — the pixel coordinates of the exact center of the image (the "principal point"). Not always exactly width/2, height/2, but close.

**Why these values matter:** they're the whole basis for turning "pixels moved" into "meters moved." Get them wrong, and every velocity number is scaled wrong by a fixed percentage. Per your memory notes, these were recently updated to `fx=fy=381.36, cx=320, cy=240` for the new 640×480 camera. If you use the old 1280×720 numbers (762.72, 640, 360) with the new image size, your velocities will be off by roughly 2x.

### `max_corners=150, quality=0.01, min_distance=12`
These control **how many trackable points get picked**, using OpenCV's `goodFeaturesToTrack` (called later in `_detect`):
- `max_corners` — the most corner points it will ever pick per frame. More points = smoother/more robust median, but slower to compute. 150 is a "plenty but still fast" number.
- `quality` — a threshold (0 to 1) for how "cornery" a spot must be relative to the single best corner in the image, to be accepted. Lower = accepts weaker/blurrier corners too (more points, more noise). Higher = only the sharpest corners (fewer points, cleaner). 0.01 is quite permissive — it's tuned to make sure a boring/low-texture floor still yields enough points.
- `min_distance` — minimum pixel gap enforced between any two selected corners, so they don't all cluster in one spot. 12 px keeps them spread out a bit even if there's one very "cornery" area.

**If you change these:** raise `max_corners` for smoother data but slower processing; raise `quality` if you're getting bad shaky data from too many weak points; raise `min_distance` if points are clumping in one texture-rich patch and going blind everywhere else.

**A little extra on `min_distance` specifically:** it's a parameter OpenCV's `goodFeaturesToTrack` uses *during corner selection itself*, not a filter applied afterward. Here's how the algorithm actually works: it first finds every pixel that scores well as a "corner" (using the Shi-Tomasi corner-quality measure), then sorts all of them by strength — best corner first. It walks through that sorted list and greedily accepts a corner only if it's **at least `min_distance` pixels away from every corner already accepted**. So if a picture has a really rich, cornery patch (say, a scuffed tile intersection with lots of edges), the raw scoring would want to hand you 20 corners all crammed into that one small area — `min_distance` is what stops that: once one is picked, its 12-pixel-radius neighborhood is effectively "off limits" for further picks, even if a slightly weaker candidate exists there.

A couple of practical consequences worth knowing:
- **It trades quality for coverage.** A corner accepted under this rule might genuinely be a *weaker* corner than one that got excluded for being too close to a stronger one already picked. That's the intended trade — a mediocre corner on the far side of the image is more useful to this system than another great corner sitting right next to one you already have, since spread-out points make the median/outlier-rejection math more reliable and (per the Vz-compensation comment below) keeps points roughly symmetric around the image center.
- **It's measured in the full, uncropped image** — when grid-mode is on, `min_distance` still applies only *within* each grid cell (since `goodFeaturesToTrack` is called separately per cell), so points from two different cells could in theory end up closer together than 12 px right at a cell boundary.
- **12 px is fairly small** at 640×480 resolution — it's mostly there to stop pure duplicate/overlapping corners on the same physical feature rather than to aggressively force wide spacing; the grid-cell splitting (when enabled) does the heavier lifting for actual spread across the whole image.

### `swap_xy=False, sign_x=1.0, sign_y=1.0`
```python
self.swap_xy = swap_xy
self.sign_x = sign_x
self.sign_y = sign_y
```
These exist purely to correct for **how the camera is physically mounted** on the vehicle. Depending on which way the camera is bolted in, "image right" might correspond to "vehicle left" instead of "vehicle right," or the axes might be swapped entirely (image up = vehicle sideways instead of forward). Rather than guess, these are three knobs you tune empirically:
- `swap_xy` — swaps forward/sideways if the camera is rotated 90°.
- `sign_x`, `sign_y` — flip the sign (+1 or -1) if an axis is backwards.

The comment says these come from a "hand-push test" — literally picking up the vehicle and pushing it by hand in a known direction while watching what the flow estimator reports, then adjusting these three numbers until it matches. Default is "no correction" (1.0, 1.0, false).

**If you change these wrong:** velocity comes out flipped or rotated 90°, which would look like the vehicle drifting in a totally wrong direction even though the raw tracking is fine.

### `min_features=25`
```python
self.min_features = min_features
```
The minimum number of successfully-tracked points required before the estimator trusts the result at all. Below this, there's too little data to get a reliable median. Used in two places later: after tracking (`n_tracked`) and after outlier-rejection (`n_inliers`).

**If you lower it:** you'll get estimates more often (fewer dropouts) but each one is noisier/riskier. **If you raise it:** fewer, but more trustworthy, estimates — more dropouts on a boring floor.

**How this differs from `max_corners`:** the two control opposite ends of the same pipeline. `max_corners` is the ceiling on how many corner points `_detect()` is *allowed to pick* in the first place, back when scanning a fresh reference frame — a supply-side cap: "don't hand me more than 150 candidates." `min_features` is the floor on how many points must *survive tracking* by the time `process()` is deciding whether to trust the result — a demand-side minimum: "don't give me an answer unless at least 25 points made it through alive." So `max_corners=150` is the most points you ever start with; `min_features=25` is the fewest you're allowed to end with. Points get lost along the way — some corners fail to track at all, some fail the forward-backward round-trip, some get rejected as outliers — so you naturally want `max_corners` well above `min_features`, giving the pipeline a comfortable margin to lose points and still clear the bar.

### Hold / dropout-recovery settings: `max_hold_frames=5, max_hold_dt=0.5, fb_max_err=1.0`
```python
self.max_hold_frames = max_hold_frames
self.max_hold_dt = max_hold_dt
self.fb_max_err = fb_max_err
```
This is explaining a whole bug-fix documented in the big comment above it (lines 50–63). Here's the story in plain terms:

Old behavior: if tracking failed for one frame (couldn't find a good match), the code would just move on and use the CURRENT frame as the new starting point next time — silently throwing away the fact that the vehicle moved during that failed frame. Do this repeatedly and your tracked position falls further and further behind reality. Measured result: the flow-based position estimate ended up at 44% of the real distance traveled, because *almost half of all moving intervals were silently zeroed out*.

New behavior: instead of giving up and resetting, a failure **holds** onto the last known-good reference frame, and keeps trying to match the CURRENT frame against that same old reference. As soon as a match succeeds, the whole gap (however many frames it spanned) gets treated as one big time interval, and the velocity divides displacement by the WHOLE elapsed time (not just the last single frame), so nothing is lost.

- `max_hold_frames` (default 5) — if it goes this many consecutive frames without a successful match, give up and accept the loss (declare "track lost" and start fresh from the current frame). This limits how "stale" the held reference frame can get — a frame from 5 frames ago might look too different for LK to match reliably at all.
- `max_hold_dt` (default 0.5 seconds) — same idea, but measured in time instead of frame count, in case the camera framerate is irregular. 0.5s is chosen to stay under a related sanity check elsewhere in the pipeline (the position integrator ignores any single time-gap ≥ 1.0s as "obviously broken data" — see the comment).
- `fb_max_err` (default 1.0 pixel) — a completely separate reliability check called "forward-backward": after tracking a point forward (old frame → new frame), it also tracks it backward (new frame → old frame). If a point is a genuinely good match, going forward then backward should land you back almost exactly where you started. If it lands more than `fb_max_err` pixels away, that point is thrown out as an unreliable/false match. 1.0 px is a strict, "must round-trip almost perfectly" threshold.

**If you raise `max_hold_frames`/`max_hold_dt`:** the system tries harder to recover from dropouts (less silent data loss), but risks trying to match frames that are now too different, causing false matches (see the anti-aliasing section below). **If you lower `fb_max_err`:** stricter, throws out more borderline points, fewer false matches but possibly fewer usable points. Raise it and you'll accept sloppier matches.

**What "holding" actually means, concretely:** normally, every time `process()` succeeds, it calls `_advance_ref()`, which throws away the old reference frame and makes the *current* frame the new reference for next time. So under normal conditions the reference frame is always just "the previous frame" — nothing is being held onto. But if tracking fails on a given frame (say, too few points survived, or LK couldn't find a match), instead of still calling `_advance_ref()` and moving the reference forward anyway (the old, buggy version of this code did exactly that, and it's what silently threw away 48% of the real motion), the new code does the opposite: it *keeps the old reference frame exactly as it was* and just increments two counters — `hold_frames` and `hold_dt`. So "holding" literally means: the reference frame is frozen in place — not updated — while the estimator waits for a frame that tracks successfully against it. Frame after frame, if tracking keeps failing, the code keeps *not* moving the reference forward, piling up `hold_frames`/`hold_dt` to record how long it's been frozen. This can go on for up to `max_hold_frames` or `max_hold_dt` — whichever limit is hit first — before it gives up and force-advances anyway (accepting the loss). If, before those limits are hit, a frame comes along that *does* track successfully against that same old, frozen reference frame — that's the recovery: the displacement is measured between the old held reference and this new successful frame, spanning however many frames were skipped in between. For example: reference frame set at frame 1 → frames 2, 3, 4, 5 all fail to track (each one bumps `hold_frames`, reference stays frozen at frame 1) → frame 6 finally succeeds, matching directly against that still-frozen frame-1 reference, spanning the whole 5-frame gap in one velocity measurement.

**How can the old reference frame still successfully match the current frame, despite frames having been lost in between?** LK tracking doesn't need the in-between frames at all — it only ever needs *two* images: a starting image and an ending image, plus a rough idea of where to search. It literally never looks at what happened in the missed frames; it just asks "where, in THIS current image, does this same patch of floor texture now appear, compared to where it was in the OLD reference image?" As long as the floor texture hasn't drifted *too* far or changed *too* much between those two specific frames (which is exactly why `max_hold_frames`/`max_hold_dt` exist as a limit), LK can jump straight from frame 1 to frame 6, skipping 2–5 entirely, and still successfully find the match. The predictive seeding described in Step 3 below (giving LK a head-start guess based on the last known pixel rate) is what makes this bigger jump feasible instead of LK getting lost searching too small a window.

**Displacement over the whole gap, divided by the whole gap's time, gives velocity** — yes, exactly: displacement = (position in current frame) − (position in the old held reference frame); time = `dt_eff`, the *entire* span from when that reference frame was set until now, not just the last single frame's `dt`. So `velocity = total displacement ÷ total elapsed time across the whole gap`. That's precisely why nothing gets lost: instead of discarding the motion during the failed frames, it gets folded into one wider-but-still-correct velocity measurement.

**Why both `max_hold_frames` and `max_hold_dt` exist, given that one already limits how long a hold can go on:** "5 frames" isn't a fixed amount of *time* — camera framerate can vary (dropped frames, CPU hiccups, variable publish rate). If the camera briefly stutters, 5 frames could span 0.15s (fine) or 2 full seconds (way too stale, texture has drifted too much, LK will likely mismatch or alias onto a repeating tile). `max_hold_frames` guards against "too many gaps in a row" assuming normal timing; `max_hold_dt` independently guards against "too much real time passed" regardless of how many frames that took. Whichever limit gets hit first ends the hold — they're two different failure modes (frame-count-based vs. wall-clock-based) covering for each other.

### Anti-aliasing state: `px_rate`, `last_v`, `v_jump_max=0.8`
```python
self.px_rate = None
self.last_v = None
self.v_jump_max = 0.8
```
This addresses a specific danger of the "hold and recover" strategy above: **pool floor tiles repeat**. If the camera drifted by exactly one tile-width during a held gap, the tracker might confidently — but wrongly — lock onto the identical-looking neighboring tile edge instead of the true matching one. The forward-backward check above wouldn't catch this, because the false match is self-consistent (forward and backward both agree on the wrong tile).

Two defenses against this, both using these variables:
- `self.px_rate` — remembers the last known good "pixels moved per second" rate, so that on the next attempt, the tracker's search can be given a head start / prediction of where to look, steering it toward the correct (true) match instead of a repeating pattern one tile over.
- `self.last_v` — remembers the last known good body-frame velocity (meters/second), used as a sanity check: if the newly-recovered velocity is wildly different from the last trusted one, that's suspicious.
- `self.v_jump_max` (0.8 m/s) — the threshold for "wildly different." An AUV physically can't change speed by 0.8 m/s instantly, so if a recovered estimate implies that, it's almost certainly a tile-aliasing error, not real motion, and gets rejected.

**If you raise `v_jump_max`:** more tolerant of sudden apparent speed changes (fewer good measurements thrown out, but tile-aliasing errors get through more easily). **Lower it:** stricter, more protection against aliasing, but a genuinely fast real acceleration might get wrongly rejected too.

### Texture-robustness options: `use_clahe`, `grid_rows`, `grid_cols`
```python
self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
self.grid_rows = int(grid_rows)
self.grid_cols = int(grid_cols)
```
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — brightens up murky/washed-out/caustic-lit footage so corners are easier to find, by boosting local contrast in small tiles across the image (here, an 8×8 grid of tiles, each allowed to boost contrast up to a limit of 2.0 so it doesn't create fake noise-corners). **Off by default** — the comment explains why: CLAHE's contrast-boosting is different in each little tile of the image, so if the vehicle drifts, the *same physical patch of floor* ends up sitting in a *different* CLAHE tile on the next frame and gets brightened differently — breaking the forward-backward matching trick above. So it's a genuine trade-off: helps in low-visibility water, hurts dropout recovery. Only turn it on if corners are genuinely too scarce otherwise.
- `grid_rows`, `grid_cols` — instead of letting `goodFeaturesToTrack` pick its top corners wherever it likes (which tends to cluster all points on the single richest-texture patch, e.g. one bright grout intersection, leaving the rest of the image "blind"), this splits the image into an R×C grid and picks a quota of corners from *each* cell separately, forcing spread-out coverage. Default is `0`, `0`, meaning this feature is off (behaves like plain `goodFeaturesToTrack` over the whole image) unless explicitly turned on (memory shows it's set to 3 rows × 4 cols by the node).

**Why spread-out coverage matters, in physical terms:** when the vehicle moves purely sideways/forward, every tracked point on the floor appears to slide by roughly the same amount — that's the actual signal you want (the median flow). But when the vehicle is *also* rising or sinking at the same time, there's a second effect layered on top: the whole image subtly "zooms" in or out, like a photo taken while walking toward or away from a wall. This zoom effect isn't uniform across the image — a point sitting far to the right of center appears to shift right; a point far to the left of center appears to shift left; a point near the center barely moves at all from this effect. The further a point sits from the image's center, the bigger this zoom-caused apparent shift is, and the *direction* of that shift depends on which side of center it's on.

The key trick the code relies on: if the tracked points are scattered roughly evenly on both sides of the center, the zoom-shifts point in opposite directions and are similar in size — so when you take the median across all points, they cancel each other out, and only the true sideways-motion signal survives. That's what "antisymmetric ... cancels in the median" means in the comment below on Vz compensation. Now suppose instead all the tracked points happen to cluster in one corner of the image — say, all sitting to the right of center, because that's the only patch of floor with good texture that frame. Now every point's zoom-shift points the same direction and is a similar size. There's nothing left to cancel out — the median flow now includes a chunk of leftover "fake" motion caused purely by the vehicle rising/sinking, mixed in as if it were real sideways motion. So the whole flow measurement — not just the optional Vz-compensation code below — is more trustworthy when the tracked points are spread out roughly symmetrically around the image center, which is exactly the problem the grid-based detection (`grid_rows`/`grid_cols`) is there to fix, by forcing points to come from all parts of the image instead of whichever single patch had the richest texture.

### Vertical-motion (Vz) compensation: `compensate_vz=False`
```python
self.compensate_vz = bool(compensate_vz)
self.ref_altitude = None
```
This is explaining a subtle physics detail. The basic flow-to-velocity formula assumes the vehicle only translates sideways/forward. But if the vehicle is ALSO rising or sinking (changing altitude) at the same time as moving sideways, there's a secondary "zoom" effect: features near the edges of the image appear to move faster/slower than features near the center, purely because of the changing distance to the floor (like a photo "zooming in" as a drone rises). This is exactly zero only if either (a) the vehicle isn't changing altitude, or (b) the tracked points happen to sit exactly symmetrically around the center of the image (so the zoom-effect cancels itself out in the median, as described just above).

`compensate_vz` (off by default) turns on an extra correction step that removes this "zoom" leakage directly using the altitude readings the class already receives. It's off by default because it needs validating first (per the comment, with a dedicated hand-push test) — an untested correction could introduce a bug rather than fix one.

`self.ref_altitude` — remembers the altitude that was true at the moment the current reference frame was captured, so it can later measure "how much did the altitude change across this specific interval."

### Bookkeeping / diagnostic state
```python
self.hold_frames = 0
self.hold_dt = 0.0
self.last_failure = None
self.fail_counts = {}

self.prev_gray = None
self.prev_pts = None
```
- `hold_frames`, `hold_dt` — live counters (not settings) tracking how long the current "gap" has been open. Reset to 0 every time a real reference-advance happens.
- `last_failure` — a string recording *why* the most recent call failed (e.g. `'few_tracked'`, `'bad_altitude'`), so a human debugging this can see the actual cause instead of just "it returned None."
- `fail_counts` — a running tally, per failure-reason, of how many times each kind of failure has happened — useful for a summary log ("87% of failures were 'few_tracked', floor texture is too weak").
- `prev_gray` — the actual previous grayscale image (the "reference frame") to compare the next frame against.
- `prev_pts` — the actual pixel coordinates of the corners that were found in that reference frame.

### LK tracker settings
```python
self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                30, 0.01))
```
These are passed straight into OpenCV's `calcOpticalFlowPyrLK` (the actual point-tracking function) every time it's called:
- `winSize=(21, 21)` — the size (in pixels) of the little search window LK looks at around each point to figure out where it moved to. Bigger window = can handle bigger/faster motion between frames without losing the point, but is slower and can get confused by nearby different textures. 21×21 is a moderate, commonly-used size.
- `maxLevel=3` — LK works on an "image pyramid" (the same image shrunk down repeatedly, e.g. full-size, half-size, quarter-size, etc.), tracking first on the tiny blurry version to catch big motions, then refining on bigger, sharper versions. `maxLevel=3` means 4 pyramid levels total (0 through 3). More levels = can track larger movements between frames, at some computational cost.
- `criteria=...` — tells the internal LK math "stop refining once you've either done 30 iterations, OR the estimate stops improving by more than 0.01 (pixels), whichever comes first." This is a convergence/precision-vs-speed tradeoff; these are fairly standard OpenCV defaults.

**If you increase `winSize`:** tracking survives faster motion / bigger frame gaps, but is slower and can grab the wrong nearby texture more easily. **If you increase `maxLevel`:** same trade — handles bigger motions, costs more CPU.

---

## `_detect(self, gray)` — finding corners to track (lines 112–136)

This is called whenever the class needs to pick a fresh batch of trackable points from an image (i.e., whenever the reference frame advances).

```python
if self.grid_rows > 0 and self.grid_cols > 0:
```
If grid-distributed detection is turned on (see above), it goes into the grid branch:
```python
h, w = gray.shape[:2]
quota = max(self.max_corners // (self.grid_rows * self.grid_cols), 3)
```
Splits the total corner budget (`max_corners`) evenly across every cell of the R×C grid, but guarantees each cell gets *at least 3* points even if the math would round down to fewer — so no cell is left completely empty.

```python
found = []
for r in range(self.grid_rows):
    for c in range(self.grid_cols):
        y0, y1 = h * r // self.grid_rows, h * (r + 1) // self.grid_rows
        x0, x1 = w * c // self.grid_cols, w * (c + 1) // self.grid_cols
        pts = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1],
                                      maxCorners=quota,
                                      qualityLevel=self.quality,
                                      minDistance=self.min_distance)
```
Loops through every grid cell, computes that cell's pixel boundaries (`y0:y1, x0:x1`), and runs `goodFeaturesToTrack` on just that little sub-image slice, asking for up to `quota` corners with the class's `quality`/`min_distance` settings.

**The four arguments passed to `goodFeaturesToTrack` here, explained:**
- **image** (first, positional arg — `gray[y0:y1, x0:x1]`): the input image to search for corners in. Must be grayscale. Here it's not the full frame but a cropped slice — one cell of the grid.
- **maxCorners** (`quota`): the most corners to return from this call, sorted strongest-first. This is the per-cell version of the class-wide `max_corners` — split evenly across all grid cells (with a minimum of 3 per cell, as seen above).
- **qualityLevel** (`self.quality`, i.e. `0.01`): a relative threshold. OpenCV first finds the single strongest possible corner in the image, multiplies its score by this fraction, and then rejects any candidate corner scoring below that cutoff. So `0.01` means "accept anything at least 1% as strong as the best corner here" — quite permissive, needed because a plain pool floor doesn't have super sharp corners everywhere.
- **minDistance** (`self.min_distance`, i.e. `12`): the minimum pixel separation enforced between any two accepted corners (discussed above) — applied *within this cell's crop*, not the whole image.

Under the hood, `goodFeaturesToTrack` computes a "cornerness" score for every pixel using the Shi-Tomasi method (a refinement of the classic Harris corner detector), then applies the `qualityLevel` cutoff, then greedily picks corners strongest-first while enforcing `minDistance` between picks, up to `maxCorners` total. It returns an array of shape `(N, 1, 2)` — pixel `(x, y)` coordinates — or `None` if nothing passed the quality bar.

```python
if pts is not None:
    pts[:, 0, 0] += x0
    pts[:, 0, 1] += y0
    found.append(pts)
```
`goodFeaturesToTrack` returns coordinates *relative to the sub-image slice it was given* (i.e., starting from 0,0 in that little cell) — so this adds back the cell's offset (`x0`, `y0`) to convert those coordinates back into full-image coordinates. Then it's added to a running list.

```python
if not found:
    return None
return np.vstack(found).astype(np.float32)
```
If literally no cell found any corners at all, return `None` (nothing to track). Otherwise, stack all the per-cell corner lists into one big array and make sure it's the `float32` type OpenCV expects.

```python
pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners, ...)
return pts
```
If grid mode is off, this is the plain, simple version: just ask OpenCV for the best `max_corners` corners across the whole image at once, no cell-splitting. Returns either an array of shape (N, 1, 2) — N points, each an (x, y) pixel coordinate — or `None` if it found nothing.

---

## `_advance_ref(self, gray, altitude=None)` — moving the reference frame forward (lines 139–152)

```python
def _advance_ref(self, gray, altitude=None):
    self.prev_gray = gray
    self.prev_pts = self._detect(gray)
    self.hold_frames = 0
    self.hold_dt = 0.0
    self.ref_altitude = altitude
```
This is the function that says "okay, THIS frame is now our new starting point for comparison." It:
1. Stores the current image as the new reference (`prev_gray`).
2. Re-detects a fresh batch of corner points on it (`prev_pts`) — old tracked points from the previous reference are discarded and replaced.
3. Resets the "how long have we been stuck in a gap" counters back to zero.
4. Remembers the altitude at this exact moment (for the optional Vz-compensation math later).

Per the comment above it, this only gets called in two situations: (a) a successful velocity estimate was just produced, or (b) a failure happened that's fundamentally *not about tracking* (like bad altitude data) where holding the old reference wouldn't help anyway.

---

## `_fail(self, gray, dt, reason, hold, altitude=None)` — handling a failed frame (lines 154–170)

```python
def _fail(self, gray, dt, reason, hold, altitude=None):
    self.last_failure = reason
    self.fail_counts[reason] = self.fail_counts.get(reason, 0) + 1
```
Every failure, no matter the cause, gets logged: remember the reason as the most-recent failure, and bump the running tally for that specific reason (creating the tally entry at 0 first if it's the first time that reason has ever occurred).

```python
    if not hold:
        self._advance_ref(gray, altitude)
        return None
```
If this particular kind of failure isn't worth "holding" for (e.g. bad altitude data — tracking itself was fine, so there's nothing to gain by keeping the old reference), just advance the reference to the current frame anyway, then return `None` (no estimate this frame).

```python
    self.hold_frames += 1
    self.hold_dt += max(dt, 0.0)
```
Otherwise, this is a genuine "hold and try again next time" situation: bump the frame-count and the elapsed-time counters for how long we've been stuck (clamping `dt` at 0 in case of a weird negative time-step from bad timestamps). This is the exact moment "holding" happens — the reference frame is left untouched, and these two counters are the only thing that changes.

```python
    if self.hold_frames > self.max_hold_frames or self.hold_dt > self.max_hold_dt:
        self.last_failure = reason + '+track_lost'
        self.fail_counts['track_lost'] = self.fail_counts.get('track_lost', 0) + 1
        self._advance_ref(gray, altitude)
    return None
```
If we've now held for too long (past either limit set in `__init__`), give up: tag the failure reason with `+track_lost` so it's visible in logs that this wasn't just a normal recoverable blip but an actual accepted loss of data, tally it under `'track_lost'`, and finally advance the reference (start fresh from here). Either way — recovered later or truly lost — this function always returns `None` for this particular frame, since no velocity could be computed right now.

---

## `process(self, gray, dt, gyro_xy_cam, altitude)` — the main function, called every frame (lines 172–325)

This is the one method everything else exists to support. It's called once per camera frame and either returns a dictionary of results, or `None` if nothing usable could be computed this frame.

**Inputs:**
- `gray` — the current camera frame, grayscale.
- `dt` — seconds elapsed since the previous frame.
- `gyro_xy_cam` — the current rotation rates (`wx`, `wy`) of the camera, already converted into the camera's own coordinate frame (x = image-right, y = image-down).
- `altitude` — how many meters the camera currently is above the pool floor.

**What `gyro_xy_cam`'s `wx`, `wy` actually are:** `wx`, `wy` are not positions or directions of travel — they're spin rates. `w` (omega) conventionally means angular velocity — "how fast is this thing rotating," in radians per second, around a given axis. So `wx` = how fast it's spinning around the x-axis, `wy` = spinning around the y-axis. These come straight from the gyroscope inside the IMU; they describe *rotation*, not *position or orientation*.

"The camera's own coordinate frame" is just standard image-processing convention, not the vehicle's body frame. For any camera, computer vision defines its local axes like this: **x** = along the image, to the right; **y** = along the image, downward (because image row-numbers increase going down — pixel row 0 is the top row); **z** = straight out of the lens, into the scene (the "optical axis"). This is a fixed convention regardless of how the camera is mounted on anything — it's not describing forward/backward/left/right of the vehicle, it's describing "along the image" and "into the image." So "y is down" doesn't mean "backward" — it means "toward the bottom edge of the picture," the same way "down" means toward the bottom of a piece of paper; it's about the 2D image, not the 3D world. There's no "backward" axis in a camera's own frame, because a camera frame only has three axes and one of them (z) is already claimed by "straight out through the lens." And for a downward-looking camera, z genuinely does point toward the floor (world-down) — it's just that `process()` never needs `wz` (spin around the optical axis) for the derotation math below; only the two axes that visibly make the image *slide* (`wx`, `wy`) matter, since spinning around the boresight just rotates the image in place, which the corner-tracking math handles fine on its own.

**Why the gyro reading has to be converted into this frame at all:** the IMU reports rotation rates in the *vehicle's* body frame (forward/right/down or similar), because that's a natural way to describe how a vehicle is spinning. But the derotation formula in Step 9 below is written in terms of *image* pixel motion (`du`, `dv`), so it needs the rotation rates expressed in that *same* image-aligned axis system (camera x/y), not the vehicle's own body axes. Since the down-camera is mounted at some fixed rotation relative to the vehicle body, someone has to do a one-time conversion from "vehicle body spin rates" into "camera-frame spin rates" before this function can use them. That conversion (`gyro_xy_cam=(-wy, -wx)` per your memory notes) happens *before* `process()` is called — `process()` just receives the already-converted `(wx, wy)` pair as its `gyro_xy_cam` input and assumes they're already in the right frame.

### Step 0: color conversion + optional contrast boost
```python
if gray.ndim == 3:
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
if self.clahe is not None:
    gray = self.clahe.apply(gray)
```
If a color image was accidentally passed in (3 dimensions instead of 2), convert it to grayscale first. Then, if CLAHE contrast-boosting was enabled in the constructor, apply it now.

### Step 1: bootstrap check
```python
if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.min_features:
    self.last_failure = 'bootstrap'
    self._advance_ref(gray, altitude)
    return None
```
This handles the very first frame ever received (nothing to compare against yet), or a situation where the current reference frame simply doesn't have enough trackable points stored. In either case: there's nothing to compute a velocity from yet, so just set up THIS frame as the new reference and return `None`. Tagged `'bootstrap'` so it's clearly distinguished in logs from a real tracking failure.

### Step 2: bad timestamp check
```python
if dt <= 0:
    self.last_failure = 'bad_dt'
    self.fail_counts['bad_dt'] = self.fail_counts.get('bad_dt', 0) + 1
    return None
```
If the time-elapsed value is zero or negative (a corrupted/duplicate timestamp, clock issue, etc.), there's no sane way to compute a "distance per unit time." Note this does NOT hold or advance the reference — since no valid time has "provably passed," the reference and gap-timers are left completely untouched, waiting for a frame with a sane timestamp.

### Step 3: predictive seeding for the tracker
```python
dt_span = dt + self.hold_dt
lk_kwargs = dict(self.lk_params)
guess = None
if self.px_rate is not None:
    pred = np.array(self.px_rate, dtype=np.float32) * dt_span
    guess = (self.prev_pts + pred.reshape(1, 1, 2)).astype(np.float32)
    lk_kwargs['flags'] = cv2.OPTFLOW_USE_INITIAL_FLOW
```
`dt_span` is the total time since the reference frame was set (this frame's own gap, plus any additional time already accumulated from previous failed hold attempts).

If we have a previous known pixel-movement-rate (`px_rate`, set at the end of a previous successful `process()` call), it's used to *predict* where each tracked point probably is now: take each old point's position and add `(rate × elapsed time)`. This prediction (`guess`) is then handed to OpenCV as a starting point/head-start for its search (`OPTFLOW_USE_INITIAL_FLOW` tells LK "start searching from here instead of the point's own old position"). If there's no previous rate yet (first frame ever tracked), `guess` stays `None` and LK just uses its normal default starting point (the point's own old position).

**The problem this "predictive seeding" solves:** LK tracking works by searching a small window (`winSize=(21,21)`, so ~10 px in each direction) around a point's *old* position to find where it moved to. Normally, LK assumes "the point probably didn't move very far" and starts its search centered right on the point's old coordinates. That assumption is fine frame-to-frame at 30 fps, where things only shift a few pixels. But if, say, 4 frames got skipped during a hold, the point might have actually moved 40–50 pixels — way outside that little 21×21 search window. LK would search in the wrong place entirely and either fail to match, or worse, latch onto some *other* nearby texture that happens to look similar (like the next tile over).

**The fix:** instead of blindly starting the search at the point's old position, give LK a smarter starting guess — "based on how fast this point was moving last time we had a good measurement (`px_rate`, in pixels/second), and how much time has now passed (`dt_span`), here's roughly where I'd expect it to be *now*." That's the `pred = px_rate * dt_span` line — literally `distance = speed × time`. Adding that predicted offset to the old point positions (`self.prev_pts + pred`) gives a `guess` array of *expected current positions*. Passing that `guess` in, along with the `OPTFLOW_USE_INITIAL_FLOW` flag, tells OpenCV: "don't search around the old position — search around THIS predicted position instead." Now even a big multi-frame jump lands inside LK's small search window, because the search window moved along with the prediction instead of staying anchored to stale coordinates.

**Why this also fights tile-aliasing:** on a repeating tile floor, if you search near the *wrong* starting point (the old position), you might find a plausible-looking match at the neighboring tile purely because it looks similar. But if you start the search near the *correctly predicted* position, the nearby, correct tile edge is what's actually sitting inside the search window — steering LK toward the true match instead of a lookalike one tile over. It's a head start, not a guarantee — LK still does the real matching/refinement from that starting guess; the prediction just points it at the right neighborhood first.

### Step 4: the actual forward tracking call
```python
nxt, status, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                             self.prev_pts, guess, **lk_kwargs)
if nxt is None:
    return self._fail(gray, dt, 'lk_none', hold=True, altitude=altitude)
```
This is the core OpenCV call: given the reference image, the current image, and the reference points, find where each of those points has moved to in the current image. `nxt` = the new positions found, `status` = per-point flag (1 = successfully tracked, 0 = lost). `_err` (the per-point tracking error) is computed but intentionally unused (the underscore prefix signals "not needed here").

If LK totally fails (returns `None` outright, e.g. corrupted images), that's treated as a hold-worthy failure tagged `'lk_none'`.

### Step 5: forward-backward consistency check
```python
back, bstat, _berr = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray,
                                              nxt, None, **self.lk_params)
if back is None:
    return self._fail(gray, dt, 'lk_none', hold=True, altitude=altitude)
```
Track each point *backward* — from its new position in the current frame, back toward where it should land in the old reference frame. If this also totally fails, treat it the same as a forward tracking failure.

```python
fb_err = np.linalg.norm(
    self.prev_pts.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
good = ((status.reshape(-1) == 1) & (bstat.reshape(-1) == 1)
        & (fb_err < self.fb_max_err))
p0 = self.prev_pts.reshape(-1, 2)[good]
p1 = nxt.reshape(-1, 2)[good]
n_tracked = len(p0)
```
For every point, measure the straight-line pixel distance (`fb_err`) between where it *started* (its true old position) and where the backward-track landed it — ideally this distance should be ~0 for a trustworthy point. A point only counts as `good` if: (a) the forward track succeeded (`status==1`), AND (b) the backward track succeeded (`bstat==1`), AND (c) the round-trip error is under the `fb_max_err` threshold (1.0 px by default). `p0`/`p1` then keep only the surviving old/new coordinate pairs, and `n_tracked` is simply how many points passed this filter.

### Step 6: minimum-points check
```python
if n_tracked < self.min_features:
    return self._fail(gray, dt, 'few_tracked', hold=True, altitude=altitude)
```
If too few points survived the forward-backward filter, that's a hold-worthy failure — the floor texture probably didn't give enough good corners this frame (murky water, blank patch, etc.).

### Step 7: altitude sanity check
```python
if altitude is None or altitude < 0.1:
    return self._fail(gray, dt, 'bad_altitude', hold=False, altitude=None)
```
If there's no valid altitude reading (or it's absurdly close to zero — under 10 cm, basically "touching the floor," which would make the scaling math blow up toward infinity), the tracking itself was fine, but there's no way to convert pixel movement into meters. This is deliberately NOT a hold-worthy failure (`hold=False`) — since the points tracked fine, there's no benefit to keeping the old reference around; better to advance now so the pixel-gap doesn't keep growing for no reason. Note `altitude=None` is passed to `_advance_ref`, so `ref_altitude` stays `None` too — this deliberately disables the optional Vz-compensation feature until a genuinely good altitude reading comes back.

### Step 8: compute the actual displacement
```python
dt_eff = dt + self.hold_dt
d = p1 - p0
med = np.median(d, axis=0)
mad = np.median(np.abs(d - med), axis=0) + 1e-6
inlier = np.all(np.abs(d - med) < 4.0 * mad + 1.0, axis=1)
n_inliers = int(inlier.sum())
if n_inliers < self.min_features:
    return self._fail(gray, dt, 'few_inliers', hold=True, altitude=altitude)
du, dv = np.median(d[inlier], axis=0)
```
- `dt_eff` — the total real time this measurement spans (same as `dt_span` computed earlier — current frame's gap plus any accumulated hold time). This is the actual denominator used for all velocity math below.
- `d = p1 - p0` — for every surviving tracked point, how far it moved in pixels (a per-point `[du, dv]` vector).
- `med` — the median movement across all points — this is the initial best guess at "the floor's overall movement," robust to a handful of bad points.
- `mad` (median absolute deviation) — a measure of how spread-out/noisy the individual point movements are relative to that median. The tiny `+ 1e-6` just prevents a division-by-zero-like situation if all points agree exactly.
- `inlier` — marks each point as trustworthy only if it's within `4.0 × mad + 1.0` pixels of the median (in both directions). This is the outlier-rejection step: throws out points whose motion doesn't match the crowd — caustic light sparkles, a fish swimming through, a loose rope, anything moving independently of the floor. The `4.0` multiplier and `+1.0` fixed pixel allowance are tuning constants — 4 MAD is a fairly generous "you have to be a real outlier to get thrown out" bound, and the `+1.0` avoids being overly strict when `mad` itself is tiny (near-perfect agreement).
- `n_inliers` — the count of points that survived this filter. If fewer than `min_features` (25 by default) survived, that's a hold-worthy failure tagged `'few_inliers'` — meaning enough points were tracked, but they disagreed too much with each other to trust.
- `du, dv` — finally, the median pixel movement of ONLY the trustworthy inlier points — this is the number that represents "the floor moved this many pixels" for this whole interval.

**If you raise the `4.0` MAD multiplier:** more tolerant of noisy/disagreeing points (fewer rejected as outliers, but more risk of contamination). **Lower it:** stricter, cleans data more aggressively but risks throwing away genuinely good points during real motion.

### Step 9: derotation
```python
wx, wy = gyro_xy_cam
du_t = du - (-self.fx * wy * dt_eff)
dv_t = dv - (+self.fy * wx * dt_eff)
```
When the camera *rotates* (without translating at all), the image still appears to slide — e.g. tilting forward makes the floor pattern appear to slide "up" in the image, purely from rotation, not actual sideways travel. This subtracts that fake, rotation-caused apparent motion, using the known current spin rates (`wx`, `wy`) times the elapsed time (`dt_eff`), scaled by the focal lengths (since the apparent pixel speed from a given rotation rate depends on how zoomed-in the lens is). What's left (`du_t`, `dv_t`) is (approximately) the movement caused by actual translation only.

**Where these equations come from:** this isn't arbitrary — it comes from the standard equations of how a rigid, non-moving scene appears to flow across an image when a camera rotates, evaluated at the special case of "near the center of the image."

*Starting point — the pinhole camera model.* For a 3D point at camera-frame coordinates $(X, Y, Z)$, its pixel position is:
$$u = f_x \frac{X}{Z}, \qquad v = f_y \frac{Y}{Z}$$
(measuring $u, v$ relative to the principal point, i.e. $u - c_x$, $v - c_y$, so the center of the image is $(0,0)$).

*The general "motion field" equation.* If the camera itself moves with translational velocity $T = (T_x, T_y, T_z)$ and rotational velocity $\omega = (\omega_x, \omega_y, \omega_z)$ relative to a static scene, classic projective geometry gives the resulting apparent pixel velocity of any tracked point as the sum of a *translational* part and a *rotational* part:

$$\dot u = \underbrace{\frac{-f_x T_x + x\, T_z}{Z}}_{\text{translation}} \;+\; \underbrace{\frac{xy}{f_x}\omega_x - \left(f_x + \frac{x^2}{f_x}\right)\omega_y + y\,\omega_z}_{\text{rotation}}$$

$$\dot v = \underbrace{\frac{-f_y T_y + y\, T_z}{Z}}_{\text{translation}} \;+\; \underbrace{\left(f_y + \frac{y^2}{f_y}\right)\omega_x - \frac{xy}{f_y}\omega_y - x\,\omega_z}_{\text{rotation}}$$

(here $x, y$ mean the pixel's position relative to the center, i.e. $u-c_x$, $v-c_y$; this exact form appears in any structure-from-motion textbook — the Longuet-Higgins & Prazdny motion field equation).

*The key simplification: evaluate near the image center.* Notice the rotational part has terms in $x$, $y$, $xy$, $x^2$, $y^2$ — but *all of them vanish or become negligible* if you plug in $x \approx 0, y \approx 0$ (i.e., you're looking at a point right near the center of the image). What survives is just:
$$\dot u_{\text{rot}} \approx -f_x\,\omega_y \qquad \dot v_{\text{rot}} \approx +f_y\,\omega_x$$
This is exactly why the "spread your tracked points around the center" design from earlier matters *twice over* — once for the Vz-zoom-cancellation, and again here: this simplified rotation formula is only accurate if the points being averaged over sit close to the principal point (or symmetrically around it, so the higher-order $x^2$/$xy$ terms average out).

*Turning this into the code.* Multiply by elapsed time to get pixel *displacement* instead of pixel *velocity*:
$$du_{\text{rot}} \approx -f_x \,\omega_y\, dt_{\text{eff}} \qquad dv_{\text{rot}} \approx +f_y\,\omega_x\, dt_{\text{eff}}$$
To get the translation-only part, subtract this predicted rotational contribution from the raw measured flow:
$$du_t = du - du_{\text{rot}} = du - (-f_x\,\omega_y\,dt_{\text{eff}})$$
$$dv_t = dv - dv_{\text{rot}} = dv - (+f_y\,\omega_x\,dt_{\text{eff}})$$
...which is exactly the two lines of code above.

*Sanity-checking the pieces intuitively:*
- **Focal length scaling** — a given rotation rate (say, 1 rad/s) sweeps the same *angular* field of view regardless of lens, but a more zoomed-in lens (bigger $f_x$/$f_y$) maps that same angular sweep to *more pixels*. So the pixel-speed caused by pure rotation scales directly with focal length — hence the $f_x$, $f_y$ multipliers.
- **The cross-pairing (wy → du, wx → dv)** — rotating about the axis that's aligned with the image's vertical (y) direction sweeps the view sideways, shifting pixels horizontally ($du$); rotating about the axis aligned with the image's horizontal (x) direction tips the view up/down, shifting pixels vertically ($dv$). So each rotation axis drives the *opposite* image axis — which is exactly the "cross" pattern in the two equations.
- **The signs** — the $-f_x\omega_y$ vs $+f_y\omega_x$ come directly from working through the geometry/sign conventions of the pinhole projection and the image y-down axis convention; they're not arbitrary, but they are the kind of thing worth empirically double-checking once (e.g. the hand-push/stationary-wiggle test mentioned elsewhere in your codebase — hold the vehicle still and rock it in pitch/roll; a correct derotation should leave the reported velocity near zero).

This is a first-order (small-angle-near-center) approximation, not an exact closed form — good enough here because the outlier-rejection and grid-spread already keep the tracked points reasonably close to center, which is precisely where this simplification is valid.

### Step 10: optional Vz (vertical motion) compensation
```python
if self.compensate_vz and self.ref_altitude is not None:
    delta_alt = self.ref_altitude - altitude
    x_bar = float(np.mean(p0[inlier, 0])) - self.cx
    y_bar = float(np.mean(p0[inlier, 1])) - self.cy
    du_t -= x_bar * delta_alt / altitude
    dv_t -= y_bar * delta_alt / altitude
```
Only runs if this optional feature is turned on AND there's a valid altitude recorded from when the reference frame was set. `delta_alt` = how much the altitude changed across this interval (positive means the vehicle got closer to the floor, i.e. descended). `x_bar`, `y_bar` = how far off-center (on average) the trustworthy tracked points sit. The correction subtracts a small amount from the flow proportional to both (a) how off-center the points are and (b) how much the altitude changed — removing the "zoom" leakage explained earlier.

### Step 11: convert pixels-per-time into real camera-frame velocity
```python
vx_cam = -(du_t / dt_eff) * altitude / self.fx
vy_cam = -(dv_t / dt_eff) * altitude / self.fy
```
This is the heart of the whole "optical flow as a DVL" idea. `du_t / dt_eff` gives "pixels moved per second." Multiplying by `altitude` and dividing by focal length (`fx`/`fy`) converts that into "meters moved per second" — the further you are from the floor, the more real-world distance a single pixel of apparent movement represents. The leading minus sign flips the sign because ground features appear to move opposite to the camera's own direction of travel (like scenery outside a moving train window flowing backward relative to your motion, even though you're the one moving forward) — this is explained explicitly in `flow_estimator.py`'s overlay comments too.

**Where these equations come from — building on the same motion-field equation used for derotation, keeping the translation half this time:**
$$\dot u = \frac{-f_x T_x + x\, T_z}{Z} + (\text{rotation terms, already removed})$$
$$\dot v = \frac{-f_y T_y + y\, T_z}{Z} + (\text{rotation terms, already removed})$$
Here $(T_x, T_y, T_z)$ is the camera's own translational velocity (in the camera's frame), $Z$ is the distance from the camera to that particular 3D point along the optical axis, and $x, y$ are the pixel's position relative to the image center.

*Step 1 — drop the $x T_z$, $y T_z$ terms.* Just like with rotation, evaluate near the image center ($x \approx 0$, $y \approx 0$), or assume the tracked points are roughly symmetric around it. This term is exactly the "zoom leakage" from vertical motion ($T_z$) discussed earlier — the thing `compensate_vz` (Step 10) optionally corrects the leftover of. Dropping it here gives:
$$\dot u_{\text{trans}} \approx \frac{-f_x T_x}{Z} \qquad \dot v_{\text{trans}} \approx \frac{-f_y T_y}{Z}$$

*Step 2 — what is $Z$ here, and where the tilt-compensation goes.* $Z$ is supposed to be the true, per-point distance from the camera to that specific bit of floor, measured along the optical axis. In reality this varies slightly point-to-point (the floor isn't infinitely flat, and if the camera is tilted, points on one side of the image are genuinely closer than points on the other side). The equation as coded assumes this is well approximated by one single number for the whole patch: `altitude`. That assumption only holds if the floor patch is locally flat and the camera is looking close to straight down at it — which is exactly why the *tilt-compensation* has to happen **before** this function is ever called, not inside it.

This is worth being explicit about: **`flow_core.py` does not do tilt-compensation itself.** It just trusts whatever number is handed to it as `altitude`. The actual correction — turning "how deep the vehicle is, minus how deep the floor is" into "true range along the tilted optical axis" — happens upstream, in `flow_eval_node._camera_altitude()`:
```python
alt = alt / max(cos(roll) * cos(pitch), 0.5)
```
Dividing by $\cos(\text{roll})\cos(\text{pitch})$ is exactly the correction for "the optical axis isn't perfectly vertical, so the straight-down depth difference undercounts the true diagonal distance to the floor." Once that's done upstream, the `altitude` this function receives is already the corrected, tilt-adjusted $Z$ — so inside `process()`, the tilt problem has already vanished by the time it gets here; the equation just gets to pretend $Z$ = `altitude`, flat and uniform, no tilt term anywhere in sight. In other words: **tilt doesn't vanish through some approximation inside this equation — it's never in this equation to begin with**, because it's assumed already baked into the single `altitude` value passed in from outside, by the caller's separate tilt-compensation step.

*Step 3 — solve for velocity instead of pixel rate.* We actually want $T_x, T_y$ (the camera's real velocity), and we have $\dot u_{\text{trans}}, \dot v_{\text{trans}}$ (pixel rate) approximated by `du_t / dt_eff`, `dv_t / dt_eff` (the derotated pixel displacement, divided by elapsed time). Rearranging:
$$T_x = \frac{-\dot u_{\text{trans}} \cdot Z}{f_x} \qquad T_y = \frac{-\dot v_{\text{trans}} \cdot Z}{f_y}$$
Substituting $Z = \text{altitude}$ and $\dot u_{\text{trans}} = du_t / dt_{\text{eff}}$:
$$T_x = -\left(\frac{du_t}{dt_{\text{eff}}}\right)\frac{\text{altitude}}{f_x} \qquad T_y = -\left(\frac{dv_t}{dt_{\text{eff}}}\right)\frac{\text{altitude}}{f_y}$$
...which is exactly the two lines of code above.

*Intuition check on each piece:*
- **Why divide by $f_x$/$f_y$ (opposite of the rotation term, which multiplied):** a more zoomed-in lens (bigger $f$) makes the *same real-world motion* produce *more* pixels of apparent shift. So to recover real-world speed from pixel speed, you divide out the zoom factor — the reverse of the rotation case, where you were going the other direction (converting a known real rotation rate *into* pixels).
- **Why multiply by altitude:** the further the camera is from the floor, the *more real-world distance* a single pixel of apparent shift represents (like how a drone flying higher makes the ground below appear to move more slowly across the frame for the same real speed). So altitude scales pixel-speed *up* into real speed.
- **The leading minus sign:** same reasoning as before — a stationary ground point appears to move opposite to the camera's own direction of travel (scenery streaming backward past a moving train), so recovering the camera's actual direction of travel requires flipping the sign of the apparent image motion.

### Step 12: camera-frame → body-frame conversion
```python
bx, by = -vy_cam, -vx_cam
if self.swap_xy:
    bx, by = by, bx
bx *= self.sign_x
by *= self.sign_y
```
Converts from "camera's own left/right, up/down" into "vehicle's own forward/sideways" using the default mounting assumption (image-up = vehicle-forward), then applies whichever mounting corrections (`swap_xy`, `sign_x`, `sign_y`) were set in the constructor to match the *actual* physical camera mount.

### Step 13: tile-aliasing sanity check on the recovered velocity
```python
recovered = self.hold_dt
if recovered > 0.0 and self.last_v is not None:
    jump = math.hypot(bx - self.last_v[0], by - self.last_v[1])
    if jump > self.v_jump_max:
        return self._fail(gray, dt, 'alias_reject', hold=True, altitude=altitude)
```
`recovered` being greater than zero means this estimate had to span a held gap (i.e. tracking failed at least once before this success). In that case, compare the new velocity against the last known-good velocity (`last_v`) — if the jump between them is bigger than physically plausible (`v_jump_max = 0.8` m/s), this is almost certainly the tile-aliasing problem described earlier, not real acceleration. In that case, reject this estimate too (as a hold-worthy failure tagged `'alias_reject'`) and try again next frame — with the predictive seeding, the next attempt is more likely to converge on the true match.

### Step 14: success — update state and return the answer
```python
self.px_rate = (du / dt_eff, dv / dt_eff)
self.last_v = (float(bx), float(by))
self.last_failure = None
self._advance_ref(gray, altitude)
```
Now that we have a trustworthy velocity estimate: remember the pixel rate (for next frame's predictive seeding), remember this velocity (for next time's aliasing sanity check), clear the failure flag (nothing wrong this time), and finally advance the reference frame to the current one — meaning the NEXT call to `process()` will compare against THIS frame.

```python
spread = float(np.mean(np.std(d[inlier], axis=0)))
return dict(vx=float(bx), vy=float(by),
            n_tracked=n_tracked, n_inliers=n_inliers,
            spread_px=spread, flow_px=(float(du), float(dv)),
            dt_eff=float(dt_eff), recovered_gap_s=float(recovered))
```
`spread` measures how much the individual inlier points' movements varied from each other (their standard deviation) — a rough "how noisy/confident was this frame" quality indicator (used elsewhere, e.g. to scale down trust in this measurement when feeding it into the EKF). Finally, it packages everything useful into a dictionary and returns it:
- `vx, vy` — the actual answer: body-frame velocity in meters/second.
- `n_tracked` — how many points survived the forward-backward check.
- `n_inliers` — how many of those also survived the outlier-rejection.
- `spread_px` — the noisiness/quality indicator.
- `flow_px` — the raw median pixel movement (before scaling to meters) — useful for debugging/visualization.
- `dt_eff` — the actual time span this measurement covers (helpful to know if it recovered a multi-frame gap).
- `recovered_gap_s` — non-zero if this estimate had to bridge a held gap, telling the caller "this one spans more than just the last single frame."

---

## Summary of what happens if you tweak the main knobs

| Parameter | Raise it | Lower it |
|---|---|---|
| `max_corners` | Smoother/more robust, slower | Faster, noisier |
| `quality` | Fewer but cleaner corners | More corners incl. weak/noisy ones |
| `min_distance` | More spread out points | Points can cluster |
| `min_features` | Fewer estimates, more trustworthy | More estimates, more risk |
| `max_hold_frames`/`max_hold_dt` | Recovers from longer dropouts, more alias risk | Gives up (and loses data) sooner, less alias risk |
| `fb_max_err` | Accepts sloppier matches | Stricter, more points thrown out |
| `v_jump_max` | Tolerates bigger sudden "speed changes" (more alias risk) | Rejects more aggressively (may reject real fast moves) |
| `use_clahe` | Helps in murky water | (default off; hurts gap-recovery when on) |
| `grid_rows/cols` | Better point spread, better Vz-compensation validity | (0 = off, points cluster naturally) |
| `compensate_vz` | Removes zoom-leakage during climbs/dives | (default off, unvalidated) |
