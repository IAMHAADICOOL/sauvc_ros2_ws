# `on_image` Fusion Logic, Hardening, `offset`, and the Mod-90° Disambiguation — Full Explanation

## Part 1: Exhaustive Line-by-Line Walkthrough

### Line-by-line

```python
self.pub_meas.publish(Float32(data=float(ang)))
```
Publishes the raw, ambiguous mod-90° line angle to the `/heading/line_meas` debug topic — purely for external monitoring/logging, doesn't affect any internal logic below.

```python
hardening = bool(self.get_parameter('enable_hardening').value)
```
Reads the `enable_hardening` parameter once into a local variable, used repeatedly below to branch between the "hardened" and "original" behavior.

```python
if hardening:
    yaw_img = self._yaw_at(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    if yaw_img is None:
        return
else:
    yaw_img = self.gyro_yaw
```
Gets the yaw belief to use as the disambiguation tiebreaker. With hardening: look up the *interpolated* yaw at the image's own timestamp (via `_yaw_at`, covered earlier) — more accurate, avoids rate×latency bias. If `_yaw_at` somehow returns `None` (shouldn't normally happen given its fallbacks, but this guards against it), abandon this frame entirely — `return` exits `on_image` right here, skipping everything below, including the debug-image publish at the very end of the function. Without hardening: just use whatever `self.gyro_yaw` is right now — the older, simpler, un-interpolated behavior.

```python
cur = wrap(yaw_img + self.offset)
```
Combine that yaw belief with the currently-learned drift correction — this is "what I currently think the true absolute heading is," used as the tiebreaker for disambiguation.

```python
meas = -ang
```
Negate the raw detector output — the empirically-validated sign relating image-frame line angle to vehicle body yaw.

```python
k = round((cur - meas) / (math.pi / 2))
meas_unwrapped = meas + k * (math.pi / 2)
err = wrap(meas_unwrapped - cur)
```
The disambiguation step covered at length earlier: pick the multiple of 90° (`k`) that puts the ambiguous `meas` closest to `cur`, producing a full unambiguous heading `meas_unwrapped`, then compute the small residual disagreement `err` against `cur`.

```python
turning = hardening and self._rate_lp > float(self.get_parameter('freeze_rate').value)
```
A boolean: are we hardening-enabled *and* is the smoothed turn rate above the freeze threshold? If hardening is off, `turning` is always `False` (short-circuits), so the whole freeze mechanism is inert without hardening.

```python
if turning:
    self._n_rate_frozen += 1
    status = (f'frozen (turning {math.degrees(self._rate_lp):.0f} deg/s '
              f'> freeze_rate)')
```
**Branch A.** If turning fast: don't touch `offset` at all — just increment a diagnostic counter and set a human-readable status string (used later for the debug overlay). Nothing else happens; this frame's vision measurement is discarded for correction purposes even though it was successfully detected and disambiguated.

```python
elif abs(err) < math.radians(20):             # sanity gate
    g = float(self.get_parameter('gain').value)
    if hardening and bool(self.get_parameter('gain_r_scaling').value):
        R = self._last_R if self._last_R is not None else 0.6
        g *= min(1.0, max(0.3, (R - 0.6) / 0.3))
    self.offset = wrap(self.offset + g * err)
    self._n_accepted += 1
    self._reject_streak = 0
    status = 'ACCEPTED'
```
**Branch B — only reached if not turning, and the residual error is small (< 20°).**
- `g = gain` (default `0.02`) — the base correction strength.
- If hardening and `gain_r_scaling` are both on: scale `g` by line quality. `R` is the concentration score from `detect_line_angle` (recall it's always `≥ 0.6`, since anything lower was already rejected there). The formula `(R - 0.6) / 0.3` maps `R=0.6 → 0`, `R=0.9 → 1.0`, clamped by `max(0.3, ...)` so it never drops below 30%, and `min(1.0, ...)` so it never exceeds 100%. So a barely-passing frame (`R=0.6`) pulls at 30% strength, a crisp frame (`R≥0.9`) pulls at full strength — quality-weighted trust.
- `self.offset = wrap(self.offset + g * err)` — **the actual update**, discussed extensively before: nudge `offset` a small fraction of the way toward fully explaining `err`.
- Increment the accepted counter, **reset `_reject_streak` to 0** (this matters for Branch C below — any successful acceptance clears the "stuck" counter), set status.

```python
else:
    self._n_gate_rejected += 1
    self._reject_streak += 1
    status = f'gate-rejected ({math.degrees(err):+.0f} deg)'
    n_need = int(self.get_parameter('relock_after_rejects').value)
    if (hardening and n_need > 0 and self._reject_streak >= n_need
            and self._rate_lp < float(self.get_parameter('relock_max_rate').value)):
        self.offset = wrap(self.offset + err)
        self._n_relocks += 1
        self._reject_streak = 0
        self.get_logger().warn(...)
        status = 'RE-LOCKED'
```
**Branch C — not turning, but `err` is too large (≥ 20°).** Count the rejection, increment the consecutive-rejection streak, set status. Then check stuck-lock recovery conditions: hardening on, feature enabled (`n_need > 0`), streak has reached the threshold (default 90), *and* the vehicle is nearly stationary (`_rate_lp < relock_max_rate`). If all true: snap `offset` by the **full** `err` (no `gain` scaling — a complete, one-shot correction, not a gradual nudge), reset the streak, log a warning explaining what just happened and its caveat (a residual multiple of 90° still can't be detected), and override the status to `'RE-LOCKED'`.

---

## Part 2: What "Hardening" Actually Is

Strip away the mechanism and hardening is: **"don't let the vision correction touch `offset` in situations where the disambiguation (`k`) is more likely to be wrong, and add a recovery path in case it happens anyway."**

Recall from earlier: the whole disambiguation trick (`k = round((cur-meas)/90°)`) only works correctly if `cur` (the gyro-based belief) is within about 45° of the truth — comfortably closer to the right 90°-branch than any neighboring one. That assumption is normally safe because gyro drift is slow. But it gets shaky specifically during **fast rotation**: motion blur degrades the line detection, and any small timing mismatch between when the image was captured and when the gyro was sampled gets *amplified* by the rotation speed (a 50ms lag at 0.3 rad/s is a small ~1.5° error; the same lag at 2 rad/s is ~10° — much closer to eating into that 45° margin). If `cur` drifts too close to a 45°/135°/etc. boundary during a fast turn, `k` can round to the *wrong* branch, and the resulting `err` can look small enough to slip under the 20° gate even though `meas_unwrapped` is actually 90° off from truth. That would silently corrupt `offset` with a large, wrong value — and because the gate re-derives `err` the same way every subsequent frame, once corrupted it tends to *stay* corrupted (every future frame keeps failing the same way).

Hardening is four coordinated defenses against exactly that failure mode:

1. **`freeze_rate` / the `turning` check** — the primary defense: simply refuse to update `offset` at all while the smoothed turn rate is high. This is a "don't even try" strategy — rather than trying to make disambiguation more robust during fast turns, just skip vision correction entirely during them, and trust the gyro to coast (justified because gyro drift over a few seconds of turning is negligible — far smaller than the ~20-90° error a wrong branch pick could inject).
2. **`gain_r_scaling`** — a softer defense: even when not turning, weight the correction by how confident the line detection was (`R`). Marginal, barely-passing detections get less influence than crisp, unambiguous ones.
3. **Image-timestamp interpolation (`_yaw_at`)** — reduces the rate×latency bias in `cur` itself, so `cur` is a more accurate belief to begin with, making a wrong `k` pick less likely in the first place even before the freeze kicks in.
4. **`relock_after_rejects` / stuck-lock recovery** — a safety net for if corruption happens anyway (or hardening was off, or an unlucky slow-turn edge case): detect "many consecutive rejections while basically stationary" as the signature of a stuck/corrupted offset, and deliberately allow one large, ungated correction to escape it — specifically *while stationary*, because that's when a wrong branch pick becomes implausible (crisp, unambiguous lines).

`enable_hardening=False` switches all four off and reproduces the plain original behavior — useful for comparison/rollback, or diagnosing whether a problem is caused by the hardening logic itself.

---

## Part 3: What `self.offset` Is

You've got it right: **`self.offset` is purely a slowly-updated correction that compensates for the gyro's drift over time.** It starts at `0.0` (assume no drift), and every time a trustworthy vision frame comes through, it inches a little further toward fully canceling whatever drift has accumulated since — never snapping instantly (except the rare relock case), always gradually, so no single noisy or ambiguous frame can throw it far off course. It's not a measurement of the current heading itself — it's a *correction term* that gets added to the fast, always-fresh `gyro_yaw` (in `on_imu`) to produce the actual published heading. Everything discussed above — the turning freeze, the R-scaling, the 20° gate, the stuck-lock recovery — exists entirely to protect the *quality* of that one slowly-evolving number, because a single bad update to it can propagate forward and corrupt every heading published afterward until it's somehow corrected.

---

## Part 4: The Mod-90° Disambiguation, Worked with Numbers

### The setup: the four candidates are spaced 90° apart

Say the true heading is **100°**. The camera's mod-90° reading, after disambiguation-attempt, gives `meas`, and the code tries all the nearby 90°-multiples of it to find which one is closest to `cur`. The candidates consistent with the raw measurement `10°` (which is `100° mod 90°`) are:

```
..., -170°, -80°, 10°, 100°, 190°, ...
```

each one exactly 90° from its neighbors.

### What `k = round((cur - meas) / 90°)` is actually doing

This formula measures the distance from `cur` to `meas` (in units of 90°) and rounds to the nearest whole number — i.e., it picks whichever candidate on that list above is **closest to `cur`**.

Picture the number line with those candidates marked as dots, 90° apart: `...-80°, 10°, 100°, 190°...`. Now think about the **boundary** between two neighboring dots — the point exactly halfway between them. Between `10°` and `100°`, that halfway point is `55°`. Between `100°` and `190°`, it's `145°`.

**If `cur` falls anywhere between 55° and 145°, the "closest dot" is 100° — correctly picking the true answer.** That's a 90°-wide safe zone, centered on the truth. In other words: `cur` can be off from the true value (100°) by up to **45° in either direction** (down to 55°, or up to 145°) and the rounding will still correctly land on `k` corresponding to `100°`.

### Where "45°" comes from

45° is exactly half of the 90° spacing between candidates. Since `round()` always picks the *nearest* candidate, the boundary between "nearest is the correct one" and "nearest is a neighboring, wrong one" sits exactly halfway between two candidates — which is 45° away from each. So:

- If `cur` is off from the truth by **less than 45°**, it's still closer to the correct candidate than to either neighbor → `k` comes out right.
- If `cur` is off from the truth by **more than 45°**, it's now actually closer to a neighboring (wrong) candidate → `k` comes out wrong, picking a heading that's a full 90° off from reality.
- If `cur` is off by **exactly 45°**, it's tied — right on the boundary, a coin-flip between correct and 90°-wrong.

### Why this matters practically

This is exactly why the earlier point about "gyro drift is slow" is load-bearing, and why fast turns are dangerous: the whole disambiguation scheme is a bet that **the gyro's current belief (`cur`) is never wrong by anywhere close to 45°** at the moment a vision frame is being processed. Under normal, slow-drift conditions, that's a very safe bet — gyro error accumulates gradually, nowhere near 45° between corrections. But during a fast, blurry turn, timing lag and drift can conspire to push `cur`'s error much closer to that 45° cliff-edge — and if it crosses it, `round()` doesn't fail gracefully or give a "low confidence" warning, it just confidently returns the wrong `k`, silently producing a `meas_unwrapped` that's a full 90° off from the truth, potentially still slipping under the 20° `err` gate if the numbers land unluckily. That's the precise mechanism the "hardening" freeze logic is designed to prevent, by refusing to trust `cur` for this purpose whenever the vehicle is spinning fast enough that a 45°-scale error becomes plausible.

---

## Part 5: The Full Worked Example, With Actual `cur` Values

### Setup

True heading = **100°**. Camera measurement (already established): `meas = 10°`.

### Case 1: `cur = 95°` (gyro is off by only 5° — normal, healthy situation)

```python
k = round((cur - meas) / (math.pi / 2))
```
Plug in degrees (same math, easier to read than radians):
```
k = round((95° - 10°) / 90°)
  = round(85° / 90°)
  = round(0.944)
  = 1
```
```python
meas_unwrapped = meas + k * 90° = 10° + 1×90° = 100°
```
✅ **Correct.** `k=1` correctly recovered the true heading.

### Now let's understand *why* that worked, by trying every candidate `k` by hand

The formula is really just testing: "how far is `cur` (95°) from each possible unwrapped candidate?" Let's list a few candidates and their distance from `cur=95°`:

| `k` | `meas + k×90°` | distance from `cur=95°` |
|---|---|---|
| -1 | 10° − 90° = **-80°** | \|95 − (−80)\| = 175° |
| 0 | 10° + 0 = **10°** | \|95 − 10\| = 85° |
| **1** | 10° + 90° = **100°** | \|95 − 100\| = **5°** ← smallest |
| 2 | 10° + 180° = **190°** | \|95 − 190\| = 95° |

`k=1` gives the smallest distance (5°) — that's why `round()` picks it. The formula `round((cur-meas)/90°)` is just a fast mathematical shortcut for "find which candidate is nearest," instead of manually computing all these distances.

### Case 2: `cur = 60°` (gyro is off by 40° — still under the 45° limit, borderline)

```
k = round((60° - 10°) / 90°) = round(50°/90°) = round(0.556) = 1
meas_unwrapped = 10° + 90° = 100°
```
✅ Still correct! Even though `cur` (60°) is quite far from the truth (100°) — off by 40° — it's still closer to `100°` than to `10°` (distance to 100° is 40°; distance to 10° is 50°), so it still rounds correctly.

### Case 3: `cur = 50°` (gyro is off by 50° — now past the 45° limit)

```
k = round((50° - 10°) / 90°) = round(40°/90°) = round(0.444) = 0
meas_unwrapped = 10° + 0×90° = 10°
```
❌ **Wrong!** The formula now returns `k=0`, giving `meas_unwrapped = 10°` — a full 90° away from the true 100°. Let's verify why, using the distance-table approach again:

| `k` | candidate | distance from `cur=50°` |
|---|---|---|
| 0 | 10° | \|50 − 10\| = 40° |
| 1 | 100° | \|50 − 100\| = 50° |

Since `cur=50°` is now closer to the candidate `10°` (40° away) than to the true candidate `100°` (50° away), `round()` correctly-by-its-own-logic — but *wrongly overall* — picks `k=0`. The math did exactly what it's designed to do (pick the nearest candidate); the problem is that `cur` itself had drifted too far from the truth for "nearest" to still mean "correct."

### Finding the exact tipping point

The two candidates bracketing the truth are `10°` and `100°`. The switchover point — where `cur` is *exactly equally distant* from both — is their midpoint:
```
(10° + 100°) / 2 = 55°
```
- If `cur > 55°` (closer to 100° side): picks `100°` ✅ (this covers our Case 1 at 95°, and Case 2 at 60°)
- If `cur < 55°` (closer to 10° side): picks `10°` ❌ (this is our Case 3 at 50°)

Similarly, on the other side, the next candidate up is `190°`, and the midpoint between `100°` and `190°` is `145°`. If `cur > 145°`, it would incorrectly pick `190°` instead.

So the **safe zone for `cur`** — the range where it correctly resolves to `100°` — is `55° < cur < 145°`. That's a 90°-wide window, and its center is exactly `100°` (the truth). The two edges, `55°` and `145°`, are each exactly **45° away** from the center (`100° - 55° = 45°`, `145° - 100° = 45°`).

### Tying it back to "45°"

That's the entire origin of the "45° limit" statement: as long as `cur`'s error from the true value is **strictly less than 45°** (i.e., `cur` stays inside `(55°, 145°)` in our example), the nearest-candidate arithmetic lands on the right answer. The moment `cur`'s error reaches or exceeds 45° (as in Case 3, where it was 50° off), `cur` has crossed over the midpoint into being nearer to a *neighboring, wrong* 90°-multiple, and `round()` — doing exactly what it's told — confidently hands back the wrong branch, silently producing a `meas_unwrapped` that's off from the truth by a full 90°.

---

## Part 6: Correcting What `err` Actually Is — and Why the 45° Threshold Matters

### Correcting one piece: what `err` actually is

`cur` and `meas_unwrapped` are **not** "IMU yaw" vs "image's yaw at a different instant." They are two **independent estimates of the exact same instant** — both are trying to answer "what was the yaw at the moment this image was captured?" — just computed from two completely different sources:
- `cur` = IMU's answer (gyro reading + learned correction), for that instant.
- `meas_unwrapped` = vision's answer (disambiguated line reading), for that same instant.

`err = meas_unwrapped − cur` is the **disagreement between two independent sensors about the same moment**, not a difference across time. That's an important distinction, because the whole point of this system is: "IMU is fast but drifts; vision is precise but ambiguous — let's use their disagreement to correct the IMU's drift." If `err` were comparing two different time instants, it wouldn't make sense as a drift-correction signal at all.

### Why should you even care about this 45° thing?

Here's the concern in one sentence: **`k` is not "verified" to be correct — it's just a guess, produced by rounding, and rounding can silently guess wrong.**

Nothing in the code checks "is `k` actually right?" There's no independent way to confirm it — the whole reason we needed `k` in the first place is that the raw camera measurement *alone* can't tell you the true heading. So `k`'s correctness is entirely riding on one unverified assumption: that `cur` (the IMU's guess) was already close enough to the truth that "round to nearest" happens to land on the right multiple of 90°.

**If that assumption ever breaks — even once — nothing catches it automatically the way you'd hope.** A wrong `k` produces a `meas_unwrapped` that's a full 90° away from the truth. But then `err = meas_unwrapped - cur` gets computed from that wrong value, and — this is the scary part — `err` doesn't necessarily come out looking huge and obviously wrong. Depending on exactly how close `cur` was to the 45° boundary, `err` might still come out looking like a small, plausible number (this is what the 20° sanity gate is trying to catch, but it's not foolproof right at the boundary). So a bad `k` can sneak past every safeguard and corrupt `offset` with a confidently-wrong 90° error, and — since `offset` feeds back into `cur` for the *next* frame — that corruption can persist and compound.

**That's why you should care about the 45° threshold**: it's the precise line between "this whole mechanism is silently trustworthy" and "this whole mechanism can silently fail." You're not just learning a math curiosity — you're identifying exactly *when* this system's core assumption is safe versus exactly *when* it's gambling.

### The setup and the walkthrough, once more, fully worked

**Ground truth:** the vehicle's real heading is **100°**. Nobody in the code "knows" this number — it's just physical reality.

**What the camera produces:** `detect_line_angle` runs, and (through the ×4 trick) reports a clean, confident, mod-90° reading. Since `100° mod 90° = 10°` (100 minus one 90 is 10), the camera reports `ang`, and after negation, `meas = 10°`.

Because of the mod-90° symmetry, this single number `10°` is honestly consistent with *any* of these being the real truth:
```
..., -170°,  -80°,   10°,   100°,   190°,   280°, ...
```
Every one of these differs from its neighbor by exactly 90°. The camera has zero way, on its own, to tell you which one of these is real — they'd all produce the identical photo.

**Enter `cur`:** this is the IMU's independent guess at the true heading, at this same instant. Say the gyro has drifted a little, so:
```
cur = 95°
```
(5° away from the real 100° — a normal, small drift, nothing alarming.)

**Computing `k`:**
```
k = round((cur - meas) / 90°) = round((95° - 10°) / 90°) = round(85°/90°) = round(0.944) = 1
```
```
meas_unwrapped = meas + k×90° = 10° + 90° = 100°
```
This correctly reconstructed the true heading. **But notice — the code has no way to *know* this is correct.** It just trusted that `cur=95°` was a good enough guess, and it happened to be.

**Now here's the "why should I care" part, made concrete: let's slide `cur` slowly away from 95° and watch what happens, with no other change to the code's logic.**

| `cur` (IMU's guess) | how far `cur` is from truth (100°) | `k = round((cur-10°)/90°)` | `meas_unwrapped` | Correct? |
|---|---|---|---|---|
| 95° | 5° off | round(0.944)=1 | 100° | ✅ |
| 70° | 30° off | round(0.667)=1 | 100° | ✅ |
| 56° | 44° off | round(0.511)=1 | 100° | ✅ (barely) |
| **54°** | **46° off** | round((54-10)/90)=round(0.489)=**0** | **10°** | ❌ |
| 40° | 60° off | round(0.333)=0 | 10° | ❌ |

Look at that jump between `cur=56°` and `cur=54°` — just a 2° change in `cur`'s error crossed a hidden cliff-edge, and the output silently flipped from "100° (correct)" to "10° (wrong by a full 90°)." There's no gradual degradation, no warning sign in the math itself — `round()` doesn't hesitate or flag uncertainty near the boundary, it just picks a side, confidently, every time.

**Why 55° is exactly that cliff-edge:** it's the midpoint between the two candidates flanking the truth — `10°` and `100°`. Midpoint = `(10+100)/2 = 55°`. As long as `cur` stays on the "100°-side" of that midpoint (i.e., `cur > 55°`), the nearest-candidate logic favors `100°`. The instant `cur` crosses to the other side (`cur < 55°`), it becomes numerically closer to `10°` than to `100°`, and `round()` — which only ever does "pick whichever is nearer" — switches its answer, with total confidence, to the wrong one. The distance from the truth (`100°`) to that cliff-edge (`55°`) is exactly `45°` — hence: **`cur` is safe as long as it's wrong by less than 45°; the moment it's wrong by 45° or more, the disambiguation can silently flip to a confidently-wrong answer.**

**This is precisely why you should be concerned about it**, and precisely why the "hardening" logic (freeze during fast turns) exists: fast rotation is the one realistic scenario where `cur`'s error can actually grow large enough, quickly enough, to approach or cross that 45° cliff — and when it does, the code has no built-in way to notice the cliff was crossed; it just quietly starts feeding wrong 90°-off corrections into `offset` as if they were normal, correct ones.

---

## Part 7: `meas` Also Has Its Own (Separate, Milder) Error

### What we were assuming

In the worked example, `meas = 10°` was treated as a perfect, noise-free reading of `100° mod 90°`. That let us isolate and study *just* the `k`-selection problem in isolation, without a second source of error muddying the picture. But `meas` itself is a **measurement**, produced by real cameras and real edge-detection on real (possibly imperfect) floor textures — so it has its own error too, separate from the `k`-selection error.

### Where `meas`'s own error comes from

`detect_line_angle` is built from Hough line segments, each with pixel-level noise, imperfect edges, maybe a partially-obscured grout line, lighting variation, etc. The ×4 circular-mean trick and the `R` concentration check exist specifically to keep this error small — by averaging many segments and rejecting frames where they disagree too much — but "small" isn't "zero." So realistically, if the true heading is 100°, the camera might report `meas = 8°` or `meas = 13°` instead of a perfect `10°` — off by a couple degrees due to normal sensor/detection noise, even on a frame that passes the `R ≥ 0.6` quality gate.

### Why this matters, layered on top of the `k` problem

So there are actually **two separate, stacked sources of error** here, not one:

1. **Noise within `meas` itself** — small, a few degrees typically, from imperfect line detection. This is a "fine-grained" error.
2. **The branch-selection (`k`) error** — either exactly right, or wrong by a clean multiple of 90°. This is a "coarse, catastrophic" error — either 0° of error, or a huge 90°/180°/270° error, with essentially nothing in between.

These are fundamentally different *kinds* of error, and that's actually why the code treats them so differently:

- Type 1 (noise in `meas`) is handled by *averaging and quality-weighting* — the `R` concentration check, and (with hardening) the `gain_r_scaling`, which reduces trust in a correction when line agreement is marginal.
- Type 2 (wrong `k`) can't be handled by averaging at all — a wrong `k` doesn't look like "slightly noisy," it looks like a completely different, internally-consistent-looking answer that happens to be 90° off. That's why it needs an entirely separate defense (the freeze-while-turning logic), rather than just "trust it a bit less."

### Redoing the example honestly, with both errors present

Say true heading = 100°, and this time the camera has a realistic 3° measurement error: it reports `meas = 13°` instead of the perfect `10°` (`103° mod 90° = 13°`, roughly — i.e., as if the "true mod-90 signal" itself has 3° of noise riding on it).

With `cur = 95°`:
```
k = round((95° - 13°) / 90°) = round(82°/90°) = round(0.911) = 1
meas_unwrapped = 13° + 90° = 103°
err = 103° - 95° = 8°
```
Still passes the 20° gate, still gets accepted, and nudges `offset` toward correcting an 8° gap — close to, but not exactly, the "true" 5° gap you'd get with a perfect measurement. That small discrepancy (8° vs. 5°) is Type-1 noise leaking through — harmless, self-correcting over many frames because of the tiny `gain`, and exactly what the gate/gain/R-scaling machinery is designed to tolerate gracefully.

**But now compare that to what a Type-2 (wrong `k`) error looks like — from the earlier table, `cur=54°` gave `meas_unwrapped=10°` instead of `100°`, a 90° error.** That's not a small nudge in the wrong direction — it's a completely different, catastrophically wrong answer that the gate might or might not catch depending on exact numbers, and if it slips through, it corrupts `offset` by nearly 90° in one shot rather than a gentle few-degree drift.

### The bottom line

`meas` implicitly assumed to be perfect was a deliberate simplification to isolate the concept being taught (the `k`-rounding cliff-edge). In reality, `meas` carries its own small noise on top, but that noise is the *mild*, well-handled kind of error — a few degrees, gently smoothed away by averaging, quality-weighting, and the tiny gain. The 45°-boundary problem with `k` is a fundamentally *different*, much more dangerous kind of error — not "somewhat off," but "confidently, catastrophically wrong by a clean 90°" — and that's precisely why it gets its own dedicated defense mechanism (turn-freezing) rather than being left to the same noise-averaging tools that handle ordinary measurement error in `meas`.

---

## Part 8: What "Freeze" Actually Means and Why It Exists

### What "freeze" literally means in the code

```python
if turning:
    self._n_rate_frozen += 1
    status = (f'frozen ...')
```
That's it — that's the entire "freeze" action. When `turning` is `True`, the code does **nothing** to `self.offset`. It just bumps a counter and sets a log message. Compare this to the "accepted" branch, which actually runs `self.offset = wrap(self.offset + g * err)`. In the frozen branch, that line simply never executes.

So "freezing" doesn't mean stopping the robot, pausing the camera, or halting any motion. It specifically means: **`self.offset` is not allowed to change this frame.** Whatever value `offset` currently holds, it stays exactly that value, untouched, for this camera frame — as if this frame's vision measurement never happened at all, from `offset`'s point of view.

Meanwhile, everything else keeps running completely normally: `on_imu` keeps firing, `gyro_yaw` keeps updating, and the published heading (`gyro_yaw + offset - pool_axis_offset`) keeps coming out on every IMU tick — just using the *old, unchanged* `offset` instead of a possibly-updated one.

### Why freeze it — what problem this is preventing

Think about what `offset` actually represents: it's the node's best running estimate of "how much has the gyro drifted so far." It gets built up slowly and carefully, a tiny nudge at a time, precisely so that no single bad frame can throw it far off course.

Now recall what was established earlier: **during a fast turn, the vision measurement for this one frame has an elevated chance of being badly wrong** — not "a little noisy," but *catastrophically* wrong, off by a clean 90°, because of the `k`-selection cliff-edge problem. Motion blur makes the line detection less reliable, and any small IMU/camera timing mismatch gets amplified by the fast rotation into a bigger error in `cur`, pushing it closer to that dangerous 45° boundary where `round()` can silently pick the wrong branch.

If the code *didn't* freeze, and just ran the normal acceptance logic on a frame captured during a fast turn, here's the bad case: `k` picks the wrong branch, `meas_unwrapped` ends up 90° away from the truth, and — if you're unlucky — the resulting `err` still happens to look small enough to sneak under the 20° gate. Then this bad frame does the exact same thing a good frame would do: `self.offset = wrap(self.offset + g * err)` runs, nudging `offset` toward a value that's actually wrong by a large amount. And because `offset` carries forward into every future frame's `cur` calculation, that corruption doesn't just affect this one frame — it pollutes the reference point for every frame after it, potentially triggering the exact same wrong-branch mistake again and again (a self-reinforcing bad lock).

### Freezing is a "when in doubt, don't touch it" decision

The logic is essentially: *"I can't fully trust vision measurements captured during a fast turn — the risk of a catastrophic, hard-to-detect wrong-branch error is elevated right now. Rather than gamble and possibly poison my best running estimate, I'll just refuse to use this frame for correction at all, and lean entirely on the gyro instead."*

That last part is the key justification for why this is *safe* to do, not just cautious: gyro drift is a slow process. Over the few seconds a typical turn takes, the gyro on its own accumulates only a tiny, sub-1° amount of additional drift — negligible compared to the ~90° error a bad vision correction could inject. So "doing nothing" (freezing) during the risky window costs almost nothing (a few seconds of slightly-uncorrected drift), while potentially saving `offset` from a large, persistent corruption that would take a long time (or a rare relock event) to undo. It's a straightforward cost-benefit tradeoff: the downside of skipping a few frames is small and temporary; the downside of accepting one bad frame during a turn can be large and lasting.

### Summary

**What's frozen:** just one variable, `self.offset` — it's held at its current value, unchanged, for the duration of the fast turn.

**Why:** because vision measurements captured mid-turn are unusually likely to be catastrophically (not just mildly) wrong, due to the mod-90° branch-selection mechanism being fragile exactly when rotation is fast. Freezing avoids feeding a potentially badly-wrong correction into the one variable (`offset`) that persists and compounds across every future frame, at the cost of simply trusting the gyro alone (which barely drifts over a few seconds) until the turn is over and normal, reliable corrections can resume.
