# `self._rate_lp` — What It Measures, Why It's Smoothed, and Its Startup Lag

## What `self._rate_lp` is measuring

It's a number that answers one question: **"how fast is the vehicle spinning right now?"** — specifically, the speed of rotation around the vertical axis (turning left/right, like spinning on the spot), measured in radians per second. It comes from the gyro's raw angular velocity reading (`msg.angular_velocity.z`), smoothed out so a single jittery sensor sample can't fool it (covered in an earlier discussion of the low-pass filter — a running, self-updating average that mostly reflects recent reality but doesn't overreact to one noisy blip).

Think of it like a speedometer, but for spinning instead of driving forward. `0` means not turning at all. A small number means turning slowly. A bigger number means turning fast.

## Why it's used in this specific line

```python
turning = hardening and self._rate_lp > float(self.get_parameter('freeze_rate').value)
```

This line is just asking: *"is the vehicle currently spinning fast enough that I shouldn't trust this frame's vision correction?"* It compares the smoothed spin-speed (`_rate_lp`) against a threshold (`freeze_rate`, default `0.15` rad/s ≈ 8.6°/s). If the vehicle is spinning faster than that threshold, `turning` becomes `True`.

## What "purpose" it's serving here — the plain-English reason

Recall the core mechanism: the node figures out the true heading from a line reading by picking whichever of 4 possible headings (90° apart) is closest to what the gyro already believes. That trick only works if the gyro's belief is reasonably close to the truth at that exact moment.

Here's the problem with fast spinning specifically: when the vehicle is turning quickly, two things go wrong at once. First, the camera image itself gets blurry (things are moving across the frame fast during the exposure), which makes the line-detection less reliable. Second, there's always a tiny delay between "when the photo was actually taken" and "when the gyro sample used to interpret it was measured" — and if the vehicle is spinning fast, even a small delay means the vehicle has already turned a meaningful amount in that gap, throwing off the comparison. Put together, fast spinning is exactly the situation where the "pick the closest of the 4 options" trick is most likely to accidentally pick the wrong option — which would badly corrupt the correction.

So this line is a simple safety check: *"is the vehicle spinning fast right now? If so, don't trust this frame's correction — sit this one out."* That's why, a few lines later, when `turning` is `True`, the code skips updating `offset` entirely rather than risk feeding it a possibly-wrong correction.

## Why "smoothed" instead of raw

If this check used the raw, jittery sensor reading instead of the smoothed `_rate_lp`, it could flicker unpredictably — falsely triggering "turning fast!" from a single noisy sensor spike even while the vehicle is actually holding still, or missing a real fast turn because one lucky low-noise sample slipped through right when it mattered. The smoothed version reacts to genuine, sustained spinning (within about a tenth of a second) while ignoring single-sample noise, making this "should I trust this frame" decision far more reliable.

---

## Is the startup/onset lag a problem?

You're circling back to a real gap here, and the phrase "is it alright to use it in the start only" suggests a slight misread — let's clarify one thing first, then address whether this lag is actually a problem in practice.

### Clarifying: it's not "only at the start"

The lag isn't a one-time startup quirk — it re-happens **every single time the true rotation rate changes**, not just at node boot. If the vehicle is calmly sitting still (`_rate_lp ≈ 0`) and then suddenly starts spinning fast, `_rate_lp` lags behind the *new* true value for about 0.1 second, exactly the same way it lagged behind at `t=0`. Startup is just the first instance of this general pattern (starting from 0 is a special case, but the *lag itself* is a recurring, structural property of the filter, happening at the beginning of every acceleration).

### Is it "alright"? — Yes, but with a specific, bounded caveat, not a free pass

You're right to be suspicious: **during that ~0.1s catch-up window, `_rate_lp` genuinely does understate how fast the vehicle is actually turning.** That means there's a small window, right at the *onset* of a fast turn, where:

```python
turning = hardening and self._rate_lp > freeze_rate
```

could evaluate `False` (not yet flagged as "turning") even though the vehicle has, in the last instant, actually started spinning fast. In that narrow window, a camera frame could still slip through into the "not turning" branch and attempt a vision correction — right when the real-world conditions (possible blur, timing lag) are becoming exactly the risky ones discussed throughout this whole conversation.

So no, this isn't a perfect, airtight guard. It has a genuine blind spot.

### Why this is still a reasonable engineering tradeoff, not a bug

A few things bound how bad this actually is in practice:

1. **The lag window is short (~0.1–0.2s) relative to camera frame timing.** At typical camera rates (say 15–30 fps), that's only 2-6 frames where the risk is elevated but not yet flagged — not a large exposure window compared to, say, a multi-second turn.

2. **Rotation doesn't reach dangerous speed instantly.** Physically, a vehicle accelerating its spin rate still has to ramp up — it doesn't teleport from 0 to a fast turn-rate in zero time. So while `_rate_lp` is lagging behind the *true instantaneous* rate, the true rate itself is *also* still relatively low in that same brief window (it's still accelerating too). The two lags partially offset each other in practice — `_rate_lp`'s error is largest exactly when the true rate itself hasn't fully "arrived" at its dangerous value yet either.

3. **This isn't the only layer of defense** — remember, this sits alongside the 20° `err` sanity gate and the image-timestamp interpolation (`_yaw_at`). Even if a frame slips through the freeze check during this brief lag window, it still has to pass the 20° gate to actually corrupt `offset`. It's not guaranteed to be caught, but it's not undefended either — this is "defense in depth," where several imperfect layers overlap rather than any single layer being airtight.

4. **The alternative (using the raw, un-smoothed rate) is worse, not better.** If you removed the smoothing to eliminate this lag, you'd reintroduce the original problem: single noisy spikes triggering false "turning!" flags constantly, which would make the freeze logic unreliable in the *opposite* direction — freezing legitimate, safe corrections unnecessarily, all the time, due to sensor noise rather than real turns.

### Honest summary

You've correctly found a real, acknowledged-by-design limitation: `_rate_lp`'s ~0.1s catch-up lag means there's a brief window at the *onset* of any fast rotation where the freeze mechanism hasn't "noticed" yet, and a risky frame could theoretically slip through. It's "alright" not because the gap doesn't exist, but because (a) the window is short, (b) the true rate is also still low during most of that window, and (c) it's backed up by the separate 20° error gate as a second line of defense — not because this particular piece of code is bulletproof on its own. This is a genuinely imperfect but deliberately-accepted tradeoff, the same category as "the mod-90 ambiguity resolution assumes `cur` is within 45° of truth" — a probabilistic risk-reduction measure, not an absolute guarantee.

---

## Why is the smoothing argument even needed? Why not just use the raw value directly?

The short answer: because the raw value directly would make the freeze decision *worse*, not simpler — it would fail in the opposite direction, and more often.

Think about what "using the raw value directly" actually means here: on every single camera frame, you'd check the *one* IMU sample nearest to that moment and ask "was the vehicle turning fast right at this instant?" The problem is that a single IMU sample is noisy — vibration, thruster wash, electrical noise can all make one sample read a spurious spike (e.g., a true rate of `0.05 rad/s` momentarily reading `0.4 rad/s` on one sample). If the freeze check used that raw sample directly:

- It would **falsely freeze** perfectly good frames constantly — the vehicle sitting still or turning slowly, but one noisy sample crosses the `freeze_rate` threshold and the code needlessly discards a trustworthy correction. Over a long mission, this means `offset` gets far fewer opportunities to correct drift than it should, even though nothing was actually wrong.
- It could also **fail to freeze** during an actual fast turn, if one lucky low-noise sample happened to land right when a camera frame was being processed — letting a genuinely risky frame through the exact door the freeze check exists to close.

Neither failure mode is acceptable, and using the raw value directly gives you *both*, unpredictably, frame to frame. The smoothing argument isn't an added complication for its own sake — it's what makes the freeze decision actually track *real, sustained* turning rather than sensor noise, which is the only thing the freeze logic is supposed to be reacting to in the first place. The 0.1s lag is the cost paid for that reliability, and it's a small, bounded cost compared to the alternative of a constantly flickering, unreliable freeze signal.
