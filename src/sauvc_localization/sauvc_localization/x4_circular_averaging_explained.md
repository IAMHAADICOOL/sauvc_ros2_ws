# The ×4 Trick for Mod-90° Circular Averaging — Explained

Let's build this from the ground up with a clean geometric picture, then generalize beyond just the 0°/90° special case.

## Step 1: Ordinary circular averaging (no ×4) — why angles need vectors, not raw numbers

You can't just average angles like normal numbers because of wraparound: average of 359° and 1° should be 0° (they're 2° apart, centered at 0°), but naive arithmetic averaging gives `(359+1)/2 = 180°` — completely wrong, on the opposite side of the circle.

The fix (standard, nothing to do with our 90° problem yet): treat each angle as a point on a circle — an arrow of length 1 pointing in that direction, i.e. `(cos θ, sin θ)`. To average, add up all the arrows as vectors (add the x's, add the y's), then find the angle of the *resulting* arrow with `atan2`. Averaging 359° and 1°: their arrows point almost the same direction (both near "3 o'clock, slightly up/down"), so adding them gives a combined arrow also pointing almost exactly at 0°. This correctly handles wraparound because vector addition naturally respects the circular geometry — there's no artificial jump at the 0°/360° seam.

## Step 2: Our extra problem — it's not 360°-periodic, it's 90°-periodic

Ordinary circular averaging assumes: if two angles are far apart on the circle, they're genuinely different directions, and should pull the average toward a compromise. That's true for a normal compass direction. But it's **false** for our floor-grid measurement — as established, a 0° reading and a 90° reading aren't "different directions that disagree," they're the *same* underlying grid rotation, just seen through two different (row vs. column) line families. If you feed them into ordinary vector-averaging as-is, the arrows for 0° and 90° point in genuinely different directions (one points "east," one points "north") and would partially cancel/compromise — which is exactly the wrong behavior here, since we want them to *reinforce*, not compromise.

## Step 3: The fix — relabel the circle before averaging, so "same fact" angles become "same point"

Here's the geometric trick, stated as directly as possible: **before doing the vector-average, stretch every angle by multiplying it by 4.** Why 4? Because our real period is 90°, and a full circle is 360°, and `360° ÷ 90° = 4`. Multiplying by 4 takes something that repeats every 90° and turns it into something that repeats every 360° — i.e., it converts our 90°-periodic quantity into an ordinary 360°-periodic quantity, which is exactly the kind ordinary vector-averaging (Step 1) already knows how to handle correctly.

**Why does ×4 specifically achieve "0° and 90° become the same point"?** Because on the circle, two points are "the same point" precisely when they differ by a multiple of 360°. So we need: whenever two original angles differ by a multiple of 90° (i.e., they represent the "same fact" per our earlier discussion), their *stretched* versions should differ by a multiple of 360°. Check: if two angles differ by 90°, their ×4 versions differ by `4 × 90° = 360°` — exactly one full lap, landing back on the same spot. If they differ by 180°, ×4 gives `720°` — two full laps, still the same spot. This is precisely why multiplying by 4 was the right number to choose — it's engineered so that "differs by any multiple of 90°" becomes "differs by a multiple of 360°" (i.e. the same point), automatically.

## Step 4: Walking through more than just 0°/90° — showing it's general, not a coincidence

Let's check a *non-special-case* pair — say 10° and 100° (differ by 90°, but not aligned with the axes like before):

```
10° × 4  = 40°
100° × 4 = 400°,  and 400° mod 360° = 40°
```

Both land on **40°** — identical, just like 0°/90° landed on the identical point before. This confirms it's not a fluke of the 0°/90° example — *any* two angles 90° apart collapse to the same stretched point, because the ×4 relationship works structurally, not just for round numbers.

Now contrast with two angles that are **genuinely different**, not 90°-related — say 10° and 35°:

```
10° × 4 = 40°
35° × 4 = 140°
```

These do **not** land on the same point (40° ≠ 140°) — correctly, since 10° and 35° really are different, disagreeing readings (not related by any multiple of 90°), so the stretch correctly keeps them apart, and averaging them will produce a genuine compromise value, as it should.

## Step 5: Undoing the stretch at the end

Since everything was stretched by ×4 before averaging, the *result* of that averaging is also sitting in "stretched space" — you have to shrink it back down by dividing by 4 to get a real, normal-sized angle back:

```python
mean4 = math.atan2(s, c)   # this is a stretched-space angle
return mean4 / 4.0         # shrink back down to real angle
```

This is why, in the 0°/90° example, `mean4` came out as `0°` (both segments landed exactly on the stretched point `0°`, so their vector-sum also points at `0°`), and dividing by 4 still gives `0°/4 = 0°` — landing correctly back on "grid-aligned," not on some artificial 45° compromise.

## The one-sentence summary

The ×4 multiplication is a **relabeling trick**: it renames every angle so that two angles which used to look "90° apart" (and would be wrongly treated as disagreeing) now look "0° apart, or a full lap apart" (correctly recognized as agreeing) — turning our nonstandard "same-fact-every-90°" problem into an ordinary, already-solved "same-fact-every-360°" problem, which the standard vector-averaging technique from Step 1 handles automatically and correctly, before finally shrinking the answer back down (÷4) to a real-world angle at the very end.
