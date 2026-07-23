Explanation for the logic below:

This part trips a lot of people up because it highlights the difference between human perception and computer vision. 
When you look at a grid of pool tiles, your brain instantly maps it as a "floor," and you know which way is forward.
The camera and OpenCV don't know what a floor is, or which way the robot is pointing. They just see a flat image with straight lines.
Here is exactly why OpenCV returns four different angles, and why naive math breaks down when trying to average them.

---

## 1. Why OpenCV Sees 4 Different Angles

Imagine looking straight down at a perfectly square grid of pool tiles. Let's say the robot is rotated exactly **$10^\circ$** off from the grid lines.

When OpenCV runs its line-detecting algorithm (Hough Transform), two things happen that create the 4-fold ambiguity:

### The Grid Has Two Axes
A square grid is made of vertical grout lines and horizontal grout lines. If the robot is offset by $10^\circ$, the lines going "roughly forward" will measure at **$10^\circ$**. But OpenCV will also detect the horizontal lines crossing them. Because the tiles are square, those crossing lines are exactly $90^\circ$ perpendicular. OpenCV will measure those at $10^\circ + 90^\circ = \mathbf{100^\circ}$.

### Lines Have No Inherent "Forward" Direction
In computer vision, a line segment is just a connection between Point A $(x_1, y_1)$ and Point B $(x_2, y_2)$.
OpenCV arbitrarily picks which end is Point A and which is Point B.

* If it measures a horizontal grout line from left-to-right, the angle is $100^\circ$.
* If it measures that exact same horizontal grout line from right-to-left, the angle flips to $100^\circ + 180^\circ = \mathbf{280^\circ}$.
* If it measures the forward grout line backward (top-to-bottom), it gets $10^\circ + 180^\circ = \mathbf{190^\circ}$.

**The Result:** From a single grid that is rotated by $10^\circ$, OpenCV hands you a bucket of line segments with angles at roughly **$10^\circ$**, **$100^\circ$**, **$190^\circ$**, and **$280^\circ$**.

Every single one of those angles is correctly telling you "the grid is rotated by $10^\circ$." But to a computer, $10$ and $100$ are completely different numbers.

---

## 2. Why Direct Averaging Fails

Now, let's look at why a standard arithmetic average destroys this data.

Let's say the robot is **perfectly aligned** with the pool tiles. The true offset is **$0^\circ$**.
OpenCV detects two very strong, clean lines:
1. One going straight ahead: **$0^\circ$**
2. One going perfectly sideways: **$90^\circ$**

Both of these lines are perfectly aligned with the grid. Both lines are effectively screaming: 
*"The robot is perfectly aligned!"*

But watch what happens if you take a standard average:

$$\text{Average Angle} = \frac{0^\circ + 90^\circ}{2} = 45^\circ$$

### The $45^\circ$ Contradiction
Your math just concluded the robot is facing $45^\circ$.

If the robot turns to $45^\circ$, it will be driving diagonally across the tiles, cutting right through the middle of the squares where there are no grout lines at all. By averaging a vertical line and a horizontal line together, you created a "ghost" line that doesn't actually exist on the grid.

* **$0^\circ$** means "aligned with the grid."
* **$90^\circ$** means "aligned with the grid."
* But averaging them gives you **$45^\circ$**, which means "completely unaligned with the grid."

Because a square grid looks identical every time you rotate it by $90^\circ$, angles that are exactly $90^\circ$ apart are mathematically equivalent in this context. A standard average doesn't know that. It treats $0^\circ$ and $90^\circ$ as competing values that need to meet in the middle, rather than agreeing values that should reinforce each other.

That is exactly why the code multiplies the angles by 4: it forces $0^\circ$, $90^\circ$, $180^\circ$, and $270^\circ$ to mathematically become the exact same number before the average is taken.

To see exactly how this mathematical sleight of hand works, let's run the numbers on a grid that is rotated exactly **$10^\circ$** from the robot's camera.

If OpenCV detects two lines of equal length (we'll call their length $L=1$ to keep it simple):
* **Line A:** $10^\circ$ (A forward-pointing tile line)
* **Line B:** $100^\circ$ (A horizontal crossing tile line)

If we take a naive average: $(10^\circ + 100^\circ) / 2 = \mathbf{55^\circ}$. This is completely wrong—it tells the robot the grid is at $55^\circ$, pointing it straight into the diagonal of the tiles.

Here is how the $\times 4$ algorithm runs those exact same numbers and outputs the correct answer.

1. **Multiply Angles by 4:**
First, the algorithm takes the raw angles and scales them up by 4.
* **Line A:** $10^\circ \times 4 = \mathbf{40^\circ}$
* **Line B:** $100^\circ \times 4 = \mathbf{400^\circ}$
Because a circle is $360^\circ$, an angle of $400^\circ$ wraps around and lands on the exact same position as $40^\circ$ ($400^\circ - 360^\circ = 40^\circ$). **This is the magic moment:** the $90^\circ$ difference between the two lines has been mathematically erased.

2. **Convert to Vector Components (s and c):**
Next, instead of adding degrees directly, we convert these new angles into 2D Cartesian vectors using sine (for the Y-axis) and cosine (for the X-axis).

**Line A ($40^\circ$):**
$$x_A = 1 \cdot \cos(40^\circ) \approx 0.766$$
$$y_A = 1 \cdot \sin(40^\circ) \approx 0.643$$

**Line B ($400^\circ$, which is just $40^\circ$):**
$$x_B = 1 \cdot \cos(400^\circ) \approx 0.766$$
$$y_B = 1 \cdot \sin(400^\circ) \approx 0.643$$

3. **Sum the Vectors (The Averaging Step):**
Now we add up all the X components (the `c` accumulator in the code) and all the Y components (the `s` accumulator).

$$c_{\text{total}} = 0.766 + 0.766 = \mathbf{1.532}$$
$$s_{\text{total}} = 0.643 + 0.643 = \mathbf{1.286}$$

Notice that because both lines mapped to the same angle, their vectors *added* together constructively, giving us a longer, stronger vector pointing in the agreed-upon direction.

4. **Recover the Angle:**
We use `atan2(y, x)` to turn our total X and Y values back into an angle in $\times 4$ space.

$$\phi = \text{atan2}(1.286, 1.532) = \mathbf{40^\circ}$$

Even with multiple lines, the combined vector still points exactly at $40^\circ$.

5. **Divide by 4 for the Final Offset:**
Because we multiplied by 4 in Step 1, our angle is currently four times too large. We divide by 4 to map it back to the real world.

$$\text{Final Yaw} = \frac{40^\circ}{4} = \mathbf{10^\circ}$$

By converting the angles to $\times 4$ space, the orthogonal horizontal line ($100^\circ$) acted as a **perfect second vote** for the $10^\circ$ forward line, rather than contradicting it. The algorithm mathematically grouped them together, proving the true grid offset is exactly $10^\circ$.

This code calculates **$R$**, known in directional statistics as the **Mean Resultant Length** (or circular concentration). It acts as a built-in confidence score—a "BS-detector"—to prevent the robot from trusting bad data.

Here is exactly what this quantity signifies and how the math breaks it down.

## 1. The Numerator: The Vector Tug-of-War
`math.hypot(s, c)` calculates the magnitude (length) of the final combined vector in $\times 4$ space: $\sqrt{s^2 + c^2}$. 

Because vectors have direction, they can cancel each other out. If one line says the grid is at $10^\circ$ and another says it's at $30^\circ$, their vectors will pull in slightly different directions, resulting in a final combined vector that is shorter than if they had perfectly agreed.

## 2. The Denominator: Maximum Potential Length
`sum(math.hypot(x2 - x1, y2 - y1) ...)` calculates the raw sum of all the individual line segment lengths in pixels, ignoring their angles completely. 

This represents the "perfect scenario." If every single line segment in the image pointed in the *exact same direction*, there would be zero cancellation. The final vector magnitude would exactly equal this total sum.

## 3. The Ratio ($R$): Measuring Agreement
By dividing the actual vector length by the maximum potential length, you get $R$:

$$R = \frac{\text{Actual Combined Vector Magnitude}}{\text{Sum of All Individual Line Lengths}}$$

$R$ always evaluates to a value between $0$ and $1$, representing how strongly the lines agree with each other.

| $R$ Value | Meaning | Vector Behavior |
| :--- | :--- | :--- |
| **$1.0$** | **Perfect Agreement** | Every line points in the exact same (transformed) direction. Zero cancellation. |
| **$0.8$ - $0.9$** | **Strong Grid** | A clear tile floor with minor noise from water ripples or pixel jitter. |
| **$\approx 0.0$** | **Total Chaos** | Lines point in random directions, completely cancelling each other out. |

## 4. The `0.6` Threshold 
In a real swimming pool, OpenCV won't just see the floor grid. It will detect lane ropes, shadows, curved pool walls, debris, and reflections. 

When the camera points at a clean tile floor, the lines will strongly agree, and $R$ will easily clear $0.80$. But if the robot tilts upward and the camera sees a jumble of overlapping waves and curved walls, the Hough Transform will return lines pointing in a hundred different directions. The vectors will cancel each other out, and $R$ will drop rapidly.

The logic `if R < 0.6:` translates to: 
*"If the angles are so scattered that the vectors lose more than 40% of their potential strength to internal contradictions, this is not a valid tile grid."*

Instead of feeding a hallucinated, random heading to the robot's navigation system, it gracefully rejects the frame (`low_concentration`) and returns `None`, telling the robot to rely on its IMU/gyroscope until it can see a clear grid again.