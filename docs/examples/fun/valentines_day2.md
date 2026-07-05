# Happy Valentine's Day! (again)

**Original:** [fun/ValentinesDay2](https://www.chebfun.org/examples/fun/ValentinesDay2.html)
**Author(s):** Anonymous, February 2013

---

Happy Valentine's Day to all Chebfun2 users! This example constructs a 3D
heart-shaped surface, wraps a greeting message around it, and produces a
rotating animation.

## Parametric heart surface

The heart is defined as a parametric surface using Chebfun2, with parameters
$t \in [0,1]$ and $\theta \in [0, 4\pi]$:

$$X = \sin(\pi t)\cos(\theta/2),$$
$$Y = 0.7\,\sin(\pi t)\sin(\theta/2),$$
$$Z = \frac{(t-1)(-49 + 50t + 30t\cos\theta + \cos 2\theta)}{-25 + \cos^2\theta}.$$

## Colour scheme and message

A colour map is defined by

$$C = \sin(10X)\cos\!\bigl((Y-0.1)^2\bigr) + (Z+1),$$

giving warm tones when rendered with a "hot" colourmap.

The text "Happy Valentines Day!" is created using `scribble` and mapped onto
the 3D surface via `plot3`, wrapping the message around the heart.

## Rotating animation

The view angle is swept from $-1.25$ to $3$ (in multiples of $180^\circ$)
while keeping the elevation fixed at $6^\circ$, producing a smooth rotation
that reveals the heart from all sides. The animation can optionally be saved
as a GIF.


![Happy Valentine's Day! (again)](../../images/fun/valentines_day2.png)

1. Anonymous, "Happy Valentine's Day!," Chebfun Example [fun/ValentinesDay](https://www.chebfun.org/examples/fun/ValentinesDay.html), February 2013.

## Code

```python
import numpy as np

v = np.linspace(0, 2 * np.pi, 5)
hx = np.sin(v) * (15 * np.sin(v) - 4 * np.sin(3 * v)) / 16
hz = (15 * np.cos(v) - 5 * np.cos(2 * v) - 2 * np.cos(3 * v)
      - np.cos(4 * v)) / 16
print("heart curve samples:", np.round(hx, 3), np.round(hz, 3))
```


## References

## Figures (chebfun.org parity)

![ValentinesDay2 figure 1](../../images/fun/ValentinesDay2_01.png)
