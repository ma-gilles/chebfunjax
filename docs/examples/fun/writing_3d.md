# Writing a Message in 3D

**Original:** [fun/Writing3D](https://www.chebfun.org/examples/fun/Writing3D.html)
**Author(s):** Nick Trefethen, November 2010

---

The `scribble` command produces a chebfun defined on the domain $[-1,1]$ that
takes piecewise-linear complex values. This example shows how to lift such
"scribble" text into three dimensions.

## From complex plane to 3D

The use of complex variables in `scribble` is a convenience: the real and
imaginary parts of $s(t)$ give the $x$- and $y$-coordinates of the lettering.
Including `'jumpline','none'` in the plot command prevents dotted lines from
being drawn between disconnected letter strokes (Chebfun has different
defaults for plotting gaps in real and complex functions).

## Text on a wavy surface

With real and imaginary parts separated as

$$r(t) = \operatorname{Re}\bigl(s(t)\bigr), \qquad
  i(t) = \operatorname{Im}\bigl(s(t)\bigr),$$

a `plot3` call of the form $(r, \sin(6r), i)$ drapes the message over a wavy
surface whose height varies sinusoidally with the horizontal coordinate.

## Text on a cylinder

A longer message -- the poem by British poet Kate McLoughlin,

> *There is no fun like chebfun. Try it and you'll see.
> It does your calculation, and makes a cup of tea!*

-- is wrapped around a cylinder. Scaling $s$ by a factor of 6 and using

$$\bigl(\cos(r),\; \sin(r),\; \operatorname{Im}(s) + 0.05\,r\bigr)$$

maps the horizontal coordinate onto a circle while the vertical coordinate
retains the letter shapes with a gentle spiral offset $0.05\,r$. The
`camorbit` command then rotates the camera 360 degrees so the viewer can read
the message as it circles around.


![Writing a Message in 3D](../../images/fun/writing_3d.png)

## Code

```python
from examples.fun.writing_3d import run
run()
```
