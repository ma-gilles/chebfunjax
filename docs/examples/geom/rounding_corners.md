# Rounding Corners by Convolution

**Original:** [geom/RoundingCorners](https://www.chebfun.org/examples/geom/RoundingCorners.html)
**Author(s):** Nick Trefethen, November 2012

---

This example demonstrates how convolution can be used to smooth corners of
a function or a parametric curve in the plane.

## A W-shaped function

We start with a function that has the shape of a _W_:

$$
f(t) = 3\min\bigl(|t + 0.4|,\;|t - 0.3|\bigr),
$$

which has sharp corners where the two absolute-value branches meet.

## A narrow tent function

Next we define a narrow "tent" function $g$ with integral equal to 1 and
half-width $h = 0.1$:

$$
g(s) = \frac{h - |s|}{h^2}, \qquad s \in [-h, h].
$$

## Convolution rounds the corners

If we convolve the two functions, we get a _W_ with rounded corners. The
radius of rounding is controlled by the width parameter $h$ of the tent
function. At the ends, the "rounding" has brought the values down to 0.

## Complex convolution

A similar but different computation uses a complex-valued function of a
real parameter, where the _W_ is a curve in the plane:

$$
W(t) = t + if(t).
$$

Convolving $W$ with $g$ rounds the corners of the plane curve. The result
looks different from the real-valued case because the convolution now
operates on both the $x$ and $y$ components simultaneously, shifting
the entire curve horizontally at the endpoints rather than just pulling
the values toward zero.


![Rounding Corners](../../images/geom/rounding_corners.png)

## Code

```python
from examples.geom.rounding_corners import run
run()
```
