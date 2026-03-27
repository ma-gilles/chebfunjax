# Dual Points, Lines, Polygons and Curves

**Original:** [geom/DualCurves](https://github.com/chebfun/examples/blob/master/geom/DualCurves.m)
**Author(s):** Alex Townsend, August 2011

---

In geometry many statements have an equally valid dual statement [1, 2].
For example, "Two non-parallel lines intersect at exactly one point" is
dual to "Two distinct points are joined by exactly one line" and "Three
points lie on a line iff they are collinear" is dual to "Three lines meet
at a point iff they are concurrent".

## Point-line duality

The dual of a point is a line: the point $(a,b)$ is dual to the line
with gradient $-a/b$ passing through $(-a/(a^2+b^2),\, -b/(a^2+b^2))$.
Conversely, the dual of a line is a point: the line $y = mx + c$ is dual
to the point $(m/c,\, -1/c)$.

## Dual of a polygon

A polygon consists of edges (lines) and vertices (points). The dual of an
edge is a point and the dual of a vertex is a line. Hence the dual of a
polygon is also a polygon. In particular, odd-sided regular polygons are
self-dual.

## Dual of a curve

A curve is the limit of piecewise linear curves, so we can extend the
definition of duality to curves. Given any curve $C$, the **dual curve**
is the set of points which are dual to the tangent lines of $C$.

Given a parameterized curve $z(t) = x(t) + iy(t)$, the dual curve can
be computed from the formula

$$
p(t) = \frac{-y'(t)}{x'(t)\,y(t) - x(t)\,y'(t)}, \qquad
q(t) = \frac{x'(t)}{y'(t)\,x(t) - y(t)\,x'(t)},
$$

so the dual curve is $p(t) + iq(t)$.

## Example: heart curve

A smoothed heart-shaped curve is defined by

$$
x(t) = 2\sin t, \quad
y(t) = 2\cos t - \tfrac{1}{2}\cos 2t - \tfrac{1}{4}\cos 3t
       - \tfrac{1}{8}\cos 4t,
$$

for $t \in [0, 2\pi]$. Its dual reveals the envelope of tangent lines.

## References

1. [Wikipedia: Dual curve](http://en.wikipedia.org/wiki/Dual_curve)

2. [Wikipedia: Dual polygon](http://en.wikipedia.org/wiki/Dual_polygon)

## Code

```python
from examples.geom.dual_curves import run
run()
```

## Output

![Dual Curves](../../images/geom/dual_curves.png)
