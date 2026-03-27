# Hilbert Curve and Chebsnake

**Original:** [fun/HilbertSurfaceChebsnake2](https://github.com/chebfun/examples/blob/master/fun/HilbertSurfaceChebsnake2.m)
**Author(s):** Georges Klein, March 2013

---

Functions of two variables can be plotted easily in Chebfun2. This example
combines Chebfun and Chebfun2 to draw space-filling curves on surfaces and
introduces the Chebsnake2 game.

## A hilly surface

A Chebfun2 function of two variables defines a hilly landscape:

$$h(x,y) = \cos\!\bigl(2(x^2 + y)\bigr)\,\sin\!\bigl(3(-x + y^2)\bigr).$$

## The Hilbert curve

The fifth iterate of the Hilbert space-filling curve has $2^{10} = 1024$
corner points. These are constructed recursively by the rule that each
iteration applies four affine transformations (involving rotations, reflections,
and scalings by $1/2$) to the previous set of corners. The resulting points
are defined as values in the complex plane and interpolated by a single
Chebfun.

Note that the figure does *not* display a true Hilbert curve -- it is a
smooth Chebyshev interpolant through the corner points that define the
Hilbert curve.

## A curve on the surface

Chebfun and Chebfun2 can be combined to study curves on surfaces. Given a
curve $(x(t), y(t))$ in the plane (here the Hilbert interpolant) and a
surface $h(x,y)$, one can evaluate $h$ along the curve and produce a 3D
space curve $(x(t), y(t), h(x(t),y(t)))$.

## Chebsnake2

The `chebsnake2` function is the 2D-surface analogue of Chebfun's classic
snake game. The user can specify a Chebfun2 surface, the kind of nodes for
the one-dimensional chebsnake, and a speed parameter. Since navigating on
a surface is more challenging than in the plane, the shadow of the snake
and the "food" are displayed as well.

## Code

```python
from examples.fun.hilbert_surface_chebsnake import run
run()
```

![Hilbert Curve and Chebsnake](../../images/fun/hilbert_surface_chebsnake.png)

## References

1. D. Hilbert, "Uber die stetige Abbildung einer Linie auf ein Flachenstuck,"
   *Mathematische Annalen* 38 (1891), 459--460.