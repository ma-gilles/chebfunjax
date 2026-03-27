# Chapter 14: Chebfun2: Rootfinding and Optimisation

*Based on [Chebfun Guide Chapter 14](https://www.chebfun.org/docs/guide/guide14.html)*

This chapter covers computational techniques for finding zeros and extrema of bivariate functions using `Chebfun2`.

## 14.1 Zero Contours

The `roots()` method of `Chebfun2` finds the zero level-set of a function $f(x,y) = 0$. For a function of two variables, the zero set is typically a curve (or collection of curves) in the $(x, y)$ plane.

```python
import jax.numpy as jnp
from chebfunjax.chebfun2d import chebfun2

# Circle of radius 0.7
f = chebfun2(lambda x, y: x**2 + y**2 - 0.49)
curves = f.roots()
print(f"Number of components: {len(curves)}")
print(f"Number of points: {curves[0].shape[0]}")
```

The result is a list of point clouds, one per connected component. Each element is an array of shape `(n_pts, 2)` with $(x, y)$ coordinates along the zero contour.

The implementation uses a marching-squares algorithm on a fine grid (500 x 500 by default), with linear interpolation to locate sign changes. This provides good initial approximations but is not refined to machine precision via Newton iteration (unlike the MATLAB version).

### Example: Lemniscate of Bernoulli

```python
f = chebfun2(lambda x, y: (x**2 + y**2)**2 - (x**2 - y**2))
curves = f.roots()
# The lemniscate is a figure-eight curve
```

## 14.2 Solving Systems of Two Equations

To solve a system $f(x,y) = 0,\; g(x,y) = 0$ simultaneously, one approach is to find the zero contours of each function and compute their intersections:

```python
f = chebfun2(lambda x, y: x**2 + y**2 - 0.49)
g = chebfun2(lambda x, y: x - y)

# Find zero contours separately
curves_f = f.roots()
curves_g = g.roots()

# The intersections are where the circle x^2+y^2=0.49 meets y=x
# Analytically: x = y = +/- sqrt(0.245)
```

For more sophisticated bivariate rootfinding (common zeros of two `Chebfun2` objects), the MATLAB Chebfun uses Bezoutian resultant methods. In chebfunjax, you can combine the marching-squares approach with numerical refinement.

## 14.3 Critical Points

Critical points of a function $f(x,y)$ satisfy $\partial f/\partial x = 0$ and $\partial f/\partial y = 0$ simultaneously. Using `Chebfun2` differentiation:

```python
f = chebfun2(lambda x, y: (1 - x)**2 + 100 * (y - x**2)**2,
             domain=(-2.0, 2.0, -1.0, 3.0))

# Compute gradient components
fx = f.diff(dim=2)  # df/dx
fy = f.diff(dim=1)  # df/dy

# The critical point of Rosenbrock's function is at (1, 1)
print(f"fx(1,1) = {fx(1.0, 1.0):.2e}")
print(f"fy(1,1) = {fy(1.0, 1.0):.2e}")
```

## 14.4 Global Optimisation

While `Chebfun2` does not have built-in `max2` / `min2` methods in chebfunjax, you can find extrema by evaluating on a fine grid:

```python
import numpy as np

f = chebfun2(lambda x, y: jnp.cos(10 * x * y) * jnp.exp(-x**2 - y**2))

# Evaluate on a fine grid
n = 200
xs = jnp.linspace(-1, 1, n)
ys = jnp.linspace(-1, 1, n)
xx, yy = jnp.meshgrid(xs, ys)
vals = f(xx.ravel(), yy.ravel()).reshape(n, n)

# Find global max/min
idx_max = jnp.argmax(vals)
idx_min = jnp.argmin(vals)
i_max, j_max = jnp.unravel_index(idx_max, (n, n))
i_min, j_min = jnp.unravel_index(idx_min, (n, n))

print(f"Max value: {vals[i_max, j_max]:.10f} at ({xs[j_max]:.4f}, {ys[i_max]:.4f})")
print(f"Min value: {vals[i_min, j_min]:.10f} at ({xs[j_min]:.4f}, {ys[i_min]:.4f})")
```

Alternatively, use JAX's automatic differentiation for gradient-based optimization:

```python
import jax

# Use JAX's gradient for local optimization
def neg_f(xy):
    return -f(xy[0], xy[1])

xy0 = jnp.array([0.1, 0.1])

# Simple gradient descent to find local maximum
lr = 0.01
xy = xy0
for _ in range(1000):
    g = jax.grad(neg_f)(xy)
    xy = xy - lr * g

print(f"Local maximum at ({xy[0]:.6f}, {xy[1]:.6f})")
print(f"Value: {f(xy[0], xy[1]):.10f}")
```

## 14.5 The Infinity Norm

The infinity norm $\|f\|_\infty = \max_{(x,y) \in \Omega} |f(x,y)|$ can be approximated by evaluation on a fine grid:

```python
f = chebfun2(lambda x, y: jnp.sin(10 * x) * jnp.cos(7 * y))

n = 500
xs = jnp.linspace(-1, 1, n)
ys = jnp.linspace(-1, 1, n)
xx, yy = jnp.meshgrid(xs, ys)
vals = f(xx.ravel(), yy.ravel()).reshape(n, n)
inf_norm = float(jnp.max(jnp.abs(vals)))
print(f"||f||_inf ~ {inf_norm:.10f}")
```

## 14.6 Gradient Fields

The gradient of a scalar `Chebfun2` can be constructed as a `Chebfun2v`:

```python
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun2d.separable_approx import SeparableApprox

f = chebfun2(lambda x, y: jnp.sin(x) * jnp.cos(y))

# Build gradient manually using diff
fx_approx = SeparableApprox.from_function(
    lambda x, y: jnp.cos(x) * jnp.cos(y))
fy_approx = SeparableApprox.from_function(
    lambda x, y: -jnp.sin(x) * jnp.sin(y))

grad_f = Chebfun2v([fx_approx, fy_approx])
```

Or more directly, using `diff`:

```python
fx = f.diff(dim=2)  # df/dx as Chebfun2
fy = f.diff(dim=1)  # df/dy as Chebfun2
```

## 14.7 References

1. A. Townsend and L. N. Trefethen, "An extension of Chebfun to two dimensions", *SIAM J. Sci. Comput.*, 35(6), C495--C518, 2013.

2. Y. Nakatsukasa, V. Noferini, and A. Townsend, "Computing the common zeros of two bivariate functions via Bezout resultants", *Numer. Math.*, 129(1), 2015.
