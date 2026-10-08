"""JAX cubic splines shared by public interpolation and contour refinement.

The callers follow @chebfun/spline.m and @separableApprox/roots.m at
Chebfun 7574c77. The kernel solves the cubic interpolation equations with
not-a-knot or prescribed endpoint slopes; it uses linear storage and a
tridiagonal solve, including for complex and array-valued data.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def spline_coefficients(x, y, slopes=None):
    """Return ascending local-power coefficients, shape ``(n-1, 4, ...)``.

    Sites must be real, strictly increasing, and have length at least two.
    Samples have their site axis first. Optional slopes have shape
    ``(2, ...)``. Validation of values belongs to the eager public adapter;
    this fixed-shape kernel can be traced and compiled.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y)
    dtype = jnp.result_type(y.dtype, jnp.float64)
    if slopes is not None:
        dtype = jnp.result_type(dtype, jnp.asarray(slopes).dtype)
    y = y.astype(dtype)
    n = x.shape[0]
    values = y.reshape((n, -1))
    h = jnp.diff(x)
    delta = jnp.diff(values, axis=0) / h[:, None]
    if slopes is None and n == 2:
        second = jnp.zeros_like(values)
    elif slopes is None and n == 3:
        curvature = 2 * (delta[1] - delta[0]) / (h[0] + h[1])
        second = jnp.broadcast_to(curvature, values.shape)
    elif slopes is None:
        # Eliminate endpoint second derivatives using equality of the
        # third derivatives on the first/last two intervals.
        diagonal = 2 * (h[:-1] + h[1:])
        lower = jnp.concatenate((jnp.zeros(1), h[1:-1]))
        upper = jnp.concatenate((h[1:-1], jnp.zeros(1)))
        diagonal = diagonal.at[0].add(h[0] * (1 + h[0] / h[1]))
        upper = upper.at[0].add(-h[0] * (h[0] / h[1]))
        diagonal = diagonal.at[-1].add(h[-1] * (1 + h[-1] / h[-2]))
        lower = lower.at[-1].add(-h[-1] * (h[-1] / h[-2]))
        rhs = 6 * jnp.diff(delta, axis=0)
        middle = jax.lax.linalg.tridiagonal_solve(
            lower.astype(dtype), diagonal.astype(dtype), upper.astype(dtype), rhs)
        first = middle[0] + (h[0] / h[1]) * (middle[0] - middle[1])
        last = middle[-1] + (h[-1] / h[-2]) * (middle[-1] - middle[-2])
        second = jnp.concatenate((first[None], middle, last[None]), axis=0)
    else:
        slopes = jnp.asarray(slopes, dtype=dtype).reshape((2, -1))
        diagonal = jnp.concatenate((2 * h[:1], 2 * (h[:-1] + h[1:]), 2 * h[-1:]))
        lower = jnp.concatenate((jnp.zeros(1), h))
        upper = jnp.concatenate((h, jnp.zeros(1)))
        rhs = 6 * jnp.concatenate(((delta[0] - slopes[0])[None],
                                  jnp.diff(delta, axis=0),
                                  (slopes[1] - delta[-1])[None]), axis=0)
        second = jax.lax.linalg.tridiagonal_solve(
            lower.astype(dtype), diagonal.astype(dtype), upper.astype(dtype), rhs)
    linear = delta - h[:, None] * (2 * second[:-1] + second[1:]) / 6
    quadratic = second[:-1] / 2
    cubic = jnp.diff(second, axis=0) / (6 * h[:, None])
    coeffs = jnp.stack((values[:-1], linear, quadratic, cubic), axis=1)
    return coeffs.reshape((n - 1, 4) + y.shape[1:])


@jax.jit
def spline_evaluate(x, coefficients, points):
    """Evaluate local-power splines, extrapolating the first/last cubic."""
    x = jnp.asarray(x, dtype=jnp.float64)
    points = jnp.asarray(points, dtype=jnp.float64)
    indices = jnp.clip(jnp.searchsorted(x, points, side="right") - 1, 0, x.size - 2)
    c = coefficients[indices]
    t = (points - x[indices]).reshape(points.shape + (1,) * (coefficients.ndim - 2))
    a, b, c2, d = (jnp.take(c, k, axis=points.ndim) for k in range(4))
    return a + t * (b + t * (c2 + t * d))
