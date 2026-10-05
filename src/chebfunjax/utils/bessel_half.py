"""JAX half-integer Bessel prerequisites for generalized Laguerre RH.

This bounded helper covers orders -3/2, -1/2, 1/2 and 3/2 on real
nonnegative arguments. For x<1 it evaluates the defining power series to
avoid the cancellation in J_{3/2}; for x>=1 it uses elementary sine/cosine
forms. The half-order positive roots have exact arithmetic progressions.

Provenance
----------
The downstream algorithm is ``asyBessel`` in ``lagpts.m`` and calls
``besselroots(alpha, itric)`` (Chebfun source commit
7574c77680d7e82b79626300bf255498271a72df). The local source files are pinned
``lagpts.m`` and ``besselroots.m``. Half-order identities follow DLMF 10.16.1;
the stable near-zero series follows DLMF 10.2.2; adjacent-order recurrence
follows DLMF 10.6.1:
https://dlmf.nist.gov/10.16.E1
https://dlmf.nist.gov/10.2.E2
https://dlmf.nist.gov/10.6.E1

``_bessel_roots_half`` intentionally returns exact mathematical roots. The
pinned MATLAB ``besselroots.m`` uses Piessens polynomial approximations for
the first six roots when -1 <= order <= 5, so this helper is not bitwise or
rounded-value parity with that approximation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

_HALF_ORDERS = (-1.5, -0.5, 0.5, 1.5)
_SQRT_2_OVER_PI = 0.797884560802865355879892119868763737
_PI = 3.141592653589793238462643383279502884


def _half_series(x: jax.Array, order: float) -> jax.Array:
    """Evaluate 25 terms of the DLMF defining series at 0 < x < 1."""
    prefactors = {
        -1.5: -_SQRT_2_OVER_PI,
        -0.5: _SQRT_2_OVER_PI,
        0.5: _SQRT_2_OVER_PI,
        1.5: _SQRT_2_OVER_PI / 3.0,
    }
    term0 = jnp.ones_like(x)

    def series_step(k, carry):
        term, total = carry
        kf = jnp.asarray(k, dtype=x.dtype)
        term = term * (-(x * x) / (4.0 * kf * (kf + order)))
        return term, total + term

    # k=0..24 gives 25 terms; the series is used only below x=1.
    _, total = lax.fori_loop(1, 25, series_step,
                             (term0, jnp.ones_like(x)))
    power = x**order
    return prefactors[order] * power * total


def _half_closed_form(x: jax.Array, order: float) -> jax.Array:
    """Elementary forms valid for positive real x."""
    scale = jnp.sqrt(2.0 / (_PI * x))
    if order == -1.5:
        return -scale * (jnp.sin(x) + jnp.cos(x) / x)
    if order == -0.5:
        return scale * jnp.cos(x)
    if order == 0.5:
        return scale * jnp.sin(x)
    return scale * (jnp.sin(x) / x - jnp.cos(x))


def _bessel_j_half_scalar(order: float, x: jax.Array) -> jax.Array:
    """Scalar implementation with safe inactive branch inputs."""
    valid = jnp.isfinite(x) & (x >= 0.0)
    safe = jnp.where(valid, x, 1.0)
    zero = safe == 0.0
    positive = jnp.where(zero, 1.0, safe)
    small_active = positive < 1.0

    # The series sees x itself on its active branch. The closed-form branch
    # receives the safe value 1 below its boundary to avoid singular inactive
    # calculations; where preserves the active branch's tangent at x=1.
    series_x = jnp.where(small_active, positive, 0.5)
    closed_x = jnp.where(small_active, 1.0, positive)
    series_value = _half_series(series_x, order)
    closed_value = _half_closed_form(closed_x, order)
    value = jnp.where(small_active, series_value, closed_value)

    zero_limit = jnp.asarray(0.0, dtype=x.dtype)
    if order == -1.5:
        zero_limit = jnp.asarray(-jnp.inf, dtype=x.dtype)
    elif order == -0.5:
        zero_limit = jnp.asarray(jnp.inf, dtype=x.dtype)
    value = jnp.where(x == 0.0, zero_limit, value)
    return jnp.where(valid, value, jnp.asarray(jnp.nan, dtype=x.dtype))


def _bessel_j_half(order: float, x: jax.Array) -> jax.Array:
    """Return J_order(x), for static order in {-3/2,-1/2,1/2,3/2}.

    Inputs are real; negative and nonfinite values produce NaN. At zero the
    two negative orders return their mathematical signed infinities, while the
    positive orders return zero. JVPs are intended for x>0; derivatives at
    zero are singular for some orders.

    Provenance: DLMF 10.16.1, 10.2.2 and 10.6.1; source motivation is
    ``lagpts.m``/``besselroots.m`` at Chebfun commit 7574c776.
    """
    if order not in _HALF_ORDERS:
        raise ValueError(f"unsupported half-integer Bessel order {order!r}")
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        raise TypeError("half-integer Bessel helper accepts real arguments only")
    x = x.astype(jnp.float64)
    if x.ndim == 0:
        return _bessel_j_half_scalar(order, x)
    flat = jnp.ravel(x)
    values = jax.vmap(lambda value: _bessel_j_half_scalar(order, value))(flat)
    return values.reshape(x.shape)


def _bessel_jhalf_orders(x: jax.Array) -> tuple[jax.Array, ...]:
    """Return J_{-3/2}, J_{-1/2}, J_{1/2}, J_{3/2} in that order."""
    return tuple(_bessel_j_half(order, x) for order in _HALF_ORDERS)


def _bessel_roots_half(order: float, n: int) -> jax.Array:
    """Return the first n positive exact roots of J_{±1/2}.

    J_{-1/2} roots are (k-1/2)pi and J_{1/2} roots are k*pi, k=1..n.
    This is an exact mathematical root adapter, not the rounded Piessens
    approximation selected by the pinned MATLAB ``besselroots.m``.

    Provenance: DLMF 10.16.1 and Chebfun ``besselroots.m`` (commit 7574c776).
    """
    if order not in (-0.5, 0.5):
        raise ValueError("roots are supported only for orders +/-1/2")
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError("n must be a nonnegative integer")
    n = int(n)
    if order == -0.5:
        index = jnp.arange(n, dtype=jnp.float64) + 0.5
    else:
        index = jnp.arange(1, n + 1, dtype=jnp.float64)
    return index * _PI

