"""JAX half-integer Bessel extension for RH complex continuation.

Finite arguments on the nonnegative real or positive imaginary axes are
supported, with explicit zero/invalid contracts. This supplies the imaginary
Bessel argument of a negative real Laguerre RH Newton trial; arbitrary
complex-plane input is unported. Real helper behavior is unchanged.

Provenance
----------
MATLAB source: lagpts.m (asyBessel), besselroots.m; Chebfun commit:
7574c77680d7e82b79626300bf255498271a72df. Mathematical identities: DLMF
10.2.E2, 10.16.E1 and 10.6.E1. This uses the qualified half-order series
and closed forms; it does not implement source Piessens root approximations.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.utils.bessel_half import _HALF_ORDERS, _half_closed_form, _half_series


def _bessel_j_half_complex(order: float, x: jax.Array) -> jax.Array:
    """Return J_order(x) on the positive real/imaginary axes.

    This extension accepts finite x on either axis with nonnegative
    coordinate, including zero. Other complex-plane inputs and nonfinite
    inputs return complex NaN. At zero the negative orders return signed
    infinities and the positive orders return zero. The output dtype is always
    complex128, including for positive-real inputs.

    The small branch uses the DLMF defining series and is used only for
    ``abs(x) < 1``. The other branch uses the closed forms in DLMF 10.16.1.
    Static order must be one of -3/2, -1/2, 1/2, 3/2. No arbitrary complex
    continuation is claimed: the positive imaginary axis is the downstream
    RH negative Newton-trial case.

    Provenance: DLMF 10.2.2 and 10.16.1, together with the ``asyBessel``
    branch in Chebfun ``lagpts.m`` at commit 7574c77680d7e82b79626300bf255498271a72df.
    """
    if order not in _HALF_ORDERS:
        raise ValueError(f"unsupported half-integer Bessel order {order!r}")
    x = jnp.asarray(x, dtype=jnp.complex128)
    real_part = jnp.real(x)
    imag_part = jnp.imag(x)
    on_real_axis = (imag_part == 0.0) & (real_part >= 0.0)
    on_imag_axis = (real_part == 0.0) & (imag_part >= 0.0)
    valid = jnp.isfinite(real_part) & jnp.isfinite(imag_part) & (on_real_axis | on_imag_axis)
    safe = jnp.where(valid, x, jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128))
    is_zero = safe == 0.0
    positive = jnp.where(is_zero, jnp.asarray(1.0 + 0.0j), safe)
    small_active = jnp.abs(positive) < 1.0

    # Keep both inactive branches away from their zero/singular endpoints.
    # The where masks depend only on primal values, so active analytic branch
    # derivatives remain the ordinary complex derivatives.
    series_x = jnp.where(small_active, positive, jnp.asarray(0.5 + 0.0j))
    closed_x = jnp.where(small_active, jnp.asarray(1.0 + 0.0j), positive)
    series_value = _half_series(series_x, order)
    closed_value = _half_closed_form(closed_x, order)
    value = jnp.where(small_active, series_value, closed_value)

    if order == -1.5:
        zero_value = jnp.asarray(-jnp.inf + 0.0j, dtype=jnp.complex128)
    elif order == -0.5:
        zero_value = jnp.asarray(jnp.inf + 0.0j, dtype=jnp.complex128)
    else:
        zero_value = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)
    value = jnp.where(x == 0.0, zero_value, value)
    return jnp.where(valid, value, jnp.asarray(jnp.nan + 1j * jnp.nan))

def _bessel_j_half_complex_array(order: float, x: jax.Array) -> jax.Array:
    """Shape-preserving elementwise wrapper for ``_bessel_j_half_complex``.

    Provenance and domain are the same as ``_bessel_j_half_complex``; ``order``
    is a static Python value so callers can JIT the closure over it.
    """
    if order not in _HALF_ORDERS:
        raise ValueError(f"unsupported half-integer Bessel order {order!r}")
    x = jnp.asarray(x, dtype=jnp.complex128)
    if x.ndim == 0:
        return _bessel_j_half_complex(order, x)
    flat = jnp.ravel(x)
    values = jax.vmap(lambda value: _bessel_j_half_complex(order, value))(flat)
    return values.reshape(x.shape)
