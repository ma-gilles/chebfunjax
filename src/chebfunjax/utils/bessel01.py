"""JAX-only J0/J1 prerequisite for the RH Laguerre quadrature ASY path.

This is a bounded real, nonnegative argument helper, not a general Bessel
function port. It follows the standard DLMF 10.17 asymptotic expansion for
large positive arguments and uses a regular power series near zero. The
middle interval uses JAX's Miller-recurrence implementation.

Provenance
----------
The downstream motivation is the `asyBessel` branch in root-level
``lagpts.m`` (Chebfun source commit 7574c776). The formulas here follow
DLMF 10.17.E1, E2, and E3:
https://dlmf.nist.gov/10.17.E1
https://dlmf.nist.gov/10.17.E2
https://dlmf.nist.gov/10.17.E3
This helper does not claim MATLAB/SciPy builtin behavior or full arbitrary
order/complex-argument support.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import lax

_PI = 3.141592653589793238462643383279502884


def _series_j01(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sixteen-term regular power series, evaluated only for 0 <= x < 1."""
    z = -(x * x) / 4.0

    def add_term(k, carry):
        term0, sum0, term1, sum1 = carry
        kf = jnp.asarray(k, dtype=x.dtype)
        term0 = term0 * z / (kf * kf)
        term1 = term1 * z / (kf * (kf + 1.0))
        return term0, sum0 + term0, term1, sum1 + term1

    init = (jnp.ones_like(x), jnp.ones_like(x),
            jnp.ones_like(x), jnp.ones_like(x))
    # k=0 through k=15: sixteen terms total, including the initial 1.
    _, j0, _, series1 = lax.fori_loop(1, 16, add_term, init)
    return j0, (0.5 * x) * series1


def _miller_j01(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """JAX Miller recurrence, used on the interval [1, 32]."""
    values = jsp.bessel_jn(x, v=1, n_iter=80)
    return values[0], values[1]


def _asymptotic_jnu(x: jax.Array, nu: int) -> jax.Array:
    """DLMF 10.17 expansion with 20 even and 20 odd asymptotic terms."""
    mu = 4.0 * nu * nu
    omega = x - (nu * 0.5 + 0.25) * _PI

    # Generate a_j recursively. P contains a_0,...,a_38 and Q contains
    # a_1,...,a_39. The alternating signs are those in DLMF 10.17.E1.
    def add_asymptotic_term(j, carry):
        previous, p, q = carry
        jf = jnp.asarray(j, dtype=x.dtype)
        aj = previous * (mu - (2.0 * jf - 1.0) ** 2) / (8.0 * jf)

        def add_p(state):
            p0, q0 = state
            k = j // 2
            sign = jnp.where(k % 2 == 0, 1.0, -1.0).astype(x.dtype)
            return p0 + sign * aj / x**j, q0

        def add_q(state):
            p0, q0 = state
            k = (j - 1) // 2
            sign = jnp.where(k % 2 == 0, 1.0, -1.0).astype(x.dtype)
            return p0, q0 + sign * aj / x**j

        p, q = lax.cond(j % 2 == 0, add_p, add_q, (p, q))
        return aj, p, q

    a0 = jnp.ones_like(x)
    _, p, q = lax.fori_loop(
        1, 40, add_asymptotic_term,
        (a0, jnp.ones_like(x), jnp.zeros_like(x)))
    return jnp.sqrt(2.0 / (_PI * x)) * (jnp.cos(omega) * p
                                         - jnp.sin(omega) * q)


def _asymptotic_j01(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _asymptotic_jnu(x, 0), _asymptotic_jnu(x, 1)


def _bessel_j01_scalar(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Scalar core; surrogate inputs keep inactive branches regular."""
    valid = jnp.isfinite(x) & (x >= 0.0)
    safe_x = jnp.where(valid, x, 0.0)

    series_active = safe_x < 1.0
    middle_active = (safe_x >= 1.0) & (safe_x <= 32.0)
    far_active = safe_x > 32.0
    # Selection by where, rather than clip, preserves the identity tangent on
    # an active branch exactly at x=1 or x=32. Each inactive branch still sees
    # a regular in-range surrogate argument.
    series_x = jnp.where(series_active, safe_x, 0.0)
    middle_x = jnp.where(middle_active, safe_x, 1.0)
    far_x = jnp.where(far_active, safe_x, 32.0)

    j0s, j1s = _series_j01(series_x)
    j0m, j1m = _miller_j01(middle_x)
    j0a, j1a = _asymptotic_j01(far_x)

    j0 = jnp.where(safe_x < 1.0, j0s,
                   jnp.where(safe_x <= 32.0, j0m, j0a))
    j1 = jnp.where(safe_x < 1.0, j1s,
                   jnp.where(safe_x <= 32.0, j1m, j1a))
    nan = jnp.asarray(jnp.nan, dtype=x.dtype)
    return jnp.where(valid, j0, nan), jnp.where(valid, j1, nan)


def _bessel_j01(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return `(J0(x), J1(x))` for real nonnegative `x`.

    Supports scalar or array inputs and JIT/vmap/JVP. Negative and nonfinite
    arguments return NaN. The branch boundaries are 1 and 32. Accuracy targets
    are documented by the independent tests. This is a private prerequisite
    rather than a general-order Bessel API.

    Parameters
    ----------
    x : array-like
        Real, nonnegative argument(s).

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Float64 arrays matching the input shape.

    Provenance
    ----------
    DLMF 10.17.E1–E3; downstream source motivation is
    ``lagpts.m:800`` (``asyBessel``) at Chebfun commit 7574c776.
    """
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        raise TypeError("bessel_j01 accepts real arguments only")
    x = x.astype(jnp.float64)
    if x.ndim == 0:
        return _bessel_j01_scalar(x)
    flat = jnp.ravel(x)
    j0, j1 = jax.vmap(_bessel_j01_scalar)(flat)
    return j0.reshape(x.shape), j1.reshape(x.shape)
