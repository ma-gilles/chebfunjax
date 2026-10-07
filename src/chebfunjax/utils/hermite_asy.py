"""Asymptotic Newton Gauss--Hermite nodes from Chebfun's ``hermpts``.

This module implements the MATLAB ASY path for ``n >= 21``. Method selection
and probabilists' rescaling belong to the public quadrature wrapper.

Provenance
----------
MATLAB source : hermpts.m (``HermiteInitialGuesses``, ``hermpts_asy0``,
    ``hermpoly_asy_airy``)
Chebfun commit: 7574c77
Original author: Nick Hale.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from chebfunjax.utils._gradual import (
    gradual_exp_negative,
    gradual_positive_divide,
    gradual_positive_multiply,
)
from chebfunjax.utils._signed_gradual import (
    flip_sign_bits,
    source_barycentric_half,
    source_barycentric_normalize,
)
from chebfunjax.utils.airy import _airy_negative
from chebfunjax.utils.hermite_rec import _initial_guesses


def _hermite_asy_value_derivative(n: int, theta: jax.Array):
    """Evaluate the Airy asymptotic Hermite polynomial and derivative.

    Literal JAX translation of MATLAB ``hermpoly_asy_airy``. ``theta`` is
    real and lies in the oscillatory interval ``(0, pi)``.

    Provenance
    ----------
    MATLAB source : hermpts.m (``hermpoly_asy_airy``)
    Chebfun commit: 7574c77
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    musq = jnp.asarray(2 * n + 1, dtype=jnp.float64)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    sin_2t = 2.0 * cos_t * sin_t
    eta = 0.5 * theta - 0.25 * sin_2t
    chi = -(3.0 * eta / 2.0) ** (2.0 / 3.0)
    phi = (-chi / sin_t**2) ** 0.25
    ai, aip = _airy_negative(musq ** (2.0 / 3.0) * chi)
    ai = jnp.real(ai)
    aip = jnp.real(aip)

    a0 = 1.0
    b0 = 1.0
    a1 = 15.0 / 144.0
    b1 = -7.0 / 5.0 * a1
    a2 = 5.0 * 7.0 * 9.0 * 11.0 / 2.0 / 144.0**2
    b2 = -13.0 / 11.0 * a2
    a3 = 7.0 * 9.0 * 11.0 * 13.0 * 15.0 * 17.0 / 6.0 / 144.0**3
    b3 = -19.0 / 17.0 * a3

    u0 = 1.0
    u1 = (cos_t**3 - 6.0 * cos_t) / 24.0
    u2 = (-9.0 * cos_t**4 + 249.0 * cos_t**2 + 145.0) / 1152.0
    u3 = (
        -4042.0 * cos_t**9 + 18189.0 * cos_t**7
        - 28287.0 * cos_t**5 - 151995.0 * cos_t**3
        - 259290.0 * cos_t
    ) / 414720.0

    val = ai
    b0_term = -(a0 * phi**6 * u1 + a1 * u0) / chi**2
    val = val + b0_term * aip / musq ** (4.0 / 3.0)
    a1_term = (
        b0 * phi**12 * u2 + b1 * phi**6 * u1 + b2 * u0
    ) / chi**3
    val = val + a1_term * ai / musq**2
    b1_term = -(
        phi**18 * u3 + a1 * phi**12 * u2 + a2 * phi**6 * u1 + a3 * u0
    ) / chi**5
    val = val + b1_term * aip / musq ** (4.0 / 3.0 + 2.0)
    val = 2.0 * jnp.sqrt(jnp.pi) * musq ** (1.0 / 6.0) * phi * val

    v0 = 1.0
    v1 = (cos_t**3 + 6.0 * cos_t) / 24.0
    v2 = (15.0 * cos_t**4 - 327.0 * cos_t**2 - 143.0) / 1152.0
    v3 = (
        259290.0 * cos_t + 238425.0 * cos_t**3
        - 36387.0 * cos_t**5 + 18189.0 * cos_t**7
        - 4042.0 * cos_t**9
    ) / 414720.0

    c0 = -(b0 * phi**6 * v1 + b1 * v0) / chi
    dval = c0 * ai / musq ** (2.0 / 3.0)
    dval = dval + a0 * v0 * aip
    c1 = -(
        phi**18 * v3 + b1 * phi**12 * v2 + b2 * phi**6 * v1 + b3 * v0
    ) / chi**4
    dval = dval + c1 * ai / musq ** (2.0 / 3.0 + 2.0)
    d1 = (
        a0 * phi**12 * v2 + a1 * phi**6 * v1 + a2 * v0
    ) / chi**3
    dval = dval + d1 * aip / musq**2
    dval = jnp.sqrt(2.0 * jnp.pi) * musq ** (1.0 / 3.0) * dval / phi
    return val, dval


@partial(jax.jit, static_argnames=("n",))
def _hermpts_asy(n: int):
    """Return ASY physicists' Hermite nodes, weights, and barycentric weights.

    The positive-half roots are initialized by the same Airy/Tricomi blend
    as the REC path, then Newton-refined in theta for at most 20 iterations.
    Folding and normalization reproduce ``hermpts_asy``.

    Parameters
    ----------
    n : int
        Number of nodes; this ASY implementation requires ``n >= 21``.

    Returns
    -------
    x, w, v : tuple of jax.Array
        Nodes, Gauss--Hermite weights normalized to ``sqrt(pi)``, and
        max-normalized barycentric weights.

    Provenance
    ----------
    MATLAB source : hermpts.m (``hermpts_asy0``, ``hermpts_asy``)
    Chebfun commit: 7574c77
    """
    if n < 21:
        raise ValueError("hermpts_asy: ASY is supported for n >= 21")

    x0 = _initial_guesses(n)
    musq = jnp.asarray(2 * n + 1, dtype=jnp.float64)
    theta0 = jnp.arccos(x0 / jnp.sqrt(musq))
    threshold = jnp.sqrt(jnp.finfo(jnp.float64).eps) / 10.0
    initial = (
        jnp.asarray(0, dtype=jnp.int32),
        theta0,
        jnp.asarray(False),
        jnp.zeros_like(theta0),
        jnp.zeros_like(theta0),
    )

    def condition(state):
        count, _theta, converged, _val, _dval = state
        return (count < 20) & ~converged

    def newton_step(state):
        count, theta, _converged, _val, _dval = state
        val, dval = _hermite_asy_value_derivative(n, theta)
        dt = -val / (
            jnp.sqrt(2.0) * jnp.sqrt(musq) * dval * jnp.sin(theta)
        )
        theta = theta - dt
        converged = jnp.max(jnp.abs(dt)) < threshold
        return count + 1, theta, converged, val, dval

    _, theta, _, val, dval = lax.while_loop(condition, newton_step, initial)
    t0 = jnp.cos(theta)
    x_half = jnp.sqrt(musq) * t0
    ders = x_half * val + jnp.sqrt(2.0) * dval
    w_half = gradual_positive_divide(gradual_exp_negative(-(x_half**2)), ders**2)
    v_half = source_barycentric_half(x_half, ders)

    if n % 2:
        x = jnp.concatenate((-x_half[::-1], x_half[1:]))
        w = jnp.concatenate((w_half[::-1], w_half[1:]))
        v = jnp.concatenate((v_half[::-1], v_half[1:]))
    else:
        x = jnp.concatenate((-x_half[::-1], x_half))
        w = jnp.concatenate((w_half[::-1], w_half))
        v = jnp.concatenate((v_half[::-1], flip_sign_bits(v_half)))

    w = gradual_positive_multiply(w, jnp.sqrt(jnp.pi) / jnp.sum(w))
    v = source_barycentric_normalize(v)
    return x, w, v
