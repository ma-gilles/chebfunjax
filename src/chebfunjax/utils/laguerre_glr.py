"""Pure-JAX Laguerre GLR nodes and weights from Chebfun's lagpts.

This module translates the source helpers ``alg0_Lag``, ``alg1_Lag``,
``alg3_Lag``, ``eval_Lag``, and ``rk2_Lag`` without NumPy/SciPy numerical
bodies. It supports the source GLR domain alpha=0; method selection,
normalization, intervals, and barycentric weights stay in ``quadrature.py``.

Provenance
----------
MATLAB source : ``lagpts.m`` (GLR routines, source lines 263--405)
Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df``
Original algorithm: Glaser, Liu, and Rokhlin (2007).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def _eval_lag(x, n):
    """Evaluate the source scaled Laguerre polynomial and derivative."""
    initial = (jnp.asarray(0.0, dtype=jnp.float64),
               jnp.exp(-x / 2.0),
               jnp.asarray(0.0, dtype=jnp.float64),
               jnp.asarray(0.0, dtype=jnp.float64))

    def recur(k, state):
        lm2, lm1, lpm2, lpm1 = state
        kf = jnp.asarray(k, dtype=jnp.float64)
        den = kf + 1.0
        lval = ((2.0 * kf + 1.0 - x) * lm1 - kf * lm2) / den
        lpval = ((2.0 * kf + 1.0 - x) * lpm1 - lm1 - kf * lpm2) / den
        return lm1, lval, lpm1, lpval

    lm2, lm1, lpm2, lpm1 = lax.fori_loop(0, n, recur, initial)
    del lm2, lpm2
    return lm1, lpm1


def _rk2_lag(t, tn, x, n):
    """Ten-step second-order Runge--Kutta phase integration from MATLAB."""
    step = (tn - t) / 10.0

    def advance(_, state):
        phase, value = state
        f1 = n + 0.5 - 0.25 * value
        k1 = -step / (jnp.sqrt(f1 / value)
                      + 0.25 * (1.0 / value - 0.25 / f1)
                      * jnp.sin(2.0 * phase))
        phase_next = phase + step
        value_next = value + k1
        f2 = n + 0.5 - 0.25 * value_next
        k2 = -step / (jnp.sqrt(f2 / value_next)
                      + 0.25 * (1.0 / value_next - 0.25 / f2)
                      * jnp.sin(2.0 * phase_next))
        return phase_next, value_next + 0.5 * (k2 - k1)

    return lax.fori_loop(0, 10, advance, (t, x))[1]


def _alg3_lag(n, xs):
    """Source GLR initial-root Newton solve and derivative."""
    u, up = _eval_lag(xs, n)
    theta = jnp.arctan(jnp.sqrt(xs / (n + 0.5 - 0.25 * xs)) * up / u)
    x1 = _rk2_lag(theta, -jnp.pi / 2.0, xs, n)
    eps = jnp.finfo(jnp.float64).eps
    init = (x1, u, up, jnp.asarray(jnp.inf), jnp.asarray(0, jnp.int32))

    def condition(state):
        _x, value, _derivative, step, count = state
        return ((jnp.abs(step) > eps) | (jnp.abs(value) > eps)) & (count < 200)

    def iterate(state):
        x, _value, _derivative, _step, count = state
        value, derivative = _eval_lag(x, n)
        step = value / derivative
        return x - step, value, derivative, step, count + 1

    x1, _, _, _, _ = lax.while_loop(condition, iterate, init)
    _, d1 = _eval_lag(x1, n)
    return x1, d1


def _alg1_lag(roots, ders, n, n1):
    """Propagate consecutive roots by source Taylor/Newton steps."""
    m = 30
    hh_ones = jnp.ones((m + 1,), dtype=jnp.float64)
    zz = jnp.zeros((m,), dtype=jnp.float64)
    x0 = roots[n1 - 1]
    eps = jnp.finfo(jnp.float64).eps

    def advance(j, state):
        root_values, derivative_values, x = state
        h = _rk2_lag(jnp.pi / 2.0, -jnp.pi / 2.0, x, n) - x
        scale = 1.0 / h
        scale2, scale3, scale4 = scale**2, scale**3, scale**4
        r = x * (n + 0.5 - 0.25 * x)
        p = x**2
        u = jnp.zeros((m + 1,), dtype=jnp.float64)
        up = jnp.zeros((m + 1,), dtype=jnp.float64)
        u = u.at[0].set(0.0).at[1].set(derivative_values[j] / scale)
        u3 = (-0.5 * u[1] / (scale * x)
              - (n + 0.5 - 0.25 * x) * u[0] / (x * scale2))
        u = u.at[2].set(u3)
        u4 = (-u3 / (scale * x)
              + (-(1.0 + r) * u[1] / 6.0 / scale2
                 - (n + 0.5 - 0.5 * x) * u[0] / scale3) / p)
        u = u.at[3].set(u4)
        up = up.at[0].set(u[1]).at[1].set(2.0 * u[2] * scale)
        up = up.at[2].set(3.0 * u[3] * scale)

        def taylor(k, pair):
            coeffs, derivs = pair
            next_u = (
                -x * (2.0 * k + 1.0) * (k + 1.0) * coeffs[k + 1] / scale
                - (k**2 + r) * coeffs[k] / scale2
                - (n + 0.5 - 0.5 * x) * coeffs[k - 1] / scale3
                + 0.25 * coeffs[k - 2] / scale4
            ) / (p * (k + 2.0) * (k + 1.0))
            coeffs = coeffs.at[k + 2].set(next_u)
            derivs = derivs.at[k + 1].set((k + 2.0) * next_u * scale)
            return coeffs, derivs

        u, up = lax.fori_loop(2, m - 1, taylor, (u, up))
        up = up.at[-1].set(0.0)
        u, up = u[::-1], up[::-1]
        hh0 = hh_ones.at[-1].set(scale)

        def unit_scale(_):
            products = jnp.cumprod(scale * h + zz)
            return jnp.concatenate((jnp.reshape(scale, (1,)), products))[::-1]

        hh0 = lax.cond(scale == 1.0, unit_scale, lambda _: hh0, operand=None)
        init = (h, hh0, jnp.asarray(jnp.inf), jnp.asarray(0, jnp.int32))

        def condition(loop_state):
            _h, _hh, step, count = loop_state
            return (jnp.abs(step) > eps) & (count < 10)

        def newton(loop_state):
            step_h, _hh, _step, count = loop_state
            # The source inner products are row-vector products.
            step = jnp.vdot(u, _hh) / jnp.vdot(up, _hh)
            step_h = step_h - step
            hh = jnp.concatenate((jnp.reshape(scale, (1,)),
                                  jnp.cumprod(scale * step_h + zz)))[::-1]
            return step_h, hh, step, count + 1

        h, hh, _, _ = lax.while_loop(condition, newton, init)
        x_next = x + h
        root_values = root_values.at[j + 1].set(x_next)
        derivative_values = derivative_values.at[j + 1].set(jnp.vdot(up, hh))
        return root_values, derivative_values, x_next

    roots, ders, _ = lax.fori_loop(n1 - 1, n - 1, advance, (roots, ders, x0))
    return roots, ders


@partial(jax.jit, static_argnames=('n',))
def _laguerre_glr(n: int):
    """Return GLR nodes and unnormalized alpha=0 weights for static ``n``.

    Public ``lagpts`` performs source weight normalization and interval mapping.
    The source GLR method is defined only for alpha=0.
    """
    if n < 1:
        raise ValueError("laguerre GLR requires n >= 1")
    n1 = min(20, n)
    roots = jnp.zeros((n,), dtype=jnp.float64)
    ders = jnp.zeros((n,), dtype=jnp.float64)
    xs0 = jnp.asarray(1.0 / (2.0 * n + 1.0), dtype=jnp.float64)

    def seed(k, state):
        root_values, derivative_values, xs = state
        root, derivative = _alg3_lag(n, xs)
        root_values = root_values.at[k].set(root)
        derivative_values = derivative_values.at[k].set(derivative)
        return root_values, derivative_values, 1.1 * root

    roots, ders, _ = lax.fori_loop(0, n1, seed, (roots, ders, xs0))
    roots, ders = _alg1_lag(roots, ders, n, n1)
    weights = jnp.exp(-roots) / (roots * ders**2)
    return roots, weights
