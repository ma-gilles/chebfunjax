"""Port of MATLAB Chebfun tests/trigtech/test_mldivide.m (Opus 4.8[1m]).

mldivide (A\\B) solves the continuous-L^2 least-squares problem A X = B.

Provenance
----------
MATLAB source : tests/trigtech/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechMldivide:
    def test_scalar_exact(self):
        # f \ f == 1 (known exact solution).
        f = _tt(lambda x: jnp.cos(jnp.sin(jnp.pi * x)))
        X = Trigtech.mldivide(f, f)
        assert abs(float(jnp.asarray(X).ravel()[0]) - 1) < 10 * f.vscale * EPS

    def test_array_exact_coeff(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.sin(jnp.pi * x + jnp.pi / 4))
        X = Trigtech.mldivide(f, g)
        exact = jnp.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        assert _ninf(jnp.asarray(X).ravel() - exact) < 10 * f.vscale * EPS

    def test_array_exact_residual(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.sin(jnp.pi * x + jnp.pi / 4))
        X = Trigtech.mldivide(f, g)
        err = g - (f @ X)
        assert _ninf(err.values) < 10 * f.vscale * EPS

    def test_least_squares(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.ones_like(x), jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x)],
            axis=-1))
        g = _tt(lambda x: jnp.cos(jnp.pi * x))
        X = Trigtech.mldivide(f, g)
        exact = jnp.array([0.0, 1.0, 0.0])
        assert _ninf(jnp.asarray(X).ravel() - exact) < 10 * f.vscale * EPS

    def test_error_nontrigtech(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x),
             jnp.exp(1j * jnp.pi * x)], axis=-1))
        with pytest.raises(ValueError) as exc:
            Trigtech.mldivide(f, 2)
        assert "trigtechMldivideUnknown" in str(exc.value)

    def test_error_identifier(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        with pytest.raises(ValueError) as exc:
            Trigtech.mldivide(f, 2)
        assert "CHEBFUN:TRIGTECH:mldivide:trigtechMldivideUnknown" \
            in str(exc.value)
