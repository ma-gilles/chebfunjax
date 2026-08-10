"""Port of MATLAB Chebfun tests/chebtech/test_mldivide.m (Fable 5).

Array-valued techs and a tech-level ``qr``/``mldivide`` both exist now, so
every MATLAB assertion is ported directly at MATLAB's tolerances.  The MATLAB
file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we parametrize
over ``[Chebtech1, Chebtech2]``.

No gaps: all six MATLAB passes are exercised.

Provenance
----------
MATLAB source : tests/chebtech/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2, _collapse_single_column

EPS = float(np.finfo(np.float64).eps)

BOTH = [Chebtech1, Chebtech2]


def _max_abs(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechMldivide:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_self_divide_is_one(self, Tech):
        # pass(n, 1:2): f \ f == 1 and the residual vanishes.
        f = Tech.from_function(jnp.sin)
        x = f.mldivide(f)
        tol = 10 * f.vscale * EPS

        assert abs(float(jnp.reshape(jnp.asarray(x), ())) - 1.0) < tol

        err = f - f * jnp.reshape(jnp.asarray(x), ())
        assert _max_abs(err.coeffs) < tol

    @pytest.mark.parametrize("Tech", BOTH)
    def test_known_exact_solution(self, Tech):
        # pass(n, 3:4): [sin cos] \ sin(x + pi/4) == [1/sqrt(2); 1/sqrt(2)].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        g = Tech.from_function(lambda x: jnp.sin(x + math.pi / 4))
        tol = 10 * f.vscale * EPS

        x = f.mldivide(g)
        expected = jnp.full((2,), 1.0 / math.sqrt(2.0))
        assert _max_abs(jnp.reshape(x, (2,)) - expected) < tol

        err = g - _collapse_single_column(f @ jnp.reshape(x, (2, 1)))
        assert _max_abs(err.coeffs) < tol

    @pytest.mark.parametrize("Tech", BOTH)
    def test_known_least_squares_solution(self, Tech):
        # pass(n, 5): [1 x x^2 x^3] \ (x^4 + x^3 + x + 1).
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.ones_like(x), x, x**2, x**3], axis=-1))
        g = Tech.from_function(lambda x: x**4 + x**3 + x + 1)
        tol = 10 * f.vscale * EPS

        x = f.mldivide(g)
        expected = jnp.array([32.0 / 35.0, 1.0, 6.0 / 7.0, 1.0])
        assert _max_abs(jnp.reshape(x, (4,)) - expected) < 1e1 * tol

    @pytest.mark.parametrize("Tech", BOTH)
    def test_error_non_tech_operand(self, Tech):
        # pass(n, 6): f \ 2 raises chebtechMldivideUnknown.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        with pytest.raises(ValueError, match="chebtechMldivideUnknown"):
            f.mldivide(2)
