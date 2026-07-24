"""Port of MATLAB Chebfun tests/chebtech/test_poly.m (Opus 4.8).

MATLAB ``poly(f)`` returns the *monomial* (power-basis) coefficients of a
chebtech, highest degree first.  chebfunjax now implements ``poly`` on both
tech classes (Chebyshev-T -> monomial via the ``T_k`` power-basis recurrence),
so every assertion is exercised at the SAME tolerance MATLAB uses.

Provenance
----------
MATLAB source : tests/chebtech/test_poly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechPoly:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_zeros(self, Tech, kind):
        # pass(n, 1): poly(0) == 0.
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        p = f.poly()
        assert _ninf(p) <= 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_constant(self, Tech, kind):
        # pass(n, 2): poly(3) == 3.
        f = Tech.from_function(lambda x: 3 * jnp.ones_like(x))
        p = f.poly()
        assert _ninf(p - 3.0) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_linear_complex(self, Tech, kind):
        # pass(n, 3): poly(6.4*x - 3i) == [6.4  -3i].
        f = Tech.from_function(lambda x: 6.4 * x - 3j)
        p = f.poly()
        assert _ninf(p - jnp.array([6.4, -3j])) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_quintic_complex(self, Tech, kind):
        # pass(n, 4): poly(2i x^5 - 3.2 x^4 + 2 x^2 - (1.2+3i)).
        f = Tech.from_function(
            lambda x: 2j * x ** 5 - 3.2 * x ** 4 + 2 * x ** 2 - (1.2 + 3j)
        )
        p = f.poly()
        p_exact = jnp.array([2j, -3.2, 0, 2, 0, -(1.2 + 3j)])
        assert _ninf(p - p_exact) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_valued(self, Tech, kind):
        # pass(n, 5): array-valued poly; rows correspond to columns of f.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [
                    3 * jnp.ones_like(x),
                    (6.4 * x - 3j),
                    (4 * x ** 2 - 2j * x + 3.7),
                ],
                axis=-1,
            )
        )
        p = f.poly()
        # Highest degree first, one row per column of f (length n = f.n).
        p_exact = np.array(
            [
                [0, 0, 3],
                [0, 6.4, -3j],
                [4, -2j, 3.7],
            ]
        )
        assert p.shape == p_exact.shape
        assert _ninf(np.asarray(p) - p_exact) < 10 * f.vscale * EPS
