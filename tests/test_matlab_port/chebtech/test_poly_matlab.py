"""Port of MATLAB Chebfun tests/chebtech/test_poly.m (Opus 4.8).

MATLAB ``poly(f)`` returns the *monomial* (power-basis) coefficients of a
chebtech.  chebfunjax has NO ``poly`` method (no Chebyshev->monomial power
coefficient conversion), so every assertion is marked xfail with a precise
reason.  We do not hand-roll a monomial converter.

Provenance
----------
MATLAB source : tests/chebtech/test_poly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

_NO_POLY = "chebfunjax lacks poly() (Chebyshev->monomial power coefficients)"


class TestChebtechPoly:
    @pytest.mark.xfail(reason=_NO_POLY, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_zeros(self, Tech, kind):
        # pass(n, 1): poly(0) == 0.
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        f.poly()

    @pytest.mark.xfail(reason=_NO_POLY, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_constant(self, Tech, kind):
        # pass(n, 2): poly(3) == 3.
        f = Tech.from_function(lambda x: 3 * jnp.ones_like(x))
        f.poly()

    @pytest.mark.xfail(reason=_NO_POLY, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_linear_complex(self, Tech, kind):
        # pass(n, 3): poly(6.4*x - 3i) == [6.4  -3i].
        f = Tech.from_function(lambda x: 6.4 * x - 3j)
        f.poly()

    @pytest.mark.xfail(reason=_NO_POLY, strict=False)
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_quintic_complex(self, Tech, kind):
        # pass(n, 4): poly(2i x^5 - 3.2 x^4 + 2 x^2 - (1.2+3i)).
        f = Tech.from_function(
            lambda x: 2j * x ** 5 - 3.2 * x ** 4 + 2 * x ** 2 - (1.2 + 3j)
        )
        f.poly()

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_valued(self, Tech, kind):
        # pass(n, 5): array-valued poly.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; also lacks poly() "
            "(Chebyshev->monomial power coefficients)"
        )
