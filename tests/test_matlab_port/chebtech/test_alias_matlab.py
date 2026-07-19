"""Port of MATLAB Chebfun tests/chebtech/test_alias.m (Opus 4.8).

MATLAB ``testclass.alias(coeffs, k)`` aliases a Chebyshev coefficient vector
down/up to length ``k``.  chebfunjax now implements ``Chebtech{1,2}.alias`` as
an exact port of ``@chebtech{1,2}/alias.m``, so the scalar assertions run;
the array-valued assertions are skipped (chebfunjax Chebtech is scalar-valued).

Provenance
----------
MATLAB source : tests/chebtech/test_alias.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts

CASES = [(Chebtech1, 1), (Chebtech2, 2)]

EPS = float(np.finfo(np.float64).eps)

_SCALAR = "chebfunjax Chebtech is scalar-valued; no array-valued techs"


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechAlias:
    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_alias_up_to_11(self, Tech, kind):
        # pass(n, 1): alias(sin coeffs, 11) reproduces sin on the k=11 grid.
        f = Tech.from_function(jnp.sin)
        c = Tech.alias(f.coeffs, 11)
        x = chebpts(11, kind)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert _ninf(values - jnp.sin(x)) < 10 * vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_alias_down_to_2(self, Tech, kind):
        # pass(n, 2): alias(sin coeffs, 2) -> length 2 with aliased values.
        f = Tech.from_function(jnp.sin)
        c = Tech.alias(f.coeffs, 2)
        x = chebpts(2, kind)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert c.shape[0] == 2
        assert _ninf(values - jnp.sin(x)) < 1e1 * vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin_up_to_11(self, Tech, kind):
        # pass(n, 3)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin_down_to_2(self, Tech, kind):
        # pass(n, 4)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_32(self, Tech, kind):
        # pass(n, 5)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_100(self, Tech, kind):
        # pass(n, 6)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_2(self, Tech, kind):
        # pass(n, 7)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_cos1000_to_1(self, Tech, kind):
        # pass(n, 8)
        pytest.skip(_SCALAR)

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_cos1000_to_2(self, Tech, kind):
        # pass(n, 9)
        pytest.skip(_SCALAR)
