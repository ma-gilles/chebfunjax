"""Port of MATLAB Chebfun tests/chebtech/test_alias.m (Opus 4.8).

MATLAB ``testclass.alias(coeffs, k)`` aliases a Chebyshev coefficient vector
down/up to length ``k``.  chebfunjax now implements ``Chebtech{1,2}.alias`` as
an exact port of ``@chebtech{1,2}/alias.m``, and array-valued techs landed in 2026-07, so
every MATLAB assertion (scalar and array-valued) is ported at MATLAB's
tolerances.

No gaps: all nine MATLAB passes are exercised.

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

def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _pm(F):
    """MATLAB ``@(x) [F(x), -F(x)]`` as a two-column operator."""
    return lambda x: jnp.stack([F(x), -F(x)], axis=-1)


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
        # pass(n, 3): f = [sin(x), -sin(x)]; alias to k = 11.
        f = Tech.from_function(_pm(jnp.sin))
        c = Tech.alias(f.coeffs, 11)
        x = chebpts(11, kind)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert _ninf(values - _pm(jnp.sin)(x)) < 10 * vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin_down_to_2(self, Tech, kind):
        # pass(n, 4): alias to 2 -> length 2 with aliased values.
        f = Tech.from_function(_pm(jnp.sin))
        c = Tech.alias(f.coeffs, 2)
        y = chebpts(2, kind)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert c.shape[0] == 2
        assert _ninf(values - _pm(jnp.sin)(y)) < 10 * vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_32(self, Tech, kind):
        # pass(n, 5): F = sin(1000*x), alias to k = 32.
        F = lambda x: jnp.sin(1000 * x)  # noqa: E731
        f = Tech.from_function(_pm(F))
        c = Tech.alias(f.coeffs, 32)
        x = chebpts(32, kind)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert _ninf(values - _pm(F)(x)) < 1e3 * vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_100(self, Tech, kind):
        # pass(n, 6): same, alias to k = 100.
        F = lambda x: jnp.sin(1000 * x)  # noqa: E731
        f = Tech.from_function(_pm(F))
        c = Tech.alias(f.coeffs, 100)
        x = chebpts(100, kind)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert _ninf(values - _pm(F)(x)) < 1e4 * vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_sin1000_to_2(self, Tech, kind):
        # pass(n, 7): alias to 2; MATLAB's bound here is a bare 1e3*eps.
        F = lambda x: jnp.sin(1000 * x)  # noqa: E731
        f = Tech.from_function(_pm(F))
        c = Tech.alias(f.coeffs, 2)
        y = chebpts(2, kind)
        values = Tech.coeffs2vals(c)
        assert c.shape[0] == 2
        assert _ninf(values - _pm(F)(y)) < 1e3 * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_cos1000_to_1(self, Tech, kind):
        # pass(n, 8): F = cos(1000*x) aliased to a single coefficient -> [1, -1].
        F = lambda x: jnp.cos(1000 * x)  # noqa: E731
        f = Tech.from_function(_pm(F))
        c = Tech.alias(f.coeffs, 1)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert c.shape[0] == 1
        assert _ninf(values - jnp.array([[1.0, -1.0]])) < 1e2 * vscale * EPS

    @pytest.mark.parametrize("Tech,kind", CASES)
    def test_array_cos1000_to_2(self, Tech, kind):
        # pass(n, 9): alias to 2.
        F = lambda x: jnp.cos(1000 * x)  # noqa: E731
        f = Tech.from_function(_pm(F))
        c = Tech.alias(f.coeffs, 2)
        y = chebpts(2, kind)
        values = Tech.coeffs2vals(c)
        vscale = _ninf(values)
        assert c.shape[0] == 2
        assert _ninf(values - _pm(F)(y)) < 1e3 * vscale * EPS
