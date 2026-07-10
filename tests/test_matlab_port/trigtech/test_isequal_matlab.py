"""Port of MATLAB Chebfun tests/trigtech/test_isequal.m (Opus 4.8).

isequal compares two trigtechs for identical coefficients (and realness).
chebfunjax has no ``isequal`` method, so equality is defined here as same
length, bit-identical coefficients, and matching realness flag — the same
semantics MATLAB uses.

Provenance
----------
MATLAB source : tests/trigtech/test_isequal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech


def _tt(f):
    return Trigtech.from_function(f)


def _isequal(f, g):
    if f.n != g.n or f.is_real != g.is_real:
        return False
    return bool(jnp.all(f.coeffs == g.coeffs))


class TestTrigtechIsequal:
    def test_equal_to_self(self):
        f = _tt(lambda x: jnp.sin(200 * jnp.pi * x))
        g = f
        assert _isequal(f, g) and _isequal(g, f)

    def test_not_equal_sin_cos(self):
        f = _tt(lambda x: jnp.sin(200 * jnp.pi * x))
        g = _tt(lambda x: jnp.cos(200 * jnp.pi * x))
        assert not _isequal(f, g)

    def test_equal_after_reassignment(self):
        g = _tt(lambda x: jnp.sin(200 * jnp.pi * x))
        f = g
        assert _isequal(f, g)

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_scalar_vs_array(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_vs_array(self):
        raise AssertionError("array-valued trigtech not implemented")
