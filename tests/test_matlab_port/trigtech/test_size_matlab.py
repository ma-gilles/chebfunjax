"""Port of MATLAB Chebfun tests/trigtech/test_size.m (Opus 4.8).

size(f) == size(f.coeffs).  For a scalar-valued chebfunjax trigtech the
coefficient array is 1-D, so we check the single (length) dimension; the
column dimension requires array-valued trigtechs, which chebfunjax lacks.

Provenance
----------
MATLAB source : tests/trigtech/test_size.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech


def _tt(f, n=None):
    return Trigtech.from_function(f, n=n)


class TestTrigtechSize:
    def test_scalar_size(self):
        f = _tt(lambda x: jnp.sin(10 * jnp.pi * x))
        assert f.coeffs.shape[0] == f.n

    def test_scalar_fixed_length(self):
        f = _tt(lambda x: jnp.sin(19 * jnp.pi * x), n=101)
        assert f.coeffs.shape == (101,)

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_size(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_fixed_length_size(self):
        raise AssertionError("array-valued trigtech not implemented")
