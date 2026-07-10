"""Port of MATLAB Chebfun tests/trigtech/test_length.m (Opus 4.8).

length(f) is the number of Fourier coefficients (rows of f.coeffs).

Provenance
----------
MATLAB source : tests/trigtech/test_length.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech


def _tt(f, n=None):
    return Trigtech.from_function(f, n=n)


class TestTrigtechLength:
    def test_length_equals_ncoeffs(self):
        f = _tt(lambda x: jnp.tanh(jnp.sin(jnp.pi * x)))
        assert len(f) == f.coeffs.shape[0]

    def test_fixed_length(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x), n=101)
        assert len(f) == 101

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_valued_length(self):
        raise AssertionError("array-valued trigtech not implemented")
