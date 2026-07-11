"""Port of MATLAB Chebfun tests/chebfun3/test_complex.m (Fable 5).

Complex construction was fixed in the Fable 5 audit (the imaginary part
was previously silently dropped, as in Chebfun2).  real/imag method
assertions are skipped (methods absent).

Provenance
----------
MATLAB source : tests/chebfun3/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 1000 * EPS


class TestChebfun3Complex:
    def test_complex_combination(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x * y * z))
        g = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        h = f + 1j * g
        assert maxdiff(
            h, lambda x, y, z: jnp.sin(x * y * z)
            + 1j * jnp.cos(x * y * z)) < TOL

    def test_direct_complex_construction(self):
        h = Chebfun3.from_function(
            lambda x, y, z: jnp.exp(1j * x * y * z))
        assert maxdiff(h,
                       lambda x, y, z: jnp.exp(1j * x * y * z)) < TOL

    def test_real_imag_methods(self):
        pytest.skip("Chebfun3 has no real()/imag() methods")
