"""Port of MATLAB Chebfun tests/chebfun3/test_complex.m (Fable 5).

Complex construction was fixed in the Fable 5 audit (the imaginary part
was previously silently dropped, as in Chebfun2).  FIXED (Fable 5): the
real/imag/complex methods are now present and exercised below.

Provenance
----------
MATLAB source : tests/chebfun3/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

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
        # MATLAB test_complex.m: h == complex(f, g), i.e. real(h)==f and
        # imag(h)==g for h = f + 1i*g.
        f = Chebfun3.from_function(lambda x, y, z: jnp.sin(x * y * z))
        g = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        h = f + 1j * g
        assert maxdiff(h.real(), lambda x, y, z: jnp.sin(x * y * z)) < TOL
        assert maxdiff(h.imag(), lambda x, y, z: jnp.cos(x * y * z)) < TOL
        # complex(f, g) reconstructs h.
        assert maxdiff(
            Chebfun3.complex(f, g),
            lambda x, y, z: jnp.sin(x * y * z)
            + 1j * jnp.cos(x * y * z)) < TOL
