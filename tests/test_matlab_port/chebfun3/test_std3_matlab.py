"""Port of MATLAB Chebfun tests/chebfun3/test_std3.m (Fable 5).

FIXED: Chebfun3.std3 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_std3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.chebpref import ChebfunPref


class TestChebfun3Std3:
    def test_std3_of_x(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        # var(x) over [-1,1]^3 = 1/3
        assert abs(float(f.std3()) - np.sqrt(1 / 3)) < 1e-10

    def test_native_constant_assertion(self):
        # tests/chebfun3/test_std3.m: chebfun3(10), exact=0,
        # norm(h-exact) < 1e3*pref.cheb3Prefs.chebfun3eps.
        f = chebfun3(10)
        tol = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
        assert jnp.abs(f.std3()) < tol

    def test_complex_constant_std3_is_zero(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.zeros_like(x, dtype=jnp.complex128) + (2 + 3j)
        )
        value = f.std3()
        assert value.shape == ()
        assert bool(jnp.isfinite(value))
        assert float(jnp.abs(jnp.real(value))) < 1e-12
        assert float(jnp.abs(jnp.imag(value))) < 1e-12
        assert float(jnp.real(value)) >= 0.0

    def test_complex_centered_magnitude_uses_conjugate(self):
        f = Chebfun3.from_function(lambda x, y, z: (3 + 2j) + x + 1j * y)
        value = f.std3()
        expected = jnp.sqrt(2.0 / 3.0)
        assert value.shape == ()
        assert bool(jnp.isfinite(value))
        assert float(jnp.abs(value - expected)) < 1e-10
        assert float(jnp.abs(jnp.imag(value))) < 1e-10
        assert float(jnp.real(value)) >= 0.0

    def test_std3_uses_physical_domain_volume(self):
        domain = (2.0, 6.0, -1.0, 2.0, 3.0, 7.0)
        f = Chebfun3.from_function(lambda x, y, z: x, domain=domain)
        expected = 2.0 / jnp.sqrt(3.0)
        assert float(jnp.abs(f.std3() - expected)) < 1e-10
