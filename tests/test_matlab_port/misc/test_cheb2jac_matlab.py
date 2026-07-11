"""Port of MATLAB Chebfun tests/misc/test_cheb2jac.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_cheb2jac.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.transforms import cheb2jac, jac2cheb

TOL = 5e-11


class TestCheb2jac:
    def test_roundtrip(self):
        rng = np.random.default_rng(2)
        c = jnp.asarray(rng.standard_normal(30) / np.arange(1, 31) ** 2)
        back = jac2cheb(cheb2jac(c, 0.2, -0.4), 0.2, -0.4)
        assert float(jnp.max(jnp.abs(back - c))) < TOL

    def test_legendre_special_case(self):
        # P_n^{(0,0)} = Legendre: cheb2jac(c,0,0) == cheb2leg(c)
        from chebfunjax.utils.transforms import cheb2leg
        rng = np.random.default_rng(3)
        c = jnp.asarray(rng.standard_normal(20) / np.arange(1, 21) ** 2)
        a = cheb2jac(c, 0.0, 0.0)
        b = cheb2leg(c)
        assert float(jnp.max(jnp.abs(a - b))) < TOL
