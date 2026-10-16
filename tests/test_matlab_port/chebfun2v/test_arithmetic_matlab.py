"""Port of MATLAB Chebfun tests/chebfun2v/test_arithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

TOL = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps


def _field(fn):
    return Chebfun2v.from_functions(fn, fn)


class TestChebfun2vArithmetic:
    def test_all_matlab_assertions(self):
        f = _field(lambda x, y: jnp.cos(x))
        g = _field(lambda x, y: jnp.sin(y))
        plus_exact = _field(lambda x, y: jnp.cos(x) + jnp.sin(y))
        minus_exact = _field(lambda x, y: jnp.cos(x) - jnp.sin(y))
        mult_exact = _field(lambda x, y: jnp.cos(x) * jnp.sin(y))

        # Keep the source expressions and continuous vector-field norm.
        assert float((f + g - plus_exact).norm()) < TOL  # pass(1)
        assert float((f - g - minus_exact).norm()) < TOL  # pass(2)
        assert float((f * g - mult_exact).norm()) < TOL  # pass(3)
