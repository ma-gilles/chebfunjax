"""Port of MATLAB Chebfun tests/cheb/test_bernoulli.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_bernoulli.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax import cheb

jax.config.update("jax_enable_x64", True)


class TestChebBernoulli:
    def test_all_matlab_assertions(self):
        # MATLAB: pass = doesNotCrash(@() cheb.bernoulli(4)).
        B = cheb.bernoulli(4)
        assert len(B) == 5
        # Sanity beyond MATLAB's crash test: B_2(x) = x^2 - x + 1/6.
        x = jnp.linspace(0.0, 1.0, 7)
        assert float(jnp.max(jnp.abs(B[2](x) - (x ** 2 - x + 1.0 / 6.0)))) < 1e-13
