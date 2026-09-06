"""Port of MATLAB Chebfun tests/cheb/test_bspline.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_bspline.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax import cheb

jax.config.update("jax_enable_x64", True)


class TestChebBspline:
    def test_all_matlab_assertions(self):
        # MATLAB: pass = doesNotCrash(@() cheb.bspline(4)).
        B = cheb.bspline(4)
        assert len(B) == 4
        # Sanity beyond MATLAB's crash test: B_2 is the hat on [-1, 1].
        assert abs(float(B[1](jnp.array(0.0))) - 1.0) < 1e-13
        assert abs(float(B[1](jnp.array(0.5))) - 0.5) < 1e-13
        assert abs(float(B[3].sum()) - 1.0) < 1e-12
