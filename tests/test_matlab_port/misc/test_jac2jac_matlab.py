"""Port of MATLAB Chebfun tests/misc/test_jac2jac.m (Fable 5).

MATLAB sweeps 5^4 (alpha,beta,gam,delta) combinations against a direct
cheb2jac(jac2cheb(.)) reference at tol 2e-10; the port uses the same
composition identity on a reduced deterministic sweep (3^4) for
runtime, at the same tolerance per combination.

Provenance
----------
MATLAB source : tests/misc/test_jac2jac.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.transforms import cheb2jac, jac2cheb, jac2jac

TOL = 2e-10
RNG = np.random.default_rng(0)
V = jnp.asarray(RNG.standard_normal(10))
PARAMS = np.linspace(-0.99, 1.1, 3)


class TestJac2jac:
    @pytest.mark.parametrize("a", PARAMS)
    @pytest.mark.parametrize("b", PARAMS)
    def test_jac2jac_matches_composition(self, a, b):
        for g in PARAMS:
            for d in PARAMS:
                exact = cheb2jac(jac2cheb(V, a, b), g, d)
                w = jac2jac(V, a, b, g, d)
                err = float(jnp.max(jnp.abs(w - exact)))
                assert err < TOL, f"(a,b,g,d)=({a},{b},{g},{d})"
