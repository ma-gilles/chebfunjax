"""Port of MATLAB Chebfun tests/chebop/test_linearize_init_fails.m (Fable 5).

MATLAB raises 'CHEBFUN:CHEBOP:linearize:invalidInitialGuess' when the
operator cannot be evaluated at the initial guess; chebfunjax raises a
``ValueError`` carrying the same identifier.

Provenance
----------
MATLAB source : tests/chebop/test_linearize_init_fails.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


def _n(f):
    xs = jnp.linspace(1e-6, 1.0 - 1e-6, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopLinearizeInitFails:
    def test_all_matlab_assertions(self):
        # pass(1): lbc supplies a usable initial guess -> solves.
        N = Chebop(lambda u: u.diff() - u.sqrt(), domain=(0.0, 1.0))
        N.lbc = 1.0
        u = N.solve(1.0)
        assert _n(N(u) - 1.0) < 1e-10

        # pass(2): general bc -> zero init -> sqrt(0) singular linearization.
        N = Chebop(lambda u: u.diff() - u.sqrt(), domain=(0.0, 1.0))
        N.bc = lambda u: u(0.0) - 1.0
        with pytest.raises(ValueError, match="invalidInitialGuess"):
            N.solve(1.0)

        # pass(4): second order with sqrt.
        N = Chebop(lambda u: u.diff(2) - u.sqrt(), domain=(0.0, 1.0))
        N.bc = 1.0
        with pytest.raises(ValueError, match="invalidInitialGuess"):
            N.solve(1.0)

        # pass(5): 1/u at zero init.
        N = Chebop(lambda u: u.diff(2) - 1.0 / u, domain=(0.0, 1.0))
        N.bc = 1.0
        with pytest.raises(ValueError, match="invalidInitialGuess"):
            N.solve(1.0)
