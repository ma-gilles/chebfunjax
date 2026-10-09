"""Port of MATLAB Chebfun tests/chebop/test_autoVectorize.m (Fable 5).

MATLAB checks that scalar-style ops (u^2, a*(1-u^2)*diff(u)) run
after automatic vectorization; Python chebfun arithmetic is already
elementwise, so the same problems must simply solve.  As in MATLAB,
only successful execution is asserted, not accuracy.

Provenance
----------
MATLAB source : tests/chebop/test_autoVectorize.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

D = (0.0, 1.0)
A = 5.0


def _x():
    return cj.chebfun(lambda t: t, domain=D)


class TestChebopAutovectorize:
    @pytest.mark.timeout(880)
    def test_scalar_ivp(self):
        # pass(1): van der Pol-style IVP.
        N = Chebop(lambda u: u.diff(2)
                   - A * (1 - u ** 2) * u.diff() + u, D)
        N.lbc = lambda u: [u - 2, u.diff()]
        u = N.solve(0.0)
        assert u is not None

    @pytest.mark.timeout(880)
    def test_scalar_ivp_coefficient(self):
        # pass(2): chebfun coefficient in the nonlinear term.
        x = _x()
        f = (4 * x).sin().exp()
        N = Chebop(lambda xx, u: u.diff(2)
                   - f * (1 - u ** 2) * u.diff() + u, D)
        N.lbc = lambda u: [u - 2, u.diff()]
        u = N.solve(0.0)
        assert u is not None

    @pytest.mark.timeout(880)
    def test_scalar_bvp(self):
        # pass(3): BVP with interior-point conditions via N.bc.
        N = Chebop(lambda u: u.diff(2)
                   - A * (1 - u ** 2) * u.diff() + u, D)
        N.bc = lambda x, u: [u(jnp.asarray(0.0)) - 2,
                             u(jnp.asarray(1.0)) - 3]
        u = N.solve(0.0)
        assert abs(float(u(jnp.asarray(0.0))) - 2) < 1e-6
        assert abs(float(u(jnp.asarray(1.0))) - 3) < 1e-6

    @pytest.mark.timeout(880)
    def test_scalar_bvp_coefficient(self):
        # pass(4).
        x = _x()
        f = (4 * x).sin().exp()
        N = Chebop(lambda xx, u: u.diff(2)
                   - 0.2 * f * (1 - u ** 2) * u.diff() + u, D)
        N.bc = lambda xx, u: [u(jnp.asarray(0.0)) - 2,
                              u(jnp.asarray(1.0)) - 3]
        u = N.solve(0.0)
        assert abs(float(u(jnp.asarray(0.0))) - 2) < 1e-6
        assert abs(float(u(jnp.asarray(1.0))) - 3) < 1e-6
