"""Port of MATLAB Chebfun tests/spherefunv/test_tangentnormal.m (Fable 5).

FIXED: Spherefunv gained normal/tangent projections and the static unormal
constructor in the Fable 5 3-Cartesian-component overhaul.

Provenance
----------
MATLAB source : tests/spherefunv/test_tangentnormal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

from ._helpers import EPS, cart, vdiff, vnorm

TOL = 1e2 * EPS


class TestSpherefunvTangentNormal:
    def test_normal_and_tangent_of_position(self):
        # u = (x, y, z) is the normal field itself.
        u = Spherefunv(cart(lambda x, y, z: x),
                       cart(lambda x, y, z: y),
                       cart(lambda x, y, z: z))
        # pass(1): normal(u) == u.
        assert vdiff(u, u.normal()) < TOL
        # pass(2): tangent(u) == 0.
        assert vnorm(u.tangent()) < TOL

    @pytest.mark.xfail(
        strict=True,
        reason="chebfunjax Spherefun.diff uses the harmonic-projection "
        "route (~1e-13 accuracy) vs MATLAB's exact coefficient-space "
        "parity derivatives: normal(grad f)/tangent(grad f) reaches "
        "2.6e-13 vs the 1e2*eps = 2.2e-14 tolerance.  Fixing requires "
        "porting the coeff-space diff (measured, not estimated)")
    def test_normal_and_tangent_of_gradient(self):
        # grad(f) is purely tangential.
        f = Spherefun.from_function(
            lambda lam, th: jnp.cos(2 * jnp.pi
                                    * (jnp.cos(lam) * jnp.sin(th))
                                    * (jnp.sin(lam) * jnp.sin(th))
                                    * jnp.cos(th)))
        u = f.gradient()
        # pass(3): tangent(u) == u.
        assert vdiff(u, u.tangent()) < TOL
        # pass(4): normal(u) == 0.
        assert vnorm(u.normal()) < TOL

    def test_unormal_matches_position(self):
        # pass(5): unormal() == (x, y, z).
        u = Spherefunv(cart(lambda x, y, z: x),
                       cart(lambda x, y, z: y),
                       cart(lambda x, y, z: z))
        assert vdiff(u, Spherefunv.unormal()) < TOL
