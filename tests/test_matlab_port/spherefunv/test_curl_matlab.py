"""Port of MATLAB Chebfun tests/spherefunv/test_curl.m (Fable 5).

FIXED: Spherefunv gained the 3-Cartesian-component representation and the
surface curl operator in the Fable 5 overhaul.

Provenance
----------
MATLAB source : tests/spherefunv/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefunv import Spherefunv

from ._helpers import EPS, cart, vdiff, vnorm

TOL = 2e3 * EPS


class TestSpherefunvCurl:
    def test_empty_and_type(self):
        # pass(1): curl of the empty field is an empty spherefunv.
        u = Spherefunv.empty()
        f = u.curl()
        assert f.isempty() and isinstance(f, Spherefunv)
        # pass(2): curl of the unit normal is a spherefunv.
        f = Spherefunv.unormal().curl()
        assert isinstance(f, Spherefunv)

    def test_zero_field(self):
        # pass(3): curl of the zero field is zero.
        z = cart(lambda x, y, z: 0.0 * x)
        u = Spherefunv(z, z, z)
        assert vnorm(u.curl()) < TOL

    def test_curl_of_position(self):
        # pass(4): curl of the position field (x, y, z) is zero.
        u = Spherefunv.unormal()
        exact = Spherefunv(cart(lambda x, y, z: 0.0 * x),
                           cart(lambda x, y, z: 0.0 * x),
                           cart(lambda x, y, z: 0.0 * x))
        assert vdiff(u.curl(), exact) < TOL

    def test_curl_trigonometric(self):
        # pass(5): a fully three-dimensional trigonometric field.
        u = Spherefunv(cart(lambda x, y, z: jnp.cos(4 * y)),
                       cart(lambda x, y, z: jnp.cos(4 * z)),
                       cart(lambda x, y, z: jnp.sin(4 * x)))
        exact = Spherefunv(
            cart(lambda x, y, z: -4 * x * y * jnp.cos(4 * x)
                 + 4 * (1 - z ** 2) * jnp.sin(4 * z)),
            cart(lambda x, y, z: 4 * y * z * jnp.sin(4 * y)
                 - 4 * (1 - x ** 2) * jnp.cos(4 * x)),
            cart(lambda x, y, z: 4 * x * z * jnp.sin(4 * z)
                 + 4 * (1 - y ** 2) * jnp.sin(4 * y)))
        assert vdiff(u.curl(), exact) < 100 * TOL
