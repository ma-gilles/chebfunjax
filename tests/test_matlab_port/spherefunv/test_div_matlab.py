"""Port of MATLAB Chebfun tests/spherefunv/test_div.m (Fable 5).

FIXED: Spherefunv gained the 3-Cartesian-component representation and the
surface divergence operator in the Fable 5 overhaul.

Provenance
----------
MATLAB source : tests/spherefunv/test_div.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

from ._helpers import EPS, X, Y, Z, cart, sdiff, snorm

TOL = 1e2 * EPS


class TestSpherefunvDiv:
    def test_empty_and_type(self):
        # pass(1): div of the empty field is an empty spherefun.
        f = Spherefunv.empty().div()
        assert f.isempty() and isinstance(f, Spherefun)
        # pass(2): div of the unit normal is a spherefun.
        f = Spherefunv.unormal().div()
        assert isinstance(f, Spherefun)

    def test_zero_field(self):
        # pass(3): div of the zero field is zero.
        z = cart(lambda x, y, z: 0.0 * x)
        assert snorm(Spherefunv(z, z, z).div()) < TOL

    def test_div_polynomial(self):
        # pass(4): exact = -6 (x - y) z.
        u = Spherefunv(
            cart(lambda x, y, z: (1 + 2 * x * (y - x)) * z),
            cart(lambda x, y, z: (-1 + 2 * y * (y - x)) * z),
            cart(lambda x, y, z: -(x - y) * (2 * z ** 2 - 1)))
        assert sdiff(u.div(), -6 * (X - Y) * Z) < TOL

    def test_div_trig_z(self):
        # pass(5): exact = -8 (2 (1-z^2) cos(4z) - z sin(4z)).
        u = Spherefunv(cart(lambda x, y, z: 4 * x * z * jnp.sin(4 * z)),
                       cart(lambda x, y, z: 4 * y * z * jnp.sin(4 * z)),
                       cart(lambda x, y, z: -4 * (1 - z ** 2) * jnp.sin(4 * z)))
        exact = -8 * (2 * (1 - Z ** 2) * jnp.cos(4 * Z) - Z * jnp.sin(4 * Z))
        assert sdiff(u.div(), exact) < 100 * TOL

    def test_div_trig_x(self):
        # pass(6): exact = -8 (2 (1-x^2) cos(4x) - x sin(4x)).
        u = Spherefunv(cart(lambda x, y, z: 4 * (x ** 2 - 1) * jnp.sin(4 * x)),
                       cart(lambda x, y, z: 4 * x * y * jnp.sin(4 * x)),
                       cart(lambda x, y, z: 4 * x * z * jnp.sin(4 * x)))
        exact = -8 * (2 * (1 - X ** 2) * jnp.cos(4 * X) - X * jnp.sin(4 * X))
        assert sdiff(u.div(), exact) < 200 * TOL

    def test_div_trig_y_and_alias(self):
        # pass(7): exact = -8 (2 (1-y^2) cos(4y) - y sin(4y)).
        u = Spherefunv(cart(lambda x, y, z: 4 * x * y * jnp.sin(4 * y)),
                       cart(lambda x, y, z: 4 * (y ** 2 - 1) * jnp.sin(4 * y)),
                       cart(lambda x, y, z: 4 * y * z * jnp.sin(4 * y)))
        exact = -8 * (2 * (1 - Y ** 2) * jnp.cos(4 * Y) - Y * jnp.sin(4 * Y))
        assert sdiff(u.div(), exact) < 200 * TOL
        # pass(8): div is the alias for divergence.
        assert sdiff(u.divergence(), exact) < 200 * TOL
