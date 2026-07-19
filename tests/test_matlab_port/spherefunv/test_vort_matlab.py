"""Port of MATLAB Chebfun tests/spherefunv/test_vort.m (Fable 5).

FIXED: Spherefunv gained the 3-Cartesian-component representation and the
surface vorticity operator (normal component of the surface curl) in the
Fable 5 overhaul.

Provenance
----------
MATLAB source : tests/spherefunv/test_vort.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

from ._helpers import EPS, X, Y, Z, cart, sdiff, snorm

TOL = 2e3 * EPS


class TestSpherefunvVort:
    def test_empty_and_type(self):
        # pass(1): vorticity of the empty field is an empty spherefun.
        f = Spherefunv.empty().vort()
        assert f.isempty() and isinstance(f, Spherefun)
        # pass(2): vorticity of the unit normal is a spherefun.
        assert isinstance(Spherefunv.unormal().vort(), Spherefun)

    def test_zero_field(self):
        # pass(3): vorticity of the zero field is zero.
        z = cart(lambda x, y, z: 0.0 * x)
        assert snorm(Spherefunv(z, z, z).vort()) < TOL

    def test_vort_polynomial(self):
        # pass(4): exact = -6 (x - y) z.
        u = Spherefunv(cart(lambda x, y, z: (x - y) * y + z ** 2),
                       cart(lambda x, y, z: (y - x) * x + z ** 2),
                       cart(lambda x, y, z: -(x + y) * z))
        assert sdiff(u.vort(), -6 * (X - Y) * Z) < TOL

    def test_vort_trig_z(self):
        # pass(5): exact = -8 (2 (1-z^2) cos(4z) - z sin(4z)).
        u = Spherefunv(cart(lambda x, y, z: -4 * y * jnp.sin(4 * z)),
                       cart(lambda x, y, z: 4 * x * jnp.sin(4 * z)),
                       cart(lambda x, y, z: 0.0 * x))
        exact = -8 * (2 * (1 - Z ** 2) * jnp.cos(4 * Z) - Z * jnp.sin(4 * Z))
        assert sdiff(u.vort(), exact) < 100 * TOL

    def test_vort_trig_x(self):
        # pass(6): exact = -8 (2 (1-x^2) cos(4x) - x sin(4x)).
        u = Spherefunv(cart(lambda x, y, z: 0.0 * x),
                       cart(lambda x, y, z: -4 * z * jnp.sin(4 * x)),
                       cart(lambda x, y, z: 4 * y * jnp.sin(4 * x)))
        exact = -8 * (2 * (1 - X ** 2) * jnp.cos(4 * X) - X * jnp.sin(4 * X))
        assert sdiff(u.vort(), exact) < 100 * TOL

    def test_vort_trig_y_and_alias(self):
        # pass(7): exact = -8 (2 (1-y^2) cos(4y) - y sin(4y)).
        u = Spherefunv(cart(lambda x, y, z: 4 * z * jnp.sin(4 * y)),
                       cart(lambda x, y, z: 0.0 * x),
                       cart(lambda x, y, z: -4 * x * jnp.sin(4 * y)))
        exact = -8 * (2 * (1 - Y ** 2) * jnp.cos(4 * Y) - Y * jnp.sin(4 * Y))
        assert sdiff(u.vort(), exact) < 100 * TOL
        # pass(8): vort is the alias for vorticity.
        assert sdiff(u.vorticity(), exact) < 100 * TOL
