"""Port of MATLAB Chebfun tests/ballfun/test_ballfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_ballfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import EPS, X0, Y0, Z0, val


class TestBallfunBallfun:
    def test_construction_and_eval(self):
        f = Ballfun.from_function(lambda x, y, z: 1 + x * y + z ** 2)
        exact = 1 + X0 * Y0 + Z0 ** 2
        assert abs(val(f) - exact) < 1e3 * EPS


@pytest.mark.parametrize("clause", range(1, 16))
def test_native_constructor_clause(clause):
    # Literal native operands and epsilon bound. Python named adapters expose
    # vectorize, fixed-size, coefficient and copy construction explicitly.
    cart = {
        1: lambda x,y,z: x, 2: lambda x,y,z: y, 3: lambda x,y,z: z,
        4: lambda x,y,z: x*z, 5: lambda x,y,z: jnp.sin(x*y*z),
        7: lambda x,y,z: jnp.cos(x*y*z),
        9: lambda x,y,z: x*z, 10: lambda x,y,z: jnp.sin(x*y*z),
    }
    sph = {
        1: lambda r,l,t: r*jnp.sin(t)*jnp.cos(l),
        2: lambda r,l,t: r*jnp.sin(t)*jnp.sin(l),
        3: lambda r,l,t: r*jnp.cos(t),
        4: lambda r,l,t: r*jnp.sin(t)*jnp.cos(l)*r*jnp.cos(t),
        5: lambda r,l,t: jnp.sin(r*jnp.sin(t)*jnp.cos(l)*r*jnp.sin(t)*jnp.sin(l)*r*jnp.cos(t)),
        7: lambda r,l,t: jnp.cos(r*jnp.sin(t)*jnp.cos(l)*r*jnp.sin(t)*jnp.sin(l)*r*jnp.cos(t)),
    }
    sph[9], sph[10] = sph[4], sph[5]
    if clause in cart:
        f = Ballfun.from_function(cart[clause])
        exact = Ballfun.from_function(sph[clause], spherical=True)
    elif clause == 6:
        c = jnp.zeros((10,11,12), dtype=jnp.complex128)
        c = c.at[1,5,5].set(.5).at[1,5,7].set(.5)
        f = Ballfun.from_coeffs(c)
        exact = Ballfun.from_function(lambda r,l,t: r*jnp.cos(t), spherical=True)
    elif clause in (8,11,12):
        op = {8: lambda x,y,z: jnp.cos(x*y*z),
              11: lambda x,y,z: x*y, 12: lambda x,y,z: x*z*y}[clause]
        f = Ballfun.from_function(op, vectorize=True)
        exact = Ballfun.from_function(op)
    elif clause == 13:
        op = lambda x,y,z: jnp.cos(x*y)
        f = Ballfun.from_function(op, fixed_size=(50,51,52))
        exact = Ballfun.from_function(op)
    elif clause == 14:
        exact = Ballfun.from_function(lambda x,y,z: jnp.cos(x*y))
        f = Ballfun.from_function(exact, fixed_size=(51,50,49), vectorize=True)
    else:
        exact = Ballfun.from_function(lambda x,y,z: jnp.sin(x*y))
        c = exact.coeffs3(49,50,51)
        f = Ballfun.from_coeffs(c, fixed_size=(51,53,51))
    assert (f-exact).norm() < EPS
