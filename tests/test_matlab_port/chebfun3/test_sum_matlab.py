"""Literal 13 predicates from MATLAB tests/chebfun3/test_sum.m.

The native ``sum(F,dim)`` returns a Chebfun2 over the two surviving
physical domains. Python ``F.sum(dim=dim)`` is the corresponding API.

Provenance
----------
MATLAB source : tests/chebfun3/test_sum.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e4 * ChebfunPref().cheb3Prefs.chebfun3eps


def test_original_polynomial_dimension_reductions():
    dom = (11.0, 14.0, 7.0, 10.0, 5.0, 9.0)
    f = Chebfun3.from_function(lambda x, y, z: x**2 + 4*y - 5*z,
                               domain=dom)
    exact1 = Chebfun2.from_function(
        lambda y, z: 12*y - 15*z + 471, domain=dom[2:6])
    assert (f.sum() - exact1).norm() < TOL                  # pass(1)
    assert (f.sum(dim=1) - exact1).norm() < TOL             # pass(2)

    exact2 = Chebfun2.from_function(
        lambda x, z: 3*x**2 - 15*z + 102,
        domain=dom[0:2] + dom[4:6])
    assert (f.sum(dim=2) - exact2).norm() < TOL             # pass(3)
    exact3 = Chebfun2.from_function(
        lambda x, y: 4*x**2 + 16*y - 140, domain=dom[0:4])
    assert (f.sum(dim=3) - exact3).norm() < TOL             # pass(4)


def test_original_integrate_x_with_missing_coordinates():
    dom = (-2.0, 3.0, -2.0, 2.0, -5.0, -3.0)
    f = Chebfun3.from_function(
        lambda x, y, z: jnp.cos(y) + jnp.sin(z), domain=dom)
    exact = Chebfun2.from_function(
        lambda y, z: (dom[1]-dom[0])*(jnp.cos(y)+jnp.sin(z)),
        domain=dom[2:6])
    assert (f.sum() - exact).norm() < TOL                    # pass(5)
    cases = [
        (lambda x, y, z: jnp.cos(x) + jnp.sin(z),
         lambda y, z: jnp.sin(3) + jnp.sin(2) + 5*jnp.sin(z)),
        (lambda x, y, z: jnp.cos(x) + jnp.sin(y),
         lambda y, z: jnp.sin(3) + jnp.sin(2) + 5*jnp.sin(y)),
    ]
    for index, (fn, exact_fn) in enumerate(cases, start=6):
        f = Chebfun3.from_function(fn, domain=dom)
        exact = Chebfun2.from_function(exact_fn, domain=dom[2:6])
        assert (f.sum() - exact).norm() < TOL, f"pass({index})"


def test_original_integrate_y_with_missing_coordinates():
    dom = (-2.0, 3.0, -2.0, 2.0, -5.0, -3.0)
    cases = [
        (lambda x, y, z: jnp.cos(y)+jnp.sin(z),
         lambda x, z: 2*jnp.sin(2)+4*jnp.sin(z)),
        (lambda x, y, z: jnp.cos(x)+jnp.sin(z),
         lambda x, z: (dom[3]-dom[2])*(jnp.cos(x)+jnp.sin(z))),
        (lambda x, y, z: jnp.cos(x)+jnp.sin(y),
         lambda x, z: 4*jnp.cos(x)),
    ]
    for index, (fn, exact_fn) in enumerate(cases, start=8):
        f = Chebfun3.from_function(fn, domain=dom)
        exact = Chebfun2.from_function(
            exact_fn, domain=dom[0:2]+dom[4:6])
        assert (f.sum(dim=2) - exact).norm() < TOL, f"pass({index})"


def test_original_integrate_z_with_missing_coordinates():
    dom = (-2.0, 3.0, -2.0, 2.0, -5.0, -3.0)
    cases = [
        (lambda x, y, z: jnp.cos(y)+jnp.sin(z),
         lambda x, y: 2*jnp.cos(y)+jnp.cos(5)-jnp.cos(3)),
        (lambda x, y, z: jnp.cos(x)+jnp.sin(z),
         lambda x, y: 2*jnp.cos(x)+jnp.cos(5)-jnp.cos(3)),
        (lambda x, y, z: jnp.cos(x)+jnp.sin(y),
         lambda x, y: (dom[5]-dom[4])*(jnp.cos(x)+jnp.sin(y))),
    ]
    for index, (fn, exact_fn) in enumerate(cases, start=11):
        f = Chebfun3.from_function(fn, domain=dom)
        exact = Chebfun2.from_function(exact_fn, domain=dom[0:4])
        assert (f.sum(dim=3) - exact).norm() < TOL, f"pass({index})"
