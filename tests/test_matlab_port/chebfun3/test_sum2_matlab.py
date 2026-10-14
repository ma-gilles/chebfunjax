"""Literal 10 predicates from MATLAB tests/chebfun3/test_sum2.m.

MATLAB dimension vectors map to Python ``dims`` tuples. Comparisons use the
native one-dimensional Chebfun continuous norm.

Provenance
----------
MATLAB source : tests/chebfun3/test_sum2.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e5 * ChebfunPref().cheb3Prefs.chebfun3eps


def test_original_real_dimension_pairs_and_orderings():
    dom = (11.0, 14.0, 7.0, 10.0, 5.0, 9.0)
    f = Chebfun3.from_function(lambda x, y, z: x**2+4*y-5*z,
                               domain=dom)
    exact12 = Chebfun.from_function(lambda z: 1719-45*z,
                                    Domain(dom[4:6]))
    assert (f.sum2() - exact12).norm() < TOL                  # pass(1)
    assert (f.sum2((1, 2)) - exact12).norm() < TOL           # pass(2)
    assert (f.sum2((2, 1)) - exact12).norm() < TOL           # pass(3)

    exact23 = Chebfun.from_function(lambda x: 12*x**2-12,
                                    Domain(dom[0:2]))
    assert (f.sum2((2, 3)) - exact23).norm() < TOL           # pass(4)
    assert (f.sum2((3, 2)) - exact23).norm() < TOL           # pass(5)

    exact13 = Chebfun.from_function(lambda y: 48*y+1464,
                                    Domain(dom[2:4]))
    assert (f.sum2((1, 3)) - exact13).norm() < TOL           # pass(6)
    assert (f.sum2((3, 1)) - exact13).norm() < TOL           # pass(7)


def test_original_complex_valued_dimension_pairs():
    f = Chebfun3.from_function(
        lambda x, y, z: 1j*x**2 + 2j*y**2 + z**2)
    exact_z = Chebfun.from_function(lambda z: 4*z**2 + 4j,
                                    Domain((-1.0, 1.0)))
    assert (f.sum2() - exact_z).norm() < TOL                 # pass(8)

    exact_y = Chebfun.from_function(
        lambda y: 8j*y**2 + (4/3)*(1+1j), Domain((-1.0, 1.0)))
    assert (f.sum2((1, 3)) - exact_y).norm() < TOL           # pass(9)

    exact_x = Chebfun.from_function(
        lambda x: 1j*(4*x**2+8/3) + 4/3, Domain((-1.0, 1.0)))
    assert (f.sum2((2, 3)) - exact_x).norm() < TOL           # pass(10)
