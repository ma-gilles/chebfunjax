"""Literal predicates from MATLAB tests/chebfun3/test_sum3.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_sum3.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

from chebfunjax import cheb
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3

from ._helpers import EPS

TOL = 1e4 * EPS

def test_source_empty():
    assert jnp.size(chebfun3().sum3()) == 0

def test_source_constant():
    f = Chebfun3.from_function(lambda x, y, z: 1)
    assert abs(f.sum3() - 8) < TOL

def test_source_runge():
    f = cheb.gallery3('runge')
    assert abs(f.sum3() - 4.28685406230184188268) < TOL

@pytest.mark.parametrize('axis,expected', [(0, 3), (1, 6), (2, 9)])
def test_source_box(axis, expected):
    f = Chebfun3.from_function(lambda x, y, z: (x, y, z)[axis],
                             domain=(0, 1, 0, 2, 0, 3))
    assert abs(f.sum3() - expected) < TOL
