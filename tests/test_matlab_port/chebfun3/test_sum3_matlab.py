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
from chebfunjax.tech.chebtech import Chebtech2

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

def test_complex_core_with_real_factors_preserves_sum3_and_mean3():
    real_factor = Chebtech2.from_coeffs(jnp.array([1.0]))
    f = Chebfun3(
        cols=[real_factor], rows=[real_factor], tubes=[real_factor],
        core=jnp.array([[[2.0 + 3.0j]]]),
        domain=(0.0, 1.0, 0.0, 2.0, 0.0, 3.0),
    )
    assert abs(f.sum3() - 6.0 * (2.0 + 3.0j)) < TOL
    assert abs(f.mean3() - (2.0 + 3.0j)) < TOL

@pytest.mark.parametrize('core_value', [1.0, 2.0 + 3.0j])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_complex_factor_axis_preserves_sum3_and_mean3(axis, core_value):
    real_factor = Chebtech2.from_coeffs(jnp.array([1.0]))
    complex_factor = Chebtech2.from_coeffs(jnp.array([1.0j]))
    factors = [real_factor, real_factor, real_factor]
    factors[axis] = complex_factor
    f = Chebfun3(
        cols=[factors[0]], rows=[factors[1]], tubes=[factors[2]],
        core=jnp.array([[[core_value]]]),
        domain=(0.0, 1.0, 0.0, 2.0, 0.0, 3.0),
    )
    expected_mean = core_value * 1.0j
    # One factor has integral 2i instead of 2; physical volume is six.
    expected_sum = 6.0 * core_value * 1.0j
    assert abs(f.sum3() - expected_sum) < TOL
    assert abs(f.mean3() - expected_mean) < TOL
