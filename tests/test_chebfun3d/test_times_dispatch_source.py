"""Supplemental multiplication dispatch controls, Chebfun7574c77.

MATLAB source: @chebfun3/times.m and mtimes.m.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


def test_scalar_zero_reconstructs_source_factors_and_reflected_dispatch():
    f = Chebfun3.from_function(lambda x, y, z: x+y*z)
    zero = Chebfun3.from_function(lambda x, y, z: 0*x)
    for result in (f*0, 0*f, f.times(0), f.mtimes(0)):
        assert result.isequal(zero)
        for actual, expected in zip((result.cols, result.rows, result.tubes),
                                    (zero.cols, zero.rows, zero.tubes)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.coeffs, b.coeffs)
    for result in (Chebfun3.times(2., f), Chebfun3.mtimes(2., f)):
        np.testing.assert_array_equal(result.core, 2*f.core)
    with pytest.raises(ValueError, match="mtimes:size"):
        f.times(jnp.ones(2))
    with pytest.raises(ValueError, match="times:unknown"):
        f.times(object())
    g = Chebfun3.from_function(lambda x, y, z: x, domain=(-2., 2., -1., 1., -1., 1.))
    with pytest.raises(ValueError, match="times:domain"):
        f.times(g)


def test_scalar_vector_product_retains_orientation():
    f = Chebfun3.from_function(lambda x, y, z: x+2.)
    vector = Chebfun3v(f, f).T
    for result in (f.times(vector), f.mtimes(vector)):
        assert result.is_transposed == vector.is_transposed
        np.testing.assert_allclose(result(.25, .3, .4), [2.25**2]*2,
                                   rtol=0, atol=1e-13)
