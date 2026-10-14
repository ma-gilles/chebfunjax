"""Supplemental source controls for Chebfun3 division APIs.

Provenance
----------
MATLAB source : @chebfun3/{mrdivide,mldivide,ldivide}.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2


def fixture(core):
    factors = [Chebtech2(coeffs=jnp.asarray([1., 0.])),
               Chebtech2(coeffs=jnp.asarray([0., 1.]))]
    return Chebfun3(factors, factors, factors, jnp.asarray(core),
                    (-1., 1., -1., 1., -1., 1.))


def test_literal_empty_returns_and_errors():
    f = fixture(jnp.ones((2, 2, 2)))
    empty = Chebfun3.empty()
    a = jnp.empty((0,))
    assert f.mrdivide(a) is f
    assert empty.mrdivide(2) is empty
    assert Chebfun3.mldivide(a, f) is a
    assert Chebfun3.mldivide(2, empty) is empty
    with pytest.raises(ValueError, match="mrdivide:divisionByZero"):
        f.mrdivide(0)
    with pytest.raises(ValueError, match="mldivide:nonScalar"):
        f.mldivide(2)
    with pytest.raises(ValueError, match="mrdivide:mrdivide"):
        f.mrdivide(f)
    with pytest.raises(ValueError, match="two-dimensional"):
        f.mrdivide(jnp.eye(2))


def test_core_matrix_division_and_factors():
    f = fixture([[1., 2.], [3., 4.]])
    out = f.mrdivide(jnp.diag(jnp.asarray([2., 4.])))
    np.testing.assert_array_equal(out.core, [[.5, .5], [1.5, 1.]])
    assert all(a is b for a, b in zip(f.cols, out.cols))
    out = f.mrdivide(jnp.asarray([[1., 0.], [0., 1.], [0., 0.]]))
    np.testing.assert_array_equal(out.core, [[1., 2., 0.], [3., 4., 0.]])


def test_scalar_core_division_has_direct_binary64_rounding():
    data = np.asarray([3.0355227320705427, -1.4142135623730954]).reshape(2, 1, 1)
    f = fixture(data)
    out = f.mrdivide(jnp.asarray(3.))
    np.testing.assert_array_equal(np.asarray(out.core), data / 3.)
