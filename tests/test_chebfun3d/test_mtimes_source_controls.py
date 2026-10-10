"""Scalar and adapter controls for @chebfun3/mtimes.m, Chebfun7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d._mtimes import _native_squeeze
from chebfunjax.chebfun3d.chebfun3 import chebfun3


@pytest.fixture(scope='module')
def field():
    return chebfun3(lambda x, y, z: x+y+z)


def test_numeric_left_and_array_scalar(field):
    left = 2 @ field
    right = field @ jnp.array([[2.]])
    assert jnp.array_equal(left.core, 2*field.core)
    assert jnp.array_equal(right.core, left.core)
    assert all(a is b for a, b in zip(left.cols, field.cols, strict=True))


def test_zero_reconstructs_rank(field):
    zero = field @ 0
    assert jnp.max(jnp.abs(zero.core)) == 0
    assert zero.domain == field.domain
    assert zero.core.shape == (1, 1, 1)


def test_size_error(field):
    with pytest.raises(ValueError, match='mtimes:size'):
        field @ jnp.ones(2)


def test_matlab_squeeze_shape_controls():
    for original, target in [((1, 3, 1), (1, 3)), ((1, 1, 3), (3, 1)), ((2, 1, 3), (2, 3)), ((1, 1, 1), (1, 1))]:
        assert _native_squeeze(jnp.zeros(original)).shape == target
