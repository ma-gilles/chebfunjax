"""Typed scalar integers honor the public Python Chebfun scaling contract.

The lower Chebtech adapter retains explicit MATLAB class diagnostics.
"""
import jax.numpy as jnp

# uses-numpy: typed Python interop operands and independent assertions.
import numpy as np
import pytest

from chebfunjax import chebfun


@pytest.mark.parametrize("value", [np.int64(2), np.uint64(3),
                                   jnp.asarray(4, dtype=jnp.int32),
                                   jnp.asarray(5, dtype=jnp.uint64)])
@pytest.mark.parametrize("transposed", [False, True])
def test_typed_integer_scalar_scaling_preserves_complex_values_and_orientation(value, transposed):
    field = chebfun(lambda x: 1.+2j*x, domain=(-1., 1.))
    if transposed:
        field = field.T
    points = jnp.asarray([-.75, -.125, .25, .625])
    scaled = field * value
    assert scaled.is_transposed is transposed
    expected = np.asarray((1.+2j*points)*float(value))
    np.testing.assert_allclose(scaled(points), expected, rtol=0, atol=1e-12)
