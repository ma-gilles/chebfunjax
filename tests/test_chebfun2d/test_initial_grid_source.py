"""Constructor grid defaults from Chebfun 7574c77 tech preferences."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


@pytest.mark.parametrize("flags,shape", [
    ({}, (17, 17)),
    ({"trig": True}, (16, 16)),
    ({"trigx": True}, (17, 16)),
    ({"trigy": True}, (16, 17)),
])
def test_source_initial_tensor_grid(flags, shape):
    # @chebtech/techPref.m and @trigtech/techPref.m set minSamples=17.
    # @chebfun2/constructor.m rounds down to 2**k+1 or 2**k respectively.
    # Native quotient callback capture independently observes (17,17).
    tensor_shapes = []

    def function(x, y):
        if x.ndim == 2 and min(x.shape) > 1:
            tensor_shapes.append(x.shape)
        return 3 + jnp.zeros_like(x + y)

    result = Chebfun2.from_function(function, **flags)
    assert tensor_shapes[0] == shape
    assert float(result(.125, -.25)) == 3.0
