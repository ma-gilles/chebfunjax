"""Native isempty means an empty coefficient tensor, not zero function value."""

import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun


@pytest.mark.parametrize("shape,expected", [((0, 3, 4), True),
                                             ((2, 0, 4), True),
                                             ((2, 3, 0), True),
                                             ((0, 0, 0), True),
                                             ((1, 1, 1), False)])
def test_public_coefficient_emptiness_and_nonempty_zero(shape, expected):
    function = Ballfun.from_coeffs(jnp.zeros(shape))
    assert function.isempty() is expected
