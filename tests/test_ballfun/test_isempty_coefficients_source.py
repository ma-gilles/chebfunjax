"""Native isempty means an empty coefficient tensor, not zero function value."""

import jax
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


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4)])
@pytest.mark.parametrize("compiled", [False, True])
def test_empty_input_precedes_flags_and_fixed_size(shape, compiled):
    # @ballfun/ballfun.m bypasses constructor parsing for any empty input.
    def construct(coefficients):
        return Ballfun.from_coeffs(coefficients, is_real=True,
                                   fixed_size=(3, 3, 3))

    operation = jax.jit(construct) if compiled else construct
    result = operation(jnp.empty(shape))
    reference = Ballfun.empty()
    assert result.isempty()
    assert result.shape == reference.shape
    assert result.domain == reference.domain
    assert result.is_real == reference.is_real
