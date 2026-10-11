"""Literal @singfun/plus.m tolSmth threshold, Chebfun7574c77."""

import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("factor", [999, 1000, 1001])
@pytest.mark.parametrize("case", [1, 2])
def test_source_zero_threshold(factor, case):
    delta = factor * jnp.finfo(jnp.float64).eps
    if case == 1:
        f = Singfun(Chebtech2.from_coeffs(jnp.array([1.0])), (-2.0, 0.0))
        g = Singfun(Chebtech2.from_coeffs(jnp.array([-1.0 + delta])), (-2.0, 0.0))
    else:
        f = Singfun(Chebtech2.from_coeffs(jnp.array([1.0 + delta, 1.0])), (-2.0, 0.0))
        g = Singfun(Chebtech2.from_coeffs(jnp.array([-1.0])), (-1.0, 0.0))
    value = f + g
    if factor < 1000:
        assert isinstance(value, Chebtech2) and value.iszero()
    else:
        assert isinstance(value, Singfun) and value.exponents == (-2.0, 0.0)
        assert value.smoothPart.normest() == delta
