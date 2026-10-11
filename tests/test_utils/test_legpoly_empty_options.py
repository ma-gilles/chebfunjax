"""Empty optional argument order from legpoly.m at Chebfun 7574c77.

Provenance: literal isempty normalization precedes short-form routing.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import legpoly


@pytest.mark.parametrize("empty", [[], np.array([]), jnp.array([])])
def test_empty_optional_arguments(empty):
    x = jnp.linspace(-1, 1, 17)
    degrees = [0, 1, 4]
    default = legpoly(degrees)(x)
    normalized = legpoly(degrees, 'norm')(x)
    cases = [
        (legpoly(degrees, [-1, 1], None, empty), default),
        (legpoly(degrees, [-1, 1], empty), legpoly(degrees, [-1, 1], False)(x)),
        (legpoly(degrees, [-1, 1], empty, empty), default),
        (legpoly(degrees, [-1, 1], 'norm', empty), normalized),
    ]
    for actual, expected in cases:
        assert jnp.array_equal(actual(x), expected)
    with pytest.raises(ValueError, match="legpoly method must be 1, 2 or 3"):
        legpoly(degrees, 'norm', empty)
    assert jnp.array_equal(default, legpoly(degrees, [-1, 1], False, 2)(x))
    assert jnp.array_equal(normalized, legpoly(degrees, [-1, 1], 'norm', 2)(x))
