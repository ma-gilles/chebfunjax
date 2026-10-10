"""Empty property types and source validation order."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2


def get_property(f, name, route):
    return f.get(name) if route == 'get' else f.subsref({'type': '.', 'subs': name})


@pytest.mark.parametrize('name', ['cols', 'rows', 'tubes', 'core', 'domain'])
@pytest.mark.parametrize('route', ['get', 'subsref'])
def test_empty_native_numeric_properties(name, route):
    f = Chebfun3.empty()
    value = get_property(f, name, route)
    assert isinstance(value, jax.Array)
    assert value.shape == (0, 0)
    assert value.dtype == jnp.float64
    assert f.isempty()


@pytest.mark.parametrize('name', ['invalid', 'Cols'])
@pytest.mark.parametrize('route', ['get', 'subsref'])
@pytest.mark.parametrize('empty', [False, True])
def test_invalid_property_still_errors(name, route, empty):
    if empty:
        f = Chebfun3.empty()
    else:
        one = Chebtech2.from_coeffs(jnp.ones(1))
        f = Chebfun3([one], [one], [one], jnp.ones((1, 1, 1)), (-1., 1.)*3)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:get:propName:'):
        get_property(f, name, route)
