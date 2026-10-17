"""Scalar one-column storage follows native size(f,2), not array rank."""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts


@pytest.mark.parametrize('Tech,kind', [(Chebtech1, 1), (Chebtech2, 2)])
@pytest.mark.parametrize('mode', ['inner_singleton', 'outer_singleton', 'both_singleton'])
def test_scalar_column_composition(Tech, kind, mode):
    inner = Tech.from_function(lambda x: x**2)
    outer = Tech.from_function(jnp.sin)
    if mode == 'inner_singleton':
        inner = Tech.from_coeffs(inner.coeffs[:, None])
        outer = Tech.from_function(lambda x: jnp.column_stack((jnp.sin(x), jnp.cos(x))))
    elif mode == 'outer_singleton':
        inner = Tech.from_function(lambda x: jnp.column_stack((x, x**2)))
        outer = Tech.from_coeffs(outer.coeffs[:, None])
    else:
        inner = Tech.from_coeffs(inner.coeffs[:, None])
        outer = Tech.from_coeffs(outer.coeffs[:, None])
    result = inner.compose(outer)
    x = chebpts(result.n, kind)
    values = result.coeffs2vals(result.coeffs)
    expected = (jnp.column_stack((jnp.sin(x**2), jnp.cos(x**2))) if mode == 'inner_singleton'
                else jnp.column_stack((jnp.sin(x), jnp.sin(x**2))) if mode == 'outer_singleton'
                else jnp.sin(x**2))
    assert values.shape == expected.shape
    error = np.linalg.norm(np.asarray(values - expected), ord=np.inf)
    assert error < 10 * float(jnp.max(result.vscale_columns)) * np.finfo(float).eps
