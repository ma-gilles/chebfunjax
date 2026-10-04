"""Controls for paired Clenshaw steps and the odd-degree final step.

Provenance
----------
MATLAB source : @chebtech/clenshaw.m, clenshaw_scl/clenshaw_vec
Chebfun commit: 7574c77
These additional Python JIT/AD controls complement the unchanged five
original assertions in tests/chebtech/test_clenshaw.m.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import _clenshaw


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('complex_data', [False, True])
def test_jitted_columns_at_matrix_points(n, complex_data):
    c = np.arange(1, n + 1, dtype=np.float64)
    x = np.asarray([[-0.75, -0.1], [0.25, 0.8]])
    if complex_data:
        c = c + 1j * c[::-1]
        x = x + 0.2j
    columns = np.column_stack([c, -c, 0.25 * c])
    expected = np.moveaxis(np.polynomial.chebyshev.chebval(x, columns), 0, -1)
    actual = jax.jit(_clenshaw)(jnp.asarray(columns), jnp.asarray(x))
    assert actual.shape == (2, 2, 3)
    assert float(jnp.max(jnp.abs(actual - expected))) < (
        10 * np.finfo(np.float64).eps * np.max(np.abs(expected)))


@pytest.mark.parametrize('n', [2, 3, 6])
def test_coefficient_derivatives_are_chebyshev_basis(n):
    x = 0.3
    c = jnp.arange(1.0, n + 1)
    actual = jax.jit(jax.grad(lambda a: _clenshaw(a, jnp.asarray(x))))(c)
    expected = np.cos(np.arange(n) * np.arccos(x))
    assert float(jnp.max(jnp.abs(actual - expected))) < 10 * np.finfo(np.float64).eps
