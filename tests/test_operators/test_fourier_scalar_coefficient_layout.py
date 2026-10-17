"""Scalar coefficient layouts in native Fourier collocation multiplication.

Provenance: Chebfun7574c77 @valsDiscretization/mult.m constructs
DIAG(toValues(disc,f)); @trigcolloc/toValues.m evaluates f on the grid.
These authored controls supplement the unchanged original quantumstates test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.chebop import _FourierProxy


@pytest.mark.parametrize('layout', ['flat', 'column', 'row'])
@pytest.mark.parametrize('complex_values', [False, True])
def test_scalar_sample_layout(layout, complex_values):
    n = 8
    grid = jnp.arange(n, dtype=jnp.float64)/4
    matrix = (jnp.arange(n*n, dtype=jnp.float64).reshape(n, n)-13)/8
    values = 1 + grid + (1j*grid/2 if complex_values else 0)
    shaped = {'flat': values, 'column': values[:, None], 'row': values[None, :]}[layout]
    proxy = _FourierProxy(n, 2., matrix, grid=grid)
    observed = proxy._scale(lambda x: shaped).mat
    assert observed.shape == (n, n)
    np.testing.assert_array_equal(observed, jnp.diag(values) @ matrix)


def test_sample_column_remains_jittable():
    n = 8
    proxy = _FourierProxy(n, 2., jnp.eye(n), grid=jnp.arange(n))
    scale = jax.jit(lambda values: proxy._scale(values).mat)
    values = (jnp.arange(n, dtype=jnp.float64)+1)[:, None]
    np.testing.assert_array_equal(scale(values), jnp.diag(values[:, 0]))


def test_true_matrix_coefficient_rejected():
    proxy = _FourierProxy(8, 2., jnp.eye(8), grid=jnp.arange(8))
    with pytest.raises(ValueError, match='one value per grid point'):
        proxy._scale(jnp.ones((8, 2)))
