"""Independent Fourier-grid identities for the JAX source alias kernel."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import _alias_trigtech


def _evaluate(c, x):
    n = c.shape[0]
    modes = np.arange(-(n//2), (n+1)//2)
    basis = np.exp(1j*np.pi*x[:,None]*modes)
    if n % 2 == 0:
        basis[:,0] = np.cos(np.pi*(n//2)*x)
    return basis@c


@pytest.mark.parametrize('n', [1,2,3,4,9,10])
@pytest.mark.parametrize('m', [1,2,3,4,7,12])
def test_alias_preserves_values_on_target_grid_and_outer_jit(n,m):
    k = np.arange(n,dtype=float)
    c = np.stack((np.cos(k)+1j*np.sin(2*k), np.sin(k)-2j*np.cos(3*k)),axis=1)
    result = jax.jit(lambda values: _alias_trigtech(values,m))(jnp.asarray(c))
    x = -1 + 2*np.arange(m)/m
    expected = _evaluate(c,x)
    actual = _evaluate(np.asarray(result),x)
    assert result.shape == (m,2)
    assert np.max(np.abs(actual-expected)) < 100*np.finfo(float).eps*max(1,np.max(np.abs(expected)))


@pytest.mark.parametrize('n,m', [(2,7),(9,2),(10,3),(9,1)])
def test_vector_and_matrix_column_consistency(n,m):
    c = jnp.arange(n,dtype=jnp.float64)+1j*jnp.cos(jnp.arange(n))
    a = _alias_trigtech(c,m)
    b = _alias_trigtech(c[:,None],m)
    assert a.shape == (m,)
    assert bool(jnp.array_equal(a,b[:,0]))
