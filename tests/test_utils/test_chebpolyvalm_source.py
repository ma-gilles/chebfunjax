"""Source scalar/vector contracts for chebpolyvalm.m at 7574c77."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.polynomials import chebpolyvalm

jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize("p", [3.0, 3.0+2.0j])
def test_scalar_coefficient(p):
    A = jnp.array([[0.0, 2.0], [0.0, 1.0]])
    np.testing.assert_array_equal(chebpolyvalm(p, A), p*np.eye(2))

@pytest.mark.parametrize("x", [0.5, 0.5+0.25j])
def test_scalar_matrix_argument(x):
    actual = chebpolyvalm([2.0, 3.0, 4.0], x)
    assert actual.shape == (1, 1)
    np.testing.assert_allclose(actual, [[2*(2*x*x-1)+3*x+4]], rtol=0, atol=1e-15)

@pytest.mark.parametrize("p", [[], jnp.empty((0, 0)), jnp.ones((1, 2, 3))])
def test_invalid_coefficient_shape(p):
    with pytest.raises(ValueError, match="chebpolyvalm:vector"):
        chebpolyvalm(p, jnp.eye(2))

def test_complex_column_vector_and_jit():
    A = jnp.array([[0.0, 1.0j], [0.0, 0.5]])
    p = jnp.array([[2.0+1j], [3.0], [4.0]])
    expected = (2+1j)*(2*A@A-jnp.eye(2))+3*A+4*jnp.eye(2)
    np.testing.assert_allclose(jax.jit(chebpolyvalm)(p, A), expected, rtol=0, atol=1e-15)
