"""Beta scales at fractional-kernel lengths; independent scalar oracles."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import beta

from chebfunjax.tech._fractional import _fractional_beta


@pytest.mark.parametrize("shift, mu", [(0., .3), (.3, .7), (0., .7),
                                     (-.30000000000000004, .30000000000000004)])
def test_captured_parameter_scales(shift, mu):
    a = np.arange(23, dtype=np.float64) + shift + 1
    expected = beta(a, mu)
    actual = _fractional_beta(jnp.asarray(a), mu)
    np.testing.assert_allclose(actual, expected, rtol=30*np.finfo(float).eps, atol=0)


def test_integer_identity_and_symmetry():
    n = jnp.arange(1, 40, dtype=jnp.float64)
    expected = 1 / n
    np.testing.assert_allclose(_fractional_beta(1., n), expected,
                               rtol=30*np.finfo(float).eps, atol=0)
    np.testing.assert_array_equal(_fractional_beta(1., n), _fractional_beta(n, 1.))


def test_jit_beta_scales():
    a = jnp.array([.7, 7.9, 8., 8.1, 23.])
    actual = jax.jit(_fractional_beta)(a, jnp.asarray(.3))
    np.testing.assert_allclose(actual, beta(np.asarray(a), .3),
                               rtol=30*np.finfo(float).eps, atol=0)
