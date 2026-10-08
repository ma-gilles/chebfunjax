"""Independent numerical controls for the JAX Bessel J adapter.

Provenance: DLMF 10.2, 10.6, 10.9; MATLAB @chebfun/besselj.m,
Chebfun commit: 7574c77. SciPy is an independent test oracle only.
These controls qualify sampled orders -10..50 and arguments through radius100,
not universal accuracy or extreme exponential scaling.
"""
# uses-numpy: independent SciPy oracle and host-side error measurements.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jn_zeros, jv, jvp

from chebfunjax.utils.besselj import besselj


@pytest.mark.parametrize('order', [-10., -2.5, -.1, 0., .1, 1., 2.5, 3.5, 10., 20., 50.])
def test_independent_real_and_complex_regimes(order):
    radii = np.r_[1e-8, .1, 1., 3.999999, 4., 4.000001,
                  10., 19.99999, 20., 20.00001, 31.99999, 32., 32.00001,
                  49.99999, 50., 50.00001, 75., 100.]
    z = np.r_[radii, radii*np.exp(.7j), radii*1j,
              -radii+.01j, -radii-.01j]
    actual = np.asarray(besselj(order, jnp.asarray(z)))
    expected = jv(order, z)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-13)


@pytest.mark.parametrize('order', [0, 1, 2, 10, 50])
def test_real_kernel_transition_and_zeros(order):
    x = np.r_[0., .01, 3.999999, 4., 4.000001, 31.999999, 32., 32.000001,
              order+np.array([-.000001, 0., .000001]), jn_zeros(order, 4)]
    actual = np.asarray(besselj(order, jnp.asarray(x)))
    np.testing.assert_allclose(actual, jv(order, x), rtol=5e-12, atol=5e-14)


@pytest.mark.parametrize('order', [0., 1., 2.5, 10.])
def test_analytic_jvp_jit_and_vmap(order):
    z = jnp.asarray([.2+.1j, 3.9+.1j, 4.1-.1j, 5+12j, -2+.1j])
    value, derivative = jax.jvp(lambda x: besselj(order, x), (z,), (jnp.ones_like(z),))
    np.testing.assert_allclose(derivative, jvp(order, np.asarray(z)), rtol=5e-12, atol=5e-13)
    nested = jax.jit(jax.vmap(lambda x: besselj(order, x)))(z)
    np.testing.assert_allclose(value, nested, rtol=2e-14, atol=2e-14)


def test_negative_real_principal_branch_and_integer_zero():
    x = jnp.asarray([-8., -1., 0., 1., 8.])
    np.testing.assert_allclose(besselj(2.5, x), jv(2.5, np.asarray(x, complex)), rtol=5e-13, atol=5e-14)
    for order, expected in [(0, 1), (1, 0), (2, 0), (-1, 0)]:
        assert besselj(order, 0.) == expected
    assert jax.grad(lambda x: besselj(1., x))(0.) == .5
