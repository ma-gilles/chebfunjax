"""Independent interpolation/shape controls for the JAX PCHIP migration."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.utils._pchip import pchip_coefficients
from chebfunjax.utils._spline import spline_evaluate


@pytest.mark.parametrize("disabled", [True, False])
@pytest.mark.parametrize("n", [2, 3, 7])
@pytest.mark.parametrize("complex_values", [False, True])
def test_nonuniform_array_coefficients_and_extrapolation(disabled, n, complex_values):
    x = np.linspace(-1, 1, n)**3
    y = np.stack((np.exp(x), np.sin(4*x)), axis=1)
    if complex_values:
        y = y + 1j * np.stack((x*x, np.cos(3*x)), axis=1)
    z = np.linspace(-1.2, 1.2, 91)
    expected = PchipInterpolator(x, y.real)(z)
    if complex_values:
        expected = expected + 1j * PchipInterpolator(x, y.imag)(z)
    with jax.disable_jit(disabled):
        actual = spline_evaluate(x, pchip_coefficients(x, y), z)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("transpose", [True, False])
def test_public_complex_and_requested_domain(transpose):
    x = jnp.asarray([-1., -.3, .2, 1.])
    y = jnp.stack((x**2 + 1j*x, jnp.sin(x) + 2j*x**2), axis=1)
    f = Chebfun.pchip(x, y.T if transpose else y, (-.7, .1, 1.3))
    assert f.domain.breakpoints == (-.7, -.3, .1, .2, 1., 1.3)
    assert len(f) == 20
    z = np.linspace(-.7, 1.3, 53)
    expected = PchipInterpolator(x, np.asarray(y).real)(z) + 1j*PchipInterpolator(x, np.asarray(y).imag)(z)
    np.testing.assert_allclose(f(z), expected, rtol=2e-14, atol=2e-14)


def test_plateaus_are_constant_and_slopes_do_not_overshoot():
    x = jnp.arange(-3., 4.)
    y = jnp.asarray([-1., -1., -1., 0., 1., 1., 1.])
    c = pchip_coefficients(x, y)
    assert bool(jnp.all(c[jnp.asarray([0, 1, 4, 5]), 1:] == 0))
    f = Chebfun.pchip(x, y)
    z = jnp.linspace(-3, 3, 601)
    assert float(jnp.min(f(z))) >= -1 - 2e-15
    assert float(jnp.max(f(z))) <= 1 + 2e-15
    assert float(jnp.min(f.diff()(z))) >= -2e-14
