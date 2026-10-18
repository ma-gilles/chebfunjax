"""Independent panel extrema controls for native minimax.m rootsdiff."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._minimax_rootsdiff_source import rootsdiff


def test_native_keeps_endpoint_derivative_roots():
    x = jnp.cos(jnp.pi * jnp.arange(8, -1, -1) / 8)
    result = rootsdiff(x**3 - 3 * x, (-1.0, 1.0), lambda z: z**3 - 3 * z)
    np.testing.assert_allclose(jnp.sort(result), [-1.0, 1.0], rtol=0, atol=4e-15)


def test_native_T_tail_resampling_and_seven_exact_extrema():
    x = jnp.cos(jnp.pi * jnp.arange(8, -1, -1) / 8)
    seen = []

    def error(z):
        seen.append(z.size)
        return 1 + jnp.cos(8 * jnp.arccos(z))

    result = rootsdiff(1 + jnp.cos(8 * jnp.arccos(x)), (-1.0, 1.0), error)
    assert seen == [17]
    np.testing.assert_allclose(
        jnp.sort(result), jnp.sort(jnp.cos(jnp.pi * jnp.arange(1, 8) / 8)), rtol=0, atol=2e-14
    )


def test_native_constant_derivative_has_no_candidate_extrema():
    x = jnp.cos(jnp.pi * jnp.arange(8, -1, -1) / 8)
    assert rootsdiff(1 + x, (-1.0, 1.0), lambda z: 1 + z).size == 0


@pytest.mark.parametrize("amplitude", [1.0, -3.0, 1e-100, 1e100])
@pytest.mark.parametrize("domain", [(-1.0, 1.0), (2.0, 5.0)])
def test_scaled_quartic_extrema_on_affine_panel(amplitude, domain):
    # T4'(t) = 16*t*(2*t**2-1): the three roots are independent
    # of amplitude or the added constant 2. The constant keeps the native
    # tail/DC convergence ratio well defined. Exercise both the coefficient truncation and colleague
    # branches without using another numerical root finder as the oracle.
    a, b = domain
    t = jnp.cos(jnp.pi * jnp.arange(8, -1, -1) / 8)
    requests = []

    def error(x):
        requests.append(x.size)
        z = (2*x-a-b)/(b-a)
        return amplitude*(8*z**4-8*z**2+3)

    values = amplitude*(8*t**4-8*t**2+3)
    result = rootsdiff(values, domain, error)
    expected = (a+b)/2 + np.asarray([-1/np.sqrt(2), 0, 1/np.sqrt(2)])*(b-a)/2
    assert requests == []
    np.testing.assert_allclose(jnp.sort(result), expected, rtol=0, atol=4e-15)
