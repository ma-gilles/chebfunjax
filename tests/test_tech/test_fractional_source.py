"""JAX fractional coefficient path, @chebtech/@singfun/fracInt.m7574c77."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma

from chebfunjax import chebfun
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("mu", [0.5, 0.37, 0.75])
@pytest.mark.parametrize("b", [0.0, 0.3])
def test_weighted_complex_constant(Tech, mu, b):
    f = Tech(coeffs=jnp.array([1.+2j]), ishappy=False)
    g = f.fracInt(mu, b)
    expected = (1.+2j) * gamma(b + 1) / gamma(b + mu + 1)
    np.testing.assert_allclose(g(jnp.array([-.8, 0., .6])), expected,
                               rtol=0, atol=100*np.finfo(float).eps*abs(expected))
    assert g.ishappy is False
    assert isinstance(g, Tech)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_half_integral_jit_linear_complex(Tech):
    c = jnp.array([1.+2j, .25-.5j])
    run = jax.jit(lambda coefficients: Tech.from_coeffs(coefficients).fracInt(.5).coeffs)
    g = Tech.from_coeffs(run(c))
    x = jnp.array([-.9, -.2, .8])
    expected = (c[0]-c[1])/gamma(1.5) + c[1]*(x+1)/gamma(2.5)
    np.testing.assert_allclose(g(x), expected, rtol=0, atol=100*np.finfo(float).eps)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_native_array_error(Tech):
    with pytest.raises(ValueError, match="CHEFBFUN:CHEBTECH:fracInt:arrayvalued"):
        Tech.from_coeffs(jnp.ones((2, 2))).fracInt(.5)


@pytest.mark.parametrize("mu", [0.5, 0.37])
def test_complex_physical_domain(mu):
    f = chebfun(lambda x: (1.+2j)*jnp.ones_like(x), domain=(2., 5.))
    x = jnp.array([2.1, 3., 4.8])
    expected = (1.+2j)*(x-2)**mu/gamma(1+mu)
    np.testing.assert_allclose(f.fracInt(mu)(x), expected, rtol=0,
                               atol=100*np.finfo(float).eps*float(jnp.max(jnp.abs(expected))))


def test_singfun_simplifies_right_integer_before_integrating():
    f = Singfun(Chebtech2.from_coeffs(jnp.array([1.])), (0., 1.))
    x = jnp.array([-.8, 0., .7])
    expected = 2*(x+1)**.5/gamma(1.5) - (x+1)**1.5/gamma(2.5)
    np.testing.assert_allclose(f.fracInt(.5)(x), expected, rtol=0, atol=100*np.finfo(float).eps)
    with pytest.raises(ValueError, match="CHEBFUN:SINGFUN:fracInt:exponents"):
        Singfun(f.smoothPart, (0., .2)).fracInt(.5)
