"""Native scalar-column factoring/evaluation with Python query shape adapter."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.quadrature import chebpts, trigpts


def samples(tech):
    if tech is Trigtech:
        x = trigpts(9)[0]
        return x, 2+jnp.cos(jnp.pi*x)
    x = chebpts(3, kind=1 if tech is Chebtech1 else 2)
    return x, 3+x


def smooth_value(tech, x):
    return 2+jnp.cos(jnp.pi*x) if tech is Trigtech else 3+x


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
@pytest.mark.parametrize('shape', ['scalar', 'vector', 'column'])
@pytest.mark.parametrize('multiplier', [1., 1.+2j])
def test_numeric_column_values_and_shapes(tech, shape, multiplier):
    _, values = samples(tech)
    values = (multiplier*values)[:, None]
    f = Singfun.constructor(values, {'exponents': (.5, -.5)}, {'tech': tech})
    reference = tech.from_values(values)
    np.testing.assert_array_equal(f.smoothPart.coeffs, reference.coeffs)
    x = jnp.array(.25) if shape == 'scalar' else jnp.array([-.75, -.25, .25, .75])
    if shape == 'column':
        x = x[:, None]
    actual = f(x)
    assert actual.shape == x.shape
    expected = multiplier*smooth_value(tech, x)*(1+x)**.5*(1-x)**-.5
    assert float(jnp.max(jnp.abs(actual-expected))) < 100*jnp.finfo(float).eps


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_callable_column_factoring(tech):
    exponent = 0. if tech is Trigtech else -.5

    def operator(x):
        x = jnp.atleast_1d(x)
        return (smooth_value(tech, x)*(1+x)**exponent)[:, None]
    f = Singfun.constructor(operator, {'exponents': (exponent, 0.)},
                           {'tech': tech, 'fixedLength': 17})
    x = jnp.array([-.75, -.25, .25, .75])
    assert f.smoothPart.coeffs.shape == (17, 1)
    assert f(x).shape == x.shape
    assert float(jnp.max(jnp.abs(f(x)-operator(x)[:, 0]))) < 100*jnp.finfo(float).eps


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2, Trigtech])
def test_smooth_column_identity_and_ad(tech):
    _, values = samples(tech)
    smooth = tech.from_values(values[:, None])
    coeffs = np.asarray(smooth.coeffs).copy()
    f = Singfun.constructor(smooth, {'exponents': (.5, -.5)})
    assert f.smoothPart is smooth
    np.testing.assert_array_equal(smooth.coeffs, coeffs)
    def analytic(x):
        return smooth_value(tech, x)*(1+x)**.5*(1-x)**-.5
    assert abs(float(jax.grad(f)(jnp.array(.2)))-float(jax.grad(analytic)(jnp.array(.2)))) < 100*jnp.finfo(float).eps
    scalar = Singfun.constructor(values, {'exponents': (.5, -.5)}, {'tech': tech})
    x = jnp.array([-.4, .3])
    assert scalar(x).shape == f(x).shape == x.shape
    assert float(jnp.max(jnp.abs(scalar(x)-f(x)))) < 100*jnp.finfo(float).eps
