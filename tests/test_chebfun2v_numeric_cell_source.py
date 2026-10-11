"""Supplemental numeric/cell constructor contracts, native Chebfun 7574c77.

Provenance: @chebfun2v/chebfun2v.m cell unwrapping and double dispatch.
Python lists/tuples adapt MATLAB cells; numeric matrices use JAX arrays.
These analytic checks supplement, rather than replace, native constructor tests.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


@pytest.mark.parametrize("domain", [(-1., 1., -1., 1.), (-2., 3., 0., 1.)])
def test_cell_callable_and_numeric_components(domain):
    field = Chebfun2v.from_functions([lambda x, y: x + y, 2.], domain=domain)
    x = jnp.asarray([domain[0], .125, domain[1]])
    y = jnp.asarray([domain[2], .25, domain[3]])
    assert field.n_components == 2
    assert field.domain == domain
    assert jnp.max(jnp.abs(field.components[0](x, y) - (x + y))) < 100*jnp.finfo(float).eps
    assert jnp.max(jnp.abs(field.components[1](x, y) - 2.)) < 100*jnp.finfo(float).eps


def test_complex_numeric_components_with_vectorize():
    field = Chebfun2v.from_functions(2., -3.+4j, 0., vectorize=True)
    for component, value in zip(field.components, [2., -3.+4j, 0.]):
        assert jnp.max(jnp.abs(component(.125, -.25) - value)) < 100*jnp.finfo(float).eps


def test_numeric_matrix_orientation():
    # Native numeric input is sampled on the ascending tensor Chebyshev grid.
    x = -jnp.cos(jnp.pi*jnp.arange(5)/4)
    y = -jnp.cos(jnp.pi*jnp.arange(3)/2)
    values = x[None, :]**2 + 2*y[:, None]
    field = Chebfun2v.from_functions(values)
    xx, yy = jnp.asarray([-.8, .1, .9]), jnp.asarray([.7, -.3, .2])
    assert jnp.max(jnp.abs(field.components[0](xx, yy) - (xx**2 + 2*yy))) < 100*jnp.finfo(float).eps


def test_cell_component_count_is_checked_after_unwrapping():
    with pytest.raises(ValueError, match="1, 2 or 3"):
        Chebfun2v.from_functions([1., 2., 3., 4.])
