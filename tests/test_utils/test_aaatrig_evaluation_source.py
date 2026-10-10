"""Native revaltrig.m infinity limits; Chebfun commit 7574c77."""
import importlib

import jax
import jax.numpy as jnp
import pytest

module = importlib.import_module('chebfunjax.utils.aaa')


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_closed_form_imaginary_infinity_limits(form):
    # Two supports 0,pi, weights1,2 and values2,5 yield independent limits:
    # odd (2-/+10i)/(1-/+2i)=4.4-/+1.2i; even12/3=4.
    points = jax.lax.complex(jnp.zeros(2), jnp.array([jnp.inf, -jnp.inf]))
    r = module._make_trig_callable(jnp.array([0., jnp.pi]),
                                    jnp.array([2., 5.]), jnp.array([1., 2.]), form)
    expected = jnp.array([4.4-1.2j, 4.4+1.2j] if form == 'odd' else [4., 4.])
    actual = r(points.reshape(1, 2))
    assert actual.shape == (1, 2)
    assert jnp.max(jnp.abs(actual-expected)) < 2e-15
    assert jnp.isnan(r(jnp.array(jnp.nan)))
    assert jnp.isnan(r(jnp.array(jnp.inf)))
    assert jnp.isnan(r(jnp.array(-jnp.inf)))
    assert jnp.array_equal(r(jnp.array([0., jnp.pi])), jnp.array([2., 5.]))


@pytest.mark.parametrize('form', ['odd', 'even'])
def test_native_public_special_evaluations(form):
    # Original test_aaatrig.m predicates1,2,3 and the overwritten first4.
    z = jnp.linspace(0., 2*jnp.pi, 1000)
    f = jnp.exp(jnp.cos(z))
    r, *_ = module.aaatrig(f, z, form=form)
    tol = 1e4*jnp.finfo(jnp.float64).eps
    assert jnp.max(jnp.abs(f-r(z))) < 2*tol
    assert jnp.isnan(r(jnp.array(jnp.nan)))
    assert jnp.isnan(r(jnp.array(jnp.inf)))
    points = jax.lax.complex(jnp.zeros(2), jnp.array([jnp.inf, -jnp.inf]))
    assert jnp.all(~jnp.isinf(r(points)))
    # Complement native weak assertion: NaN is not a finite limit.
    assert jnp.all(jnp.isfinite(r(points)))
