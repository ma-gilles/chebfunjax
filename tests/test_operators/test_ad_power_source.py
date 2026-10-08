"""Source power branch, zero-Jacobian and updateDomain controls.

Provenance: @adchebfun/adchebfun.m, power/pow2/updateDomain,
Chebfun 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun


@pytest.mark.parametrize('operation', [lambda a: a**3, lambda a: 2**a, lambda a: a.pow2()])
def test_source_zero_jacobian_is_linear(operation):
    constant = ADChebfun(chebfun(lambda x: 2+x))**0
    result = operation(constant)
    assert result.is_linear
    assert float(result.jacobian.apply(chebfun(1.)).norm(jnp.inf)) == 0


@pytest.mark.parametrize('base', [2., -2., 2.+1j])
def test_source_scalar_base_complex_log(base):
    u = ADChebfun(chebfun(lambda x: .55+.01*x))
    p = chebfun(lambda x: .05+.002*x)
    result = base**u
    x = jnp.linspace(-1., 1., 21)
    logarithm = jnp.log(jnp.asarray(base, dtype=jnp.complex128))
    expected = jnp.exp(logarithm*u.func(x))
    assert float(jnp.max(jnp.abs(result.func(x)-expected))) < 1e-12
    assert float(jnp.max(jnp.abs(result.jacobian.apply(p)(x)-logarithm*expected*p(x)))) < 1e-12


@pytest.mark.parametrize('operation', [lambda a: a**2, lambda a: 2**a])
def test_source_power_updates_breakpoints_without_mutating_input(operation):
    u = ADChebfun(chebfun(lambda x: .55+.01*x, domain=[-1, 0, 1]))
    old_domain = u.domain
    old_jacobian_domain = u.jacobian.domain
    result = operation(u)
    assert result.domain == (-1., 0., 1.)
    assert result.jacobian.domain == result.domain
    assert u.domain == old_domain and u.jacobian.domain == old_jacobian_domain
    p = chebfun(lambda x: .05+.002*x, domain=[-1, 0, 1])
    # Compare with an independent centered directional difference on this
    # piecewise domain, in addition to the exact metadata invariant.
    h = 1e-5
    finite_difference = (operation(u+h*p).func-operation(u-h*p).func)/(2*h)
    assert float((result.jacobian.apply(p)-finite_difference).norm(jnp.inf)) < 1e-9


def test_source_numeric_zero_one_early_returns():
    u = ADChebfun(chebfun(lambda x: 2+x, domain=[-1, 0, 1]))
    assert u**1 is u
    result = u**0
    assert result.domain == u.domain
    assert result.is_linear and result.jacobian.iszero
    assert float((result.func-1).norm(jnp.inf)) == 0


def test_source_pow2_scalar_seed_column():
    u = ADChebfun(chebfun(lambda x: .55+.01*x, domain=[-1, 0, 1])).seed(1, 0)
    result = u.pow2()
    assert isinstance(result.jacobian, Chebfun)
    assert result.domain == (-1., 0., 1.)
    expected = result.func*jnp.log(2.)
    assert float((result.jacobian-expected).norm(jnp.inf)) < 1e-12
