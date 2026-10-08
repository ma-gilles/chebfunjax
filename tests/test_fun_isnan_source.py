"""Source coefficient/endpoint predicates and complex exponent cancellation.

Provenance
----------
MATLAB source : @chebtech/isnan.m, @classicfun/isnan.m, @singfun/isnan.m,
    @singfun/cancelExponents.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('kind', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('coeffs,expected', [([1., 2.], False),
    ([float('inf')], False), ([complex(1, float('nan'))], True),
    ([[1., 2.], [0., float('nan')]], True)])
def test_tech_all_coefficients(kind, coeffs, expected):
    assert kind.from_coeffs(jnp.asarray(coeffs)).isnan() is expected


@pytest.mark.parametrize('kind', [Chebtech1, Chebtech2, Bndfun, Singfun])
def test_empty_predicate(kind):
    assert not kind.empty().isnan()


@pytest.mark.parametrize('kind,domain', [(Bndfun, Domain((-2., 7.))),
                                       (Unbndfun, Domain((2., float('inf'))))])
def test_wrapper_uses_onefun(kind, domain):
    tech = Chebtech2.from_coeffs(jnp.asarray([float('nan')]))
    f = kind(tech, domain, 'right_inf') if kind is Unbndfun else kind(tech, domain)
    assert f.isnan()


def test_singular_endpoint_nan_even_with_finite_coefficients():
    smooth = Chebtech2.from_coeffs(jnp.asarray([1., -1.]))
    assert not smooth.isnan()
    assert Singfun(smooth, (0., -1.)).isnan()
    assert not Singfun(Chebtech2.from_coeffs(jnp.asarray([1.])), (0., -1.)).isnan()


def test_complex_boundary_root_cancellation_preserves_phase():
    smooth = Chebtech2.from_coeffs(jnp.asarray([1j, 1j]))
    f = Singfun(smooth, (-.5, 0.))
    g = f.cancelExponents()
    assert g.exponents == (.5, 0.)
    x = jnp.asarray([-.7, -.1, .8])
    assert float(jnp.max(jnp.abs(g(x) - 1j*jnp.sqrt(1+x)))) < 8*jnp.finfo(jnp.float64).eps
