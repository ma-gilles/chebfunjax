"""Distinguishing native complex/real/isreal controls (Chebfun 7574c77).

These are added controls; native test_complex/test_isreal retain their own
original predicates. Complex-zero storage follows executable native source.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d import _unary
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

DOMAIN = (-1., 1.)*3
TOL = 1000*jnp.finfo(jnp.float64).eps


def field(kind='real', domain=DOMAIN):
    one = Chebtech2.from_coeffs(jnp.ones(1))
    factors = [[Chebtech2.from_coeffs(jnp.asarray([2., 1.]))], [one], [one]]
    core = jnp.ones((1, 1, 1))
    if kind in ('core_zero', 'core_imag'):
        core = core.astype(jnp.complex128)*(1j if kind == 'core_imag' else 1.)
    elif kind.startswith('factor'):
        axis = int(kind[-1])
        factors[axis] = [Chebtech2.from_coeffs(factors[axis][0].coeffs.astype(jnp.complex128))]
    elif kind == 'real_product':
        factors[0] = [Chebtech2.from_coeffs(jnp.asarray([1j]))]
        factors[1] = [Chebtech2.from_coeffs(jnp.asarray([1j]))]
    return Chebfun3(*factors, core, domain)


@pytest.mark.parametrize('kind', ['real', 'core_zero', 'core_imag',
                                 'factor0', 'factor1', 'factor2', 'real_product'])
def test_literal_storage(kind):
    assert field(kind).isreal() is (kind == 'real')


@pytest.mark.parametrize('real_flag', [False, True])
def test_trig_uses_real_flag(real_flag):
    tech = Trigtech.from_coeffs(jnp.asarray([1.+0j]), is_real=real_flag)
    f = Chebfun3([tech], [tech], [tech], jnp.ones((1, 1, 1)), DOMAIN)
    assert f.isreal() is real_flag


def test_empty_real_returns_before_compose(monkeypatch):
    f = Chebfun3.empty()
    monkeypatch.setattr(Chebfun3, 'compose', lambda *args: pytest.fail('empty reached compose'))
    assert f.isreal() is True
    assert f.real() is f


@pytest.mark.parametrize('kind', ['core_zero', 'core_imag', 'factor0'])
@pytest.mark.parametrize('side', [0, 1])
def test_complex_rejects_before_domain(kind, side):
    operands = [field(domain=(-2., 2.)*3), field(domain=(-2., 2.)*3)]
    operands[side] = field(kind)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:complex:notReal1: Inputs must be real'):
        Chebfun3.complex(*operands)


@pytest.mark.parametrize('side', [0, 1])
def test_numeric_complex_zero_rejected(side):
    operands = [field(), field()]
    operands[side] = 1.+0j
    with pytest.raises(ValueError, match='complex:notReal1'):
        Chebfun3.complex(*operands)


@pytest.mark.parametrize('name', ['sqrt', 'log', 'abs'])
def test_complex_zero_bypasses_real_sign_guard(monkeypatch, name):
    f = field('core_zero')
    marker = object()
    monkeypatch.setattr(_unary, '_single_sign_test', lambda *args: pytest.fail('complex storage reached real sign check'))
    monkeypatch.setattr(Chebfun3, 'compose', lambda *args: marker)
    assert getattr(f, name)() is marker


@pytest.mark.parametrize('side', [0, 1])
def test_actual_numeric_component(side):
    f = field()
    result = Chebfun3.complex(3., f) if side == 0 else Chebfun3.complex(f, 3.)
    x = jnp.asarray([-.7, .1, .8])
    expected = 3.+1j*(2+x) if side == 0 else (2+x)+3j
    assert jnp.max(jnp.abs(result(x, 0*x, 0*x)-expected)) < TOL


@pytest.mark.parametrize('side', ['left', 'right', 'both'])
def test_actual_empty_complex(side):
    a = Chebfun3.empty() if side in ('left', 'both') else field()
    b = Chebfun3.empty() if side in ('right', 'both') else field()
    assert Chebfun3.complex(a, b).isempty()


def test_actual_real_domain_mismatch_delegates_plus():
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:plus:domain'):
        Chebfun3.complex(field(), field(domain=(-2., 2.)*3))


def test_actual_periodic_real_preserves_compose_technology():
    f = chebfun3(lambda x, y, z: jnp.cos(jnp.pi*x)+1j*jnp.sin(jnp.pi*y), trig=True)
    g = f.real()
    x = jnp.asarray([-.7, .1, .8])
    assert g.isPeriodicTech() and g.isreal()
    assert jnp.max(jnp.abs(g(x, x, x)-jnp.cos(jnp.pi*x))) < TOL


def test_actual_complex_zero_infinity_norm():
    f = field('core_zero')
    assert not f.isreal()
    assert jnp.abs(f.norm('inf')-3.) < TOL
