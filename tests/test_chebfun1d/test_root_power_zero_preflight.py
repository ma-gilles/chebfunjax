"""Bounded native zero semantics before fractional POWER, pin7574c77.

Native @chebfun/power.m columnPower does not evaluate a norm before root
insertion. @chebtech/iszero and @singfun/iszero test exact smooth coefficients;
tiny nonzero values and isolated pointValues must survive POWER.
"""
import importlib.util
import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = jnp.finfo(jnp.float64).eps


def column(kind,c):
    return Chebfun(funs=[_Piece(tech=kind.from_coeffs(jnp.asarray(c)),interval=(-1.,1.))],domain=Domain((-1.,1.)))


def count_norm(monkeypatch):
    calls=[]
    original=Chebfun.norm
    def observed(self,*args,**kwargs):
        calls.append(self)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Chebfun,'norm',observed)
    return calls


@pytest.mark.parametrize('kind',[Chebtech1,Chebtech2])
@pytest.mark.parametrize('b',[.5,-.5])
def test_exact_zero_keeps_positive_result_and_negative_error(monkeypatch,kind,b):
    f=column(kind,[[0.],[0.]])
    calls=count_norm(monkeypatch)
    if b > 0:
        g=f**b
        assert jnp.all(g(jnp.asarray([-.6,0.,.8]))==0)
        assert jnp.all(g.point_values==0)
    else:
        with pytest.raises(ValueError,match='CHEBFUN:CHEBTECH:extrapolate:nansInfs'):
            f**b
    assert not calls


@pytest.mark.parametrize('kind',[Chebtech1,Chebtech2])
@pytest.mark.parametrize('b',[.5,-.5])
def test_tiny_nonzero_complex_values_do_not_underflow_to_zero(monkeypatch,kind,b):
    amplitude=1e-170
    f=column(kind,[amplitude*(1+2j)])
    calls=count_norm(monkeypatch)
    g=f**b
    root=jnp.sqrt(amplitude)*(jnp.sqrt((jnp.sqrt(5.)+1)/2)+1j*jnp.sqrt((jnp.sqrt(5.)-1)/2))
    expected=root if b > 0 else 1/root
    error=jnp.max(jnp.abs(g(jnp.asarray([-.6,.2,.8]))-expected))
    assert error < 100*EPS*jnp.abs(expected)
    assert jnp.all(jnp.abs(g.point_values)>0)
    assert not calls


@pytest.mark.parametrize('value',[4.,jnp.inf,jnp.nan])
def test_isolated_point_values_prevent_zero_shortcut(monkeypatch,value):
    f=column(Chebtech2,[0.]).set_point_values(jnp.asarray([value,0.]))
    calls=count_norm(monkeypatch)
    g=f**.5
    assert jnp.all(g(jnp.asarray([-.5,0.,.5]))==0)
    actual=g(-1.)
    if jnp.isnan(value):
        assert jnp.isnan(actual)
    elif jnp.isinf(value):
        assert jnp.isinf(actual)
    else:
        assert actual==2.
    assert not calls


@pytest.mark.parametrize('kind',[Chebtech1,Chebtech2])
def test_nonzero_smooth_factor_is_not_zero_when_point_values_are_zero(monkeypatch,kind):
    f=column(kind,[4.]).set_point_values(jnp.zeros(2))
    calls=count_norm(monkeypatch)
    g=f**.5
    assert jnp.max(jnp.abs(g(jnp.asarray([-.6,.2,.8]))-2)) < 100*EPS
    assert jnp.all(g.point_values==0)
    assert not calls


@pytest.mark.parametrize('kind',[Chebtech1,Chebtech2])
def test_complex_singular_endpoint_power_preserves_pole_and_exponents(monkeypatch,kind):
    tech=Singfun(kind.from_coeffs(jnp.asarray([2+1j])),(-.5,0.))
    f=Chebfun(funs=[_Piece(tech=tech,interval=(-1.,1.))],domain=Domain((-1.,1.)))
    calls=count_norm(monkeypatch)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        g=f**.5
    x=jnp.asarray([-.6,.2,.8])
    root=jnp.sqrt((jnp.sqrt(5.)+2)/2)+1j*jnp.sqrt((jnp.sqrt(5.)-2)/2)
    expected=root*(1+x)**-.25
    assert jnp.max(jnp.abs(g(x)-expected)) < 100*EPS*jnp.max(jnp.abs(expected))
    assert isinstance(g.funs[0].tech,Singfun)
    assert g.funs[0].tech.exponents==(-.25,0.)
    # Native columnPower maps stored pointValues through POWER. Preserve
    # this operation, including the existing nonfinite complex primitive.
    source_values=jnp.power(f.point_values.astype(jnp.complex128),.5)
    assert jnp.array_equal(g.point_values,source_values,equal_nan=True)
    assert not jnp.isfinite(g(-1.))
    assert not [m for m in messages if 'minandmax:complexFail' in str(m.message)]
    assert not calls


def test_true_array_splits_columns_without_scalar_norm_preflight(monkeypatch):
    f=Chebfun.from_coeffs(jnp.asarray([[1+2j,3+.5j]]))
    calls=count_norm(monkeypatch)
    g=f**.5
    x=jnp.asarray([-.6,.2,.8])
    expected=jnp.sqrt(jnp.asarray([1+2j,3+.5j]))
    assert g(x).shape==(3,2)
    assert jnp.max(jnp.abs(g(x)-expected)) < 100*EPS*jnp.max(jnp.abs(expected))
    assert not calls


def test_periodic_representation_retains_existing_norm_fallback(monkeypatch):
    f=cj.chebfun(lambda x:2+jnp.zeros_like(x),domain=(-jnp.pi,jnp.pi),trig=True)
    calls=count_norm(monkeypatch)
    g=f**.5
    assert jnp.max(jnp.abs(g(jnp.asarray([-.6,.2,.8]))-jnp.sqrt(2.))) < 100*EPS
    assert calls


def test_original33_keeps_literal_bound_without_zero_preflight_warning(monkeypatch):
    path=Path(__file__).parents[1]/'test_matlab_port/chebfun/test_power_original41.py'
    spec=importlib.util.spec_from_file_location('literal_power33',path)
    literal=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(literal)
    calls=count_norm(monkeypatch)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        literal.test_literal_finite(33)
    assert not [m for m in messages if 'minandmax:complexFail' in str(m.message)]
    assert not calls
