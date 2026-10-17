"""Single effective column with native (n,1) coefficient storage.

Native @chebfun/rdivide.m inserts denominator roots regardless of coefficient
rank. Analytical controls supplement, rather than replace, original28 assigned
expressions. Chebfun commit7574c77.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, _real_simple_roots
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = jnp.finfo(jnp.float64).eps


def column(kind, root=0., complex_values=False, row=False):
    coeffs = jnp.asarray([[-root], [1.]])
    if complex_values:
        coeffs = (1+2j)*coeffs
    f = Chebfun(funs=[_Piece(tech=kind.from_coeffs(coeffs), interval=(-1.,1.))],
                domain=Domain((-1.,1.)))
    return f.T if row else f


@pytest.mark.parametrize('kind', [Chebtech1,Chebtech2])
@pytest.mark.parametrize('complex_values', [False,True])
@pytest.mark.parametrize('row', [False,True])
def test_single_column_root_discovery(kind,complex_values,row):
    f = column(kind,complex_values=complex_values,row=row)
    assert f.n_columns == 1 and f.funs[0].tech.coeffs.ndim == 2
    _,raw = f.getRootsForBreaks(return_all=True)
    assert raw.shape == (1,1)
    roots = _real_simple_roots(f)
    assert isinstance(roots,jax.Array) and roots.shape == (1,)
    assert jnp.max(jnp.abs(roots)) < 100*EPS


@pytest.mark.parametrize('numeric_numerator', [False,True])
@pytest.mark.parametrize('row', [False,True])
def test_single_column_root_division(numeric_numerator,row):
    f = column(Chebtech2,row=row)
    h = 1/f if numeric_numerator else f/f
    assert h.is_transposed == row
    assert h.n_columns == 1
    x = jnp.asarray([-.75,-.25,.25,.75])
    values = h(x)
    expected = (1/x if numeric_numerator else jnp.ones_like(x))
    if values.ndim == 2:
        expected = expected[None,:] if row else expected[:,None]
    assert values.shape == expected.shape
    assert jnp.max(jnp.abs(values-expected)) < 100*EPS*4
    if numeric_numerator:
        assert len(h.funs) == 2
        assert all(isinstance(p.tech,Singfun) for p in h.funs)
        assert [p.tech.exponents for p in h.funs] == [(0.,-1.),(-1.,0.)]
        assert jnp.all(jnp.isinf(h(0.)))
    else:
        assert all(not isinstance(p.tech,Singfun) for p in h.funs)
        assert jnp.all(jnp.isfinite(h.point_values))
        assert jnp.max(jnp.abs(h(jnp.asarray([0.]))-1)) < 100*EPS


def test_genuine_multiple_columns_remain_outside_scalar_helper():
    f = Chebfun.from_coeffs(jnp.asarray([[0.,-.25],[1.,1.]]))
    assert f.n_columns == 2
    assert _real_simple_roots(f).size == 0


def test_retained_native_denominator_operand():
    data = json.loads((Path(__file__).parent/'fixtures/rdivide_single_column_native.json').read_text())
    domain = data['domain']
    f = Chebfun(funs=[_Piece(tech=Chebtech2.from_coeffs(jnp.asarray(c)),
                            interval=(a,b))
                      for c,a,b in zip(data['coefficients'],domain[:-1],domain[1:])],
                domain=Domain(tuple(domain))).set_point_values(jnp.asarray(data['point_values']))
    assert all(p.tech.coeffs.ndim == 2 and p.tech.coeffs.shape[1] == 1 for p in f.funs)
    r = _real_simple_roots(f)
    expected = jnp.asarray(data['native_roots'])
    assert r.shape == expected.shape
    assert jnp.max(jnp.abs(r-expected)) < 100*EPS


@pytest.mark.parametrize('kind', [Chebtech1,Chebtech2])
@pytest.mark.parametrize('complex_values', [False,True])
@pytest.mark.parametrize('row', [False,True])
def test_scalar_layout_retains_source_words_and_metadata(kind,complex_values,row):
    from chebfunjax.chebfun1d._scalar_rdivide import _scalar_column_layout

    c = jnp.asarray([[EPS/8],[1.],[-.2],[EPS],[-EPS/16]])
    if complex_values:
        c = (1+2j)*c
    f = Chebfun(funs=[_Piece(tech=kind.from_coeffs(c),interval=(-2.3,4.7))],
                domain=Domain((-2.3,4.7)))
    f = f.T if row else f
    before_values = f.funs[0].tech.values
    before_points = f.point_values
    after = _scalar_column_layout(f)
    assert after.n_columns == 1 and after.is_transposed == row
    assert after.domain == f.domain
    a,b = f.funs[0].tech,after.funs[0].tech
    assert type(a) is type(b) and a.ishappy == b.ishappy
    for old,new in [(a.coeffs[:,0],b.coeffs),(before_values[:,0],b.values),
                    (before_points[:,0],after.point_values)]:
        assert jnp.array_equal(old.reshape(-1).view(jnp.uint64),
                               new.reshape(-1).view(jnp.uint64))
