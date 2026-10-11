"""Symmetric jacpts dispatch controls: native7574c77 jacpts/legpts/ultrapts."""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import jacpts, ultrapts


@pytest.mark.parametrize('method', [None, 'default', 'REC', 'GW', 'ASY'])
def test_native_legendre_42_references(method):
    # Original tests/misc/test_legpts.m clauses3–6, unchanged1e-14 bound.
    x, w, v = jacpts(42, 0., 0., method=method, bary=True)
    assert abs(w@x) < 1e-14
    assert abs(w@(x*x)-2/3) < 1e-14
    assert abs(x[36]-.910959724904127) < 1e-14
    assert abs(w[36]-.030479240699603) < 1e-14
    assert abs(v[36]-.265155501739424) < 1e-14


@pytest.mark.parametrize('n', [20, 21, 99, 100])
def test_legendre_dispatch_boundaries(n):
    # Contract control, not replacement for any100k/200k native predicate.
    x, w, v = jacpts(n, 0., 0., 'ASY', bary=True)
    assert jnp.all(jnp.isfinite(x+w+v))
    assert jnp.array_equal(x, -x[::-1])
    if n == 21:
        rows = json.loads(Path(__file__).with_name('native_symmetric.json').read_text())['cases']
        error = next(row['mass_error'] for row in rows if row['n'] == n)
        # Native itself fails the retained initial ideal-mass diagnostic.
        assert abs(error) > 1e-12
        assert abs((jnp.sum(w)-2)-error) < 1e-14
    else:
        assert abs(jnp.sum(w)-2) < 1e-12
    if n == 20:
        xr, wr, vr = jacpts(n, 0., 0., 'REC', bary=True)
        assert all(jnp.array_equal(a, b) for a, b in zip((x,w,v),(xr,wr,vr)))


@pytest.mark.parametrize('a', [.2, 1.])
def test_ultraspherical_native_method_ignored(a):
    # jacpts source forwards no METHOD to ultrapts for generic a==b.
    ref = ultrapts(42, a+.5, [0., 10.], bary=True)
    for method in ['REC', 'GW', 'ASY']:
        actual = jacpts(42, a, a, [0., 10.], method, bary=True)
        assert all(jnp.array_equal(x,y) for x,y in zip(actual,ref))


@pytest.mark.parametrize('n', [41, 42])
@pytest.mark.parametrize('a', [-.5, .5])
def test_chebyshev_special_cases(n, a):
    x, w, v = jacpts(n, a, a, 'ASY', bary=True)
    assert jnp.array_equal(x, -x[::-1])
    assert abs(jnp.sum(w)-(jnp.pi if a == -.5 else jnp.pi/2)) < 1e-14
    assert abs(w@(x*x)-(jnp.pi/2 if a == -.5 else jnp.pi/8)) < 1e-14
    if a == -.5:
        assert v[-1] > 0
        assert (v[0] > 0) == bool(n%2)
        expected_max = 1. if n%2 else jnp.cos(jnp.pi/(2*n))
        assert abs(jnp.max(jnp.abs(v))-expected_max) < 1e-15
    else:
        assert jnp.array_equal(jnp.abs(v), 1-x*x)
    xm, wm, vm = jacpts(n, a, a, [-2., 3.], 'GW', bary=True)
    assert jnp.array_equal(vm,v)
    assert abs(jnp.sum(wm)-2.5**(2*a+1)*jnp.sum(w)) < 1e-13


@pytest.mark.parametrize('n', [20, 21, 42, 99, 100])
def test_matched_native_public_arrays(n):
    rows = json.loads(Path(__file__).with_name('native_symmetric.json').read_text())['cases']
    row = next(row for row in rows if row['n'] == n)
    for actual, name in zip(jacpts(n, 0., 0., bary=True), ['x', 'w', 'v']):
        assert jnp.max(jnp.abs(actual-jnp.asarray(row[name]).ravel())) < 1e-14
