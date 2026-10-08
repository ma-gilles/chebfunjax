"""All 36 numbered predicates in MATLAB tests/chebfun/test_sum.m.

The seed7681 first 1000 probes are the native capture used by test_repmat,
whose source has exactly the same seed and first draw. Bounds are unchanged.
MATLAB norm treats both row and column vectors as vectors; their infinity
norm is the maximum absolute entry, including transposed scalar evaluations.

Provenance
----------
MATLAB source : tests/chebfun/test_sum.m
Chebfun commit: 7574c77
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = float(jnp.finfo(jnp.float64).eps)
XR = jnp.asarray(json.loads(Path(__file__).with_name('repmat_matlab_inputs.json').read_text())['xr'])
DOM = (-1., -.5, 0., .5, 1.)


def scale(f):
    return float(jnp.max(f.vscale() if isinstance(f, Quasimatrix) else f.vscale))


def vector_inf(err):
    return jnp.max(jnp.abs(err))


@pytest.fixture(scope='module')
def pw():
    return cj.chebfun([lambda x: jnp.exp(4j*jnp.pi*x), jnp.exp, jnp.exp],
                      domain=(-1, 0, .5, 1))


@pytest.fixture(scope='module')
def bounds():
    return cj.chebfun(lambda x: x*x-1), cj.chebfun(lambda x: -x*x+1)


@pytest.fixture(scope='module')
def scalar():
    return cj.chebfun(jnp.exp, domain=DOM)


@pytest.fixture(scope='module')
def array():
    return cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1), domain=DOM)


@pytest.fixture(scope='module')
def quasi():
    cols = [cj.chebfun(op, domain=DOM) for op in (jnp.sin, jnp.cos, jnp.exp)]
    return Quasimatrix(cols, cols[0].domain)


@pytest.mark.parametrize('clause', range(1, 7))
def test_source_01_06(clause, pw):
    if clause == 1:
        assert cj.chebfun().sum() == 0
        return
    f = pw.T if clause == 3 else pw
    value = f.sum() if clause in (2, 3) else f.sum({4: [-1, 1], 5: [-1, 0], 6: [0, 1]}[clause])
    exact = 0 if clause == 5 else jnp.exp(1.)-1
    assert jnp.abs(value-exact) < 10*scale(f)*EPS


def exact_limits(a, b, array):
    if not array:
        return jnp.exp(b)-jnp.exp(a)
    return jnp.stack([jnp.cos(a)-jnp.cos(b), jnp.sin(b)-jnp.sin(a),
                      jnp.exp(b)-jnp.exp(a)], axis=-1)


@pytest.mark.parametrize('clause', (7, 8, 9, 14, 15, 16))
def test_source_variable_limits(clause, bounds, scalar, array):
    a, b = bounds
    n = clause if clause < 10 else clause-7
    lower, upper = (a, 1.) if n == 7 else (-1., b) if n == 8 else (a, b)
    f = scalar if clause < 10 else array
    result = f.sum(lower, upper)
    av = XR*XR-1 if n != 8 else jnp.full_like(XR, -1.)
    bv = -XR*XR+1 if n != 7 else jnp.ones_like(XR)
    err = result(XR)-exact_limits(av, bv, clause > 10)
    assert jnp.max(jnp.abs(err)) < (100 if clause == 16 else 10)*scale(result)*EPS


@pytest.mark.parametrize('clause', (10, 11, 12, 13, 22, 23, 24, 25))
def test_source_array_and_quasi_integrals(clause, array, quasi):
    f = array if clause < 20 else quasi
    n = clause if clause < 20 else clause-12
    if n == 11:
        f = f.T
    value = f.sum() if n in (10, 11) else f.sum([-1, 1] if n == 12 else [-1, 0])
    exact = (jnp.array([0., 2*jnp.sin(1.), jnp.exp(1.)-jnp.exp(-1.)]) if n != 13 else
             jnp.array([jnp.cos(-1.)-1, jnp.sin(1.), 1-jnp.exp(-1.)]))
    if n == 11:
        exact = exact[:, None]
    assert vector_inf(value-exact) < 10*scale(f)*EPS


@pytest.mark.parametrize('clause', (17, 18))
def test_source_discrete_dimension(clause, array):
    f = array if clause == 17 else array.T
    g = f.sum(2 if clause == 17 else 1)
    exact = jnp.sin(XR)+jnp.cos(XR)+jnp.exp(XR)
    assert vector_inf(g(XR)-exact) < 10*scale(g)*EPS


@pytest.mark.parametrize('clause', (19, 20, 21))
def test_source_subdomain_errors(clause, array, bounds):
    a, b = bounds
    args, suffix = ((-2., 2.), 'ab') if clause == 19 else ((-2., b), 'a') if clause == 20 else ((a, 2.), 'b')
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:sum:sumSubDom:'+suffix):
        array.sum(*args)


def test_source_26_singular():
    f = cj.chebfun(lambda x: (x+2)**(-.5)*jnp.sin(100*x),
                   domain=(-2, 7), exps=(-.5, 0), splitting=True)
    exact = .17330750941063138
    assert jnp.abs(f.sum()-exact) < 1e3*EPS*abs(exact)


@pytest.mark.parametrize('clause', range(27, 35))
def test_source_27_34_unbounded(clause):
    if clause == 27:
        f = cj.chebfun([lambda x: x*x*jnp.exp(-x*x), lambda x: (1-jnp.exp(-x*x))/(x*x)],
                       domain=(-jnp.inf, 2, jnp.inf))
        exact, factor = 1.364971769155161, 1e7
    elif clause in (28, 29):
        direct = cj.chebfun(lambda x: x*jnp.exp(-x), domain=(0, jnp.inf))
        if clause == 29:
            x = cj.chebfun('x', domain=(0, jnp.inf))
            e = (-x).exp()
            f = x*e

        else:
            f = direct
        exact, factor = 1., 2e5 if clause == 28 else 2e10
    elif clause == 30:
        f = cj.chebfun('exp(-x.^2/16).*(1+.2*cos(10*x))', domain=(-jnp.inf, jnp.inf))
        exact, factor = 7.0898154036220641, 1e9
    elif clause == 31:
        f = cj.chebfun(lambda x: jnp.exp(-x)/(1+3*x), domain=(0, jnp.inf))
        exact, factor = .385602012136694, 20
    elif clause == 32:
        f = cj.chebfun(lambda x: jnp.stack([jnp.exp(-x*x), jnp.exp(-x*x)], axis=-1), domain=(0, jnp.inf))
        exact, factor = jnp.full(2, .886226925452758), 10
    elif clause == 33:
        f = cj.chebfun(lambda x: jnp.stack([jnp.exp(-x*x), 1/x**3], axis=-1), domain=(1, jnp.inf))
        exact, factor = jnp.array([.139402792640331, .5]), 1e6
    else:
        f = cj.chebfun(lambda x: jnp.stack([jnp.exp(-x*x), 0*x+1], axis=-1), domain=(1, jnp.inf))
        exact, factor = .139402792640331, 10
    value = f.sum()
    if clause == 34:
        assert jnp.abs(value[0]-exact) < factor*EPS*scale(f) and jnp.isinf(value[1])
    elif clause in (32, 33):
        assert vector_inf(value-exact) < factor*EPS*scale(f)
    else:
        assert jnp.abs(value-exact) < factor*EPS*scale(direct if clause == 29 else f)


def test_source_35_complex_limits():
    with pytest.raises((ValueError, TypeError)):
        cj.chebfun(jnp.sin).sum([0, 1j])


def test_source_36_reverse_invalid_subdomain():
    with pytest.raises((ValueError, TypeError)):
        cj.chebfun(lambda x: x, domain=(0, 1)).sum(0, -1)
