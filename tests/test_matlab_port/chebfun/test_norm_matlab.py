"""All34 literal MATLAB Chebfun norm predicates, including locations/errors.

Provenance
----------
MATLAB source : tests/chebfun/test_norm.m, @chebfun/norm.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Deterministic source functions/constants; no native RNG inputs required.
Python return_location=True represents MATLAB's requested second output.
"""
from functools import lru_cache

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = float(jnp.finfo(jnp.float64).eps)


def _scale(f):
    scale = f.vscale() if callable(f.vscale) else f.vscale
    return float(jnp.max(scale))


@lru_cache(None)
def _piecewise():
    return cj.chebfun([lambda x: jnp.exp(4*jnp.pi*1j*x), jnp.exp, jnp.exp],
                      domain=[-1., 0., .5, 1.])


@lru_cache(None)
def _array():
    return cj.chebfun(lambda x: jnp.stack((jnp.sin(x), jnp.cos(x), jnp.exp(x)), axis=-1),
                      domain=[-1., -.5, 0., .5, 1.])


@lru_cache(None)
def _g():
    return cj.chebfun(lambda x: 1/(1+(x-.1)**2), domain=[-1., -.5, 0., .5, 1.])


@lru_cache(None)
def _singular23():
    return cj.chebfun(lambda x: jnp.sin(100*x)*(x+1)**.6,
                      exps=(.6, 0.), splitting=True)


@lru_cache(None)
def _unbounded27():
    return cj.chebfun(lambda x: (1-jnp.exp(-x))/x, domain=[1., jnp.inf])


@pytest.mark.parametrize('slot', range(1, 35), ids=lambda n: f'source_{n:02d}')
def test_source_predicate(slot):
    if slot == 1:
        assert cj.chebfun().norm() == 0
    elif slot in (2, 3, 4, 5):
        f = _piecewise()
        actual = f.norm() if slot == 2 else f.norm({3: 2, 4: 'fro', 5: 1}[slot])
        exact = jnp.exp(1.) if slot == 5 else jnp.sqrt(1+(jnp.exp(2.)-1)/2)
        assert abs(actual-exact) < 10*_scale(f)*EPS
    elif slot in (6, 7, 8):
        f = _g() if slot == 8 else _piecewise()
        value, location = f.norm(-jnp.inf if slot == 7 else jnp.inf, return_location=True)
        exact = 1. if slot in (7, 8) else jnp.exp(1.)
        at_location = abs(f(location)) if slot == 7 else f(location)
        assert abs(value-exact) < 10*_scale(f)*EPS and abs(at_location-exact) < 10*_scale(f)*EPS
    elif slot == 9:
        f = _array()
        value, column = f.norm(1, return_location=True)
        assert column == 3 and abs(value-(jnp.exp(1.)-jnp.exp(-1.))) < 10*_scale(f)*EPS
    elif slot == 10:
        u = cj.chebfun(lambda x: jnp.stack((1+0*x, jnp.exp(2*jnp.pi*1j*x),
                                          jnp.exp(4*jnp.pi*1j*x)), axis=-1), domain=[0., 1.])
        s = jnp.diag(jnp.array([jnp.pi, jnp.exp(1.), 1.]))
        v = jnp.array([[1/jnp.sqrt(2.), -1/jnp.sqrt(2.), 0.],
                       [1/jnp.sqrt(2.), 1/jnp.sqrt(2.), 0.], [0., 0., 1.]])
        h = u@s@v.T
        assert abs(h.norm(2)-jnp.pi) < 10*_scale(h)*EPS
    elif slot in (11, 12):
        f = _array()
        actual = f.norm() if slot == 11 else f.norm('fro')
        assert abs(actual-2.372100421113536830) < 10*_scale(f)*EPS
    elif slot in (13, 14):
        f = _array()
        sign = 1 if slot == 13 else -1
        value, location = f.norm(sign*jnp.inf, return_location=True)
        assert location == sign and abs(value-(jnp.exp(float(sign))+jnp.sin(1.)+jnp.cos(1.))) < 10*_scale(f)*EPS
    elif 15 <= slot <= 21:
        f = _g() if slot <= 18 else _array()
        p = {15: 1, 16: 2, 17: .4, 18: 'bad', 19: 2, 20: 'fro', 21: 'bad'}[slot]
        identifier = 'CHEBFUN:CHEBFUN:norm:'+('unknownNorm' if slot in (18, 21) else 'argout')
        with pytest.raises(ValueError, match=identifier):
            f.norm(p, return_location=True)
    elif slot == 22:
        f = cj.chebfun(lambda x: jnp.sin(50*x), splitting=True)
        assert abs(f.norm(3)-.9484869030456855) < 1e1*_scale(f)*EPS
    elif slot in (23, 24):
        f = _singular23()
        exact = 1.02434346249849423 if slot == 23 else 1.20927413792339491
        assert abs(f.norm(2 if slot == 23 else 1)-exact) < (1e1 if slot == 23 else 1e7)*_scale(f)*EPS
    elif slot in (25, 26):
        if slot == 25:
            f = cj.chebfun(lambda x: jnp.sin(x)*(1-x)**.6, exps=(0., .6), splitting=True)
            p, exact, factor = jnp.inf, jnp.array([1.275431511911148, -1.]), 1e1
        else:
            f = cj.chebfun(lambda x: jnp.sin(x+1.1)*(x+1)**.8, exps=(.8, 0.), splitting=True)
            p, exact, factor = -jnp.inf, jnp.array([0., -1.]), 1e3
        actual = jnp.asarray(f.norm(p, return_location=True))
        assert jnp.max(jnp.abs(actual-exact)) < factor*_scale(f)*EPS
    elif slot in (27, 29):
        f = _unbounded27()
        actual = f.norm() if slot == 27 else f.norm(3)
        exact = .860548225417173 if slot == 27 else .631964633132246
        assert abs(actual-exact) < (1e7 if slot == 27 else 1e6)*EPS*_scale(f)
    elif slot == 28:
        f = cj.chebfun(lambda x: (1-jnp.exp(-x))/x**2, domain=[1., jnp.inf])
        assert abs(f.norm(1)-.851504493224078) < 1e6*EPS*_scale(f)
    elif slot == 30:
        f = _unbounded27()
        actual = jnp.asarray(f.norm(jnp.inf, return_location=True))
        assert jnp.max(jnp.abs(actual-jnp.array([1-jnp.exp(-1.), 1.]))) < _scale(f)*EPS
    elif slot == 31:
        f = cj.chebfun(lambda x: 1/x, domain=[1., jnp.inf])
        value, location = f.norm(-jnp.inf, return_location=True)
        assert abs(value) < _scale(f)*EPS and location == jnp.inf
    elif slot == 32:
        f = cj.chebfun(lambda x: jnp.stack((jnp.exp(x), x*jnp.exp(x), (1-jnp.exp(x))/x**2), axis=-1),
                      domain=[-jnp.inf, -1.])
        assert abs(f.restrict(-1000., -1.).simplify().norm(1)-.851504493224078) < 1e-2
    elif slot == 33:
        f = cj.chebfun(lambda x: jnp.cos(x)/(1e5+(x-30)**6), domain=[0., jnp.inf])
        assert abs(f.norm()-2.4419616835794597e-5) < 1e9*_scale(f)*EPS
    else:
        f = cj.chebfun(lambda x: jnp.exp(-jnp.stack((x, x), axis=-1)**2), domain=[0., jnp.inf])
        assert abs(f.norm()-1.119515134920248) < 1e1*EPS*_scale(f)
