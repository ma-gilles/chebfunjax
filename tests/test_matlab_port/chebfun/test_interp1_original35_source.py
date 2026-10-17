"""All35 literal predicates of Chebfun7574c77 tests/chebfun/test_interp1.m.

Python one-dimensional arrays stand for source column vectors. Explicit
1-by-n arrays exercise row inputs. MATLAB's default matrix norm is spectral,
not Frobenius. RNG6178 uses the qualified explicit MATLAB uniform stream;
the two eleven-entry draws retain their original order. Existing relaxed
mapped controls remain supplemental.
"""
from functools import cache

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.utils._matlab_twister import init_matlab_twister
from chebfunjax.utils._matlab_uniform_stream import MatlabUniformStream

EPS = float(jnp.finfo(jnp.float64).eps)


@cache
def random_sites():
    stream = MatlabUniformStream(init_matlab_twister(6178))
    return stream.uniform((11, 1))[:, 0], stream.uniform((11, 1))[:, 0]


@cache
def case(name):
    x = jnp.arange(11.)
    if name == 'linear_scalar':
        y, method, domain = jnp.sin(x), 'linear', None
    elif name == 'linear_array':
        y, method, domain = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=1), 'linear', None
    elif name == 'linear_domain':
        y, method, domain = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=1), 'linear', (.01, 10.01)
    elif name == 'poly_scalar':
        y, method, domain = jnp.sin(x), None, None
    elif name == 'poly_array':
        y, method, domain = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=1), None, None
    elif name == 'poly_domain':
        y, method, domain = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=1), (.01, 10.01), None
    elif name == 'random_linear':
        x = random_sites()[0]
        y, method, domain = jnp.sin(x), 'linear', None
    elif name == 'function_y':
        x = random_sites()[1]
        y = cj.chebfun(lambda t: t, domain=(0., 1.)).exp()
        method, domain = 'poly', (0., 1.)
    elif name == 'trig_scalar':
        y, method, domain = jnp.cos(jnp.pi*x), 'trig', None
    elif name == 'trig_array':
        y, method, domain = jnp.stack([jnp.exp(jnp.sin(x)), jnp.cos(jnp.pi*x)], axis=1), 'trig', (0., 11.)
    elif name == 'poly_wide':
        y, method, domain = jnp.stack([jnp.exp(jnp.sin(x)), jnp.cos(jnp.pi*x)], axis=1), (-.01, 10.01), None
    else:
        raise AssertionError(name)
    f = cj.Chebfun.interp1(x, y, method, domain)
    return x, y, f


def norm2(value):
    # MATLAB norm(A) is the spectral norm for a matrix, Euclidean for a vector.
    return float(jnp.linalg.norm(value, ord=2))


@pytest.mark.parametrize('clause', range(1, 36))
def test_original_interp1_predicate(clause):
    if clause <= 3:
        x, y, f = case('linear_scalar')
        checks = [lambda: norm2(f(x)-y) < 10*EPS,
                  lambda: len(f.funs) == 10, lambda: len(f) == 20]
        assert checks[clause-1]()
    elif clause <= 6:
        x, y, f = case('linear_array')
        checks = [lambda: norm2(f(x)-y) < 10*EPS,
                  lambda: len(f.funs) == 10, lambda: len(f) == 20]
        assert checks[clause-4]()
    elif clause <= 10:
        x, y, f = case('linear_domain')
        checks = [lambda: bool(jnp.array_equal(jnp.asarray(f.domain.breakpoints),
                                               jnp.concatenate([jnp.array([.01]), jnp.arange(1., 11.), jnp.array([10.01])]))),
                  lambda: norm2(f(x[1:-1])-y[1:-1]) < 10*EPS,
                  lambda: len(f.funs) == 11, lambda: len(f) == 22]
        assert checks[clause-7]()
    elif clause <= 13:
        x, y, f = case('poly_scalar')
        checks = [lambda: norm2(f(x)-y) < 100*EPS,
                  lambda: len(f.funs) == 1, lambda: len(f) == len(x)]
        assert checks[clause-11]()
    elif clause <= 16:
        x, y, f = case('poly_array')
        checks = [lambda: norm2(f(x)-y) < 100*EPS,
                  lambda: len(f.funs) == 1, lambda: len(f) == len(x)]
        assert checks[clause-14]()
    elif clause <= 20:
        x, y, f = case('poly_domain')
        checks = [lambda: norm2(f(x)-y) < 100*EPS,
                  lambda: len(f.funs) == 1, lambda: len(f) == len(x),
                  lambda: bool(jnp.array_equal(jnp.asarray(f.domain.breakpoints), jnp.array([.01, 10.01])))]
        assert checks[clause-17]()
    elif clause == 21:
        x = jnp.arange(11.)
        y = jnp.sin(x)
        f1 = cj.Chebfun.interp1(x[None, :], y, 'linear')
        f2 = cj.Chebfun.interp1(x, y[None, :], 'linear')
        f3 = cj.Chebfun.interp1(x[None, :], y[None, :], 'linear')
        assert all(norm2(f(x)-y) < 10*float(f.vscale)*EPS for f in (f1, f2, f3))
    elif clause <= 24:
        x, y, f = case('random_linear')
        checks = [lambda: norm2(f(x)-y) < 10*EPS,
                  lambda: len(f.funs) == 10, lambda: len(f) == 20]
        assert checks[clause-22]()
    elif clause == 25:
        x, y, f = case('function_y')
        assert float(jnp.linalg.norm(f(x)-y(x), ord=jnp.inf)) < 100*EPS
    elif clause <= 28:
        x, y, f = case('trig_scalar')
        checks = [lambda: norm2(f(x[1:-1])-y[1:-1]) < 100*EPS
                            and abs(float(f(x[0])-(y[0]+y[-1])/2)) < 100*EPS,
                  lambda: len(f.funs) == 1, lambda: len(f) == len(x)-1]
        assert checks[clause-26]()
    elif clause <= 31:
        x, y, f = case('trig_array')
        checks = [lambda: norm2(f(x)-y) < 100*EPS,
                  lambda: len(f.funs) == 1, lambda: len(f) == len(x)]
        assert checks[clause-29]()
    else:
        x, y, f = case('poly_wide')
        checks = [lambda: norm2(f(x)-y) < 1e3*EPS,
                  lambda: len(f.funs) == 1, lambda: len(f) == len(x),
                  lambda: bool(jnp.array_equal(jnp.asarray(f.domain.breakpoints), jnp.array([-.01, 10.01])))]
        assert checks[clause-32]()
