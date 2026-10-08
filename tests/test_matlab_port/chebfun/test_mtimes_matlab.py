"""All 80 numbered predicates from MATLAB tests/chebfun/test_mtimes.m.

MATLAB ``*`` is written ``@``; ``.'`` is ``.T`` and ``'`` is ``.H``.
The seed6178 first 100 uniform probes are native MATLAB captures. The three
later randn matrices and later unbounded probe vector need a native capture;
their five source clauses remain explicitly pending, with separate controls.

Provenance
----------
MATLAB source : tests/chebfun/test_mtimes.m
Chebfun commit: 7574c77
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, kron
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(jnp.finfo(jnp.float64).eps)
ALPHA = -0.194758928283640 + 0.075474485412665j
FIXTURES = Path(__file__).parent / 'fixtures'
X = jnp.asarray(json.loads((FIXTURES / 'logical_source_matlab.json').read_text())['x'])


def quasi(f):
    return Quasimatrix(f.mat2cell(), f.domain)


def scale(f):
    return float(jnp.max(f.vscale() if isinstance(f, Quasimatrix) else f.vscale))


def f2_op(x):
    return jnp.stack([jnp.sin(x) * jnp.abs(x - .1), jnp.exp(x)], axis=-1)


def trig_pair():
    f = cj.chebfun([
        lambda x: jnp.stack([jnp.sin(2*jnp.pi*x), jnp.sin(2*jnp.pi*x)], axis=-1),
        lambda x: jnp.stack([jnp.cos(2*jnp.pi*x), jnp.cos(4*jnp.pi*x)], axis=-1)],
        domain=(-1, 0, 1))
    g = cj.chebfun([
        lambda x: jnp.stack([jnp.sin(4*jnp.pi*x), jnp.sin(4*jnp.pi*x)], axis=-1),
        lambda x: jnp.stack([jnp.cos(2*jnp.pi*x), jnp.cos(4*jnp.pi*x)], axis=-1)],
        domain=(-1, 0, 1))
    return f, g


@pytest.fixture(scope='module')
def basic():
    f1 = cj.chebfun(lambda x: jnp.sin(x)*jnp.abs(x-.1), splitting=True)
    f2 = cj.chebfun(f2_op, splitting=True)
    f, g = trig_pair()
    f2q = Quasimatrix([
        cj.chebfun(lambda x: jnp.sin(x)*jnp.abs(x-.1), splitting=True),
        cj.chebfun(jnp.exp, splitting=True)], f2.domain)
    return f1, f2, f2q, f, g, quasi(f), quasi(g)


@pytest.mark.parametrize('clause', range(1, 21))
def test_source_01_20(clause, basic):
    f1, f2, f2q, f, g, fq, gq = basic
    if clause in (1, 2):
        f = cj.chebfun(jnp.sin, domain=(-1, 1))
        assert (f @ ([] if clause == 1 else cj.chebfun())).isempty()
    elif clause in (3, 4, 6, 7, 11, 12):
        f = f1 if clause in (3, 4) else f2 if clause in (6, 7) else f2q
        a, b = f @ ALPHA, ALPHA @ f
        if clause in (3, 6, 11):
            ac = a.cols if isinstance(a, Quasimatrix) else [a]
            bc = b.cols if isinstance(b, Quasimatrix) else [b]
            assert all(x.isequal(y) for x, y in zip(ac, bc))
        else:
            exact = (jnp.sin(X)*jnp.abs(X-.1) if clause == 4 else f2_op(X))*ALPHA
            assert jnp.max(jnp.abs(a(X)-exact)) < 1e2*scale(a)*EPS
    elif clause == 5:
        f = cj.chebfun([lambda x: jnp.sin(2*jnp.pi*x), lambda x: jnp.cos(2*jnp.pi*x)],
                       domain=(-1, 0, 1))
        g = cj.chebfun([lambda x: jnp.exp(2j*jnp.pi*x), lambda x: jnp.cos(2*jnp.pi*x)],
                       domain=(-1, 0, 1))
        assert jnp.abs(g.T @ f - (.5+.5j)).item() < 10*max(scale(f), scale(g))*EPS
    elif clause in (8, 13, 14, 15):
        left = gq if clause in (13, 15) else g
        right = fq if clause in (14, 15) else f
        err = left.T @ right - jnp.diag(jnp.array([.5, .5]))
        assert jnp.max(jnp.abs(err)) < 10*max(scale(f), scale(g))*EPS
    elif clause in (9, 10, 16, 17):
        data = native_rng()
        a = jnp.asarray(data['A_array' if clause < 16 else 'A_quasi'])
        matrix_case(f2 if clause < 16 else f2q, a, X, clause in (10, 17))
    elif clause == 18:
        with pytest.raises(TypeError, match='CHEBFUN:CHEBFUN:mtimes:unknown'):
            f @ 'X'
    elif clause == 19:
        try:
            f @ f1
        except ValueError as exc:
            assert 'CHEBFUN:CHEBFUN:dimCheck:dim' in str(exc)
    elif clause == 20:
        f @ g


def native_rng():
    path = FIXTURES / 'mtimes_source_matlab.json'
    if not path.exists():
        pytest.skip('Native MATLAB source-order randn matrices/probes pending; deterministic controls are separate.')
    return json.loads(path.read_text())


def matrix_case(f, a, x, row):
    h = a @ f.T if row else f @ a
    exact = a @ f2_op(x).T if row else f2_op(x) @ a
    assert jnp.max(jnp.abs(h(x)-exact)) < 10*scale(h)*EPS


@pytest.mark.parametrize('clause', (21, 22, 23))
def test_source_21_23(clause):
    if clause in (21, 22):
        f = cj.chebfun(lambda x: jnp.sin(20*x)/(x+1)**.5,
                       exps=(-.5, 0), splitting=True)
        if clause == 21:
            h = 3 @ f
            err = h(X) - 3*jnp.sin(20*X)/(X+1)**.5
            assert jnp.linalg.norm(err, ord=jnp.inf) < 1e4*scale(h)*EPS
        else:
            g = cj.chebfun(lambda x: jnp.cos(30*x), splitting=True)
            exact = .13033807496531659
            assert jnp.abs(f.T @ g-exact).item() < 1e2*exact*EPS
    else:
        data = native_rng()
        unbounded_case(jnp.asarray(data['A_unbounded']), jnp.asarray(data['x_unbounded']))


def unbounded_case(a, x):
    op = lambda x: jnp.stack([jnp.exp(x), x*jnp.exp(x), (1-jnp.exp(x))/x], axis=-1)
    f = cj.chebfun(op, domain=(-jnp.inf, -3*jnp.pi))
    g = f @ a
    err = g(x) - op(x) @ a
    assert jnp.linalg.norm(err, ord=jnp.inf) < 1e2*scale(g)*EPS


@pytest.fixture(scope='module')
def dimensions():
    f = cj.chebfun(jnp.sin)
    a = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
    return f, a, quasi(a)


def size(a):
    if isinstance(a, (Chebfun, Quasimatrix)):
        return a.size()
    if isinstance(a, Chebfun2):
        return a.size()
    return a.shape


@pytest.mark.parametrize('clause', range(24, 64))
def test_source_24_63(clause, dimensions):
    f, a, q = dimensions
    inf = float('inf')
    if clause <= 27:
        products = {24: lambda: f.H @ f, 25: lambda: f @ f.H,
                    26: lambda: f @ jnp.array([[1., 2.]]),
                    27: lambda: jnp.array([[1.], [2.]]) @ f.H}
        expected = {24: (1, 1), 25: (inf, inf), 26: (inf, 2), 27: (2, inf)}
        assert size(products[clause]()) == expected[clause]
    elif clause <= 47:
        n = clause if clause <= 37 else clause-10
        a = a if clause <= 37 else q
        if n == 28:
            result, expected = a.H @ a, (3, 3)
        elif n == 29:
            result, expected = a @ a.H, (inf, inf)
        elif n == 30:
            result, expected = f.H @ a, (1, 3)
        elif n == 31:
            result, expected = a.H @ f, (3, 1)
        elif n <= 34:
            k = n-30
            result, expected = a @ jnp.ones((3, k)), (inf, k)
        else:
            k = n-33
            result, expected = jnp.ones((k, 3)) @ a.H, (k, inf)
        assert size(result) == expected
    elif clause <= 59:
        a = a if clause <= 53 else q
        n = clause if clause <= 53 else clause-6
        with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:mtimes:dims'):
            if n <= 50:
                jnp.ones((3, n-46)) @ a
            else:
                a.H @ jnp.ones((n-49, 3))
    else:
        a = a if clause <= 61 else q
        if clause % 2:
            a = a.H
        a @ a


@pytest.fixture(scope='module', params=range(4))
def outer_case(request):
    case = request.param
    if case < 2:
        d = (-1., 1., -1., 1.) if case == 0 else (-2., float(jnp.pi), -float(jnp.pi), 2.)
        f = cj.chebfun(lambda x: x*x, domain=d[:2], splitting=True)
        g = cj.chebfun(jnp.sin, domain=d[2:], splitting=True)
        op = lambda x, y: x*x*jnp.sin(y)
        opr = lambda x, y: y*y*jnp.sin(x)
    else:
        d = (-1., 1., -1., 1.) if case == 2 else (-2., 1., -1., 1.)
        x = cj.chebfun('x', domain=d[:2], splitting=True)
        y = cj.chebfun('x', domain=d[2:], splitting=True)
        f = Quasimatrix([0*x+1, x, x**2, x**4], x.domain)
        g = Quasimatrix([0*y+1, y.cos(), y.sin(), y**5], y.domain)
        op = lambda x, y: 1+x*jnp.cos(y)+x*x*jnp.sin(y)+x**4*y**5
        opr = lambda x, y: 1+y*jnp.cos(x)+y*y*jnp.sin(x)+y**4*x**5
    h1 = cj.chebfun2(op, domain=d)
    h2 = cj.chebfun2(opr, domain=(*d[2:], *d[:2]))
    return case, f, g, h1, h2


@pytest.mark.parametrize('offset', range(4))
def test_source_64_79(outer_case, offset):
    case, f, g, h1, h2 = outer_case
    # pass(64+4*case+offset), preserving source norm and bound.
    expected = h1 if offset % 2 == 0 else h2
    actual = (kron(f.H, g) if offset == 0 else kron(f, g.H) if offset == 1
              else g @ f.H if offset == 2 else f @ g.H)
    tol = 20*EPS*(10 if case % 2 else 1)
    assert (expected-actual).norm() < tol


def test_source_80():
    x = cj.chebfun('x', domain=(-1, 1), splitting=True)
    f = Quasimatrix([0*x+1, x, x**2, x**4], x.domain)
    h = cj.chebfun2(lambda x, y: 1+x*jnp.cos(y)+x*x*jnp.sin(y)+x**4*y**5)
    assert ((f.H @ h) - (h.H @ f).H).norm() < 20*EPS


@pytest.mark.parametrize('quasimatrix', (False, True))
@pytest.mark.parametrize('row', (False, True))
def test_deterministic_matrix_controls(basic, quasimatrix, row):
    a = jnp.array([[.25, -1.5], [2., .75]])
    matrix_case(basic[2] if quasimatrix else basic[1], a, X, row)


def test_deterministic_unbounded_control():
    a = jnp.array([[.25, -1., .5], [2., .75, -.125], [-.5, .25, 1.]])
    unbounded_case(a, jnp.linspace(-1e6, -3*jnp.pi, 100))
