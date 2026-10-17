"""All14literal predicates from the pinned default test_minandmax.m.

The older test_minandmax_matlab.py retains supplemental/partial translations.
MATLAB extrema output (y,x) is adapted to Python's (x,y) tuple convention.

Provenance
----------
MATLAB source : tests/chebfun/test_minandmax.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun

EPS = np.finfo(float).eps
Y_REFERENCE = np.array([0.710869767377087, 1.884217141925336])
Y_LOCAL = np.array([0., 1.923771282655145, -0.342247088203205,
                    1.117294907913736, -0.971179645473729,
                    1.343997479566445, 0.284846700239241])
X_LOCAL_ARRAY = np.array([[0., 0.], [1.323339426259694, 2.646678852519388],
                          [2.220599667639221, 4.],
                          [2.781195946808315, np.nan],
                          [3.308480466603983, np.nan],
                          [3.776766383330969, np.nan], [4., np.nan]])
Y_LOCAL_ARRAY = np.column_stack([Y_LOCAL,
                                [0., 1.923771282655145, 0.070019315123878,
                                 np.nan, np.nan, np.nan, np.nan]])


def _global(f):
    (xmin, ymin), (xmax, ymax) = f.minandmax()
    return np.stack([ymin, ymax]), np.stack([xmin, xmax])


def _op(x):
    return jnp.sin(x)**2 + jnp.sin(x**2)


def _reference(x):
    return ((x-.2)**3 - (x-.2) + 1) * (1/jnp.cos(x-.2))


def test_pass_01_empty():
    y, x = _global(chebfun())
    assert y.size == 0 and x.size == 0


def test_pass_02_smooth():
    f = chebfun(_reference)
    y, x = _global(f)
    assert np.all(np.abs(y-Y_REFERENCE) <= 10*f.vscale*EPS)
    assert np.all(np.abs(np.asarray(f(x))-Y_REFERENCE) <= 10*f.vscale*EPS)


def test_pass_03_piecewise():
    f = chebfun(_reference, domain=np.linspace(-1, 1, 10))
    y, x = _global(f)
    assert np.all(np.abs(y-Y_REFERENCE) <= 10*f.vscale*EPS)
    assert np.all(np.abs(np.asarray(f(x))-Y_REFERENCE) <= 10*f.vscale*EPS)


def test_pass_04_complex_cells():
    f = chebfun([lambda x: jnp.exp((1+1j)*x), lambda x: 1-x/10],
                domain=[-1, 0, 1])
    y, x = _global(f)
    expected = np.array([np.exp(-1-1j), 1.])
    assert np.all(np.abs(y-expected) <= 10*f.vscale*EPS)
    assert np.all(np.abs(np.asarray(f(x))-expected) <= 10*f.vscale*EPS)


def _constant_cells():
    return chebfun([-1, 1, 2], domain=[-1, 0, 1, 2])


def test_pass_05_constant_cells():
    y, _ = _global(_constant_cells())
    assert np.all(y == [-1, 2])


def test_pass_06_point_values():
    f = _constant_cells()
    f = f.set_point_values(f.point_values.at[0].set(10).at[2].set(-10))
    y, x = _global(f)
    assert np.all(y == [-10, 10]) and np.all(x == [1, -1])


def test_pass_07_local_scalar():
    f = chebfun(_op, domain=[0, 4])
    x, y = f.minandmax('local')
    y = np.asarray(y)
    # Literal native numel(y == 7) tests nonemptiness.
    assert (y == 7).size and np.linalg.norm(y-Y_LOCAL, np.inf) < 10*f.vscale*EPS


@pytest.mark.parametrize('predicate', [8, 9])
def test_pass_08_09_global_array(predicate):
    if predicate == 8:
        f = chebfun(lambda x: jnp.stack([jnp.sin(10*x), jnp.cos(10*x),
                                        jnp.exp(x)], axis=-1),
                    domain=[-1, -.5, .5, 1])
        expected = np.array([[-1, -1, np.exp(-1)], [1, 1, np.exp(1)]])
    else:
        f = chebfun(lambda x: jnp.stack([jnp.exp((1+1j)*x),
                                        1/jnp.cos(1j*(x-.5))], axis=-1),
                    domain=[-1, 0, 1])
        expected = np.array([[np.exp(-1-1j), 1/np.cos(-1.5j)],
                             [np.exp(1+1j), 1]])
    y, x = _global(f)
    fx = np.asarray(f(x.ravel(order='F')))
    fx = np.column_stack([fx[2*j:2*j+2, j] for j in range(expected.shape[1])])
    assert np.all(np.abs(y.ravel(order='F')-expected.ravel(order='F')) <= 10*f.vscale*EPS)
    assert np.all(np.abs(fx.ravel(order='F')-expected.ravel(order='F')) <= 10*f.vscale*EPS)


@pytest.mark.parametrize('predicate', [10, 11])
def test_pass_10_11_local_array(predicate):
    f = chebfun(lambda x: jnp.stack([_op(x), _op(x/2)], axis=-1), domain=[0, 4])
    x, y = map(np.asarray, f.minandmax('local'))
    fx1 = np.asarray(f(X_LOCAL_ARRAY[:, 0]))
    fx2 = np.asarray(f(X_LOCAL_ARRAY[:3, 1]))
    if predicate == 10:
        assert y.shape == (7, 2)
        assert np.all(np.isnan(y[3:, 1])) and np.all(np.isnan(x[3:, 1]))
    else:
        assert np.linalg.norm(y[:, 0]-Y_LOCAL_ARRAY[:, 0], np.inf) < 10*f.vscale*EPS
        assert np.linalg.norm(y[:3, 1]-Y_LOCAL_ARRAY[:3, 1], np.inf) < 10*f.vscale*EPS
        assert np.linalg.norm(fx1[:, 0]-Y_LOCAL_ARRAY[:, 0], np.inf) < 10*f.vscale*EPS
        assert np.linalg.norm(fx2[:, 1]-Y_LOCAL_ARRAY[:3, 1], np.inf) < 10*f.vscale*EPS


def test_pass_12_singular_splitting():
    dom = [-2, 7]
    power = -.5
    def op(x):
        return (x-dom[0])**power * jnp.sin(300*x)**2
    f = chebfun(op, domain=dom, exps=[power, 0], splitting=True)
    y, x = _global(f)
    expected = np.array([0., np.inf])
    fx = np.asarray(op(jnp.asarray(x)))
    # MATLAB MAX omits NaNs, including Inf-Inf in this exact predicate.
    assert np.nanmax(np.abs(y-expected)) < 1e4*EPS
    assert np.nanmax(np.abs(fx-expected)) < 1e4*EPS


def test_pass_13_unbounded():
    f = chebfun(lambda x: (1-jnp.exp(-x**2))/x, domain=[-np.inf, np.inf])
    vals, pos = _global(f)
    expected_v = np.array([-.6381726863389515, .6381726863389515])
    expected_p = np.array([-1.120906422778534, 1.120906422778534])
    assert np.linalg.norm(vals-expected_v, np.inf) < 1e2*EPS*f.vscale
    assert np.linalg.norm(pos-expected_p, np.inf) < 1e2*EPS*f.vscale


def test_pass_14_gamma_reciprocal():
    gam = chebfun('gamma(x)', domain=[-4, 4], blowup=True, splitting=True)
    mm, _ = _global(1/gam)
    assert np.linalg.norm(mm-[-1.125953228398760, 4.079508980001102]) < 1e4*EPS
