"""Remaining literal subsref predicates, Chebfun7574c77.

Existing14source clauses/companions remain untouched. Structured records use
Python's existing Chebfun3.subsref adapter convention for native brace syntax;
missing public Chebfun.subsref is a failure, not a restrict-method surrogate.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = np.finfo(np.float64).eps


def _f(array=False):
    if array:
        return cj.chebfun(lambda x: jnp.stack((jnp.sin(x-.1), jnp.cos(x-.2)), axis=-1))
    return cj.chebfun(lambda x: jnp.sin(x-.1))


def _expect_id(call, identifier):
    with pytest.raises(Exception) as caught:
        call()
    assert getattr(caught.value, 'identifier', None) == identifier


@pytest.mark.parametrize('clause', [4, 5, 6, 17, 18, 19, 20, 21, 22])
def test_original_ordinary(clause):
    f = _f(clause == 22)
    if clause == 4:
        g = cj.chebfun(lambda x: jnp.cos(x+.2))
        assert g.compose(f).isequal(f(g))
    elif clause == 5:
        assert f.isequal(f[:])  # Python adapter for native f(:).
    elif clause == 6:
        _expect_id(lambda: f('X'), 'CHEBFUN:CHEBFUN:subsref:nonnumeric')
    elif clause == 17:
        assert f.isequal(f.subsref({'type': '{}', 'subs': [':']}))
    elif clause in (18, 22):
        ends = [-1., -.1, .2, 1.]
        result = f.subsref({'type': '{}', 'subs': ends})
        error = (result-f.restrict(ends)).norm(jnp.inf) if clause == 22 else (
            result-f.restrict(ends)).norm()
        assert float(error) < (5*EPS if clause == 22 else EPS)
    elif clause == 19:
        _expect_id(lambda: f.subsref({'type': '{}', 'subs': ['X']}),
                   'CHEBFUN:CHEBFUN:subsref:badDomain')
    elif clause == 20:
        cells = np.empty((2, 1), dtype=object)
        cells[0, 0], cells[1, 0] = np.array([1., 2.]), np.array([3., 4.])
        _expect_id(lambda: f.subsref({'type': '{}', 'subs': cells}),
                   'CHEBFUN:CHEBFUN:subsref:dimensions')
    else:
        _expect_id(lambda: f.subsref({'type': '[]', 'subs': []}),
                   'CHEBFUN:CHEBFUN:subsref:unexpectedType')


def _source_cdr_equal(left, right):
    # Native separableApprox.isequal compares actual C/R and raw pivotValues;
    # the current public2D norm-surrogate isequal alone is insufficient.
    assert left.approx.domain == right.approx.domain
    assert left.approx.pivot_values is not None and right.approx.pivot_values is not None
    assert np.array_equal(left.approx.pivot_values, right.approx.pivot_values)
    for family in ('cols', 'rows'):
        a, b = getattr(left.approx, family), getattr(right.approx, family)
        assert len(a) == len(b)
        assert all(f.isequal(g) for f, g in zip(a, b))


@pytest.mark.parametrize('clause', [24, 25])
def test_original_two_dimensional(clause):
    f = cj.chebfun2(lambda x, y: x+y)
    g = cj.chebfun(lambda t: t, domain=(-2., 2.))
    h = g(f)
    if clause == 24:
        assert float((h-f).norm()) < 10*EPS
    else:
        _source_cdr_equal(h, f.compose(g))


@pytest.mark.parametrize('clause', [26, 27, 28])
def test_original_three_dimensional(clause):
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

    f = Chebfun3.from_function(lambda x, y, z: x+y+z)
    g = cj.chebfun((lambda t: jnp.stack((t, t*t), axis=-1)) if clause == 28 else
                    (lambda t: t), domain=(-3.1, 3.1))
    h = g(f)
    if clause == 26:
        assert float((h-f).norm()) < 100*EPS
    elif clause == 27:
        assert h.isequal(f.compose(g))
    else:
        truth = Chebfun3v.from_functions(lambda x, y, z: x+y+z,
                                        lambda x, y, z: (x+y+z)**2)
        assert float((h-truth).norm()) < 100*EPS


@pytest.mark.parametrize('clause', [29, 30])
def test_original_sphere(clause):
    from chebfunjax.spherefun._constructor import spherefun
    from chebfunjax.spherefun.spherefunv import Spherefunv

    f = spherefun(lambda x, y, z: x)
    if clause == 29:
        g = cj.chebfun(lambda t: t*t, domain=(-2., 2.))
        truth = spherefun(lambda x, y, z: x*x)
    else:
        g = cj.chebfun(lambda t: jnp.stack((t, t*t, t**3), axis=-1), domain=(-2., 2.))
        truth = Spherefunv(spherefun(lambda x, y, z: x),
                           spherefun(lambda x, y, z: x*x),
                           spherefun(lambda x, y, z: x**3))
    h = g(f)
    assert float((h-truth).norm()) < EPS


@pytest.mark.parametrize('clause', [31, 32])
def test_original_disk(clause):
    from chebfunjax.diskfun.diskfun import diskfun
    from chebfunjax.diskfun.diskfunv import Diskfunv

    f = diskfun(lambda x, y: x)
    if clause == 31:
        g = cj.chebfun(lambda t: t*t)
        truth = diskfun(lambda x, y: x*x)
    else:
        g = cj.chebfun(lambda t: jnp.stack((t*t, jnp.exp(t)), axis=-1))
        truth = Diskfunv(diskfun(lambda x, y: x*x), diskfun(lambda x, y: jnp.exp(x)))
    h = g(f)
    assert float((h-truth).norm()) < EPS
