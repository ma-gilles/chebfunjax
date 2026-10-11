"""All twelve executed assignments in native tests/chebfun/test_isequal.m.

The native final vector has ten slots because the original bounded
perturbation pass(7) and singular splitting pass(8) are overwritten later.
Both overwritten predicates are retained under distinct names here.

Provenance
----------
MATLAB source : tests/chebfun/test_isequal.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj

EPS = float(jnp.finfo(jnp.float64).eps)


def op(x):
    return jnp.sin(x) * jnp.abs(x - 0.1)


@pytest.fixture(scope="module")
def f():
    return cj.chebfun(op, domain=(-1, 1), splitting=True)


@pytest.mark.parametrize("clause", range(1, 8))
def test_source_bounded_01_07(clause, f):
    if clause == 1:
        empty = cj.chebfun()
        assert empty.isequal(empty)
    elif clause == 2:
        assert f.isequal(f)
    elif clause == 3:
        assert not f.isequal(f.T)
    else:
        g = (
            cj.chebfun(jnp.sin, domain=(-1, 1), splitting=True)
            if clause == 4
            else cj.chebfun(
                lambda x: jnp.stack([op(x), jnp.cos(x)], axis=-1), domain=(-1, 1), splitting=True
            )
            if clause == 5
            else cj.chebfun(op, domain=(-1 + EPS, 1 + EPS), splitting=True)
            if clause == 6
            else cj.chebfun(lambda x: op(x) + EPS, domain=(-1, 1), splitting=True)
        )
        assert not f.isequal(g)


def test_source_singular_08_before_overwrite():
    dom = (-2, 7)
    power = -1.64
    singular = lambda x: jnp.sin(100 * x) * (x - dom[0]) ** power
    f = cj.chebfun(singular, domain=dom, exps=(power, 0), splitting=True)
    g = cj.chebfun(singular, domain=dom, exps=(power, 0), splitting=False)
    assert not f.isequal(g)


@pytest.mark.parametrize("clause", (7, 8, 9))
def test_source_unbounded_07_09(clause):
    dom = (-jnp.inf, jnp.inf)
    f = cj.chebfun(lambda x: (1 - jnp.exp(-(x**2))) / x, domain=dom)
    if clause == 7:
        assert f.isequal(f)
    else:
        g = cj.chebfun(lambda x: x**2 * (1 - jnp.exp(-(x**2))), domain=dom, exps=(2, 2))
        assert not (f.isequal(g) if clause == 8 else g.isequal(f))


def test_source_array_semibounded_10():
    f = cj.chebfun(
        lambda x: jnp.stack([jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1),
        domain=(-jnp.inf, -3 * jnp.pi),
    )
    assert f.isequal(f)
