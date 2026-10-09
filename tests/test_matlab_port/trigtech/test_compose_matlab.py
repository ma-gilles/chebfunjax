"""All 11 original predicates of tests/trigtech/test_compose.m.

MATLAB source : tests/trigtech/test_compose.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(jnp.finfo(jnp.float64).eps)
make = Trigtech.from_function


def norminf(x):
    return jnp.linalg.norm(x, ord=jnp.inf)


def col(*args):
    return jnp.stack(args, axis=-1)


def pair(x):
    return col(jnp.pi*jnp.cos(jnp.pi*x), jnp.pi*jnp.cos(2*jnp.pi*x))


@pytest.mark.parametrize("clause", [1, 2, 3, 4])
def test_unary_original(clause):
    if clause == 1:
        fun = lambda x: jnp.pi*jnp.cos(jnp.pi*(x-.1))
    elif clause in (2, 3):
        fun = pair
    else:
        fun = lambda x: col(jnp.pi*jnp.cos(jnp.pi*x), jnp.pi*jnp.cos(2*jnp.pi*x), jnp.pi*jnp.cos(3*jnp.pi*x))
    f = make(fun)
    g = f.compose(jnp.sin)
    h = make(lambda x: jnp.sin(fun(x) if clause != 4 else pair(x)))
    if clause in (1, 2):
        n = max(g.n, h.n)
        assert norminf(h.prolong(n).coeffs-g.prolong(n).coeffs) < 10*h.vscale*EPS
    else:
        assert norminf(jnp.sin(fun(trigpts(g.n)))-g.values) < 100*h.vscale*EPS


@pytest.mark.parametrize("clause", [5, 6])
def test_binary_original(clause):
    if clause == 5:
        f1 = make(lambda x: jnp.exp(jnp.sin(jnp.pi*x)))
        f2 = make(lambda x: jnp.exp(jnp.cos(jnp.pi*x)))
        g = f1.compose(jnp.add, f2)
        x = trigpts(g.n)
        h = Trigtech.from_values(jnp.exp(jnp.sin(jnp.pi*x))+jnp.exp(jnp.cos(jnp.pi*x)))
        bound = 10
    else:
        f1 = make(lambda x: jnp.exp(col(jnp.sin(jnp.pi*x), jnp.cos(jnp.pi*x))))
        f2 = make(lambda x: jnp.exp(col(jnp.cos(jnp.pi*x), jnp.sin(jnp.pi*jnp.cos(jnp.pi*x)))))
        g = f1.compose(jnp.multiply, f2)
        x = trigpts(g.n)
        h = Trigtech.from_values(col(jnp.exp(jnp.sin(jnp.pi*x)+jnp.cos(jnp.pi*x)),jnp.exp(jnp.cos(jnp.pi*x)+jnp.sin(jnp.pi*jnp.cos(jnp.pi*x)))))
        bound = 100
    assert norminf(h.values-g.values) < bound*h.vscale*EPS


@pytest.mark.parametrize("clause", [7, 8, 9])
def test_tech_of_tech_original(clause):
    if clause == 7:
        inner = lambda x: jnp.sin(jnp.pi*x)
        outer = lambda x: jnp.exp(jnp.cos(jnp.pi*x))
    elif clause == 8:
        inner = lambda x: jnp.cos(jnp.pi*jnp.sin(jnp.pi*x))
        outer = lambda x: col(jnp.sin(jnp.pi*(x-.1)),jnp.cos(jnp.pi*(x+.5)))
    else:
        inner = lambda x: col(jnp.sin(jnp.pi*(x-.1)),jnp.cos(jnp.pi*(x+.5)))
        outer = lambda x: jnp.cos(jnp.pi*jnp.sin(jnp.pi*x))
    h = make(inner).compose(make(outer))
    assert norminf(h.values-outer(inner(trigpts(h.n)))) < (10 if clause==7 else 100)*h.vscale*EPS


def test_original_error_10():
    f = make(lambda x: col(jnp.exp(jnp.pi*jnp.cos(jnp.pi*(x-.14))),jnp.exp(jnp.pi*jnp.sin(jnp.pi*(x-.14)))))
    g = make(lambda x: col(jnp.sin(jnp.pi*x),jnp.cos(jnp.pi*x)))
    with pytest.raises(ValueError, match="CHEBFUN:TRIGTECH:compose:arrval"):
        f.compose(g)


def test_original_error_11():
    f = make(lambda x: col(jnp.cos(jnp.pi*x),jnp.sin(jnp.pi*(x-.17))))
    g = make(lambda x: jnp.sin(jnp.pi*x))
    with pytest.raises(ValueError, match="CHEBFUN:TRIGTECH:compose:dim"):
        f.compose(jnp.add, g)
