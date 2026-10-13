"""All10 native tests/chebfun/test_ne.m predicates, Chebfun7574c77.

Public logical_ne adapts MATLAB~=; Python!= retains object comparison.
Native row-vector infinity norm is maxabsolute, not matrixrowsum.
RNG6178 first100 draws use retained nativefixture; second100 draws are
validated against fixture.y then mapped to native[-100,100] testinterval.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

FIX = json.loads((Path(__file__).parent / "fixtures/logical_source_matlab.json").read_text())
X = jnp.asarray(FIX["x"])
D = [-1.0, -0.5, 0.0, 0.5, 1.0]
EPS = np.finfo(float).eps


def gop(x):
    return (jnp.exp(0.5) - jnp.exp(-0.5)) * (x + 0.5) + jnp.exp(-0.5)


@pytest.mark.parametrize("clause", range(1, 11))
def test_native_ne(clause):
    if clause == 1:
        f = cj.chebfun(jnp.sin, domain=D)
        g = cj.chebfun()
        assert f.logical_ne(g).isempty() and g.logical_ne(f).isempty()
    elif clause == 2:
        f = cj.chebfun(jnp.sin, domain=D)
        h = f.logical_ne(cj.chebfun(lambda x: 0 * x + jnp.sqrt(2) / 2))
        ind = np.flatnonzero(np.asarray(h.point_values) == 0)
        assert ind.size == 1
        assert np.all(
            np.abs(np.asarray(h.domain.breakpoints)[ind] - np.pi / 4) < 10 * float(h.vscale) * EPS
        )
    elif clause in (3, 6):
        f = cj.chebfun(jnp.exp, domain=D if clause == 6 else [-1.0, 1.0])
        h = f.logical_ne(cj.chebfun(gop))
        ind = np.flatnonzero(np.asarray(h.point_values) == 0)
        # Native indexed domains are rowvectors; flatten retains native
        # vector-infinity norm (maxabs) and requires the original2entries.
        diff = np.asarray(h.domain.breakpoints)[ind] - np.array([-0.5, 0.5])
        assert diff.shape == (2,)
        assert np.max(np.abs(diff)) < 10 * float(h.vscale) * EPS
    elif clause in (4, 5):
        f = cj.chebfun(jnp.exp)
        h = f.logical_ne(f if clause == 4 else -f)
        assert len(h.funs) == 1 and np.all(np.asarray(h(X)) == (clause == 5))
    elif clause in (7, 8):
        f = cj.chebfun(lambda x: jnp.stack((jnp.sin(x), jnp.cos(x), jnp.exp(x)), axis=-1), domain=D)
        g = cj.chebfun(lambda x: jnp.exp(2 * jnp.pi * 1j * x))
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:ne:array"):
            f.logical_ne(g) if clause == 7 else g.logical_ne(f)
    elif clause == 9:
        f = cj.chebfun(lambda x: -jnp.sin(x) / (x + 1), exps=(-1, 0))
        h = f.logical_ne(f)
        assert np.asarray(h.point_values)[0] == 0 and not np.any(np.asarray(h(X)))
    else:
        rng = np.random.RandomState(6178)
        np.testing.assert_array_equal(2 * rng.rand(100) - 1, FIX["x"])
        second = rng.rand(100)
        np.testing.assert_array_equal(9 * second - 2, FIX["y"])
        x = jnp.asarray(200 * second - 100)
        f = cj.chebfun(
            lambda x: x**2 * (1 - jnp.exp(-(x**2))), domain=(-jnp.inf, jnp.inf), exps=(2, 2)
        )
        assert not np.any(np.asarray(f.logical_ne(f)(x)))


def test_native_eq_ne_unbounded_preserve_mapping():
    """@chebfun/eq/ne retain each unbounded FUN's mapping when zeroing."""
    from chebfunjax.fun.unbndfun import Unbndfun

    f = cj.chebfun(
        lambda x: x**2 * (1 - jnp.exp(-(x**2))),
        domain=(-jnp.inf, jnp.inf), exps=(2, 2))
    equal = f.logical_eq(f)
    unequal = f.logical_ne(f)
    zero = f.logical_eq(0.0)
    assert all(isinstance(piece, Unbndfun) for piece in equal.funs)
    assert all(isinstance(piece, Unbndfun) for piece in unequal.funs)
    assert all(isinstance(piece, Unbndfun) for piece in zero.funs)
    x = jnp.asarray([-100.0, -1.0, 0.0, 1.0, 100.0])
    np.testing.assert_array_equal(np.asarray(equal(x)), np.ones(5))
    np.testing.assert_array_equal(np.asarray(unequal(x)), np.zeros(5))
    # This supplemental mapping control samples where the input is nonzero.
    # The zero at x=0 needs its own native root/sign capture.
    nonzero_x = jnp.asarray([-100.0, -1.0, 1.0, 100.0])
    np.testing.assert_array_equal(np.asarray(zero(nonzero_x)), np.zeros(4))


def test_native_singular_eq_ne_pointvalues():
    """Native eq preserves NaN PVs; ne deletes them before logical()."""
    f = cj.chebfun(lambda x: -jnp.sin(x) / (x + 1), exps=(-1, 0))
    zero = f - f
    eq = f.logical_eq(f)
    ne = f.logical_ne(f)
    np.testing.assert_array_equal(
        np.isnan(np.asarray(zero.sign().point_values)), [True, False])
    np.testing.assert_array_equal(
        np.isnan(np.asarray(eq.point_values)), [True, False])
    assert np.asarray(eq.point_values)[1] == 1
    np.testing.assert_array_equal(np.asarray(ne.point_values), [0.0])
    # The pinned native result has a one-entry pointValues vector for two
    # domain endpoints; querying the missing right-end entry raises likewise.
    with pytest.raises(IndexError):
        ne(1.0)
