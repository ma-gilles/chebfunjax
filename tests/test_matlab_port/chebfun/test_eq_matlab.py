"""Port of MATLAB Chebfun tests/chebfun/test_eq.m (Fable 5).

MATLAB f == g returns a logical chebfun (pointwise equality regions);
chebfunjax has no pointwise-eq chebfun -- roots of f - g cover the
underlying assertion.

Provenance
----------
MATLAB source : tests/chebfun/test_eq.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunEq:
    def test_pointwise_eq_chebfun(self):
        # MATLAB pass(2): the crossing of sin(x) with sqrt(2)/2 shows up
        # as a pointValue of 1 at the breakpoint pi/4.
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        g = cj.chebfun(lambda x: 0 * x + float(np.sqrt(2) / 2))
        h = f.logical_eq(g)
        pv = np.asarray(h.point_values)
        dom = np.asarray(list(h.domain.breakpoints))
        ind = np.flatnonzero(pv == 1)
        assert len(ind) == 1
        assert abs(dom[ind][0] - np.pi / 4) < 10 * EPS

    def test_eq_two_crossings(self):
        # MATLAB pass(3, 6): exp(x) against its secant through +-0.5
        # crosses at -0.5 and 0.5, both with and without pre-existing
        # breakpoints there.
        secant = lambda x: (np.exp(0.5) - np.exp(-0.5)) * (
            x + 0.5) + np.exp(-0.5)  # noqa: E731
        for dom in ([-1.0, 1.0], [-1.0, -0.5, 0.0, 0.5, 1.0]):
            f = cj.chebfun(jnp.exp, domain=dom)
            g = cj.chebfun(secant)
            h = f.logical_eq(g)
            pv = np.asarray(h.point_values)
            bps = np.asarray(list(h.domain.breakpoints))
            ind = np.flatnonzero(pv == 1)
            assert float(np.max(np.abs(np.sort(bps[ind])
                                       - np.array([-0.5, 0.5])))) < 10 * EPS

    def test_eq_self_and_negation(self):
        # MATLAB pass(4, 5): f == f is identically 1 on a single piece and
        # f == -f is identically 0 on a single piece.
        rng = np.random.default_rng(6178)
        x = jnp.asarray(2 * rng.uniform(size=100) - 1)
        f = cj.chebfun(jnp.exp)
        h = f.logical_eq(f)
        assert len(h.funs) == 1 and np.all(np.asarray(h(x)) == 1)
        h = f.logical_eq(-f)
        assert len(h.funs) == 1 and np.all(np.asarray(h(x)) == 0)

    def test_eq_empty(self):
        # MATLAB pass(1): the empty case propagates.
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        g = cj.chebfun()
        assert f.logical_eq(g).isempty()
        assert g.logical_eq(f).isempty()

    def test_eq_array_valued_raises(self):
        # MATLAB pass(7, 8): 'CHEBFUN:CHEBFUN:eq:array'.
        f = cj.chebfun(lambda x: jnp.stack(
            [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
            domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        g = cj.chebfun(jnp.exp)
        with pytest.raises(ValueError):
            f.logical_eq(g)
        with pytest.raises(ValueError):
            g.logical_eq(f)

    def test_equality_locations_via_roots(self):
        # MATLAB: sin(x) == sqrt(2)/2 has solutions where sin crosses
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        c = float(np.sqrt(2) / 2)
        r = np.asarray((f - c).roots())
        exact = np.array([np.pi / 4])
        assert len(r) == 1
        assert abs(r[0] - exact[0]) < 100 * EPS
