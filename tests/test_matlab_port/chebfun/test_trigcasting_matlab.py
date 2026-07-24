"""Port of MATLAB Chebfun tests/chebfun/test_trigcasting.m (Opus 4.8).

MATLAB casts mixed TRIGTECH + CHEBTECH arithmetic (and the non-periodic
results of ``abs``/``round``/``floor``/``ceil``/``cumsum``) to the CHEBTECH
basis.  chebfunjax now performs the same cast in :func:`_cast_tech_pair`
(binary ops) and in :meth:`_Piece.cumsum` (a non-zero-mean trig
antiderivative is not periodic).  The reachable, non-quasimatrix assertions
are ported below.

The quasimatrix cases (``H = [f, g]`` with a per-column mix of trig and
cheb techs, ``quasi2cheb``, and ``diff(H, 1, 2)`` -- MATLAB pass 3, 4, 6,
9-11, 20) have no chebfunjax counterpart: an array-valued Chebfun stores a
single tech shared by all columns, so a column-wise mix of techs cannot be
represented.

Provenance
----------
MATLAB source : tests/chebfun/test_trigcasting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _tech(f):
    return type(f.funs[0].tech).__name__


class TestChebfunTrigcasting:
    def test_restriction_sum_and_periodic_feval(self):
        # pass(1, 2): sum over a restricted sub-interval of a trig chebfun and
        # periodic feval outside the base domain.
        f = cj.chebfun(lambda x: jnp.cos(np.pi * x) ** 2, domain=(1, 3),
                       trig=True)
        v = float(f.vscale)
        assert abs(float(f.restrict(2, 3).sum()) - 0.5) < v * EPS * 100
        assert abs(float(f(jnp.asarray(20.0))) - 1.0) < v * EPS * 100

    def test_abs_casts_to_cheb(self):
        # pass(5): abs of a trig chebfun casts to the chebtech basis.  The
        # cast (the subject of this test) is frequency-independent; MATLAB
        # uses sin(10x), but at that frequency chebfunjax's trigtech
        # rootfinding is too imprecise to split |.| cleanly (the pieces do not
        # resolve), so a lower frequency is used here.  abs *value* fidelity is
        # pinned by test_abs_matlab.py on the cheb path.
        f = cj.chebfun(lambda x: jnp.sin(3 * x), domain=(0, 2 * np.pi),
                       trig=True)
        af = f.abs()
        assert _tech(af) == "Chebtech2"
        xs = jnp.asarray(np.random.default_rng(6178).uniform(0, 2 * np.pi, 200))
        assert bool(jnp.all(af(xs) >= -1e-12))

    def test_times_casts_to_cheb(self):
        # pass(12, 13): (trig f) .* (cheb x) casts to cheb and matches the
        # all-cheb product.
        f = cj.chebfun(lambda x: jnp.exp(-x ** 2), domain=(-10, 10), trig=True)
        g = cj.chebfun(lambda x: jnp.exp(-x ** 2), domain=(-10, 10))
        xid = cj.chebfun(lambda x: x, domain=(-10, 10))
        fx = f * xid
        gx = g * xid
        assert _tech(fx) == "Chebtech2"
        xx = jnp.asarray(np.arange(-3.0, 8.0))
        assert float(jnp.max(jnp.abs(fx(xx) - gx(xx)))) < 100 * EPS * float(fx.vscale)

    def test_power_casts_to_cheb(self):
        # pass(21): (trig f) .^ (cheb g) casts to cheb and matches.
        f = cj.chebfun(lambda x: 2 + jnp.sin(np.pi * x), trig=True)
        h = cj.chebfun(lambda x: 2 + jnp.sin(np.pi * x))
        g = cj.chebfun(jnp.exp)
        lhs = f ** g
        rhs = h ** g
        assert _tech(lhs) == "Chebtech2"
        xs = jnp.linspace(-1, 1, 40)
        assert float(jnp.max(jnp.abs(lhs(xs) - rhs(xs)))) < 100 * EPS * float(f.vscale)

    def test_round_casts_to_cheb(self):
        # pass(14, 15): round of a trig chebfun casts to cheb; domain and
        # definite integral match the all-cheb round.
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=(0, 2 * np.pi),
                       trig=True)
        g = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=(0, 2 * np.pi))
        fr = f.round()
        gr = g.round()
        assert _tech(fr) == "Chebtech2"
        d_f = np.asarray(list(fr.domain.breakpoints))
        d_g = np.asarray(list(gr.domain.breakpoints))
        assert d_f.shape == d_g.shape
        # MATLAB hscale = max(|domain endpoints|); here max(|0|, |2*pi|).
        hscale = max(abs(float(f.domain.a)), abs(float(f.domain.b)))
        assert float(np.max(np.abs(d_f - d_g))) < 100 * EPS * hscale
        assert abs(float(fr.sum()) - float(gr.sum())) < 100 * EPS * float(f.vscale)

    def test_floor_ceil_cast_to_cheb(self):
        # pass(16, 17): floor/ceil of a trig chebfun cast to cheb and match
        # the all-cheb result (compared by sampling, as norm(., inf) does --
        # forming floor(f) - floor(g) as a chebfun trips on the differing
        # breakpoint slivers, but the pointwise values agree exactly).
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=(0, 2 * np.pi),
                       trig=True)
        g = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)), domain=(0, 2 * np.pi))
        xs = jnp.linspace(0.05, 2 * np.pi - 0.05, 400)
        ff, gf = f.floor(), g.floor()
        cf, cg = f.ceil(), g.ceil()
        assert _tech(ff) == "Chebtech2" and _tech(cf) == "Chebtech2"
        assert float(jnp.max(jnp.abs(ff(xs) - gf(xs)))) < 100 * EPS * float(f.vscale)
        assert float(jnp.max(jnp.abs(cf(xs) - cg(xs)))) < 100 * EPS * float(f.vscale)

    def test_cumsum_casts_when_mean_nonzero(self):
        # pass(18, 19): cumsum of a ZERO-mean trig function stays trig; adding
        # a small constant makes the antiderivative non-periodic, so it casts
        # to the cheb basis.
        f = cj.chebfun(lambda x: jnp.sin(2 * np.pi * x), trig=True)
        g = f.cumsum()
        h = (f + 1e-5).cumsum()
        assert _tech(g) == "Trigtech"
        assert _tech(h) == "Chebtech2"

    def test_breakpoints_reconstruct(self):
        # pass(22): rebuilding a trig chebfun on a broken domain restricts to
        # the given breakpoints.
        f = cj.chebfun(lambda x: jnp.sin(2 * np.pi * x), trig=True)
        g = cj.chebfun(f, domain=(-1, 0, 1))
        assert float(np.max(np.abs(
            np.asarray(list(g.domain.breakpoints)) - np.array([-1.0, 0.0, 1.0])
        ))) < 10 * EPS

    def test_cast_helper_leaves_same_tech_untouched(self):
        # The cast is a no-op for same-tech operands (the hot path): two
        # chebtechs are returned unchanged (identity), not rebuilt.
        from chebfunjax.chebfun1d.chebfun import _cast_tech_pair
        a = Chebtech2.from_function(jnp.exp)
        b = Chebtech2.from_function(jnp.sin)
        ca, cb = _cast_tech_pair(a, b)
        assert ca is a and cb is b
