"""Port of MATLAB Chebfun tests/singfun/test_rdivide.m (Opus 4.8).

Self-validating: each quotient is checked against ``fh(x)/gh(x)`` at the SAME
tolerance MATLAB uses.  The singfun 'vscale' maps to ``f.smoothPart.vscale``
(MATLAB ``get(f,'vscale')`` returns the smooth-part vscale).  Test points:
interior grid, MATLAB drops the 4 points nearest each endpoint.

Provenance
----------
MATLAB source : tests/singfun/test_rdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)

X = jnp.asarray(np.sort(np.linspace(-0.99, 0.99, 100)))
XI = X[4:-4]  # drop points nearest the endpoints, as MATLAB does


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _vscale(f):
    # MATLAB get(singfun, 'vscale') == vscale(smoothPart).  A smooth quotient
    # is demoted to a bare Chebtech2, whose own vscale is the smoothfun vscale.
    return getattr(f, "smoothPart", f).vscale


class TestSingfunRdivide:
    def test_empty(self):
        f = Singfun.empty()
        g = _sf(lambda x: 1.0 / (1 + x), (-1.0, 0.0))
        assert (f / g).isempty()
        assert (g / f).isempty()
        assert (f / f).isempty()

    def test_smoothfun_rdivide_singfun(self):
        # MATLAB: smoothfun ./ (smooth) singfun isa smoothfun, both ways.
        f = Chebtech2.from_function(lambda x: 2.0 + jnp.sin(x))
        g = _sf(lambda x: jnp.cos(x), (0.0, 0.0))
        assert isinstance(f / g, Chebtech2) and not isinstance(f / g, Singfun)
        assert isinstance(g / f, Chebtech2) and not isinstance(g / f, Singfun)

    def test_smooth_div_smooth_not_singfun(self):
        f = _sf(lambda x: jnp.sin(x), (0.0, 0.0))
        g = _sf(lambda x: jnp.cos(x), (0.0, 0.0))
        assert not isinstance(f / g, Singfun)

    def test_divide_by_scalar(self):
        def fh(x):
            return 1.0 / ((1 + x) * (1 - x))

        f = _sf(fh, (-1.0, -1.0))
        c = 0.37
        g = f / c
        exact = fh(X) / c
        assert _ninf(g(X) - exact) <= 2e3 * _vscale(g) * EPS

    def test_reciprocal_of_smooth(self):
        f = _sf(lambda x: 0 * x + 1.0, (0.0, 0.0))
        g = _sf(lambda x: jnp.cos(x), (0.0, 0.0))
        h = f / g
        exact = 1.0 / jnp.cos(XI)
        assert _ninf(h(XI) - exact) <= 1e3 * _vscale(h) * EPS

    def test_divide_smooth_by_itself(self):
        f = _sf(lambda x: jnp.cos(x), (0.0, 0.0))
        h = f / f
        exact = jnp.ones_like(XI)
        assert _ninf(h(XI) - exact) <= 1e3 * _vscale(h) * EPS

    def test_divide_creating_poles(self):
        f = _sf(lambda x: jnp.sin(x), (0.0, 0.0))
        g = _sf(lambda x: (1 + x) * (1 - x), (0.0, 0.0))
        h = f / g
        exact = jnp.sin(XI) / ((1 + XI) * (1 - XI))
        assert _ninf(h(XI) - exact) <= 1e3 * _vscale(h) * EPS

    def test_reciprocal_flips_exponents(self):
        # g = (1+x)^a (1-x)^b with positive integer exponents; 1/g flips them.
        a, b = 3, 4
        g = _sf(lambda x: ((1 + x) ** a) * ((1 - x) ** b), (float(a), float(b)))
        h = 1.0 / g
        assert tuple(h.exponents) == (-float(a), -float(b))

    def test_reciprocal_simplifies_exponents(self):
        a, b = 3, 4
        g = _sf(lambda x: ((1 + x) ** -a) * ((1 - x) ** -b), (-float(a), -float(b)))
        h = 1.0 / g
        # 1/((1+x)^-3 (1-x)^-4) has positive *integer* exponents (3, 4) that
        # simplify entirely into the smooth part, leaving a smooth result which
        # is demoted to a bare Chebtech2 (i.e. exponents (0, 0), all < 1).
        if isinstance(h, Singfun):
            assert all(e < 1 for e in h.exponents)

    def test_simplify_positive_exponent(self):
        f = _sf(lambda x: 1 + x, (0.0, 0.0))
        g = _sf(lambda x: jnp.sqrt(1 + x), (0.5, 0.0))
        h = f / g
        assert tuple(h.exponents) == (0.5, 0.0)

    def test_division_as_differentiation(self):
        # pass(12): on [1, Inf], diff(1/x) + (1/x)/x == 0.
        import numpy as np

        from chebfunjax.chebfun1d.chebfun import chebfun
        r = chebfun(lambda x: x, domain=(1.0, np.inf))
        f = 1.0 / r
        err = f.diff() + f / r
        assert float(err.norm(np.inf)) < 1e3 * EPS * 30.0

    def test_division_as_negative_powers(self):
        # pass(13-15): rdivide == power, rdivides == times, associativity.
        import numpy as np

        from chebfunjax.chebfun1d.chebfun import chebfun
        r = chebfun(lambda x: x, domain=(1.0, np.inf))
        f = 1.0 / r - r ** -1
        g = (1.0 / r) / r - r ** -2
        h = 1.0 / (r ** 2) - (1.0 / r) ** 2
        assert float(f.norm(np.inf)) < 30.0 * EPS
        assert float(g.norm(np.inf)) < 30.0 * EPS
        assert float(h.norm(np.inf)) < 1e2 * 30.0 * EPS
