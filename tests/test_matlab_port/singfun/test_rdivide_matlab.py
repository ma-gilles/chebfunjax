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
import pytest

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

X = jnp.asarray(np.sort(np.linspace(-0.99, 0.99, 100)))
XI = X[4:-4]  # drop points nearest the endpoints, as MATLAB does


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _vscale(f):
    # MATLAB get(singfun, 'vscale') == vscale(smoothPart)
    return f.smoothPart.vscale


class TestSingfunRdivide:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty Singfun representation")

    def test_smoothfun_rdivide_singfun(self):
        pytest.skip("chebfunjax has no separate smoothfun class")

    @pytest.mark.xfail(
        reason="chebfunjax f/g of two smooth Singfuns always returns a Singfun; "
        "it never demotes to a bare smoothfun",
        strict=True,
    )
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

    @pytest.mark.xfail(
        reason="chebfunjax does not simplify/absorb exponents >= 1 into the "
        "smooth part; 1/((1+x)^-a (1-x)^-b) keeps exponents (a, b) >= 1",
        strict=True,
    )
    def test_reciprocal_simplifies_exponents(self):
        a, b = 3, 4
        g = _sf(lambda x: ((1 + x) ** -a) * ((1 - x) ** -b), (-float(a), -float(b)))
        h = 1.0 / g
        assert all(e < 1 for e in h.exponents)

    @pytest.mark.xfail(
        reason="chebfunjax does not canonicalise exponents: (1+x)/sqrt(1+x) "
        "yields exponents (-0.5, 0) with smooth part (1+x) rather than the "
        "simplified (0.5, 0)",
        strict=True,
    )
    def test_simplify_positive_exponent(self):
        f = _sf(lambda x: 1 + x, (0.0, 0.0))
        g = _sf(lambda x: jnp.sqrt(1 + x), (0.5, 0.0))
        h = f / g
        assert tuple(h.exponents) == (0.5, 0.0)

    def test_division_as_differentiation(self):
        pytest.skip("MATLAB case uses chebfun on the unbounded domain [1, Inf]")

    def test_division_as_negative_powers(self):
        pytest.skip("MATLAB case uses chebfun on the unbounded domain [1, Inf]")
