"""Port of MATLAB Chebfun tests/singfun/test_plus.m (Opus 4.8).

Self-validating: each sum is checked against ``fh(x) + gh(x)`` at the SAME
tolerance MATLAB uses.  Where MATLAB relied on automatic exponent detection
(``singfun(fh, [])``) we supply the analytically-known exponents explicitly,
which reconstructs the same math (chebfunjax has no exponent auto-detection).
Test points: interior grid ``[-0.99, 0.99]``.

Provenance
----------
MATLAB source : tests/singfun/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

X = jnp.asarray(np.linspace(-0.99, 0.99, 100))


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _exps(f):
    # A smooth (all-zero-exponent) sum is demoted to a bare Chebtech2, which
    # carries no `exponents` attribute; that corresponds to exponents (0, 0).
    return tuple(f.exponents) if isinstance(f, Singfun) else (0.0, 0.0)


def _isequal(f, g):
    # chebfunjax has no isequal method; MATLAB isequal compares exponents and
    # the (deterministically constructed) smooth-part coefficients.
    if _exps(f) != _exps(g):
        return False
    cf, cg = f.coeffs, g.coeffs
    return cf.shape == cg.shape and bool(jnp.array_equal(cf, cg))


class TestSingfunPlus:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty Singfun representation")

    def test_smooth_plus_smooth_not_singfun(self):
        f = _sf(lambda x: jnp.sin(x), (0.0, 0.0))
        g = _sf(lambda x: jnp.cos(x), (0.0, 0.0))
        assert not isinstance(f + g, Singfun)

    def test_smoothfun_plus_singfun(self):
        pytest.skip("chebfunjax has no separate smoothfun class")

    def test_add_complex_scalar(self):
        alpha = -0.194758928283640 + 0.075474485412665j
        f = _sf(lambda x: 1.0 / ((1 + x) * (1 - x)), (-1.0, -1.0))
        g1 = f + alpha
        g2 = alpha + f
        assert _isequal(g1, g2)

    def test_add_zero_to_zero(self):
        f = _sf(lambda x: jnp.zeros_like(x), (0.0, 0.0))
        h1 = f + f
        h2 = f + f
        assert _isequal(h1, h2)
        assert _ninf(h1(X)) <= 2e3 * EPS

    def test_add_same_exponents(self):
        # f = sin(pi x)/(1-x), g = cos(pi x)/(1-x); both have exponents (0, -1)
        def fh(x):
            return jnp.sin(np.pi * x) / (1 - x)

        def gh(x):
            return jnp.cos(np.pi * x) / (1 - x)

        f = _sf(fh, (0.0, -1.0))
        g = _sf(gh, (0.0, -1.0))
        h1 = f + g
        h2 = g + f
        assert _isequal(h1, h2)
        exact = fh(X) + gh(X)
        assert _ninf(h1(X) - exact) <= 2e3 * EPS

    def test_add_integer_exponent_diff(self):
        # f exps (0, -1), g smooth cos(1e2 x) exps (0, 0)
        def fh(x):
            return jnp.sin(np.pi * x) / (1 - x)

        def gh(x):
            return jnp.cos(1e2 * x)

        f = _sf(fh, (0.0, -1.0))
        g = _sf(gh, (0.0, 0.0))
        h1 = f + g
        h2 = g + f
        assert _isequal(h1, h2)
        exact = fh(X) + gh(X)
        assert _ninf(h1(X) - exact) <= 2e3 * EPS

    def test_add_complex_function(self):
        # FIXED (Fable 5): the Chebtech1 complex-transform fix made
        # complex smooth parts work in Singfun too.
        def fh(x):
            return jnp.sin(np.pi * x) / (1 - x)

        def gh(t):
            return jnp.sinh(t * np.exp(2 * np.pi * 1j / 6))

        f = _sf(fh, (0.0, -1.0))
        g = _sf(gh, (0.0, 0.0))
        h = f + g
        exact = fh(X) + gh(X)
        assert _ninf(h(X) - exact) <= 2e3 * EPS

    def test_plus_vs_direct_construction(self):
        f = _sf(lambda x: x, (0.0, 0.0))
        g = _sf(lambda x: jnp.cos(x) - 1, (0.0, 0.0))
        h1 = f + g  # smooth sum -> demoted to a bare Chebtech2
        h2 = _sf(lambda x: x + jnp.cos(x) - 1, (0.0, 0.0))
        vs1 = getattr(h1, "smoothPart", h1).vscale
        vs2 = getattr(h2, "smoothPart", h2).vscale
        tol = 10 * max(vs1 * EPS, vs2 * EPS)
        assert _ninf(h1(X) - h2(X)) < tol

    @pytest.mark.xfail(
        reason="MATLAB test relies on pref.fixedLength=256 and get(h,'ishappy'); "
        "chebfunjax has neither a fixed-length preference nor an ishappy flag",
        strict=True,
    )
    def test_noninteger_exponent_diff_small_result(self):
        def op(x):
            return ((x + 1) / 2) ** np.pi

        f = _sf(op, (np.pi - 3, 0.0))
        g = _sf(op, (0.0, 0.0))
        h = f - g
        assert getattr(h, "ishappy") and len(h) < 1024
