"""Port of MATLAB Chebfun tests/chebtech/test_sum.m (Opus 4.8; marker audit
Fable 5).

Self-validating: definite integrals are compared against analytic exacts at
the SAME tolerances MATLAB uses.  The MATLAB test loops ``for n = 1:2`` over
``{chebtech1(), chebtech2()}``; we parametrize over ``[Chebtech1, Chebtech2]``.

Every MATLAB assertion (pass 1-11) is ported on BOTH tech kinds; there are no
gaps:

* Array-valued techs are supported ((n, m) coefficient matrices), so the
  array-valued and DIM-option assertions (pass 9-11) are real tests.
* Complex-valued construction works on Chebtech1 as well as Chebtech2
  (vals2coeffs/coeffs2vals split complex data into re/im), so the
  ``sinh(t*z)`` sub-test (pass 4) runs on both.
* The ``cos(1e4*x)`` relative-error sub-test (pass 3) now meets MATLAB's
  ``1e6*vscale*eps`` bound on both kinds (measured 0.64x on Chebtech1,
  0.18x on Chebtech2).

Provenance
----------
MATLAB source : tests/chebtech/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

BOTH = [Chebtech1, Chebtech2]

# pass(n, 3): cos(1e4*x) relative-error quadrature.  Chebtech1 once carried a
# non-strict xfail for a ~1.2x margin over MATLAB's 1e6*vscale*eps bound; that
# was removed 2026-07-30 and re-measured 2026-08-10 at 0.64x (Chebtech1) and
# 0.18x (Chebtech2), so both kinds are enforced strictly.
SUM_COS_TECHS = [Chebtech1, Chebtech2]


def _at(f, x):
    return f(jnp.array([float(x)]))[0]


class TestChebtechSum:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_integral_exp(self, Tech):
        # pass(n, 1)
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        assert abs(float(f.sum()) - 0.350402387287603) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_integral_lorentzian(self, Tech):
        # pass(n, 2)
        f = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        assert abs(float(f.sum()) - np.pi / 2) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", SUM_COS_TECHS)
    def test_integral_high_frequency(self, Tech):
        # pass(n, 3)
        f = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        exact = -6.112287777765043e-05
        assert abs(float(f.sum()) - exact) / abs(exact) < 1e6 * f.vscale * EPS

    # pass(n, 4): sinh(t*z) is complex-valued.  Both tech kinds handle complex
    # data now, so this runs on Chebtech1 and Chebtech2 (MATLAB's n = 1:2).
    @pytest.mark.parametrize("Tech", BOTH)
    def test_integral_sinh_complex_is_zero(self, Tech):
        z = np.exp(2 * np.pi * 1j / 6)
        f = Tech.from_function(lambda t: jnp.sinh(t * z))
        assert abs(complex(f.sum())) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_linearity(self, Tech):
        # pass(n, 5): sum(a*f + b*g) == a*sum(f) + b*sum(g), a=2, b=-1i.
        # f and g are real-valued; the complex combination is formed by
        # coefficient arithmetic (scalar multiply), which both techs support.
        a = 2.0
        b = -1j
        f = Tech.from_function(lambda x: x * jnp.sin(x ** 2) - 1)
        g = Tech.from_function(lambda x: jnp.exp(-x ** 2))
        tol_f = 10 * f.vscale * EPS
        tol_g = 10 * f.vscale * EPS  # MATLAB uses vscale(f) here (line 47)
        lhs = complex((a * f + b * g).sum())
        rhs = complex(a * f.sum() + b * g.sum())
        assert abs(lhs - rhs) < max(tol_f, tol_g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_integration_by_parts(self, Tech):
        # pass(n, 6)
        f = Tech.from_function(lambda x: x * jnp.sin(x ** 2) - 1)
        df = f.diff()
        g = Tech.from_function(lambda x: jnp.exp(-x ** 2))
        dg = g.diff()
        fg = f * g
        gdf = g * df
        fdg = f * dg
        tol_fg = 10 * fg.vscale * EPS
        tol_fdg = 10 * fdg.vscale * EPS
        tol_gdf = 10 * gdf.vscale * EPS
        lhs = complex(fdg.sum())
        rhs = complex(_at(fg, 1) - _at(fg, -1) - gdf.sum())
        assert abs(lhs - rhs) < max(tol_fdg, tol_gdf, tol_fg)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_ftc_f(self, Tech):
        # pass(n, 7): sum(df) == f(1) - f(-1)
        f = Tech.from_function(lambda x: x * jnp.sin(x ** 2) - 1)
        df = f.diff()
        tol_df = 10 * df.vscale * EPS
        tol_f = 10 * f.vscale * EPS
        assert abs(complex(df.sum()) - complex(_at(f, 1) - _at(f, -1))) < max(
            tol_df, tol_f
        )

    @pytest.mark.parametrize("Tech", BOTH)
    def test_ftc_g(self, Tech):
        # pass(n, 8): sum(dg) == g(1) - g(-1)
        g = Tech.from_function(lambda x: jnp.exp(-x ** 2))
        dg = g.diff()
        tol_dg = 10 * dg.vscale * EPS
        tol_g = 10 * g.vscale * EPS
        assert abs(complex(dg.sum()) - complex(_at(g, 1) - _at(g, -1))) < max(
            tol_dg, tol_g
        )

    # FIXED (Fable 5, Big-Three array-valued epic): pass 9-11 port now
    # that techs support (n, m) coefficient matrices and sum(dim=2).
    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_sum(self, Tech):
        # pass(n, 9): sum of [sin(x), x.^2, exp(1i*x)] column-wise.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), x ** 2, jnp.exp(1j * x)], axis=-1))
        I = np.asarray(f.sum())
        I_exact = np.array([0.0, 2.0 / 3.0, 2 * np.sin(1.0)])
        assert np.max(np.abs(I - I_exact)) < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_dim_option_array_valued(self, Tech):
        # pass(n, 10): sum(f, 2) collapses the columns pointwise.
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), x ** 2, jnp.exp(1j * x)], axis=-1))
        g = f.sum(dim=2)
        xs = jnp.asarray(np.linspace(-1.0, 1.0, 100))
        h = jnp.sin(xs) + xs ** 2 + jnp.exp(1j * xs)
        err = float(jnp.max(jnp.abs(g(xs) - h)))
        assert err < 10 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_dim_option_scalar_noop(self, Tech):
        # pass(n, 11): sum(h, 2) on a scalar-valued tech is a no-op.
        h = Tech.from_function(lambda x: jnp.cos(x))
        sumh2 = h.sum(dim=2)
        assert np.array_equal(np.asarray(h.coeffs),
                              np.asarray(sumh2.coeffs))


def test_chebtech1_rejects_complex():
    # FIXED (Fable 5): Chebtech1 now splits complex data into re/im
    # in vals2coeffs/coeffs2vals; this sentinel now passes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = Chebtech1.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
    assert f.ishappy
