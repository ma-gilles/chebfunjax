"""Port of MATLAB Chebfun tests/chebtech/test_diff.m (Opus 4.8).

Self-validating: each operation is checked against an analytic exact at the
SAME tolerance MATLAB uses (multiples of vscale*eps).  No .mat fixture needed.

The MATLAB test loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; we
parametrize each ported assertion over ``[Chebtech1, Chebtech2]``.

Notes on gaps (see the report):
* Array-valued / DIM-option assertions (pass 12-16) are skipped — chebfunjax
  Chebtech is scalar-valued only.
* The airy sub-test (pass 4) uses a complex-valued function; chebfunjax
  Chebtech1 discards the imaginary part in its transforms, so that case is
  ported over Chebtech2 only (see ``test_chebtech1_rejects_complex``).

Provenance
----------
MATLAB source : tests/chebtech/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
# Deterministic test points in [-1, 1] (analytic checks hold at any x).
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))

BOTH = [Chebtech1, Chebtech2]


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _airy_ai(w):
    # MATLAB airy(z*t) == Ai(z*t); scipy.special.airy returns (Ai, Aip, Bi, Bip).
    return jnp.asarray(sp.airy(np.asarray(w))[0])


def _airy_aip(w):
    # MATLAB airy(1, z*t) == Ai'(z*t).
    return jnp.asarray(sp.airy(np.asarray(w))[1])


class TestChebtechDiff:
    @pytest.mark.parametrize("Tech", BOTH)
    def test_spotcheck_exp(self, Tech):
        # pass(n, 1)
        f = Tech.from_function(lambda x: jnp.exp(x) - x)
        df = f.diff()
        err = _ninf(jnp.exp(X) - 1 - df(X))
        assert err < 100 * df.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_spotcheck_atan(self, Tech):
        # pass(n, 2)
        f = Tech.from_function(lambda x: jnp.arctan(x))
        df = f.diff()
        err = _ninf(1.0 / (1 + X ** 2) - df(X))
        tol = 500 * df.vscale * EPS
        assert err < 10 * tol

    @pytest.mark.parametrize("Tech", BOTH)
    def test_spotcheck_sin(self, Tech):
        # pass(n, 3)
        f = Tech.from_function(lambda x: jnp.sin(x))
        df = f.diff()
        err = _ninf(jnp.cos(X) - df(X))
        assert err < 100 * df.vscale * EPS

    # pass(n, 4): complex-valued airy — Chebtech2 only (Chebtech1 drops imag).
    @pytest.mark.parametrize("Tech", [Chebtech2])
    def test_spotcheck_airy_complex(self, Tech):
        z = np.exp(2 * np.pi * 1j / 3)
        f = Tech.from_function(lambda t: _airy_ai(z * t))
        df = f.diff()
        err = _ninf(z * _airy_aip(z * X) - df(X))
        assert err < 1e3 * df.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_diff_equals_direct_construction(self, Tech):
        # pass(n, 5): diff(0.5x - 0.0625 sin 8x) == sin(4x)^2
        f = Tech.from_function(lambda x: 0.5 * x - 0.0625 * jnp.sin(8 * x))
        df = Tech.from_function(lambda x: jnp.sin(4 * x) ** 2)
        err = f.diff() - df
        assert _ninf(err.coeffs) < 100 * df.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_sum_rule(self, Tech):
        # pass(n, 6): diff(f + g) - (df + dg)
        f = Tech.from_function(lambda x: x * jnp.sin(x ** 2) - 1)
        df = f.diff()
        g = Tech.from_function(lambda x: jnp.exp(-x ** 2))
        dg = g.diff()
        tol_f = 10 * df.vscale * EPS
        tol_g = 10 * dg.vscale * EPS
        err = ((f + g).diff() - (df + dg))(X)
        assert _ninf(err) < max(tol_f, tol_g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_product_rule(self, Tech):
        # pass(n, 7): diff(f.*g) - (f.*dg + g.*df)
        f = Tech.from_function(lambda x: x * jnp.sin(x ** 2) - 1)
        df = f.diff()
        g = Tech.from_function(lambda x: jnp.exp(-x ** 2))
        dg = g.diff()
        tol_f = 10 * df.vscale * EPS
        tol_g = 10 * dg.vscale * EPS
        err = ((f * g).diff() - (f * dg + g * df))(X)
        assert _ninf(err) < 10 * len(f) * max(tol_f, tol_g)

    @pytest.mark.parametrize("Tech", BOTH)
    def test_derivative_of_constant(self, Tech):
        # pass(n, 8): derivative of a constant is exactly zero.
        const = Tech.from_function(lambda x: jnp.ones_like(x))
        dconst = const.diff()
        assert _ninf(dconst(X)) == 0.0

    @pytest.mark.parametrize("Tech", BOTH)
    def test_second_derivative(self, Tech):
        # pass(n, 9)
        f = Tech.from_function(
            lambda x: x * jnp.arctan(x) - x - 0.5 * jnp.log(1 + x ** 2)
        )
        df2 = f.diff(2)
        err = _ninf(1.0 / (1 + X ** 2) - df2(X))
        assert err < 1e6 * df2.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_fourth_derivative(self, Tech):
        # pass(n, 10): 4th derivative of sin == sin
        f = Tech.from_function(lambda x: jnp.sin(x))
        df4 = f.diff(4)
        err = _ninf(jnp.sin(X) - df4(X))
        assert err < 1e7 * df4.vscale * EPS

    @pytest.mark.parametrize("Tech", BOTH)
    def test_sixth_derivative_of_quintic_is_zero(self, Tech):
        # pass(n, 11): 6th derivative of a quintic is exactly zero.
        f = Tech.from_function(lambda x: x ** 5 + 3 * x ** 3 - 2 * x ** 2 + 4)
        df6 = f.diff(6)
        assert _ninf(df6(X)) == 0.0

    @pytest.mark.parametrize("Tech", BOTH)
    def test_array_valued_and_dim_option_skipped(self, Tech):
        # pass(n, 12)-(16): array-valued diff, the DIM option (diff(f,1,2),
        # diff(f,2,2)) and issue #1641 all require array-valued (quasimatrix)
        # techs, which chebfunjax does not implement.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix "
            "techs or DIM option"
        )


@pytest.mark.xfail(
    reason="Chebtech1 drops the imaginary part in vals2coeffs/coeffs2vals; it "
    "cannot represent complex-valued functions, so the n=1 (chebtech1) "
    "iterations of the complex sub-tests are unportable",
    strict=False,
)
def test_chebtech1_rejects_complex():
    # Documents the gap that forces the complex sub-tests onto Chebtech2 only:
    # constructing a complex-valued function on Chebtech1 never converges.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = Chebtech1.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
    assert f.ishappy
