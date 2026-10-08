"""Port of MATLAB Chebfun tests/bndfun/test_diff.m (Opus 4.8).

Self-validating: each operation is checked against an analytic exact at
the SAME tolerance MATLAB uses (multiples of vscale*eps).  No .mat
fixture needed — the reference is the closed-form derivative.

Provenance
----------
MATLAB source : tests/bndfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.utils.airy_general import airy_all

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
# deterministic test points in the domain (analytic checks hold at any x)
_REF = json.loads((Path(__file__).parents[1] / "chebfun/fixtures/logical_source_matlab.json").read_text())
# MT19937 sequence agrees exactly with the retained native seed6178 capture.
_R = np.random.RandomState(6178).rand(100)
assert np.array_equal(2 * _R - 1, _REF["x"])
X = jnp.asarray(9 * _R - 2)


def _bf(f):
    return Bndfun.from_function(f, DOM)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestBndfunDiff:
    def test_spotcheck_exp(self):
        f = _bf(lambda x: jnp.exp(x / 10) - x)
        df = f.diff()
        err = jnp.exp(X / 10) / 10 - 1 - df(X)
        assert _ninf(err) < 1e3 * f.vscale * EPS


    def test_spotcheck_atan(self):
        f = _bf(lambda x: jnp.arctan(x))
        df = f.diff()
        err = 1.0 / (1 + X ** 2) - df(X)
        assert _ninf(err) < 1e3 * f.vscale * EPS


    def test_spotcheck_sin(self):
        f = _bf(lambda x: jnp.sin(x))
        df = f.diff()
        err = jnp.cos(X) - df(X)
        assert _ninf(err) < 1e3 * f.vscale * EPS


    def test_spotcheck_complex_airy(self):
        z = np.exp(2 * np.pi * 1j / 3)
        f = _bf(lambda x: airy_all(z * x)[0])
        assert _ninf(z * airy_all(z * X)[1] - f.diff()(X)) < 1e3 * f.vscale * EPS


    def test_diff_equals_direct_construction(self):
        # diff(0.5x - 0.0625 sin 8x) == sin(4x)^2
        f = _bf(lambda x: 0.5 * x - 0.0625 * jnp.sin(8 * x))
        df = _bf(lambda x: jnp.sin(4 * x) ** 2)
        err = f.diff() - df
        assert err.vscale < 1e4 * f.vscale * EPS


    def test_sum_rule(self):
        f = _bf(lambda x: x * jnp.sin(x ** 2) - 1)
        g = _bf(lambda x: jnp.exp(-x ** 2))
        df, dg = f.diff(), g.diff()
        tol = 10 * max(f.vscale, g.vscale, df.vscale, dg.vscale) * EPS
        # diff(f+g) - (df+dg)
        err = ((f + g).diff() - (df + dg))(X)
        assert _ninf(err) < tol


    def test_product_rule(self):
        f = _bf(lambda x: x * jnp.sin(x ** 2) - 1)
        g = _bf(lambda x: jnp.exp(-x ** 2))
        df, dg = f.diff(), g.diff()
        tol = 10 * max(f.vscale, g.vscale, df.vscale, dg.vscale) * EPS
        err = ((f * g).diff() - (f * dg + g * df))(X)
        assert _ninf(err) < 1e1 * tol


    def test_derivative_of_constant(self):
        const = _bf(lambda x: jnp.ones_like(x))
        dconst = const.diff()
        assert _ninf(dconst(X)) <= 10 * dconst.vscale * EPS


    def test_second_derivative(self):
        f = _bf(lambda x: x * jnp.arctan(x) - x - 0.5 * jnp.log(1 + x ** 2))
        df2 = f.diff(2)
        err = 1.0 / (1 + X ** 2) - df2(X)
        assert _ninf(err) < 1e7 * df2.vscale ** 2 * EPS


    def test_fourth_derivative(self):
        f = _bf(lambda x: jnp.sin(x))
        df4 = f.diff(4)
        err = _ninf(jnp.sin(X) - df4(X))
        assert err < 1e6 * 10 * df4.vscale * EPS


    def test_sixth_derivative_of_quintic_is_zero(self):
        f = _bf(lambda x: x ** 5 + 3 * x ** 3 - 2 * x ** 2 + 4)
        df6 = f.diff(6)
        assert _ninf(df6(X)) <= df6.vscale ** 6 * EPS


    @staticmethod
    def _array():
        return _bf(lambda x: jnp.stack([jnp.sin(x), x**2, jnp.exp(1j*x)], axis=-1))


    def test_array_derivative(self):
        f = self._array()
        exact = jnp.stack([jnp.cos(X), 2*X, 1j*jnp.exp(1j*X)], axis=-1)
        assert _ninf(f.diff()(X) - exact) < 1e3 * f.vscale * EPS


    def test_dim2_first(self):
        f = self._array()
        exact = jnp.stack([X**2-jnp.sin(X), jnp.exp(1j*X)-X**2], axis=-1)
        assert _ninf(f.diff(1, 2)(X) - exact) < 10 * f.vscale * EPS


    def test_dim2_second(self):
        f = self._array()
        exact = jnp.exp(1j*X)-2*X**2+jnp.sin(X)
        assert _ninf(jnp.ravel(f.diff(2, 2)(X)) - exact) < 10 * f.vscale * EPS


    def test_dim2_scalar_empty(self):
        f = _bf(lambda x: x**3)
        assert f.diff(1, 2).isempty()


    def test_singular_derivative(self):
        power = -0.5
        f = Bndfun.from_function(lambda x: (x-DOM.a)**power*jnp.sin(x), DOM,
                                exponents=(power, 0.0))
        exact = (X-DOM.a)**(power-1)*(power*jnp.sin(X)+(X-DOM.a)*jnp.cos(X))
        assert _ninf(f.diff()(X)-exact) < 1e5*EPS*_ninf(exact)
