"""Port of MATLAB Chebfun tests/chebfun/test_diff.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunDiff:
    def test_empty(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        assert chebfun().diff().isempty()

    def test_piecewise_sin(self):
        f1 = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        df1 = f1.diff()
        err = jnp.abs(df1(XR) - jnp.cos(XR))
        assert float(jnp.max(err)) < 100 * df1.vscale * EPS

    def test_transpose_variant(self):
        # Native pass(3): diff(f1.', 1, 2) differentiates continuously.
        f1 = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        df1t = f1.T.diff(1, dim=2)
        assert df1t.is_transposed
        err = jnp.abs(df1t(XR) - jnp.cos(XR))
        assert float(jnp.max(err)) < 100 * df1t.vscale * EPS

    def test_native_piecewise_sine_second_derivative(self):
        # Native pass(5): preserve the original piecewise-sine predicate.
        f1 = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        d2f1 = f1.diff(2)
        err = d2f1(XR) + jnp.sin(XR)
        assert float(jnp.max(jnp.abs(err))) < 1e4 * d2f1.vscale * EPS

    def test_diff_static_python_orders_are_jit_safe(self):
        from chebfunjax.chebfun1d.chebfun import Chebfun

        f = cj.chebfun(jnp.sin)

        def original_continuous_diff(fun, order):
            # For this smooth one-piece fixture, the original method's path is
            # to differentiate each piece and rebuild the same Chebfun.
            pieces = [piece.diff(order) for piece in fun.funs]
            return Chebfun(funs=pieces, domain=fun.domain)

        actual, reference = jax.jit(
            lambda fun: (fun.diff(), original_continuous_diff(fun, 1)))(f)
        assert bool(jnp.array_equal(actual.coeffs, reference.coeffs))

        actual2, reference2 = jax.jit(
            lambda fun: (fun.diff(2, dim=1), original_continuous_diff(fun, 2)))(f)
        assert bool(jnp.array_equal(actual2.coeffs, reference2.coeffs))

    def test_diff_accepts_scalar_numeric_orders(self):
        # Native isnumeric accepts numeric scalar adapters beyond Python int.
        f = cj.chebfun(jnp.sin)
        expected = f.diff(2)(XR)
        for order in (np.int64(2), jnp.asarray(2, dtype=jnp.int32)):
            got = f.diff(order)(XR)
            assert float(jnp.max(jnp.abs(got - expected))) < 100 * EPS

    def test_second_derivative(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(2 * x))
        d2 = f.diff(2)
        exact = jnp.exp(XR) * (4 * jnp.cos(2 * XR) - 3 * jnp.sin(2 * XR))
        err = jnp.abs(d2(XR) - exact)
        assert float(jnp.max(err)) < 1e4 * d2.vscale * EPS


    def test_native_array_valued_first_and_second_derivatives(self):
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
            domain=[-1.0, -0.5, 0.5, 1.0])
        first = f.diff()
        exact_first = jnp.stack([jnp.cos(XR), -jnp.sin(XR), jnp.exp(XR)], axis=-1)
        assert float(jnp.max(jnp.abs(first(XR) - exact_first))) < 1e3 * first.vscale * EPS
        second = f.diff(2)
        exact_second = jnp.stack([-jnp.sin(XR), -jnp.cos(XR), jnp.exp(XR)], axis=-1)
        assert float(jnp.max(jnp.abs(second(XR) - exact_second))) < 1e5 * second.vscale * EPS

    def test_native_column_array_finite_difference(self):
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
            domain=[-1.0, -0.5, 0.5, 1.0])
        got = f.diff(1, dim=2)(XR)
        expected = jnp.stack([jnp.cos(XR) - jnp.sin(XR),
                              jnp.exp(XR) - jnp.cos(XR)], axis=-1)
        out = f.diff(1, dim=2)
        assert float(jnp.max(jnp.abs(got - expected))) < 10 * out.vscale * EPS

    def test_native_transposed_array_finite_difference(self):
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
            domain=[-1.0, -0.5, 0.5, 1.0]).T
        got = f.diff(1, dim=1)(XR)
        expected = jnp.stack([jnp.cos(XR) - jnp.sin(XR),
                              jnp.exp(XR) - jnp.cos(XR)], axis=0)
        out = f.diff(1, dim=1)
        assert out.is_transposed
        assert float(jnp.max(jnp.abs(got - expected))) < 10 * out.vscale * EPS

    def test_native_invalid_dimension_identifier(self):
        f = cj.chebfun(lambda x: jnp.sin(x))
        try:
            f.diff(1, dim=3)
        except ValueError as exc:
            assert "CHEBFUN:CHEBFUN:diff:dim" in str(exc)
        else:
            raise AssertionError("diff with dim=3 must fail with the native identifier")

    def test_native_singular_derivative_predicate(self):
        dom = (-2.0, 7.0)
        pow_ = -0.5
        op = lambda x: (x - dom[0]) ** pow_ * jnp.sin(200 * x)
        f = cj.chebfun(op, domain=dom, exps=(pow_, 0.0), splitting=True)
        x = jnp.linspace(dom[0] + 0.1, dom[1] - 0.1, 100)
        got = f.diff()(x)
        exact = (x - dom[0]) ** (pow_ - 1) * (
            pow_ * jnp.sin(200 * x) + 200 * (x - dom[0]) * jnp.cos(200 * x))
        assert float(jnp.max(jnp.abs(got - exact))) < 1e6 * EPS * float(jnp.max(jnp.abs(exact)))

    def test_native_unbounded_derivative_predicate(self):
        def op(x):
            safe_x = jnp.where(x == 0, 1.0, x)
            value = (1 - jnp.exp(-x**2)) / safe_x
            return jnp.where(x == 0, 0.0, value)
        f = cj.chebfun(op, domain=(-jnp.inf, jnp.inf), splitting=True)
        x = jnp.linspace(-100.0, 100.0, 100)
        got = f.diff()(x)
        exact = 2 * jnp.exp(-x**2) + (jnp.exp(-x**2) - 1) / jnp.where(x == 0, 1.0, x**2)
        assert float(jnp.max(jnp.abs(got - exact))) < 1e3 * EPS * f.diff().vscale


def test_native_quasimatrix_row_continuous_and_finite_dimensions():
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    cols = [cj.chebfun(op, domain=(-1.0, 1.0))
            for op in (jnp.sin, jnp.cos, jnp.exp)]
    row = Quasimatrix(cols, cols[0].domain).T
    continuous = row.diff(1, dim=2)
    finite = row.diff(np.int64(1), dim=np.int64(1))
    expected_continuous = jnp.stack(
        [jnp.cos(XR), -jnp.sin(XR), jnp.exp(XR)], axis=0)
    expected_finite = jnp.stack(
        [jnp.cos(XR)-jnp.sin(XR), jnp.exp(XR)-jnp.cos(XR)], axis=0)
    jax_adapter = row.diff(jnp.asarray(1, dtype=jnp.int32),
                           dim=jnp.asarray(2, dtype=jnp.int32))
    assert continuous.is_transposed and finite.is_transposed
    assert bool(jnp.array_equal(jax_adapter(XR), continuous(XR)))
    assert float(jnp.max(jnp.abs(continuous(XR)-expected_continuous))) < 1e3*EPS
    assert float(jnp.max(jnp.abs(finite(XR)-expected_finite))) < 1e3*EPS

    empty = row.diff(3, dim=1)
    assert empty.isempty()
    # Native diff returns early on empty input before validating n/dim.
    assert empty.diff("invalid", dim=3).isempty()


def test_native_quasimatrix_finite_difference_dimensions():
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    cols = [cj.chebfun(op, domain=(-1.0, 1.0))
            for op in (jnp.sin, jnp.cos, jnp.exp)]
    q = Quasimatrix(cols, cols[0].domain)
    first = q.diff(1, dim=2)
    second = q.diff(2, dim=2)
    assert isinstance(first, Quasimatrix) and len(first) == 2
    assert isinstance(second, Quasimatrix) and len(second) == 1
    first_expected = jnp.stack([jnp.cos(XR) - jnp.sin(XR),
                                jnp.exp(XR) - jnp.cos(XR)], axis=-1)
    second_expected = jnp.exp(XR) - 2 * jnp.cos(XR) + jnp.sin(XR)
    assert float(jnp.max(jnp.abs(first(XR) - first_expected))) < 1e3 * EPS
    assert float(jnp.max(jnp.abs(second(XR)[:, 0] - second_expected))) < 1e3 * EPS
    assert q.diff(3, dim=2).isempty()
