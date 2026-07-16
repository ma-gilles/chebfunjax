"""Port of MATLAB Chebfun tests/trigtech/test_diff.m (Opus 4.8).

Self-validating: each derivative is checked against its analytic exact at
the SAME tolerance MATLAB uses (multiples of vscale*eps).  trigtech
represents smooth *periodic* functions via a Fourier series on [-1, 1),
so every test function here is periodic and its exact derivative is
analytic.

Provenance
----------
MATLAB source : tests/trigtech/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
# deterministic points in the domain; analytic checks hold at any x in [-1,1)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False))


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechDiff:
    def test_even_real_coeffs(self):
        # f from coeffs {[],[-.25;.75]} (even length N=2, wavenumbers -1,0)
        f = Trigtech.from_coeffs(jnp.array([-0.25, 0.75], dtype=jnp.complex128))
        df = f.diff()
        df_coeffs_exact = jnp.array([0.25 * jnp.pi * 1j, 0.0], dtype=jnp.complex128)
        assert _ninf(df.coeffs - df_coeffs_exact) < EPS * _ninf(df_coeffs_exact)

    def test_spotcheck_exp_cos(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        df = f.diff()
        exact = -jnp.pi * jnp.sin(jnp.pi * X) * jnp.exp(jnp.cos(jnp.pi * X))
        assert _ninf(exact - df(X)) < 1e3 * df.vscale * EPS

    def test_spotcheck_cos_of_sin(self):
        a, b = 10, 20
        f = _tt(lambda x: jnp.cos(a * jnp.pi * jnp.sin(b * jnp.pi * x)))
        df = f.diff()
        exact = -jnp.pi**2 * a * b * jnp.cos(b * jnp.pi * X) * jnp.sin(
            a * jnp.pi * jnp.sin(b * jnp.pi * X)
        )
        assert _ninf(exact - df(X)) < 1e4 * df.vscale * EPS

    def test_spotcheck_gaussian(self):
        # exp(-50 x^2): periodic to machine precision on [-1,1) (endpoints ~1e-22)
        f = _tt(lambda x: jnp.exp(-50 * x**2))
        df = f.diff()
        exact = -100 * X * jnp.exp(-50 * X**2)
        assert _ninf(exact - df(X)) < 100 * df.vscale * EPS

    def test_spotcheck_complex(self):
        a1, b1, a2, b2 = 4, 3, 6, 4
        f = _tt(
            lambda x: jnp.cos(a1 * jnp.pi * jnp.sin(b1 * jnp.pi * x))
            + 1j * jnp.cos(a2 * jnp.pi * jnp.sin(b2 * jnp.pi * x))
        )
        df = f.diff()
        exact = -jnp.pi**2 * a1 * b1 * jnp.cos(b1 * jnp.pi * X) * jnp.sin(
            a1 * jnp.pi * jnp.sin(b1 * jnp.pi * X)
        ) - 1j * jnp.pi**2 * a2 * b2 * jnp.cos(b2 * jnp.pi * X) * jnp.sin(
            a2 * jnp.pi * jnp.sin(b2 * jnp.pi * X)
        )
        assert _ninf(exact - df(X)) < 1e3 * df.vscale * EPS

    def test_diff_equals_direct_construction(self):
        f = _tt(lambda x: 1 / 21 / jnp.pi * jnp.cos(21 * jnp.pi * x))
        df = _tt(lambda x: -jnp.sin(21 * jnp.pi * x))
        err = f.diff() - df
        assert _ninf(err.coeffs) < 100 * df.vscale * EPS

    def test_sum_rule(self):
        f = _tt(lambda x: jnp.exp(1.0) - jnp.exp(jnp.cos(3 * jnp.pi * x)))
        df = f.diff()
        g = _tt(lambda x: jnp.exp(-jnp.sin(2 * jnp.pi * x)))
        dg = g.diff()
        tol_f = 10 * df.vscale * EPS
        tol_g = 10 * dg.vscale * EPS
        errfn = (f + g).diff() - (df + dg)
        assert _ninf(errfn(X)) < 10 * max(tol_f, tol_g)

    def test_product_rule(self):
        f = _tt(lambda x: jnp.exp(1.0) - jnp.exp(jnp.cos(3 * jnp.pi * x)))
        df = f.diff()
        g = _tt(lambda x: jnp.exp(-jnp.sin(2 * jnp.pi * x)))
        dg = g.diff()
        tol_f = 10 * df.vscale * EPS
        tol_g = 10 * dg.vscale * EPS
        errfn = (f * g).diff() - (f * dg + g * df)
        assert _ninf(errfn(X)) < f.n * max(tol_f, tol_g)

    def test_derivative_of_constant_is_zero(self):
        const = _tt(lambda x: jnp.ones_like(x))
        dconst = const.diff()
        assert _ninf(dconst(X)) == 0.0

    def test_second_derivative(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(4 * jnp.pi * x)) - 1)
        df2 = f.diff(2)
        exact = -16 * jnp.pi**2 * jnp.exp(jnp.cos(4 * jnp.pi * X)) * (
            jnp.cos(4 * jnp.pi * X) + jnp.cos(4 * jnp.pi * X) ** 2 - 1
        )
        assert _ninf(exact - df2(X)) < 1e3 * df2.vscale * EPS

    def test_sixth_derivative(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        df6 = f.diff(6)
        df6_exact = (-jnp.pi**6) * f
        assert _ninf(df6_exact(X) - df6(X)) < 100 * df6.vscale * EPS

    def test_fifth_derivative_vanishes_at_endpoints(self):
        f = _tt(lambda x: (1 / 10 / jnp.pi) * jnp.cos(10 * jnp.pi * jnp.sin(jnp.pi * x)))
        df5 = f.diff(5)
        # Odd derivatives of this function vanish at +-1
        endpts = jnp.array([-1.0, 1.0])
        assert _ninf(df5(endpts)) < 1e3 * df5.vscale * EPS

    def test_even_complex_coeffs(self):
        # f from coeffs {[],[1+1i;1-1i]} (even length N=2)
        f = Trigtech.from_coeffs(jnp.array([1 + 1j, 1 - 1j], dtype=jnp.complex128))
        df = f.diff()
        df_coeffs_exact = jnp.array([-jnp.pi * 1j * (1 + 1j), 0.0], dtype=jnp.complex128)
        assert _ninf(df.coeffs - df_coeffs_exact) < EPS * _ninf(df_coeffs_exact)

    def test_array_valued_diff(self):
        # pass(13): diff([exp(-50x^2) sin(4pi(x-.2)) 1i*exp(cos(pi x))]).
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs; diff acts
        # column-wise.
        f = _tt(
            lambda x: jnp.stack(
                [
                    jnp.exp(-50 * x**2),
                    jnp.sin(4 * jnp.pi * (x - 0.2)),
                    1j * jnp.exp(jnp.cos(jnp.pi * x)),
                ],
                axis=-1,
            )
        )
        df = f.diff()

        def df_exact(x):
            return jnp.stack(
                [
                    -100 * x * jnp.exp(-50 * x**2),
                    4 * jnp.pi * jnp.cos(4 * jnp.pi * (x - 0.2)),
                    -jnp.pi * 1j * jnp.sin(jnp.pi * x) * jnp.exp(jnp.cos(jnp.pi * x)),
                ],
                axis=-1,
            )

        assert _ninf(df(X) - df_exact(X)) < 100 * df.vscale * EPS

    def _f3(self):
        return _tt(
            lambda x: jnp.stack(
                [
                    jnp.exp(-50 * x**2),
                    jnp.sin(4 * jnp.pi * (x - 0.2)),
                    1j * jnp.exp(jnp.cos(jnp.pi * x)),
                ],
                axis=-1,
            )
        )

    def test_diff_dim2_first(self):
        # pass(14): diff(f, 1, 2) is the first difference ACROSS columns.
        # FIXED (Fable 5, Big-Three array-valued epic).  MATLAB also asserts
        # size(vscale)==[1 2]; chebfunjax vscale is a scalar, so we check the
        # 2-column result shape instead.
        f = self._f3()
        d2 = f.diff(1, dim=2)

        def g(x):
            return jnp.stack(
                [
                    jnp.sin(4 * jnp.pi * (x - 0.2)) - jnp.exp(-50 * x**2),
                    1j * jnp.exp(jnp.cos(jnp.pi * x)) - jnp.sin(4 * jnp.pi * (x - 0.2)),
                ],
                axis=-1,
            )

        assert d2.coeffs.shape[1] == 2
        assert _ninf(d2(X) - g(X)) < 100 * d2.vscale * EPS

    def test_diff_dim2_second(self):
        # pass(15): diff(f, 2, 2) is the second difference ACROSS columns.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = self._f3()
        d2b = f.diff(2, dim=2)

        def g2(x):
            return (
                jnp.exp(-50 * x**2)
                - 2 * jnp.sin(4 * jnp.pi * (x - 0.2))
                + 1j * jnp.exp(jnp.cos(jnp.pi * x))
            )

        assert d2b.coeffs.shape[1] == 1
        assert _ninf(d2b(X)[:, 0] - g2(X)) < 1e3 * d2b.vscale * EPS

    def test_diff_dim2_scalar_empty(self):
        # pass(16): diff(f, 1, 2) on scalar-valued input returns empty coeffs.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        d2 = f.diff(1, dim=2)
        assert np.asarray(d2.coeffs).size == 0
