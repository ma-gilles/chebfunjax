"""Array-valued (multi-column) trigtech foundation (Fable 5,
Big-Three item 1): column-wise transforms, Horner evaluation, and
coefficient ops on (n, m) Fourier coefficient matrices."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import (
    Trigtech,
    trig_coeffs2vals,
    trig_vals2coeffs,
)

N = 32
XG = jnp.asarray(-1.0 + 2.0 * np.arange(N) / N)
V = jnp.stack([jnp.sin(jnp.pi * XG), jnp.cos(2 * jnp.pi * XG)], axis=-1)

XS = jnp.asarray(np.linspace(-0.95, 0.95, 9))
XN = np.asarray(XS)
EXACT = np.column_stack([np.sin(np.pi * XN), np.cos(2 * np.pi * XN)])


def _mk() -> Trigtech:
    return Trigtech.from_values(V)


class TestArrayValuedTrigFoundation:
    def test_column_consistent_transforms(self):
        C = trig_vals2coeffs(V)
        assert C.shape == (N, 2)
        for j in range(2):
            np.testing.assert_allclose(
                np.asarray(C[:, j]),
                np.asarray(trig_vals2coeffs(V[:, j])), atol=1e-15)
        np.testing.assert_allclose(
            np.asarray(trig_coeffs2vals(C)).real, np.asarray(V),
            atol=1e-14)

    def test_call_and_simplify(self):
        t = _mk()
        np.testing.assert_allclose(np.asarray(t(XS)), EXACT, atol=1e-14)
        s = t.simplify()
        assert s.coeffs.ndim == 2 and s.n < t.n
        np.testing.assert_allclose(np.asarray(s(XS)), EXACT, atol=1e-14)

    def test_prolong(self):
        p = _mk().prolong(65)
        assert p.coeffs.shape == (65, 2)
        np.testing.assert_allclose(np.asarray(p(XS)), EXACT, atol=1e-14)

    def test_diff_cumsum_sum(self):
        t = _mk()
        dex = np.column_stack([np.pi * np.cos(np.pi * XN),
                               -2 * np.pi * np.sin(2 * np.pi * XN)])
        np.testing.assert_allclose(np.asarray(t.diff()(XS)), dex,
                                   atol=1e-12)
        cex = np.column_stack(
            [(-np.cos(np.pi * XN) + np.cos(-np.pi)) / np.pi,
             (np.sin(2 * np.pi * XN) - np.sin(-2 * np.pi))
             / (2 * np.pi)])
        np.testing.assert_allclose(np.asarray(t.cumsum()(XS)), cex,
                                   atol=1e-14)
        np.testing.assert_allclose(np.asarray(t.sum()), [0.0, 0.0],
                                   atol=1e-14)

    def test_multiply(self):
        t = _mk()
        np.testing.assert_allclose(np.asarray((t * t)(XS)), EXACT ** 2,
                                   atol=1e-13)
        g = Trigtech.from_values(jnp.cos(jnp.pi * XG))
        np.testing.assert_allclose(
            np.asarray((t * g)(XS)),
            EXACT * np.cos(np.pi * XN)[:, None], atol=1e-13)

    def test_roots_nan_padded(self):
        r = np.asarray(_mk().roots())
        assert r.ndim == 2 and r.shape[1] == 2
        # sin(pi x) roots: -1, 0, 1; cos(2 pi x): +-0.25, +-0.75
        assert abs(r[1, 0]) < 1e-12
        np.testing.assert_allclose(
            r[:4, 1], [-0.75, -0.25, 0.25, 0.75], atol=1e-12)

    def test_adaptive_construction(self):
        f = Trigtech.from_function(
            lambda t: jnp.stack(
                [jnp.sin(jnp.pi * t), jnp.cos(2 * jnp.pi * t)],
                axis=-1))
        assert f.ishappy and f.coeffs.ndim == 2
        np.testing.assert_allclose(np.asarray(f(XS)), EXACT, atol=1e-14)


class TestColumnOps:
    """Column operations added with the array-valued epic (mirrors the
    port-tree assertions so the core coverage gate sees them)."""

    def test_matmul_fliplr_flipud(self):
        t = _mk()
        A = np.array([[1.0, 2.0], [3.0, -1.0]])
        np.testing.assert_allclose(
            np.asarray((t @ jnp.asarray(A))(XS)), EXACT @ A, atol=1e-13)
        np.testing.assert_allclose(
            np.asarray(t.fliplr()(XS)), EXACT[:, ::-1], atol=1e-14)
        exf = np.column_stack([np.sin(-np.pi * XN),
                               np.cos(2 * np.pi * XN)])
        np.testing.assert_allclose(
            np.asarray(t.flipud()(XS)), exf, atol=1e-14)

    def test_real_imag_conj(self):
        c = Trigtech.from_function(
            lambda t: jnp.stack(
                [jnp.exp(1j * jnp.pi * t), 1j * jnp.cos(jnp.pi * t)],
                axis=-1))
        exc = np.column_stack([np.exp(1j * np.pi * XN),
                               1j * np.cos(np.pi * XN)])
        np.testing.assert_allclose(np.asarray(c.real()(XS)), exc.real,
                                   atol=1e-14)
        np.testing.assert_allclose(np.asarray(c.imag()(XS)), exc.imag,
                                   atol=1e-14)
        np.testing.assert_allclose(np.asarray(c.conj()(XS)),
                                   np.conj(exc), atol=1e-14)
        z = Trigtech.from_function(lambda t: jnp.cos(jnp.pi * t)).imag()
        assert bool(jnp.all(z.coeffs == 0))

    def test_minandmax_per_column(self):
        (mn, _), (mx, _) = _mk().minandmax()
        np.testing.assert_allclose(np.asarray(mn), [-1.0, -1.0],
                                   atol=1e-12)
        np.testing.assert_allclose(np.asarray(mx), [1.0, 1.0],
                                   atol=1e-12)

    def test_mat2cell_cell2mat_assign(self):
        t = _mk()
        parts = t.mat2cell([1, 1])
        np.testing.assert_allclose(np.asarray(parts[1](XS)),
                                   EXACT[:, 1], atol=1e-14)
        back = Trigtech.cell2mat(parts)
        np.testing.assert_allclose(np.asarray(back(XS)), EXACT,
                                   atol=1e-14)
        g = Trigtech.from_function(lambda t: jnp.cos(jnp.pi * t))
        asg = t.assign_columns(0, g)
        np.testing.assert_allclose(
            np.asarray(asg(XS)),
            np.column_stack([np.cos(np.pi * XN), EXACT[:, 1]]),
            atol=1e-14)
        dele = t.assign_columns(0, None)
        assert dele.coeffs.shape[1] == 1

    def test_dim_options(self):
        t = _mk()
        s2 = t.sum(dim=2)
        np.testing.assert_allclose(np.asarray(s2(XS)),
                                   EXACT.sum(axis=1), atol=1e-14)
        d2 = t.diff(1, dim=2)
        np.testing.assert_allclose(
            np.asarray(d2(XS))[:, 0], EXACT[:, 1] - EXACT[:, 0],
            atol=1e-14)
        scalar = Trigtech.from_function(lambda t: jnp.cos(jnp.pi * t))
        assert scalar.sum(dim=2) is scalar
        assert scalar.diff(1, dim=2).coeffs.size == 0

    def test_complex_scalar_arith_and_expansion(self):
        # is_real clearing + implicit expansion (Fable 5 audit fixes)
        f = Trigtech.from_function(lambda t: jnp.sin(jnp.pi * t))
        alpha = 0.5 + 1.5j
        np.testing.assert_allclose(
            np.asarray((f + alpha)(XS)),
            np.sin(np.pi * XN) + alpha, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray((f / 1j)(XS)),
            np.sin(np.pi * XN) / 1j, atol=1e-14)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = alpha / (Trigtech.from_function(
                lambda t: jnp.exp(jnp.cos(jnp.pi * t))))
        np.testing.assert_allclose(
            np.asarray(g(XS)),
            alpha / np.exp(np.cos(np.pi * XN)), atol=1e-13)
        r = f + jnp.asarray([1.0, 2.0, 3.0])
        assert r.coeffs.shape[1] == 3
        np.testing.assert_allclose(
            np.asarray(r(XS)),
            np.sin(np.pi * XN)[:, None] + np.array([1.0, 2.0, 3.0]),
            atol=1e-14)
        mixed = _mk() + f
        np.testing.assert_allclose(
            np.asarray(mixed(XS)),
            EXACT + np.sin(np.pi * XN)[:, None], atol=1e-14)


class TestClassicfunColumnOps:
    def test_bndfun_sum_dim(self):
        from chebfunjax.domain import Domain
        from chebfunjax.fun.bndfun import Bndfun

        dom = Domain((-2.0, 3.0))
        f = Bndfun.from_function(
            lambda x: jnp.stack([jnp.sin(x), x ** 2], axis=-1), dom)
        xs = jnp.asarray(np.linspace(-1.9, 2.9, 9))
        xn = np.asarray(xs)
        g = f.sum(dim=2)
        np.testing.assert_allclose(np.asarray(g(xs)),
                                   np.sin(xn) + xn ** 2, atol=1e-13)
        s = Bndfun.from_function(jnp.sin, dom)
        assert s.sum(dim=2) is s
        np.testing.assert_allclose(
            np.asarray(f.sum()),
            [np.cos(-2.0) - np.cos(3.0), 35.0 / 3.0], atol=1e-13)
