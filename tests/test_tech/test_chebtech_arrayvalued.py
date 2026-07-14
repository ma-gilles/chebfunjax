"""Array-valued (multi-column) chebtech foundation (Fable 5,
Big-Three item 1): column-wise transforms and Clenshaw evaluation
on (n, m) coefficient matrices."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.chebtech import Chebtech2, _clenshaw
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs

N = 17
XG = jnp.asarray(-np.cos(np.arange(N) * np.pi / (N - 1)))
V = jnp.stack([jnp.sin(XG), jnp.exp(XG), XG ** 3], axis=-1)


class TestArrayValuedFoundation:
    def test_column_consistent_transforms(self):
        C = vals2coeffs(V)
        assert C.shape == (N, 3)
        for j in range(3):
            np.testing.assert_allclose(
                np.asarray(C[:, j]),
                np.asarray(vals2coeffs(V[:, j])), atol=1e-15)
        np.testing.assert_allclose(
            np.asarray(coeffs2vals(C)), np.asarray(V), atol=1e-14)

    def test_multicolumn_clenshaw(self):
        C = vals2coeffs(V)
        xs = jnp.asarray([0.3, -0.5, 0.9])
        y = np.asarray(_clenshaw(C, xs))
        exact = np.stack([np.sin(np.asarray(xs)),
                          np.exp(np.asarray(xs)),
                          np.asarray(xs) ** 3], axis=-1)
        assert y.shape == (3, 3)
        np.testing.assert_allclose(y, exact, atol=1e-13)

    def test_from_values_matrix(self):
        t = Chebtech2.from_values(V)
        assert t.coeffs.shape[0] == N and t.coeffs.shape[1] == 3


XS = jnp.asarray(np.linspace(-1.0, 1.0, 9))
XN = np.asarray(XS)
EXACT = np.column_stack([np.sin(XN), np.exp(XN), XN ** 3])


def _mk(n: int = 33) -> Chebtech2:
    xg = jnp.asarray(-np.cos(np.arange(n) * np.pi / (n - 1)))
    return Chebtech2.from_values(
        jnp.stack([jnp.sin(xg), jnp.exp(xg), xg ** 3], axis=-1))


class TestArrayValuedOps:
    """Layer 2: coefficient-space operations on (n, m) matrices.

    Each op must agree column-wise with its scalar counterpart at the
    same accuracy (MATLAB array-valued chebtech semantics)."""

    def test_call_and_simplify(self):
        t = _mk()
        np.testing.assert_allclose(np.asarray(t(XS)), EXACT, atol=1e-14)
        s = t.simplify()
        assert s.coeffs.ndim == 2 and s.n < t.n
        np.testing.assert_allclose(np.asarray(s(XS)), EXACT, atol=1e-14)

    def test_prolong_matrix(self):
        t = _mk()
        p = t.prolong(50)
        assert p.coeffs.shape == (50, 3)
        np.testing.assert_allclose(np.asarray(p(XS)), EXACT, atol=1e-14)

    def test_diff_cumsum_sum(self):
        t = _mk()
        d = np.asarray(t.diff()(XS))
        dex = np.column_stack([np.cos(XN), np.exp(XN), 3 * XN ** 2])
        np.testing.assert_allclose(d, dex, atol=1e-12)
        c = np.asarray(t.cumsum()(XS))
        cex = np.column_stack([
            -np.cos(XN) + np.cos(-1.0),
            np.exp(XN) - np.exp(-1.0),
            (XN ** 4 - 1.0) / 4.0])
        np.testing.assert_allclose(c, cex, atol=1e-14)
        s = np.asarray(t.sum())
        np.testing.assert_allclose(
            s, [0.0, np.exp(1) - np.exp(-1), 0.0], atol=1e-14)

    def test_multiply_columnwise(self):
        t = _mk()
        m = np.asarray((t * t)(XS))
        np.testing.assert_allclose(m, EXACT ** 2, atol=1e-13)
        # scalar-column * array-valued broadcasts (MATLAB times.m)
        n = 33
        xg = jnp.asarray(-np.cos(np.arange(n) * np.pi / (n - 1)))
        g = Chebtech2.from_values(jnp.cos(xg))
        mixed = np.asarray((t * g)(XS))
        np.testing.assert_allclose(
            mixed, EXACT * np.cos(XN)[:, None], atol=1e-13)

    def test_roots_nan_padded(self):
        t = _mk()
        r = np.asarray(t.roots())
        assert r.ndim == 2 and r.shape[1] == 3
        assert abs(r[0, 0]) < 1e-12          # sin root at 0
        assert np.isnan(r[:, 1]).all()       # exp has no roots
        # x^3 has a triple root at 0: the colleague matrix resolves it
        # to O(eps^(1/3)), same as MATLAB
        assert abs(r[0, 2]) < 1e-5

    def test_minandmax_per_column(self):
        t = _mk()
        (mn, _), (mx, _) = t.minandmax()
        np.testing.assert_allclose(
            np.asarray(mn), [-np.sin(1.0), np.exp(-1.0), -1.0],
            atol=1e-13)
        np.testing.assert_allclose(
            np.asarray(mx), [np.sin(1.0), np.exp(1.0), 1.0],
            atol=1e-13)

    def test_inner_gram_matrix(self):
        t = _mk()
        gram = np.asarray(t.inner(t))
        assert gram.shape == (3, 3)
        np.testing.assert_allclose(
            gram[0, 0], 1.0 - np.sin(2.0) / 2.0, atol=1e-14)
        np.testing.assert_allclose(
            gram[1, 1], (np.exp(2) - np.exp(-2)) / 2.0, atol=1e-13)
        np.testing.assert_allclose(gram[0, 2], gram[2, 0], atol=1e-14)

    def test_happiness_and_adaptive_construction(self):
        f = Chebtech2.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.exp(x)], axis=-1))
        assert f.ishappy and f.coeffs.ndim == 2
        np.testing.assert_allclose(
            np.asarray(f(XS)), EXACT[:, :2], atol=1e-14)
