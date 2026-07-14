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
