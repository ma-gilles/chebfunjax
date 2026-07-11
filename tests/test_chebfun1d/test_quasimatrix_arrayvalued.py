"""Column-wise (array-valued) Quasimatrix API (Fable 5, gap #1).

MATLAB's array-valued chebfun semantics via the Quasimatrix container:
arithmetic, diff/cumsum/sum, matrix right-multiplication, per-column
extrema/roots, real/imag/conj/abs, horzcat/fliplr, qr/svd.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain

XS = jnp.asarray(np.linspace(-0.9, 0.9, 40))


def _q():
    return Quasimatrix([cj.chebfun(jnp.sin), cj.chebfun(jnp.cos),
                        cj.chebfun(jnp.exp)], Domain((-1.0, 1.0)))


class TestQuasimatrixArrayValued:
    def test_arithmetic_and_diff(self):
        Q = _q()
        R = (2 * Q + Q).diff()
        npt.assert_allclose(np.asarray(R[0](XS)),
                            3 * np.cos(np.asarray(XS)), atol=1e-13)
        npt.assert_allclose(np.asarray((Q - Q)[1](XS)), 0.0, atol=1e-14)

    def test_sum_is_row_of_integrals(self):
        Q = _q()
        npt.assert_allclose(
            np.asarray(Q.sum()),
            [0.0, 2 * np.sin(1.0), np.e - 1 / np.e], atol=1e-13)

    def test_matmul_with_matrix(self):
        Q = _q()
        A = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        P = Q @ A
        assert len(P) == 2
        npt.assert_allclose(
            np.asarray(P[0](XS)),
            np.sin(np.asarray(XS)) + np.exp(np.asarray(XS)), atol=1e-13)
        npt.assert_allclose(np.asarray(P[1](XS)),
                            2 * np.cos(np.asarray(XS)), atol=1e-13)

    def test_per_column_extrema(self):
        Q = _q()
        _, vmax = Q.max()
        npt.assert_allclose(np.asarray(vmax),
                            [np.sin(1.0), 1.0, np.e], atol=1e-10)
        _, vmin = Q.min()
        npt.assert_allclose(np.asarray(vmin),
                            [-np.sin(1.0), np.cos(1.0), 1 / np.e],
                            atol=1e-10)

    def test_horzcat_fliplr_roots(self):
        Q = _q()
        R = Q.horzcat(cj.chebfun(lambda x: x)).fliplr()
        assert len(R) == 4
        r = R.roots()
        assert len(r) == 4
        npt.assert_allclose(np.asarray(r[0]).ravel(), [0.0], atol=1e-12)

    def test_complex_columns(self):
        Q = Quasimatrix([cj.chebfun(lambda x: jnp.exp(1j * np.pi * x)),
                         cj.chebfun(jnp.sin)], Domain((-1.0, 1.0)))
        npt.assert_allclose(np.asarray(Q.real()[0](XS)),
                            np.cos(np.pi * np.asarray(XS)), atol=1e-12)
        npt.assert_allclose(np.asarray(Q.imag()[0](XS)),
                            np.sin(np.pi * np.asarray(XS)), atol=1e-12)

    def test_qr_orthonormal(self):
        Q = _q()
        Qo, R = Q.qr()
        cols = Qo.cols if hasattr(Qo, "cols") else Qo
        for i in range(3):
            for j in range(3):
                ip = float(cols[i].innerProduct(cols[j]))
                assert abs(ip - (1.0 if i == j else 0.0)) < 1e-11
