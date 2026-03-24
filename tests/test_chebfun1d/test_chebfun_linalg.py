"""Tests for quasimatrix linear algebra: QR and SVD on Chebfun columns.

JAX contract
------------
- qr()        jit=NO  (continuous Householder QR uses Python loops)
- svd()       jit=NO  (via QR + numpy SVD)
- Quasimatrix(x)  jit=NO  (Python dispatch per column)

MATLAB golden refs: verified against known analytic results.
  - QR of [1, x, x^2] on [-1, 1] gives L2-orthonormal polynomials (Legendre-
    like, up to sign). Specifically:
      Q[:,0] = 1/sqrt(2)         (normalised P_0)
      Q[:,1] = sqrt(3/2) * x    (normalised P_1)
      Q[:,2] = sqrt(5/2)*(3x^2-1)/2  (normalised P_2)
  - QR orthogonality: <Q[:,i], Q[:,j]> = delta_{ij} to machine precision
  - QR reconstruction: Q @ R == A to machine precision

Provenance
----------
MATLAB source : @chebfun/qr.m, @chebfun/svd.m, abstractQR.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.linalg import (
    Quasimatrix,
    _legendre_basis,
    chebfun_qr,
    chebfun_svd,
    qr_quasimatrix,
    svd_quasimatrix,
)
from chebfunjax.domain import Domain

# Standard tolerances for spectral accuracy (float64)
RTOL = 1e-10
ATOL = 1e-10


# ============================================================================
# Tier 1: Quasimatrix construction and evaluation
# ============================================================================


class TestQuasimatrix:
    """Tests for the Quasimatrix class.

    JAX contract: jit=NO (Python dispatch), vmap=NO.
    """

    def test_from_functions(self):
        """Build a quasimatrix from [1, x, x^2] on [-1, 1]."""
        dom = Domain((-1.0, 1.0))
        one = chebfun(1.0)
        x = chebfun(lambda t: t)
        x2 = chebfun(lambda t: t ** 2)
        qm = Quasimatrix(cols=[one, x, x2], domain=dom)
        assert qm.n_cols == 3

    def test_eval_single_point(self):
        """Evaluate [1, x, x^2] at x=0.5 gives [1, 0.5, 0.25]."""
        dom = Domain((-1.0, 1.0))
        one = chebfun(1.0)
        x = chebfun(lambda t: t)
        x2 = chebfun(lambda t: t ** 2)
        qm = Quasimatrix(cols=[one, x, x2], domain=dom)
        val = qm(jnp.float64(0.5))
        npt.assert_allclose(np.array(val), [1.0, 0.5, 0.25], atol=1e-13)

    def test_eval_array(self):
        """Evaluate [1, x] at multiple points."""
        dom = Domain((-1.0, 1.0))
        one = chebfun(1.0)
        x = chebfun(lambda t: t)
        qm = Quasimatrix(cols=[one, x], domain=dom)
        xs = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float64)
        vals = qm(xs)  # shape (3, 2)
        assert vals.shape == (3, 2)
        npt.assert_allclose(np.array(vals[:, 0]), [1.0, 1.0, 1.0], atol=1e-13)
        npt.assert_allclose(np.array(vals[:, 1]), [-1.0, 0.0, 1.0], atol=1e-13)

    def test_domain_mismatch_raises(self):
        """Constructing with mismatched domains raises ValueError."""
        dom1 = Domain((-1.0, 1.0))
        Domain((0.0, 1.0))
        f = chebfun(jnp.sin)       # on [-1, 1]
        g = chebfun(jnp.cos, domain=(0.0, 1.0))
        with pytest.raises(ValueError, match="domain"):
            Quasimatrix(cols=[f, g], domain=dom1)

    def test_empty_raises(self):
        """Empty column list raises ValueError."""
        dom = Domain((-1.0, 1.0))
        with pytest.raises(ValueError):
            Quasimatrix(cols=[], domain=dom)


# ============================================================================
# Tier 2: Legendre basis
# ============================================================================


class TestLegendreBasis:
    """Tests for the _legendre_basis helper.

    Verifies that the constructed basis is truly L2-orthonormal on [a, b].
    """

    def test_orthonormality_unit_interval(self):
        """L2 inner products: <P_i, P_j> = delta_{ij} on [-1, 1]."""
        dom = Domain((-1.0, 1.0))
        n = 4
        basis = _legendre_basis(n, dom)
        for i in range(n):
            for j in range(n):
                ip = float(basis[i].inner(basis[j]))
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(ip, expected, atol=1e-12,
                    err_msg=f"<P_{i}, P_{j}> = {ip} (expected {expected})")

    def test_orthonormality_custom_domain(self):
        """L2 orthonormality on [0, 2]."""
        dom = Domain((0.0, 2.0))
        n = 3
        basis = _legendre_basis(n, dom)
        for i in range(n):
            for j in range(n):
                ip = float(basis[i].inner(basis[j]))
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(ip, expected, atol=1e-11,
                    err_msg=f"<P_{i}, P_{j}> = {ip} (expected {expected})")

    def test_basis_values_degree0(self):
        """P_0 on [-1, 1] is the constant 1/sqrt(2)."""
        dom = Domain((-1.0, 1.0))
        basis = _legendre_basis(1, dom)
        val = float(basis[0](jnp.float64(0.0)))
        npt.assert_allclose(val, 1.0 / np.sqrt(2.0), rtol=1e-13)

    def test_basis_values_degree1(self):
        """P_1 on [-1, 1] is sqrt(3/2) * x."""
        dom = Domain((-1.0, 1.0))
        basis = _legendre_basis(2, dom)
        x = jnp.array([0.0, 0.5, -0.5], dtype=jnp.float64)
        vals = np.array(basis[1](x))
        expected = np.sqrt(3.0 / 2.0) * np.array([0.0, 0.5, -0.5])
        npt.assert_allclose(vals, expected, atol=1e-12)


# ============================================================================
# Tier 3: QR factorization
# ============================================================================


class TestChebfunQR:
    """Tests for QR of quasimatrices.

    JAX contract: jit=NO, vmap=NO.
    """

    def _make_monomial_qm(self, n: int, domain=(-1.0, 1.0)) -> Quasimatrix:
        """Build quasimatrix [1, x, x^2, ..., x^{n-1}]."""
        dom = Domain(domain)
        cols = []
        for k in range(n):
            k_ = k  # closure capture
            cols.append(chebfun(lambda t, k=k_: t ** k, domain=domain))
        return Quasimatrix(cols=cols, domain=dom)

    def test_single_column_normalisation(self):
        """QR of a single column: Q = f/||f||, R = [[||f||]]."""
        f = chebfun(jnp.sin)
        Q, R = f.qr()
        assert Q.n_cols == 1
        assert R.shape == (1, 1)
        # Q[:,0] is normalised
        norm_q = float(Q.cols[0].norm())
        npt.assert_allclose(norm_q, 1.0, atol=1e-12)
        # R[0,0] is the original norm
        npt.assert_allclose(float(R[0, 0]), float(f.norm()), rtol=1e-12)

    def test_single_column_nonzero_sign(self):
        """R[0,0] is non-negative."""
        f = chebfun(lambda x: -jnp.sin(x))
        Q, R = f.qr()
        assert float(R[0, 0]) >= 0.0

    def test_orthonormality_3cols(self):
        """<Q[:,i], Q[:,j]> = delta_{ij} for QR of [1, x, x^2] on [-1, 1]."""
        qm = self._make_monomial_qm(3)
        Q, R = qr_quasimatrix(qm)
        for i in range(3):
            for j in range(3):
                ip = float(Q.cols[i].inner(Q.cols[j]))
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(ip, expected, atol=1e-10,
                    err_msg=f"<Q[:,{i}], Q[:,{j}]> = {ip}")

    def test_r_upper_triangular(self):
        """R is upper triangular (lower entries are zero)."""
        qm = self._make_monomial_qm(4)
        _, R = qr_quasimatrix(qm)
        R_np = np.array(R)
        # Check lower triangle (below diagonal) is near zero
        for i in range(1, 4):
            for j in range(i):
                npt.assert_allclose(R_np[i, j], 0.0, atol=1e-10,
                    err_msg=f"R[{i},{j}] = {R_np[i,j]} should be 0")

    def test_r_diagonal_nonneg(self):
        """Diagonal entries of R are non-negative."""
        qm = self._make_monomial_qm(3)
        _, R = qr_quasimatrix(qm)
        for i in range(3):
            assert float(R[i, i]) >= -1e-12, f"R[{i},{i}] = {float(R[i,i])} < 0"

    def test_reconstruction(self):
        """A = Q * R: evaluate at test points."""
        qm = self._make_monomial_qm(3)
        Q, R = qr_quasimatrix(qm)
        R_np = np.array(R)

        xs = jnp.linspace(-1.0, 1.0, 50, dtype=jnp.float64)
        # A[:,j] = sum_i Q[:,i] * R[i,j]
        # Evaluate all columns at xs
        A_vals = qm(xs)          # shape (50, 3)
        Q_vals = Q(xs)           # shape (50, 3)
        A_reconstructed = Q_vals @ R_np  # shape (50, 3)
        npt.assert_allclose(np.array(A_reconstructed), np.array(A_vals),
                            atol=1e-10,
                            err_msg="A != Q * R at test points")

    def test_q_approximates_legendre(self):
        """QR of [1, x, x^2] gives columns close to normalised Legendre polys.

        On [-1, 1]:
          Q[:,0] ~ 1/sqrt(2)              (= normalised P_0)
          Q[:,1] ~ sqrt(3/2) * x         (= normalised P_1)
          Q[:,2] ~ sqrt(5/2)*(3x^2-1)/2  (= normalised P_2)
        (up to sign convention)
        """
        qm = self._make_monomial_qm(3)
        Q, _ = qr_quasimatrix(qm)

        xs = jnp.linspace(-1.0, 1.0, 100, dtype=jnp.float64)
        xs_np = np.array(xs)

        # Expected normalised Legendre polynomials on [-1, 1]
        P0 = np.full_like(xs_np, 1.0 / np.sqrt(2.0))
        P1 = np.sqrt(3.0 / 2.0) * xs_np
        P2 = np.sqrt(5.0 / 2.0) * (3.0 * xs_np ** 2 - 1.0) / 2.0

        q0 = np.array(Q.cols[0](xs))
        q1 = np.array(Q.cols[1](xs))
        q2 = np.array(Q.cols[2](xs))

        # Q columns match Legendre (up to global sign)
        assert (
            np.allclose(q0, P0, atol=1e-9) or np.allclose(q0, -P0, atol=1e-9)
        ), f"Q[:,0] != ±P_0, max diff = {np.max(np.abs(q0 - P0))}"
        assert (
            np.allclose(q1, P1, atol=1e-9) or np.allclose(q1, -P1, atol=1e-9)
        ), "Q[:,1] != ±P_1"
        assert (
            np.allclose(q2, P2, atol=1e-9) or np.allclose(q2, -P2, atol=1e-9)
        ), "Q[:,2] != ±P_2"

    def test_qr_custom_domain(self):
        """QR works on a non-standard domain [0, pi]."""
        dom = (0.0, float(jnp.pi))
        qm = self._make_monomial_qm(3, domain=dom)
        Q, R = qr_quasimatrix(qm)
        # Orthonormality
        for i in range(3):
            for j in range(3):
                ip = float(Q.cols[i].inner(Q.cols[j]))
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(ip, expected, atol=1e-9,
                    err_msg=f"<Q[:,{i}], Q[:,{j}]> = {ip}")

    def test_chebfun_qr_method(self):
        """Chebfun.qr() method with other_cols gives same result as qr_quasimatrix."""
        dom = Domain((-1.0, 1.0))
        one = chebfun(1.0)
        x = chebfun(lambda t: t)
        x2 = chebfun(lambda t: t ** 2)

        Q_method, R_method = one.qr([x, x2])
        qm = Quasimatrix(cols=[one, x, x2], domain=dom)
        Q_func, R_func = qr_quasimatrix(qm)

        # Both should give the same R (up to sign of each row, but since we
        # enforce non-negative diagonal they should agree exactly)
        npt.assert_allclose(np.array(R_method), np.array(R_func), atol=1e-10)

    def test_orthonormality_5cols(self):
        """QR of [1, x, x^2, x^3, x^4] gives 5 orthonormal columns."""
        qm = self._make_monomial_qm(5)
        Q, R = qr_quasimatrix(qm)
        for i in range(5):
            for j in range(5):
                ip = float(Q.cols[i].inner(Q.cols[j]))
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(ip, expected, atol=1e-9,
                    err_msg=f"<Q[:,{i}], Q[:,{j}]> = {ip}")

    def test_qr_sin_cos(self):
        """QR of [sin, cos] gives orthonormal columns."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        Q, R = f.qr([g])
        ip = float(Q.cols[0].inner(Q.cols[1]))
        npt.assert_allclose(ip, 0.0, atol=1e-11)
        npt.assert_allclose(float(Q.cols[0].norm()), 1.0, atol=1e-12)
        npt.assert_allclose(float(Q.cols[1].norm()), 1.0, atol=1e-12)


# ============================================================================
# Tier 4: SVD
# ============================================================================


class TestChebfunSVD:
    """Tests for SVD of quasimatrices.

    JAX contract: jit=NO, vmap=NO.
    """

    def _make_monomial_qm(self, n: int) -> Quasimatrix:
        dom = Domain((-1.0, 1.0))
        cols = [chebfun(lambda t, k=k: t ** k) for k in range(n)]
        return Quasimatrix(cols=cols, domain=dom)

    def test_singular_values_positive(self):
        """Singular values of [1, x, x^2] are positive."""
        qm = self._make_monomial_qm(3)
        U, S, V = svd_quasimatrix(qm)
        assert np.all(np.array(S) > 0.0)

    def test_singular_values_decreasing(self):
        """Singular values are in non-increasing order."""
        qm = self._make_monomial_qm(4)
        _, S, _ = svd_quasimatrix(qm)
        S_np = np.array(S)
        assert np.all(np.diff(S_np) <= 1e-12), f"S not non-increasing: {S_np}"

    def test_u_orthonormal(self):
        """Left singular functions U are L2-orthonormal."""
        qm = self._make_monomial_qm(3)
        U, S, V = svd_quasimatrix(qm)
        for i in range(3):
            for j in range(3):
                ip = float(U.cols[i].inner(U.cols[j]))
                expected = 1.0 if i == j else 0.0
                npt.assert_allclose(ip, expected, atol=1e-9,
                    err_msg=f"<U[:,{i}], U[:,{j}]> = {ip}")

    def test_v_orthonormal(self):
        """Right singular vectors V form an orthonormal matrix."""
        qm = self._make_monomial_qm(3)
        _, _, V = svd_quasimatrix(qm)
        V_np = np.array(V)
        VtV = V_np.T @ V_np
        npt.assert_allclose(VtV, np.eye(3), atol=1e-10)

    def test_reconstruction(self):
        """A = U * diag(S) * V^T at test points."""
        qm = self._make_monomial_qm(3)
        U, S, V = svd_quasimatrix(qm)
        S_np = np.array(S)
        V_np = np.array(V)

        xs = jnp.linspace(-1.0, 1.0, 50, dtype=jnp.float64)
        A_vals = np.array(qm(xs))        # (50, 3)
        U_vals = np.array(U(xs))         # (50, 3)
        # A = U * diag(S) * V^T
        A_reconstructed = (U_vals * S_np) @ V_np.T
        npt.assert_allclose(A_reconstructed, A_vals, atol=1e-9,
            err_msg="A != U * S * V^T at test points")

    def test_single_column_svd(self):
        """SVD of a single Chebfun column."""
        f = chebfun(jnp.sin)
        U, S, V = f.svd()
        assert U.n_cols == 1
        assert S.shape == (1,)
        assert V.shape == (1, 1)
        # S[0] = ||f||_2
        npt.assert_allclose(float(S[0]), float(f.norm()), rtol=1e-11)
        # U[:,0] is normalised
        npt.assert_allclose(float(U.cols[0].norm()), 1.0, atol=1e-12)

    def test_chebfun_svd_method(self):
        """Chebfun.svd() method matches svd_quasimatrix."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        h = chebfun(jnp.exp)
        U_m, S_m, V_m = f.svd([g, h])
        dom = Domain((-1.0, 1.0))
        qm = Quasimatrix(cols=[f, g, h], domain=dom)
        U_f, S_f, V_f = svd_quasimatrix(qm)
        # Singular values should match
        npt.assert_allclose(np.array(S_m), np.array(S_f), rtol=1e-10)

    def test_svd_gram_matrix_singular_values(self):
        """Singular values of [1, x] on [-1, 1] match sqrt of Gram matrix eigs.

        Gram matrix G[i,j] = <A[:,i], A[:,j]>:
          G = [[2,    0   ],
               [0,    2/3 ]]
        (since int_{-1}^{1} 1 dx = 2, int 1*x dx = 0, int x^2 dx = 2/3)
        eigenvalues: sqrt(2) and sqrt(2/3).
        """
        one = chebfun(1.0)
        x = chebfun(lambda t: t)
        dom = Domain((-1.0, 1.0))
        qm = Quasimatrix(cols=[one, x], domain=dom)
        _, S, _ = svd_quasimatrix(qm)
        S_np = np.sort(np.array(S))[::-1]  # descending
        expected = np.array([np.sqrt(2.0), np.sqrt(2.0 / 3.0)])
        npt.assert_allclose(S_np, expected, rtol=1e-10)


# ============================================================================
# Tier 5: Convenience wrappers
# ============================================================================


class TestConvenienceAPI:
    """Tests for chebfun_qr / chebfun_svd convenience wrappers."""

    def test_chebfun_qr_wrapper(self):
        """chebfun_qr([f, g]) works and returns Q, R."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        Q, R = chebfun_qr([f, g])
        assert isinstance(Q, Quasimatrix)
        assert Q.n_cols == 2
        assert R.shape == (2, 2)

    def test_chebfun_svd_wrapper(self):
        """chebfun_svd([f, g]) works and returns U, S, V."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        U, S, V = chebfun_svd([f, g])
        assert isinstance(U, Quasimatrix)
        assert S.shape == (2,)
        assert V.shape == (2, 2)

    def test_empty_cols_raises(self):
        """chebfun_qr([]) and chebfun_svd([]) raise ValueError."""
        with pytest.raises(ValueError):
            chebfun_qr([])
        with pytest.raises(ValueError):
            chebfun_svd([])
