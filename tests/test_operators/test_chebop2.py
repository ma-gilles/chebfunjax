"""Tests for Chebop2 (U64) — 2D PDE solver on rectangles.

Tests the following PDEs:

1. Poisson equation on [-1,1]²:
      u_xx + u_yy = f,  u|_∂Ω = 0
   Exact solution: u = (1 - x²)(1 - y²)
   RHS: f = -2(1 - x²) - 2(1 - y²) = -4 + 2x² + 2y²

2. Helmholtz equation on [-1,1]²:
      u_xx + u_yy + k²u = 0,  u|_∂Ω = g
   Exact solution chosen so that boundary data is consistent.

3. Poisson with non-zero boundary data:
      u_xx + u_yy = 0,  u = x² on all edges  →  u = x²

4. Proxy extraction test: verify the coefficient matrix is correctly
   extracted from operator lambdas.

5. Bartels-Stewart solver: unit test on a 2x2 example with known solution.

MATLAB golden refs
------------------
Poisson: u(0, 0) = 1  (maximum of (1 - x²)(1 - y²))
Helmholtz: RHS zero, BC data from exact solution.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.operators.chebop2 import (
    Chebop2,
    _Chebop2Proxy,
    _op_from_coeffs,
    _resolve_dir,
    bartels_stewart,
    diffx,
    diffy,
    lap,
    laplacian,
)

# ===========================================================================
# Proxy tests
# ===========================================================================


class TestProxy:
    """Test the _Chebop2Proxy coefficient extraction."""

    def test_laplacian_coeffs(self):
        """lambda u: u.diff(2,0) + u.diff(0,2) → A has 1 at (2,0) and (0,2)."""
        proxy = _Chebop2Proxy()
        result = proxy.diff(2, 0) + proxy.diff(0, 2)
        A = result._coeffs_matrix()
        # A[j, k] = coeff of d^j/dy^j d^k/dx^k
        assert A.shape[0] >= 3 and A.shape[1] >= 3
        npt.assert_allclose(A[2, 0], 1.0)  # u_yy
        npt.assert_allclose(A[0, 2], 1.0)  # u_xx

    def test_helmholtz_coeffs(self):
        """lambda u: u.diff(2,0) + u.diff(0,2) + 4*u → extra (0,0) = 4."""
        proxy = _Chebop2Proxy()
        result = proxy.diff(2, 0) + proxy.diff(0, 2) + 4.0 * proxy
        A = result._coeffs_matrix()
        npt.assert_allclose(A[0, 0], 4.0)  # u term
        npt.assert_allclose(A[2, 0], 1.0)  # u_yy
        npt.assert_allclose(A[0, 2], 1.0)  # u_xx

    def test_scalar_multiplication(self):
        """2 * u.diff(2, 0) → coefficient 2 at (2, 0)."""
        proxy = _Chebop2Proxy()
        result = 2.0 * proxy.diff(2, 0)
        A = result._coeffs_matrix()
        npt.assert_allclose(A[2, 0], 2.0)

    def test_negation(self):
        """-u has coefficient -1 at (0, 0)."""
        proxy = _Chebop2Proxy()
        result = -proxy
        A = result._coeffs_matrix()
        npt.assert_allclose(A[0, 0], -1.0)

    def test_subtraction(self):
        """u.diff(2,0) - u.diff(0,2) has (2,0)=+1, (0,2)=-1."""
        proxy = _Chebop2Proxy()
        result = proxy.diff(2, 0) - proxy.diff(0, 2)
        A = result._coeffs_matrix()
        npt.assert_allclose(A[2, 0], 1.0)
        npt.assert_allclose(A[0, 2], -1.0)


# ===========================================================================
# Bartels-Stewart unit test
# ===========================================================================


class TestBartelsStewart:
    """Unit test for the Bartels-Stewart solver."""

    def test_identity_rhs_zero(self):
        """AXB^T + CXD^T = 0  →  X = 0."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        C = 2.0 * np.eye(3, dtype=np.float64)
        D = 2.0 * np.eye(3, dtype=np.float64)
        E = np.zeros((3, 3), dtype=np.float64)
        X = bartels_stewart(A, B, C, D, E)
        npt.assert_allclose(X, np.zeros((3, 3)), atol=1e-13)

    def test_simple_2x2(self):
        """A X B^T + C X D^T = E with known solution X = ones(2,2)."""
        # A = I, C = I, B = I, D = I  → 2X = E  → X = E/2
        n = 3
        A = np.eye(n, dtype=np.float64)
        C = np.eye(n, dtype=np.float64)
        B = np.eye(n, dtype=np.float64)
        D = np.eye(n, dtype=np.float64)
        X_exact = np.ones((n, n), dtype=np.float64)
        E = 2.0 * X_exact  # A X B^T + C X D^T = 2X = E
        X = bartels_stewart(A, B, C, D, E)
        npt.assert_allclose(X, X_exact, atol=1e-12)

    def test_diagonal_system(self):
        """Diagonal A, B, C, D: solution is element-wise solvable."""
        n = 4
        a_diag = np.array([1.0, 2.0, 3.0, 4.0])
        b_diag = np.array([1.0, 1.0, 2.0, 2.0])
        c_diag = np.array([2.0, 1.0, 1.0, 2.0])
        d_diag = np.array([1.0, 2.0, 1.0, 3.0])
        A = np.diag(a_diag)
        B = np.diag(b_diag)
        C = np.diag(c_diag)
        D = np.diag(d_diag)
        X_exact = np.random.default_rng(42).standard_normal((n, n))
        E = A @ X_exact @ B.T + C @ X_exact @ D.T
        X = bartels_stewart(A, B, C, D, E)
        npt.assert_allclose(X, X_exact, atol=1e-11)

    @pytest.mark.parametrize("shape", [(5, 6), (7, 7), (8, 4)])
    def test_random_generalized_sylvester_residual(self, shape):
        """Dense random A, B, C, D, E: residual of A X B^T + C X D^T - E is tiny.

        Regression pin for the QZ Bartels-Stewart reduction.  Identity/diagonal
        inputs (the other tests) have trivial QZ factors and hid a transposed
        RHS transform; only dense random pencils exercise it.
        """
        m, n = shape
        rng = np.random.default_rng(12345 + m + n)
        A = rng.standard_normal((m, m))
        C = rng.standard_normal((m, m))
        B = rng.standard_normal((n, n))
        D = rng.standard_normal((n, n))
        E = rng.standard_normal((m, n))
        X = bartels_stewart(A, B, C, D, E)
        residual = np.linalg.norm(A @ X @ B.T + C @ X @ D.T - E)
        assert residual < 1e-10, f"residual={residual:.3e}"

    def test_random_matches_kronecker_solve(self):
        """Bartels-Stewart matches the direct Kronecker solve of the same system."""
        m, n = 6, 5
        rng = np.random.default_rng(2024)
        A = rng.standard_normal((m, m))
        C = rng.standard_normal((m, m))
        B = rng.standard_normal((n, n))
        D = rng.standard_normal((n, n))
        E = rng.standard_normal((m, n))
        X = bartels_stewart(A, B, C, D, E)
        # Direct solve: (B (x) A + D (x) C) vec(X) = vec(E), column-major.
        M = np.kron(B, A) + np.kron(D, C)
        X_kron = np.linalg.solve(M, E.ravel(order="F")).reshape((m, n), order="F")
        npt.assert_allclose(X, X_kron, atol=1e-9)


# ===========================================================================
# Chebop2 construction tests
# ===========================================================================


class TestChebop2Construction:
    """Test Chebop2 object construction and property setting."""

    def test_laplacian_coeffs_extracted(self):
        """Chebop2 extracts the correct coefficient matrix for Laplacian."""
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        A = N._coeffs
        assert A is not None
        npt.assert_allclose(A[2, 0], 1.0, atol=1e-14)  # u_yy
        npt.assert_allclose(A[0, 2], 1.0, atol=1e-14)  # u_xx

    def test_xorder_yorder(self):
        """xorder and yorder are correctly identified."""
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        assert N._xorder == 2
        assert N._yorder == 2

    def test_bc_assignment(self):
        """bc= sets all four boundary conditions."""
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        N.bc = 0.0
        assert N._lbc == 0.0
        assert N._rbc == 0.0
        assert N._ubc == 0.0
        assert N._dbc == 0.0

    def test_individual_bcs(self):
        """lbc, rbc, ubc, dbc can be set independently."""
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        N.lbc = 1.0
        N.rbc = 2.0
        N.ubc = 3.0
        N.dbc = 4.0
        assert N._lbc == 1.0
        assert N._rbc == 2.0
        assert N._ubc == 3.0
        assert N._dbc == 4.0

    def test_domain_validation(self):
        """Invalid domains raise ValueError."""
        with pytest.raises(ValueError, match="xa < xb"):
            Chebop2(lambda u: u.diff(2, 0), domain=(1.0, -1.0, -1.0, 1.0))

    def test_repr(self):
        """__repr__ returns a string."""
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        r = repr(N)
        assert "Chebop2" in r
        assert "domain" in r

    def test_custom_domain(self):
        """Domain is correctly stored."""
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2), domain=(0.0, 1.0, 0.0, 1.0))
        assert N.domain == (0.0, 1.0, 0.0, 1.0)


# ===========================================================================
# Poisson equation test
# ===========================================================================


class TestPoissonDirichlet:
    """Poisson: u_xx + u_yy = f,  u|_∂Ω = 0.

    Exact solution: u(x, y) = (1 - x²)(1 - y²)
    RHS:  f = u_xx + u_yy = -2(1 - y²) - 2(1 - x²)
    Boundary data: u = 0 on all edges (since u|_{x=±1} = 0, u|_{y=±1} = 0).
    """

    @classmethod
    def _u_exact(cls, x, y):
        return (1.0 - x**2) * (1.0 - y**2)

    @classmethod
    def _f_rhs(cls, x, y):
        return -2.0 * (1.0 - y**2) - 2.0 * (1.0 - x**2)

    def setup_method(self):
        self.N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        self.N.bc = 0.0

    def test_center_value(self):
        """Solution at (0, 0) should be 1.0 (the maximum)."""
        u = self.N.solve(self._f_rhs, n=12)
        u0 = float(u(jnp.array(0.0, dtype=jnp.float64),
                      jnp.array(0.0, dtype=jnp.float64)))
        npt.assert_allclose(u0, 1.0, atol=1e-9)

    def test_multiple_points(self):
        """Solution matches exact at a grid of points."""
        u = self.N.solve(self._f_rhs, n=14)
        test_pts = [-0.5, 0.0, 0.5]
        for xi in test_pts:
            for yi in test_pts:
                u_num = float(u(jnp.array(xi, dtype=jnp.float64),
                                 jnp.array(yi, dtype=jnp.float64)))
                u_ex = float(self._u_exact(xi, yi))
                npt.assert_allclose(u_num, u_ex, atol=1e-8,
                                     err_msg=f"At ({xi}, {yi})")

    def test_boundary_values(self):
        """Solution is zero (or near-zero) on the boundary."""
        u = self.N.solve(self._f_rhs, n=12)
        # Check a few boundary points
        # x = ±1 edges
        for yi in [-0.5, 0.0, 0.5]:
            v_l = float(u(jnp.array(-1.0, dtype=jnp.float64),
                           jnp.array(yi, dtype=jnp.float64)))
            v_r = float(u(jnp.array(1.0, dtype=jnp.float64),
                           jnp.array(yi, dtype=jnp.float64)))
            npt.assert_allclose(v_l, 0.0, atol=1e-8)
            npt.assert_allclose(v_r, 0.0, atol=1e-8)

    def test_mldivide_syntax(self):
        """N \\ f syntax works."""
        u = self.N / self._f_rhs
        u0 = float(u(jnp.array(0.0, dtype=jnp.float64),
                      jnp.array(0.0, dtype=jnp.float64)))
        npt.assert_allclose(u0, 1.0, atol=1e-8)

    def test_adaptive_solve(self):
        """Adaptive solve (n=None) also converges to correct answer."""
        u = self.N.solve(self._f_rhs)
        u0 = float(u(jnp.array(0.0, dtype=jnp.float64),
                      jnp.array(0.0, dtype=jnp.float64)))
        npt.assert_allclose(u0, 1.0, atol=1e-8)


# ===========================================================================
# Poisson with constant RHS (simpler exact solution)
# ===========================================================================


class TestPoissonConstantRHS:
    """Poisson with RHS = constant.

    u_xx + u_yy = -2,  u = 0 on boundary of [-1,1]².
    By symmetry, u(0,0) is the maximum.  We can check by comparing to the
    known solution u(x, y) = (1 - x²)/2 + (1 - y²)/2 - 1 = -(x² + y²)/2.
    Wait, that does NOT satisfy zero Dirichlet BCs.

    Use instead: u(x,y) = 1 - x²,  then u_xx = -2, u_yy = 0, f = -2,
    but u is not zero on y = ±1.

    Simpler: u_xx + u_yy = -4, with u = (1-x²)(1-y²) ≥ 0.
    Already tested above.  Test a different polynomial here.

    u = 1 - x^4,  u_xx = -12x², u_yy = 0, f = -12x².
    BCs: u(-1) = 0, u(1) = 0, u(y=±1) = 1 - x^4.
    """

    def test_polynomial_solution(self):
        """u = 1 - x^4 satisfies u_xx = -12x^2 with appropriate BCs."""
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        # Boundary conditions from u = 1 - x^4:
        # lbc (x=-1): u(-1, y) = 1 - 1 = 0
        # rbc (x=+1): u(+1, y) = 1 - 1 = 0
        # ubc (y=+1): u(x, +1) = 1 - x^4  (function of x)
        # dbc (y=-1): u(x, -1) = 1 - x^4  (function of x)
        N.lbc = 0.0
        N.rbc = 0.0
        N.ubc = lambda x: 1.0 - x**4
        N.dbc = lambda x: 1.0 - x**4

        def f_rhs(x, y):
            return -12.0 * x**2 * jnp.ones_like(y)

        u = N.solve(f_rhs, n=14)
        # Check at interior points
        for xi in [-0.5, 0.0, 0.5]:
            for yi in [-0.5, 0.0, 0.5]:
                u_num = float(u(jnp.array(xi, dtype=jnp.float64),
                                 jnp.array(yi, dtype=jnp.float64)))
                u_ex = 1.0 - xi**4
                npt.assert_allclose(u_num, u_ex, atol=1e-7,
                                     err_msg=f"At ({xi}, {yi})")


# ===========================================================================
# Helmholtz equation test
# ===========================================================================


class TestHelmholtz:
    """Helmholtz: u_xx + u_yy + k²u = f.

    Choose k=1, exact solution u = cos(pi*x/2) * cos(pi*y/2).
    Then u_xx = -(pi/2)^2 u, u_yy = -(pi/2)^2 u.
    f = [k² - 2*(pi/2)²] u = [1 - pi²/2] u.
    BCs: u|_{x=±1} = 0, u|_{y=±1} = 0 (since cos(pi/2) = 0).
    """

    def setup_method(self):
        k = 1.0
        self.k = k
        self.N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2) + k**2 * u)
        self.N.bc = 0.0

    def _u_exact(self, x, y):
        return jnp.cos(jnp.pi * x / 2.0) * jnp.cos(jnp.pi * y / 2.0)

    def _f_rhs(self, x, y):
        pi_half_sq = (jnp.pi / 2.0) ** 2
        return (self.k**2 - 2.0 * pi_half_sq) * self._u_exact(x, y)

    def test_center_value(self):
        """Solution at (0, 0) should equal u_exact(0, 0) = 1.0."""
        u = self.N.solve(self._f_rhs, n=14)
        u0 = float(u(jnp.array(0.0, dtype=jnp.float64),
                      jnp.array(0.0, dtype=jnp.float64)))
        npt.assert_allclose(u0, 1.0, atol=1e-7)

    def test_interior_values(self):
        """Solution matches exact at several interior points."""
        u = self.N.solve(self._f_rhs, n=16)
        for xi, yi in [(-0.3, 0.4), (0.7, -0.2), (0.0, 0.5)]:
            u_num = float(u(jnp.array(xi, dtype=jnp.float64),
                             jnp.array(yi, dtype=jnp.float64)))
            u_ex = float(self._u_exact(jnp.array(xi), jnp.array(yi)))
            npt.assert_allclose(u_num, u_ex, atol=1e-7,
                                 err_msg=f"At ({xi}, {yi})")


# ===========================================================================
# Non-default domain test
# ===========================================================================


class TestNonDefaultDomain:
    """Poisson on [0, 2] x [0, 3] with known polynomial solution."""

    def test_poisson_on_rectangle(self):
        """u = (x - 1)^2 * (y - 1.5)^2 * 4 on [0,2] x [0,3]."""
        # u(x,y) = (x-1)^2 (y-1.5)^2
        # u_xx = 2 (y-1.5)^2
        # u_yy = 2 (x-1)^2
        # f = 2(y-1.5)^2 + 2(x-1)^2
        domain = (0.0, 2.0, 0.0, 3.0)
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2), domain=domain)
        N.lbc = lambda y: (0.0 - 1.0)**2 * (y - 1.5)**2  # x=0 edge
        N.rbc = lambda y: (2.0 - 1.0)**2 * (y - 1.5)**2  # x=2 edge
        N.dbc = lambda x: (x - 1.0)**2 * (0.0 - 1.5)**2  # y=0 edge
        N.ubc = lambda x: (x - 1.0)**2 * (3.0 - 1.5)**2  # y=3 edge

        def f_rhs(x, y):
            return 2.0 * (y - 1.5)**2 + 2.0 * (x - 1.0)**2

        u = N.solve(f_rhs, n=12)
        # Check at center of rectangle (1.0, 1.5)
        u0 = float(u(jnp.array(1.0, dtype=jnp.float64),
                      jnp.array(1.5, dtype=jnp.float64)))
        npt.assert_allclose(u0, 0.0, atol=1e-7)

        # Check off-center
        xi, yi = 0.5, 1.0
        u_num = float(u(jnp.array(xi, dtype=jnp.float64),
                         jnp.array(yi, dtype=jnp.float64)))
        u_ex = (xi - 1.0)**2 * (yi - 1.5)**2
        npt.assert_allclose(u_num, u_ex, atol=1e-7)


# ===========================================================================
# Operator introspection, arithmetic, and free-function helpers (U64)
# ===========================================================================


class TestChebop2Accessors:
    """Coefficient/order accessors and operator arithmetic."""

    def test_coeffs_is_transpose_of_internal(self):
        # diffx(u,3)+diffx(diffy(u,1),2)+diffy(u,2)+pi*u  (non-symmetric).
        N = Chebop2(
            lambda u: diffx(u, 3)
            + diffx(diffy(u, 1), 2)
            + diffy(u, 2)
            + np.pi * u
        )
        # MATLAB @chebop2 layout: rows index x-order, cols index y-order.
        expected = np.array(
            [[np.pi, 0, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float64
        )
        npt.assert_allclose(N.coeffs, expected, atol=1e-15)
        assert N.xorder == 3
        assert N.yorder == 2

    def test_coeffs_empty_operator(self):
        N = Chebop2()  # op is None
        npt.assert_allclose(N.coeffs, np.zeros((1, 1)))

    def test_free_functions(self):
        N1 = Chebop2(lambda u: laplacian(u))
        N2 = Chebop2(lambda u: lap(u))
        N3 = Chebop2(lambda u: diffx(u, 2) + diffy(u, 2))
        npt.assert_allclose(N1.coeffs, N3.coeffs, atol=1e-15)
        npt.assert_allclose(N2.coeffs, N3.coeffs, atol=1e-15)

    def test_add_operators(self):
        N1 = Chebop2(lambda u: diffx(u, 2))
        N2 = Chebop2(lambda u: diffy(u, 2))
        N = N1 + N2
        assert N.xorder == 2 and N.yorder == 2
        expected = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64)
        npt.assert_allclose(N.coeffs, expected, atol=1e-15)

    def test_add_then_solve(self):
        # Combined operator must remain solvable (Laplacian, u == 0 on bdy).
        N = Chebop2(lambda u: diffx(u, 2)) + Chebop2(lambda u: diffy(u, 2))
        N.bc = 0.0
        u = N.solve(1.0, n=15)
        # Poisson u_xx+u_yy=1 with zero BC has a negative-definite interior.
        u0 = float(u(jnp.array(0.0), jnp.array(0.0)))
        assert u0 < 0.0

    def test_sub_operators(self):
        N = Chebop2(lambda u: diffx(u, 2) + diffy(u, 2))
        M = N - Chebop2(lambda u: diffy(u, 2))
        # Left with u_xx only: coeffs[2,0] == 1, everything else 0.
        expected = np.zeros((3, 3), dtype=np.float64)
        expected[2, 0] = 1.0
        npt.assert_allclose(M.coeffs, expected, atol=1e-15)

    def test_add_domain_mismatch(self):
        N1 = Chebop2(lambda u: diffx(u, 2), domain=(-1.0, 1.0, -1.0, 1.0))
        N2 = Chebop2(lambda u: diffy(u, 2), domain=(0.0, 1.0, 0.0, 1.0))
        with pytest.raises(ValueError):
            _ = N1 + N2

    def test_add_non_chebop2(self):
        N1 = Chebop2(lambda u: diffx(u, 2))
        assert N1.__add__(5) is NotImplemented
        assert N1.__sub__(5) is NotImplemented

    def test_op_from_coeffs_roundtrip(self):
        S = np.zeros((3, 3), dtype=np.float64)
        S[2, 0] = 1.0  # u_yy
        S[0, 2] = 2.0  # 2 u_xx
        op = _op_from_coeffs(S)
        proxy = op(_Chebop2Proxy())
        A = proxy._coeffs_matrix()
        npt.assert_allclose(A[2, 0], 1.0)
        npt.assert_allclose(A[0, 2], 2.0)

    def test_op_from_coeffs_zero(self):
        op = _op_from_coeffs(np.zeros((1, 1), dtype=np.float64))
        proxy = op(_Chebop2Proxy())
        npt.assert_allclose(proxy._coeffs_matrix(), np.zeros((1, 1)))


class TestResolveDir:
    """Unit tests for the per-direction adaptive resolution check."""

    def test_decayed_tail_resolves(self):
        # A coefficient matrix whose last rows/cols are tiny is resolved and
        # keeps the same grid size.
        C = np.zeros((17, 17), dtype=np.float64)
        C[0, 0] = 1.0  # a single low mode; the tails are exactly zero
        res_y, new_y = _resolve_dir(C, 0, 17, 1e-14)
        res_x, new_x = _resolve_dir(C, 1, 17, 1e-14)
        assert res_y and res_x
        assert new_y == 17 and new_x == 17

    def test_large_tail_doubles(self):
        # A non-decaying tail is unresolved and the grid is doubled.
        C = np.ones((17, 17), dtype=np.float64)
        res_y, new_y = _resolve_dir(C, 0, 17, 1e-14)
        res_x, new_x = _resolve_dir(C, 1, 17, 1e-14)
        assert (not res_y) and (not res_x)
        assert new_y == 33 and new_x == 33

    def test_direction_independence(self):
        # Tail only in the y-direction: y unresolved, x resolved.
        C = np.zeros((20, 20), dtype=np.float64)
        C[-1, 0] = 1.0  # high y-mode, low x-mode
        res_y, _ = _resolve_dir(C, 0, 20, 1e-14)
        res_x, _ = _resolve_dir(C, 1, 20, 1e-14)
        assert (not res_y) and res_x


class TestParabolicSolve:
    """The rectangular adaptive solve of a parabolic (space-time) problem."""

    def test_heat_solve_no_warning_and_bc(self):
        # u_t = u_xx on [-1,1]x[0,0.5], u(x,0)=(1-x^2), u(+-1,t)=0.
        # The parabolic solve must complete without a non-convergence warning
        # and honour the initial + boundary conditions.
        N = Chebop2(lambda u: diffy(u, 1) - diffx(u, 2),
                    domain=(-1.0, 1.0, 0.0, 0.5))
        N.dbc = lambda x: 1.0 - x ** 2
        N.lbc = 0.0
        N.rbc = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any convergence warning -> failure
            u = N.solve(0.0)
        xs = np.linspace(-1.0, 1.0, 25)
        # Initial condition at t=0.
        u0 = np.asarray(u(jnp.asarray(xs), jnp.zeros(xs.size)))
        npt.assert_allclose(u0, 1.0 - xs ** 2, atol=1e-8)
        # Dirichlet edges held at zero for all sampled times.
        for t in (0.0, 0.25, 0.5):
            lval = float(u(jnp.asarray([-1.0]), jnp.asarray([t]))[0])
            rval = float(u(jnp.asarray([1.0]), jnp.asarray([t]))[0])
            assert abs(lval) < 1e-8 and abs(rval) < 1e-8

    def test_heat_solve_interval_independence(self):
        # The heat solution must not depend on the time-interval length; solve
        # on [0,0.5] and [0,0.8] and compare on the common region.
        def solve(tmax):
            N = Chebop2(lambda u: diffy(u, 1) - diffx(u, 2),
                        domain=(-1.0, 1.0, 0.0, tmax))
            N.dbc = lambda x: 1.0 - x ** 2
            N.lbc = 0.0
            N.rbc = 0.0
            return N.solve(0.0)

        u = solve(0.5)
        v = solve(0.8)
        xx = np.linspace(-1.0, 1.0, 30)
        tt = np.linspace(0.0, 0.5, 30)
        X, Y = np.meshgrid(xx, tt)
        uv = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
        vv = np.asarray(v(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
        assert np.max(np.abs(uv - vv)) < 1e-7


class TestVariableCoefficientValidation:
    """Manufactured-solution validation of the variable-coefficient
    (CDR / ultraspherical) path and the two-argument Robin BC form.

    Note the proxy convention: ``u.diff(a, b)`` is (y-order, x-order),
    matching MATLAB's row-first ``diff`` dimension ordering.
    """

    def test_variable_coefficient_manufactured(self):
        # (1 + x^2/4) u_yy + u_xx = f, exact u = (1-x^2)(1-y^2).
        N = Chebop2(lambda x, y, u: (1.0 + x * x / 4.0) * u.diff(2, 0)
                    + u.diff(0, 2))
        N.lbc = 0.0
        N.rbc = 0.0
        N.dbc = 0.0
        N.ubc = 0.0

        def f(x, y):
            return (1.0 + x * x / 4.0) * (-2.0 * (1 - x * x)) \
                + (-2.0 * (1 - y * y))

        u = N.solve(f)
        for xv, yv in [(0.0, 0.0), (0.3, -0.4), (-0.7, 0.2)]:
            want = (1 - xv ** 2) * (1 - yv ** 2)
            got = float(u(jnp.asarray(xv), jnp.asarray(yv)))
            assert abs(got - want) < 1e-11, (xv, yv, got, want)

    def test_robin_bc_two_argument(self):
        # Laplace, exact u = x; on the left edge u - u_n + 2 = 0
        # (u(-1) = -1, u_x = 1).
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        N.lbc = lambda y, u: u - u.diff() + 2.0
        N.rbc = 1.0
        N.dbc = lambda x: x
        N.ubc = lambda x: x
        u = N.solve(0.0)
        for xv, yv in [(0.5, 0.0), (-0.5, 0.3), (0.0, -0.8)]:
            assert abs(float(u(jnp.asarray(xv), jnp.asarray(yv))) - xv) \
                < 1e-10
