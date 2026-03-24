"""Tests for chebfunjax.operators.blocks and chebfunjax.operators.chebmatrix.

Covers:
- OperatorBlock: D, I, diag, algebra (+, -, *, ^, negation)
- FunctionalBlock: eval_at, sum_functional, algebra
- ChebMatrix: matrix assembly, BVP solve
- Mathematics: D applied to sin gives cos; u'' + u eigenvalues; BVP solve

JAX contract: jit=no (operator construction is outside JIT), vmap=no, grad=no
The matrix() method calls JAX operations (diffmat, chebweights) but the
assembly logic itself is Python-level and not JIT-compiled.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import chebfunjax as cj
from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    D,
    I,
    diag,
    eval_at,
    sum_functional,
)
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.utils.quadrature import chebpts

# ===========================================================================
# Tier 1 — OperatorBlock: basic construction and matrix shapes
# ===========================================================================


class TestOperatorBlockBasic:
    """Basic construction and matrix properties of OperatorBlock.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_D_order1_shape(self):
        """D(order=1).matrix(n) has shape (n, n)."""
        for n in [4, 8, 16]:
            op = D()
            disc = ChebColloc2Disc(n)
            M = op.matrix(disc)
            assert M.shape == (n, n)

    def test_D_order2_shape(self):
        """D(order=2).matrix(n) has shape (n, n)."""
        op = D(order=2)
        M = op.matrix(8)
        assert M.shape == (8, 8)

    def test_I_is_eye(self):
        """I().matrix(n) == eye(n)."""
        for n in [4, 8]:
            M = I().matrix(n)
            npt.assert_allclose(np.array(M), np.eye(n), atol=1e-15)

    def test_D_order0_is_eye(self):
        """D(order=0) returns identity."""
        M = D(order=0).matrix(6)
        npt.assert_allclose(np.array(M), np.eye(6), atol=1e-15)

    def test_matrix_int_disc(self):
        """matrix(int) works as shorthand for matrix(ChebColloc2Disc(n))."""
        op = D()
        M1 = op.matrix(8)
        M2 = op.matrix(ChebColloc2Disc(8))
        npt.assert_allclose(np.array(M1), np.array(M2), atol=1e-15)

    def test_D_repr(self):
        """OperatorBlock has a string representation."""
        s = repr(D())
        assert "OperatorBlock" in s


# ===========================================================================
# Tier 2 — OperatorBlock: mathematical correctness
# ===========================================================================


class TestOperatorBlockMath:
    """Mathematical correctness of differentiation matrices.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_D_sin_gives_cos(self):
        """D applied to sin(x) at Chebyshev points gives cos(x).

        Tests that the spectral differentiation matrix correctly
        differentiates sin -> cos on [-1, 1] to machine precision.
        """
        n = 16
        disc = ChebColloc2Disc(n)
        D_mat = D().matrix(disc)
        x = chebpts(n, kind=2)
        sin_vals = jnp.sin(x)
        computed_deriv = D_mat @ sin_vals
        expected = jnp.cos(x)
        npt.assert_allclose(
            np.array(computed_deriv), np.array(expected), atol=1e-12
        )

    def test_D2_sin_gives_minus_sin(self):
        """D^2 applied to sin(x) gives -sin(x)."""
        n = 20
        disc = ChebColloc2Disc(n)
        D2_mat = D(order=2).matrix(disc)
        x = chebpts(n, kind=2)
        sin_vals = jnp.sin(x)
        result = D2_mat @ sin_vals
        npt.assert_allclose(
            np.array(result), np.array(-sin_vals), atol=1e-11
        )

    def test_D_polynomial_exact(self):
        """D applied to x^3 gives 3x^2 exactly."""
        n = 8
        disc = ChebColloc2Disc(n)
        D_mat = D().matrix(disc)
        x = chebpts(n, kind=2)
        vals = x ** 3
        result = D_mat @ vals
        expected = 3.0 * x ** 2
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-12)

    def test_D_on_custom_domain(self):
        """D on [0, pi] applied to sin gives cos (with domain scaling)."""
        domain = (0.0, float(jnp.pi))
        n = 20
        disc = ChebColloc2Disc(n, domain)
        D_mat = D(domain=domain).matrix(disc)
        # Chebyshev-2 points on [0, pi]
        pts_ref = chebpts(n, kind=2)
        a, b = domain
        x = 0.5 * (b - a) * pts_ref + 0.5 * (a + b)
        sin_vals = jnp.sin(x)
        result = D_mat @ sin_vals
        expected = jnp.cos(x)
        npt.assert_allclose(
            np.array(result), np.array(expected), atol=1e-11
        )

    def test_D2_plus_I_matrix(self):
        """(D^2 + I) assembled matrix has expected eigenvalue structure.

        The operator L = d^2/dx^2 + 1 on [-1,1] with Dirichlet BCs
        has known eigenvalues.  Here we just verify the matrix is assembled
        correctly as the sum of the two component matrices.
        """
        n = 12
        D2 = D(order=2)
        Id = I()
        L = D2 + Id
        M_L = L.matrix(n)
        M_D2 = D2.matrix(n)
        M_I = Id.matrix(n)
        npt.assert_allclose(
            np.array(M_L), np.array(M_D2 + M_I), atol=1e-15
        )


# ===========================================================================
# Tier 3 — OperatorBlock algebra
# ===========================================================================


class TestOperatorBlockAlgebra:
    """OperatorBlock arithmetic operations.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_add_commutativity(self):
        """D + I == I + D (their matrices are equal)."""
        n = 8
        M1 = (D() + I()).matrix(n)
        M2 = (I() + D()).matrix(n)
        npt.assert_allclose(np.array(M1), np.array(M2), atol=1e-15)

    def test_sub_D_minus_I(self):
        """(D - I).matrix = D.matrix - I.matrix."""
        n = 8
        M = (D() - I()).matrix(n)
        expected = D().matrix(n) - I().matrix(n)
        npt.assert_allclose(np.array(M), np.array(expected), atol=1e-15)

    def test_scalar_multiply(self):
        """2 * D has matrix 2 * D.matrix."""
        n = 8
        M = (2.0 * D()).matrix(n)
        expected = 2.0 * D().matrix(n)
        npt.assert_allclose(np.array(M), np.array(expected), atol=1e-15)

    def test_scalar_multiply_right(self):
        """D * 3 has matrix 3 * D.matrix."""
        n = 8
        M = (D() * 3.0).matrix(n)
        expected = 3.0 * D().matrix(n)
        npt.assert_allclose(np.array(M), np.array(expected), atol=1e-15)

    def test_composition_D_D(self):
        """D * D == D^2 (as matrices)."""
        n = 10
        M1 = (D() * D()).matrix(n)
        M2 = D(order=2).matrix(n)
        npt.assert_allclose(np.array(M1), np.array(M2), atol=1e-12)

    def test_pow_2(self):
        """D ** 2 == D * D."""
        n = 10
        M1 = (D() ** 2).matrix(n)
        M2 = (D() * D()).matrix(n)
        npt.assert_allclose(np.array(M1), np.array(M2), atol=1e-14)

    def test_pow_0_is_eye(self):
        """D ** 0 == I."""
        n = 8
        M = (D() ** 0).matrix(n)
        npt.assert_allclose(np.array(M), np.eye(n), atol=1e-15)

    def test_neg(self):
        """(-D).matrix == -D.matrix."""
        n = 8
        M = (-D()).matrix(n)
        expected = -D().matrix(n)
        npt.assert_allclose(np.array(M), np.array(expected), atol=1e-15)

    def test_order_tracking_D(self):
        """D has order 1; D+I has order 1; D*D has order 2."""
        assert D().order == 1
        assert D(order=2).order == 2
        assert (D() + I()).order == 1
        assert (D() * D()).order == 2

    def test_domain_mismatch_raises(self):
        """Adding blocks with different domains raises ValueError."""
        d1 = (-1.0, 1.0)
        d2 = (0.0, 1.0)
        with pytest.raises(ValueError, match="different domains"):
            _ = D(d1) + D(d2)


# ===========================================================================
# Tier 4 — diag (multiplication operator)
# ===========================================================================


class TestDiagOperator:
    """Multiplication-by-f operator.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_diag_identity_function(self):
        """diag(1) == I (multiplication by constant 1)."""
        f = cj.chebfun(lambda x: jnp.ones_like(x))
        M_diag = diag(f).matrix(8)
        npt.assert_allclose(np.array(M_diag), np.eye(8), atol=1e-14)

    def test_diag_x(self):
        """diag(x) is diagonal with Chebyshev-2 points on the diagonal."""
        n = 8
        f = cj.chebfun(lambda x: x)
        M = diag(f).matrix(n)
        x_pts = chebpts(n, kind=2)
        expected = jnp.diag(x_pts)
        npt.assert_allclose(np.array(M), np.array(expected), atol=1e-14)

    def test_diag_shape(self):
        """diag(f).matrix(n) has shape (n, n)."""
        f = cj.chebfun(jnp.sin)
        M = diag(f).matrix(12)
        assert M.shape == (12, 12)

    def test_diag_callable_with_domain(self):
        """diag with a plain callable and explicit domain."""
        domain = (0.0, 1.0)
        M = diag(jnp.sin, domain=domain).matrix(ChebColloc2Disc(8, domain))
        assert M.shape == (8, 8)


# ===========================================================================
# Tier 5 — FunctionalBlock
# ===========================================================================


class TestFunctionalBlock:
    """FunctionalBlock: eval_at and sum_functional.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_eval_at_left_endpoint(self):
        """eval_at(-1) row picks out the left endpoint value."""
        n = 8
        E = eval_at(-1.0)
        row = E.matrix(n)
        assert row.shape == (n,)
        # For a Chebyshev-2 grid, the first point is x=-1.
        # The barycentric interpolation row should be exactly e_0.
        x = chebpts(n, kind=2)
        # x[0] = -1, so the row is [1, 0, ..., 0]
        vals = jnp.sin(x)
        result = float(jnp.dot(row, vals))
        expected = float(jnp.sin(jnp.array(-1.0)))
        npt.assert_allclose(result, expected, atol=1e-14)

    def test_eval_at_right_endpoint(self):
        """eval_at(1) picks out the right endpoint value."""
        n = 8
        E = eval_at(1.0)
        row = E.matrix(n)
        x = chebpts(n, kind=2)
        vals = jnp.sin(x)
        result = float(jnp.dot(row, vals))
        expected = float(jnp.sin(jnp.array(1.0)))
        npt.assert_allclose(result, expected, atol=1e-14)

    def test_eval_at_interior(self):
        """eval_at(0.5) interpolates correctly at an interior point."""
        n = 16
        E = eval_at(0.5)
        row = E.matrix(n)
        x = chebpts(n, kind=2)
        vals = jnp.sin(x)
        result = float(jnp.dot(row, vals))
        expected = float(jnp.sin(jnp.array(0.5)))
        npt.assert_allclose(result, expected, atol=1e-14)

    def test_eval_at_sin_pi_over_4(self):
        """eval_at(pi/4) gives sin(pi/4) accurately."""
        n = 20
        x0 = float(jnp.pi / 4)
        E = eval_at(x0, domain=(-jnp.pi, jnp.pi))
        disc = ChebColloc2Disc(n, domain=(-float(jnp.pi), float(jnp.pi)))
        row = E.matrix(disc)
        pts_ref = chebpts(n, kind=2)
        a, b = -float(jnp.pi), float(jnp.pi)
        x_phys = 0.5 * (b - a) * pts_ref + 0.5 * (a + b)
        vals = jnp.sin(x_phys)
        result = float(jnp.dot(row, vals))
        expected = float(jnp.sin(jnp.array(x0)))
        npt.assert_allclose(result, expected, atol=1e-13)

    def test_eval_at_outside_domain_raises(self):
        """eval_at raises ValueError for points outside the domain."""
        with pytest.raises(ValueError, match="outside domain"):
            eval_at(2.0, domain=(-1.0, 1.0))

    def test_sum_functional_integral_of_one(self):
        """sum_functional applied to f=1 gives integral = 2 on [-1,1]."""
        n = 8
        S = sum_functional()
        row = S.matrix(n)
        chebpts(n, kind=2)
        ones_vals = jnp.ones(n, dtype=jnp.float64)
        result = float(jnp.dot(row, ones_vals))
        npt.assert_allclose(result, 2.0, atol=1e-14)

    def test_sum_functional_sin(self):
        """sum_functional applied to sin(x) gives 0 on [-1,1]."""
        n = 16
        S = sum_functional()
        row = S.matrix(n)
        x = chebpts(n, kind=2)
        vals = jnp.sin(x)
        result = float(jnp.dot(row, vals))
        npt.assert_allclose(result, 0.0, atol=1e-14)

    def test_sum_functional_domain_scaling(self):
        """sum_functional on [0, 1] applied to f=1 gives 1."""
        n = 8
        S = sum_functional(domain=(0.0, 1.0))
        disc = ChebColloc2Disc(n, domain=(0.0, 1.0))
        row = S.matrix(disc)
        result = float(jnp.dot(row, jnp.ones(n, dtype=jnp.float64)))
        npt.assert_allclose(result, 1.0, atol=1e-14)

    def test_functional_repr(self):
        """FunctionalBlock has a string representation."""
        s = repr(eval_at(0.5))
        assert "FunctionalBlock" in s

    def test_functional_algebra_add(self):
        """(eval_at(-1) + eval_at(1)) row sums two evaluation rows."""
        n = 8
        E1 = eval_at(-1.0)
        E2 = eval_at(1.0)
        row_sum = (E1 + E2).matrix(n)
        expected = E1.matrix(n) + E2.matrix(n)
        npt.assert_allclose(np.array(row_sum), np.array(expected), atol=1e-15)

    def test_functional_algebra_neg(self):
        """(-eval_at(0)).matrix == -eval_at(0).matrix."""
        n = 8
        E = eval_at(0.0)
        npt.assert_allclose(
            np.array((-E).matrix(n)), np.array(-E.matrix(n)), atol=1e-15
        )

    def test_functional_scalar_mul(self):
        """3 * eval_at(0) == eval_at(0) scaled by 3."""
        n = 8
        E = eval_at(0.0)
        npt.assert_allclose(
            np.array((3.0 * E).matrix(n)), 3.0 * np.array(E.matrix(n)),
            atol=1e-15
        )

    def test_functional_compose_with_op(self):
        """eval_at(0) * D gives the derivative at 0 via row @ D_matrix."""
        n = 12
        E = eval_at(0.0)
        Dop = D()
        composed = E * Dop
        row = composed.matrix(n)
        assert row.shape == (n,)
        # Verify: D sin at x=0 is cos(0) = 1.
        x = chebpts(n, kind=2)
        sin_vals = jnp.sin(x)
        result = float(jnp.dot(row, sin_vals))
        npt.assert_allclose(result, 1.0, atol=1e-12)


# ===========================================================================
# Tier 6 — ChebMatrix assembly
# ===========================================================================


class TestChebMatrixAssembly:
    """ChebMatrix block assembly.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_op_only_matrix(self):
        """ChebMatrix with a single OperatorBlock gives (n, n) matrix."""
        n = 8
        cm = ChebMatrix([[D()]])
        A, rsz = cm.matrix(n)
        assert A.shape == (n, n)
        assert rsz == [n]

    def test_op_func_stacking(self):
        """ChebMatrix([OperatorBlock], [FunctionalBlock]) gives (n+1, n) matrix."""
        n = 8
        cm = ChebMatrix([[D()], [eval_at(0.0)]])
        A, rsz = cm.matrix(n)
        assert A.shape == (n + 1, n)
        assert rsz == [n, 1]

    def test_two_bc_rows(self):
        """Two BC rows give (n+2, n) matrix."""
        n = 8
        cm = ChebMatrix([[D(order=2)], [eval_at(-1.0)], [eval_at(1.0)]])
        A, rsz = cm.matrix(n)
        assert A.shape == (n + 2, n)
        assert rsz == [n, 1, 1]

    def test_bc_rows_match_blocks(self):
        """BC rows in the assembled matrix match individual functional rows."""
        n = 8
        E0 = eval_at(-1.0)
        E1 = eval_at(1.0)
        cm = ChebMatrix([[D(order=2)], [E0], [E1]])
        A, rsz = cm.matrix(n)
        # Last two rows of A should be the functional rows
        r0 = E0.matrix(n)
        r1 = E1.matrix(n)
        npt.assert_allclose(np.array(A[n, :]), np.array(r0), atol=1e-15)
        npt.assert_allclose(np.array(A[n + 1, :]), np.array(r1), atol=1e-15)

    def test_repr(self):
        """ChebMatrix repr contains class name."""
        cm = ChebMatrix([[D()]])
        assert "ChebMatrix" in repr(cm)

    def test_empty_blocks_raises(self):
        """Empty blocks list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            ChebMatrix([])

    def test_ragged_rows_raises(self):
        """Ragged block grid raises ValueError."""
        with pytest.raises(ValueError, match="same number of columns"):
            ChebMatrix([[D(), I()], [eval_at(0.0)]])


# ===========================================================================
# Tier 7 — BVP: u'' + u eigenvalues
# ===========================================================================


class TestEigenvalues:
    """Eigenvalue test for u'' + u on [-1, 1] with Dirichlet BCs.

    The operator L = d^2/dx^2 + I on [-1, 1] with u(-1) = u(1) = 0
    has known eigenvalues lambda_k = -(k*pi/(2))^2 + 1  (k = 1, 2, 3, ...).

    Here we assemble the matrix with BC rows replacing boundary rows
    and check that the computed eigenvalues are close to exact ones.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_D2_plus_I_eigenvalues(self):
        """Eigenvalues of D^2 + I with Dirichlet BCs match theory."""
        n = 30
        # Assemble operator matrix
        L = D(order=2) + I()
        M = L.matrix(n)
        # Replace first and last rows with boundary condition rows
        M = np.array(M)
        # BC: u(-1) = 0 -> left endpoint row; u(1) = 0 -> right endpoint row
        # Chebyshev-2 ordering: x[0] = -1, x[-1] = 1
        M[0, :] = 0.0
        M[0, 0] = 1.0    # u(-1) = 0
        M[-1, :] = 0.0
        M[-1, -1] = 1.0  # u(1) = 0

        # Eigenvalues of the discretized operator
        eigvals = np.linalg.eigvals(M)
        # Sort by real part; ignore boundary rows' eigenvalues (which are ~1)
        eigvals_real = np.sort(eigvals.real)

        # Exact eigenvalues (interior): lambda_k = -(k*pi/2)^2 + 1  for k=1,2,...
        # For the first few: k=1 -> -(pi/2)^2 + 1 ≈ -1.467
        #                    k=2 -> -(pi)^2 + 1 ≈ -8.870
        #                    k=3 -> -(3*pi/2)^2 + 1 ≈ -21.21
        exact_1 = -(np.pi / 2) ** 2 + 1.0
        exact_2 = -(np.pi) ** 2 + 1.0
        exact_3 = -(3 * np.pi / 2) ** 2 + 1.0

        # Find computed eigenvalues closest to exact ones
        def nearest(eigs, target):
            return eigs[np.argmin(np.abs(eigs - target))]

        npt.assert_allclose(nearest(eigvals_real, exact_1), exact_1, rtol=1e-6)
        npt.assert_allclose(nearest(eigvals_real, exact_2), exact_2, rtol=1e-5)
        npt.assert_allclose(nearest(eigvals_real, exact_3), exact_3, rtol=1e-4)


# ===========================================================================
# Tier 8 — BVP solve: L*u = f with BCs
# ===========================================================================


class TestBVPSolve:
    """Solve a BVP using ChebMatrix.solve.

    BVP: u''(x) = -sin(x),  u(-1) = sin(1),  u(1) = -sin(1)
    Exact solution: u(x) = sin(x).  (Check: u'' = -sin(x) = f(x), and
    u(-1) = sin(-1) = -sin(1), u(1) = sin(1).)

    Wait — let us pick a simpler verifiable problem:
    u'' = -pi^2/4 * u,  u(-1) = 0,  u(1) = 0
    with u(x) = sin(pi*(x+1)/2).

    Check: u'' = -pi^2/4 * sin(pi*(x+1)/2) = -pi^2/4 * u.  Correct.
    u(-1) = sin(0) = 0.  u(1) = sin(pi) = 0.  Correct.

    JAX contract: jit=no, vmap=no, grad=no
    """

    def test_bvp_poisson(self):
        """Solve u'' = f, u(-1) = u(1) = 0 with f = -pi^2/4 * sin(pi*(x+1)/2).

        Exact solution: u(x) = sin(pi*(x+1)/2).
        """
        n = 24
        # Collocation points
        x = np.array(chebpts(n, kind=2))

        # RHS: f(x) = u''(x) = -(pi/2)^2 * sin(pi*(x+1)/2)
        exact_u = np.sin(np.pi * (x + 1.0) / 2.0)
        f_vals = -(np.pi / 2.0) ** 2 * exact_u

        # Assemble L = D^2 + (pi^2/4)*I but we want u'' = f, so L = D^2
        # with BCs u(-1) = 0, u(1) = 0.
        cm = ChebMatrix([[D(order=2)], [eval_at(-1.0)], [eval_at(1.0)]])
        u_vals = cm.solve(
            rhs=jnp.array(f_vals, dtype=jnp.float64),
            n=n,
            bc_values=[0.0, 0.0],
            bc_row_indices=[0, n - 1],
        )
        npt.assert_allclose(
            np.array(u_vals), exact_u, atol=1e-10
        )

    def test_bvp_simple_identity(self):
        """Solve I*u = f (trivial identity BVP).

        u = f everywhere; BCs are set to the exact values.
        """
        n = 10
        x = np.array(chebpts(n, kind=2))
        exact_u = np.sin(x)
        f_vals = np.sin(x)

        cm = ChebMatrix([[I()], [eval_at(-1.0)], [eval_at(1.0)]])
        u_vals = cm.solve(
            rhs=jnp.array(f_vals, dtype=jnp.float64),
            n=n,
            bc_values=[float(np.sin(-1.0)), float(np.sin(1.0))],
            bc_row_indices=[0, n - 1],
        )
        npt.assert_allclose(
            np.array(u_vals), exact_u, atol=1e-13
        )

    def test_bvp_second_order_nontrivial(self):
        """Solve u'' + u = g,  u(-1) = a,  u(1) = b.

        Exact solution: u(x) = sin(x).
        g(x) = u'' + u = -sin(x) + sin(x) = 0.
        BCs: u(-1) = sin(-1),  u(1) = sin(1).
        """
        n = 24
        x = np.array(chebpts(n, kind=2))
        exact_u = np.sin(x)
        g_vals = np.zeros(n)  # g = u'' + u = 0 for u = sin(x)

        L = D(order=2) + I()
        cm = ChebMatrix([[L], [eval_at(-1.0)], [eval_at(1.0)]])
        u_vals = cm.solve(
            rhs=jnp.array(g_vals, dtype=jnp.float64),
            n=n,
            bc_values=[float(np.sin(-1.0)), float(np.sin(1.0))],
            bc_row_indices=[0, n - 1],
        )
        npt.assert_allclose(
            np.array(u_vals), exact_u, atol=1e-10
        )
