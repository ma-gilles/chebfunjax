"""Tests for chebfunjax.discretization.chebcolloc — ChebColloc2 and ChebColloc1.

JAX contract:
  - diffmat, cumsummat, points, weights, eval_matrix: jit=yes (n must be static),
    vmap=no, grad=no.
  - These are matrix-building utilities; they are called outside JIT for BVP assembly.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.discretization.chebcolloc import ChebColloc1, ChebColloc2
from chebfunjax.domain import Domain
from chebfunjax.utils.quadrature import chebpts

# ===========================================================================
# Tier 1: Mathematical identity tests (no MATLAB needed)
# ===========================================================================


class TestChebColloc2Init:
    """Constructor and repr tests for ChebColloc2.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    def test_default_domain(self):
        """Default domain is [-1, 1]."""
        disc = ChebColloc2(n=5)
        assert disc.domain == Domain((-1.0, 1.0))
        assert disc.n == 5

    def test_tuple_domain(self):
        """Tuple domain is wrapped in Domain."""
        disc = ChebColloc2(n=5, domain=(0.0, 2.0))
        assert disc.domain.a == 0.0
        assert disc.domain.b == 2.0

    def test_domain_object(self):
        """Domain object stored correctly."""
        d = Domain((-1.0, 1.0))
        disc = ChebColloc2(n=10, domain=d)
        assert disc.domain == d

    def test_invalid_n(self):
        """n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n >= 1"):
            ChebColloc2(n=0)

    def test_repr(self):
        """repr includes n and domain."""
        disc = ChebColloc2(n=5)
        r = repr(disc)
        assert "ChebColloc2" in r
        assert "n=5" in r


class TestChebColloc2Points:
    """Points and weights are correctly mapped to the domain.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    def test_reference_domain_matches_chebpts(self):
        """On [-1, 1], points() == chebpts(n, kind=2)."""
        for n in [5, 10, 17]:
            disc = ChebColloc2(n=n)
            npt.assert_allclose(
                np.array(disc.points()),
                np.array(chebpts(n, kind=2)),
                rtol=1e-15,
            )

    def test_endpoints_on_reference_domain(self):
        """On [-1, 1], first point is -1 and last is 1."""
        disc = ChebColloc2(n=5)
        x = disc.points()
        npt.assert_allclose(float(x[0]), -1.0, atol=1e-15)
        npt.assert_allclose(float(x[-1]), 1.0, atol=1e-15)

    def test_custom_domain(self):
        """Points are affinely mapped to [a, b]."""
        disc = ChebColloc2(n=5, domain=(0.0, 2.0))
        x = disc.points()
        assert float(x[0]) == pytest.approx(0.0, abs=1e-15)
        assert float(x[-1]) == pytest.approx(2.0, abs=1e-15)
        # Interior points lie strictly between 0 and 2
        assert bool(jnp.all((x[1:-1] > 0.0) & (x[1:-1] < 2.0)))

    def test_weights_sum_to_domain_length(self):
        """Sum of quadrature weights equals domain length."""
        for a, b in [(-1.0, 1.0), (0.0, 2.0), (-3.0, 5.0)]:
            disc = ChebColloc2(n=10, domain=(a, b))
            w = disc.weights()
            npt.assert_allclose(float(jnp.sum(w)), b - a, rtol=1e-14)

    def test_weights_integrate_constant(self):
        """∫_a^b 1 dx = b - a via quadrature."""
        for a, b in [(-1.0, 1.0), (0.0, 3.0)]:
            disc = ChebColloc2(n=5, domain=(a, b))
            w = disc.weights()
            x = disc.points()
            ones = jnp.ones_like(x)
            npt.assert_allclose(float(jnp.dot(w, ones)), b - a, rtol=1e-14)

    def test_weights_integrate_polynomial(self):
        """∫_{-1}^{1} x^2 dx = 2/3 via Clenshaw-Curtis."""
        disc = ChebColloc2(n=10)
        w = disc.weights()
        x = disc.points()
        npt.assert_allclose(float(jnp.dot(w, x**2)), 2.0 / 3.0, rtol=1e-14)

    def test_equation_points_are_kind1(self):
        """Equation points are 1st-kind Chebyshev points."""
        from chebfunjax.utils.quadrature import chebpts as _chebpts

        disc = ChebColloc2(n=5)
        eq_pts = disc.equation_points()
        ref = _chebpts(5, kind=1)
        npt.assert_allclose(np.array(eq_pts), np.array(ref), rtol=1e-15)

    def test_n1_weights(self):
        """n=1: single weight = domain length."""
        disc = ChebColloc2(n=1, domain=(0.0, 3.0))
        npt.assert_allclose(float(disc.weights()[0]), 3.0, rtol=1e-14)


class TestChebColloc2Diffmat:
    """Differentiation matrix correctness.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    def test_shape(self):
        """D has shape (n, n)."""
        for n in [5, 10]:
            disc = ChebColloc2(n=n)
            D = disc.diffmat()
            assert D.shape == (n, n)

    def test_d_ones_is_zero(self):
        """D @ ones = 0 (derivative of constant is zero)."""
        for n in [5, 10, 20]:
            disc = ChebColloc2(n=n)
            D = disc.diffmat()
            result = D @ jnp.ones(n, dtype=jnp.float64)
            npt.assert_allclose(np.array(result), 0.0, atol=5e-13)

    def test_d_x_is_one(self):
        """D @ x_vals = 1 (derivative of x)."""
        for n in [5, 10, 20]:
            disc = ChebColloc2(n=n)
            D = disc.diffmat()
            x = disc.points()
            npt.assert_allclose(np.array(D @ x), 1.0, atol=5e-14)

    @pytest.mark.parametrize("n,deg", [(5, 2), (10, 3), (10, 4)])
    def test_d_monomial(self, n, deg):
        """D @ x^deg = deg * x^(deg-1)."""
        disc = ChebColloc2(n=n)
        D = disc.diffmat()
        x = disc.points()
        npt.assert_allclose(
            np.array(D @ x**deg),
            np.array(deg * x ** (deg - 1)),
            rtol=1e-11,
            atol=1e-12,
        )

    def test_d2_matches_d_squared(self):
        """D(k=2) ≈ D(k=1) @ D(k=1) (up to rounding)."""
        disc = ChebColloc2(n=10)
        D1 = disc.diffmat(k=1)
        D2 = disc.diffmat(k=2)
        npt.assert_allclose(np.array(D2), np.array(D1 @ D1), rtol=1e-10, atol=1e-11)

    def test_d_identity_for_k0(self):
        """D(k=0) is the identity matrix."""
        disc = ChebColloc2(n=7)
        D = disc.diffmat(k=0)
        npt.assert_allclose(np.array(D), np.eye(7), atol=1e-15)

    def test_domain_scaling(self):
        """Derivative matrix on [0, 2] equals that on [-1, 1] (same length).

        Both domains have b - a = 2, so the chain-rule scale factor
        (2 / (b - a))^1 = 1 is the same, meaning D is identical.
        The matrix only differs when the domain length differs.
        """
        D_ref = ChebColloc2(n=10).diffmat()
        D_scaled = ChebColloc2(n=10, domain=(0.0, 2.0)).diffmat()
        npt.assert_allclose(np.array(D_scaled), np.array(D_ref), rtol=1e-14)

    def test_domain_scaling_half_length(self):
        """Derivative matrix on [0, 1] is 2× that on [-1, 1] (half length).

        b - a = 1 on [0, 1] vs. b - a = 2 on [-1, 1].
        Chain rule: d/dx = (2/(b-a)) d/dref, so D_physical = 2 * D_ref.
        """
        D_ref = ChebColloc2(n=10).diffmat()
        D_half = ChebColloc2(n=10, domain=(0.0, 1.0)).diffmat()
        npt.assert_allclose(np.array(D_half), 2.0 * np.array(D_ref), rtol=1e-14)

    def test_jit(self):
        """diffmat is JIT-compilable with static n."""
        disc = ChebColloc2(n=8)
        jitted = jax.jit(functools.partial(disc.diffmat, 1))
        npt.assert_allclose(
            np.array(jitted()), np.array(disc.diffmat()), rtol=1e-15
        )


class TestChebColloc2Cumsummat:
    """Integration (cumsummat) correctness.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    def test_shape(self):
        """Q has shape (n, n)."""
        disc = ChebColloc2(n=8)
        Q = disc.cumsummat()
        assert Q.shape == (8, 8)

    def test_first_row_is_zero(self):
        """Q[0, :] = 0 (antiderivative is zero at left endpoint)."""
        for n in [5, 10, 17]:
            disc = ChebColloc2(n=n)
            Q = disc.cumsummat()
            npt.assert_allclose(np.array(Q[0, :]), 0.0, atol=1e-14)

    def test_cumsum_of_one_is_x_plus_1(self):
        """∫_{-1}^{x} 1 dt = x + 1."""
        disc = ChebColloc2(n=10)
        Q = disc.cumsummat()
        x = disc.points()
        ones = jnp.ones(10, dtype=jnp.float64)
        antideriv = Q @ ones
        expected = x + 1.0  # integral of 1 from -1 to x
        npt.assert_allclose(np.array(antideriv), np.array(expected), rtol=1e-13)

    def test_cumsummat_antiderivative_recoverable(self):
        """d/dx of (antiderivative of f) = f (fundamental theorem).

        Q gives values of the antiderivative F with F(-1) = 0.
        Then D @ Q @ f should give back f (since d/dx F = f).
        We check this on polynomials where this holds exactly.
        """
        n = 10
        disc = ChebColloc2(n=n)
        D = disc.diffmat()
        Q = disc.cumsummat()
        x = disc.points()
        # Test with f(x) = x^2 - 1/3 (chosen so integral starts at 0)
        # F(x) = x^3/3 - x/3, F(-1) = -1/3 + 1/3 = 0. Good.
        f_vals = x**2 - 1.0 / 3.0
        F_vals = Q @ f_vals          # antiderivative of f
        recovered = D @ F_vals       # differentiate antiderivative
        npt.assert_allclose(np.array(recovered), np.array(f_vals), rtol=1e-11, atol=1e-12)

    def test_domain_scaling(self):
        """cumsummat on [0, 2] = cumsummat on [-1, 1] (same length, scale = 1)."""
        Q_ref = ChebColloc2(n=8).cumsummat()
        Q_scaled = ChebColloc2(n=8, domain=(0.0, 2.0)).cumsummat()
        npt.assert_allclose(np.array(Q_scaled), np.array(Q_ref), rtol=1e-14)

    def test_domain_scaling_half_length(self):
        """cumsummat on [0, 1] = 0.5 * cumsummat on [-1, 1] (half length)."""
        Q_ref = ChebColloc2(n=8).cumsummat()
        Q_half = ChebColloc2(n=8, domain=(0.0, 1.0)).cumsummat()
        npt.assert_allclose(np.array(Q_half), 0.5 * np.array(Q_ref), rtol=1e-14)


class TestChebColloc2EvalMatrix:
    """Evaluation matrix (barycentric interpolation).

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    def test_shape(self):
        """E has shape (len(y), n)."""
        disc = ChebColloc2(n=8)
        y = jnp.linspace(-1.0, 1.0, 5)
        E = disc.eval_matrix(y)
        assert E.shape == (5, 8)

    def test_grid_points_are_identity_rows(self):
        """E evaluated at grid points x_i gives identity matrix."""
        disc = ChebColloc2(n=8)
        x = disc.points()
        E = disc.eval_matrix(x)
        npt.assert_allclose(np.array(E), np.eye(8), atol=1e-14)

    def test_interpolates_polynomial(self):
        """E @ p(x) reproduces p(y) for polynomials of degree < n."""
        n = 12
        disc = ChebColloc2(n=n)
        x = disc.points()
        y = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        E = disc.eval_matrix(y)
        # Test with p(x) = x^3
        p_x = x**3
        result = E @ p_x
        expected = y**3
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-12)

    def test_scalar_input(self):
        """eval_matrix accepts a scalar y."""
        disc = ChebColloc2(n=5)
        E = disc.eval_matrix(0.0)
        assert E.shape == (1, 5)
        # Should reproduce f(0) = 0 for f(x) = x
        x = disc.points()
        result = E @ x  # shape (1,)
        npt.assert_allclose(float(result[0]), 0.0, atol=1e-14)

    def test_custom_domain(self):
        """eval_matrix works on [0, 2] domain."""
        disc = ChebColloc2(n=8, domain=(0.0, 2.0))
        x = disc.points()
        y = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        E = disc.eval_matrix(y)
        # Interpolate f(x) = x, should give y back
        result = E @ x
        npt.assert_allclose(np.array(result), np.array(y), atol=1e-12)


class TestChebColloc2BVP:
    """BVP solution: u'' = -1, u(-1) = u(1) = 0, exact u = (1-x²)/2.

    This is the primary integration test — it validates that diffmat,
    points, and boundary condition imposition all work together correctly.

    JAX contract: jit=no (system assembly uses Python indexing), grad=no.
    """

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_bvp_quadratic(self, n):
        """u'' = -1 with zero BCs has exact solution u = (1-x²)/2."""
        disc = ChebColloc2(n=n)
        D2 = disc.diffmat(k=2)
        x = disc.points()

        # Assemble: replace first and last rows with boundary conditions
        A = D2
        A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
        A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)

        rhs = jnp.full(n, -1.0, dtype=jnp.float64)
        rhs = rhs.at[0].set(0.0).at[-1].set(0.0)

        u = jnp.linalg.solve(A, rhs)
        exact = (1.0 - x**2) / 2.0

        npt.assert_allclose(np.array(u), np.array(exact), atol=1e-12)

    def test_bvp_custom_domain(self):
        """u'' = -1 on [0, 1], u(0)=u(1)=0 → u = x(1-x)/2."""
        n = 15
        disc = ChebColloc2(n=n, domain=(0.0, 1.0))
        D2 = disc.diffmat(k=2)
        x = disc.points()

        A = D2
        A = A.at[0, :].set(0.0).at[0, 0].set(1.0)
        A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)

        rhs = jnp.full(n, -1.0, dtype=jnp.float64)
        rhs = rhs.at[0].set(0.0).at[-1].set(0.0)

        u = jnp.linalg.solve(A, rhs)
        exact = x * (1.0 - x) / 2.0

        npt.assert_allclose(np.array(u), np.array(exact), atol=1e-11)

    def test_bvp_second_order_neumann(self):
        """u'' = 2, u'(-1) = -2, u'(1) = 2 → u = x^2 + C.

        Tests Neumann BCs using the first row of D1 as boundary operators.
        The solution u = x^2 is the only function satisfying u'' = 2
        with u'(±1) = ±2, up to a constant (fixed by adding u(0) = 0).
        """
        n = 12
        disc = ChebColloc2(n=n)
        D1 = disc.diffmat(k=1)
        D2 = disc.diffmat(k=2)
        x = disc.points()

        # BVP: u'' = 2, u'(-1) = -2, u'(1) = 2
        # Exact solution: u = x^2 (zero constant chosen)
        # Row 0: u'(-1) = D1[0, :] @ u = -2
        # Row -1: u'(1) = D1[-1, :] @ u = 2
        # Remaining rows: D2[i, :] @ u = 2
        A = D2.at[0, :].set(D1[0, :]).at[-1, :].set(D1[-1, :])
        rhs = jnp.full(n, 2.0, dtype=jnp.float64).at[0].set(-2.0).at[-1].set(2.0)

        # System is rank-deficient (u + const is also a solution).
        # Fix by adding constraint: u(0) = 0 in one interior row.
        # Find index of x closest to 0
        mid = int(jnp.argmin(jnp.abs(x)))
        A = A.at[mid, :].set(0.0).at[mid, mid].set(1.0)
        rhs = rhs.at[mid].set(0.0)

        u = jnp.linalg.solve(A, rhs)
        exact = x**2 - float(x[mid])**2  # u(x_mid) = 0 by constraint
        npt.assert_allclose(np.array(u), np.array(exact), atol=1e-10)


# ===========================================================================
# ChebColloc1 tests
# ===========================================================================


class TestChebColloc1:
    """Basic correctness tests for ChebColloc1.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    def test_repr(self):
        """repr includes class name and n."""
        disc = ChebColloc1(n=5)
        assert "ChebColloc1" in repr(disc)
        assert "n=5" in repr(disc)

    def test_invalid_n(self):
        """n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n >= 1"):
            ChebColloc1(n=0)

    def test_reference_domain_matches_chebpts1(self):
        """On [-1, 1], points() == chebpts(n, kind=1)."""
        for n in [5, 10, 17]:
            disc = ChebColloc1(n=n)
            npt.assert_allclose(
                np.array(disc.points()),
                np.array(chebpts(n, kind=1)),
                rtol=1e-15,
            )

    def test_no_boundary_nodes(self):
        """1st-kind points do not include ±1."""
        disc = ChebColloc1(n=10)
        x = disc.points()
        assert bool(jnp.all(jnp.abs(x) < 1.0))

    def test_equation_equals_function_points(self):
        """equation_points() == points() for kind=1."""
        disc = ChebColloc1(n=7)
        npt.assert_allclose(
            np.array(disc.equation_points()),
            np.array(disc.points()),
            rtol=1e-15,
        )

    def test_weights_sum_to_domain_length(self):
        """Sum of quadrature weights equals domain length."""
        for a, b in [(-1.0, 1.0), (0.0, 4.0)]:
            disc = ChebColloc1(n=10, domain=(a, b))
            w = disc.weights()
            npt.assert_allclose(float(jnp.sum(w)), b - a, rtol=1e-14)

    def test_d_ones_is_zero(self):
        """D @ ones = 0 (derivative of constant is zero)."""
        for n in [5, 10, 20]:
            disc = ChebColloc1(n=n)
            D = disc.diffmat()
            npt.assert_allclose(
                np.array(D @ jnp.ones(n, dtype=jnp.float64)), 0.0, atol=5e-13
            )

    def test_d_x_is_one(self):
        """D @ x_vals = 1."""
        for n in [5, 10, 20]:
            disc = ChebColloc1(n=n)
            D = disc.diffmat()
            x = disc.points()
            npt.assert_allclose(np.array(D @ x), 1.0, atol=5e-14)

    def test_cumsum_of_one_is_x_plus_1(self):
        """cumsummat(kind=1) @ 1 = x_k + 1 (antiderivative from -1)."""
        disc = ChebColloc1(n=8)
        Q = disc.cumsummat()
        x = disc.points()
        ones = jnp.ones(8, dtype=jnp.float64)
        antideriv = Q @ ones
        # Q @ 1 = x + 1 (integral of 1 from -1 to x_k)
        npt.assert_allclose(np.array(antideriv), np.array(x + 1.0), rtol=1e-12)

    def test_eval_matrix_shape(self):
        """eval_matrix shape is (len(y), n)."""
        disc = ChebColloc1(n=7)
        y = jnp.linspace(-0.9, 0.9, 10)
        E = disc.eval_matrix(y)
        assert E.shape == (10, 7)

    def test_eval_matrix_interpolates(self):
        """E @ p(x) reproduces p(y) for polynomials of degree < n."""
        n = 12
        disc = ChebColloc1(n=n)
        x = disc.points()
        y = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        E = disc.eval_matrix(y)
        p_x = x**2
        result = E @ p_x
        expected = y**2
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-12)


# ===========================================================================
# Tier 2: MATLAB cross-validation
# ===========================================================================


@pytest.fixture
def matlab_chebcolloc():
    """Load MATLAB golden references for chebcolloc module."""
    from tests.conftest import load_matlab_ref
    return load_matlab_ref("chebcolloc.mat")


@pytest.mark.matlab
class TestChebColloc2VsMatlab:
    """ChebColloc2 vs. MATLAB chebcolloc2 golden references.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    @pytest.mark.parametrize("n", [5, 10, 17, 20, 32])
    def test_diffmat_k1(self, n, matlab_chebcolloc):
        """diffmat(k=1) matches MATLAB chebcolloc2.diffmat(n)."""
        disc = ChebColloc2(n=n)
        D = disc.diffmat(k=1)
        ref = matlab_chebcolloc[f"diffmat2_n{n}"]
        npt.assert_allclose(np.array(D), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [5, 10, 17, 20, 32])
    def test_diffmat_k2(self, n, matlab_chebcolloc):
        """diffmat(k=2) matches MATLAB chebcolloc2.diffmat(n, 2)."""
        disc = ChebColloc2(n=n)
        D2 = disc.diffmat(k=2)
        ref = matlab_chebcolloc[f"diffmat2_n{n}_k2"]
        npt.assert_allclose(np.array(D2), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [5, 10, 17, 20])
    def test_cumsummat(self, n, matlab_chebcolloc):
        """cumsummat() matches MATLAB chebcolloc2.cumsummat(n)."""
        disc = ChebColloc2(n=n)
        Q = disc.cumsummat()
        ref = matlab_chebcolloc[f"cumsummat2_n{n}"]
        npt.assert_allclose(np.array(Q), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [5, 10, 17])
    def test_points(self, n, matlab_chebcolloc):
        """points() matches MATLAB chebtech2.chebpts(n)."""
        disc = ChebColloc2(n=n)
        x = disc.points()
        ref = matlab_chebcolloc[f"pts2_n{n}"]
        npt.assert_allclose(np.array(x), ref, rtol=1e-15, atol=1e-15)

    @pytest.mark.parametrize("n", [5, 10, 17])
    def test_weights(self, n, matlab_chebcolloc):
        """weights() matches MATLAB chebtech2 quadrature weights."""
        disc = ChebColloc2(n=n)
        w = disc.weights()
        ref = matlab_chebcolloc[f"wts2_n{n}"]
        npt.assert_allclose(np.array(w), ref, rtol=1e-14, atol=1e-15)

    def test_eval_matrix(self, matlab_chebcolloc):
        """eval_matrix matches MATLAB barymat reference."""
        y = jnp.array(matlab_chebcolloc["evalmat2_y15"], dtype=jnp.float64)
        disc = ChebColloc2(n=10)
        E = disc.eval_matrix(y)
        ref = matlab_chebcolloc["evalmat2_n10_y15"]
        npt.assert_allclose(np.array(E), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_bvp_matches_matlab(self, n, matlab_chebcolloc):
        """BVP solution u'' = -1, u(±1)=0 matches MATLAB reference."""
        disc = ChebColloc2(n=n)
        D2 = disc.diffmat(k=2)
        A = D2.at[0, :].set(0.0).at[0, 0].set(1.0)
        A = A.at[-1, :].set(0.0).at[-1, -1].set(1.0)
        rhs = jnp.full(n, -1.0).at[0].set(0.0).at[-1].set(0.0)
        u = jnp.linalg.solve(A, rhs)
        ref = matlab_chebcolloc[f"bvp_u_n{n}"]
        npt.assert_allclose(np.array(u), ref, rtol=1e-12, atol=1e-14)


@pytest.mark.matlab
class TestChebColloc1VsMatlab:
    """ChebColloc1 vs. MATLAB chebcolloc1 golden references.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    @pytest.mark.parametrize("n", [5, 10, 17, 20])
    def test_diffmat_k1(self, n, matlab_chebcolloc):
        """diffmat(k=1) matches MATLAB chebcolloc1.diffmat(n)."""
        disc = ChebColloc1(n=n)
        D = disc.diffmat(k=1)
        ref = matlab_chebcolloc[f"diffmat1_n{n}"]
        npt.assert_allclose(np.array(D), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [5, 10, 17, 20])
    def test_diffmat_k2(self, n, matlab_chebcolloc):
        """diffmat(k=2) matches MATLAB chebcolloc1.diffmat(n, 2)."""
        disc = ChebColloc1(n=n)
        D2 = disc.diffmat(k=2)
        ref = matlab_chebcolloc[f"diffmat1_n{n}_k2"]
        npt.assert_allclose(np.array(D2), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [5, 10, 17, 20])
    def test_cumsummat(self, n, matlab_chebcolloc):
        """cumsummat() matches MATLAB chebcolloc1.cumsummat(n)."""
        disc = ChebColloc1(n=n)
        Q = disc.cumsummat()
        ref = matlab_chebcolloc[f"cumsummat1_n{n}"]
        npt.assert_allclose(np.array(Q), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [5, 10, 17])
    def test_points(self, n, matlab_chebcolloc):
        """points() matches MATLAB chebtech1.chebpts(n)."""
        disc = ChebColloc1(n=n)
        x = disc.points()
        ref = matlab_chebcolloc[f"pts1_n{n}"]
        npt.assert_allclose(np.array(x), ref, rtol=1e-15, atol=1e-15)

    def test_eval_matrix(self, matlab_chebcolloc):
        """eval_matrix matches MATLAB barymat reference."""
        y = jnp.array(matlab_chebcolloc["evalmat2_y15"], dtype=jnp.float64)
        disc = ChebColloc1(n=10)
        E = disc.eval_matrix(y)
        ref = matlab_chebcolloc["evalmat1_n10_y15"]
        npt.assert_allclose(np.array(E), ref, rtol=1e-12, atol=1e-14)
