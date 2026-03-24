"""Tests for chebfunjax.utils.diffmat — spectral differentiation and integration matrices.

JAX contract: jit=yes (n, p, kind must be static), vmap=no, grad=no.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.diffmat import cumsummat, diffmat, diffrow, intmat, introw
from chebfunjax.utils.quadrature import chebpts, chebpts_ab, chebweights

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def matlab_diffmat():
    """Load MATLAB golden references for diffmat module."""
    from tests.conftest import load_matlab_ref

    return load_matlab_ref("diffmat.mat")


# ===========================================================================
# Tier 1: Mathematical identity tests (no MATLAB needed)
# ===========================================================================


class TestDiffmatPolynomialExactness:
    """D @ polynomial_values = derivative_values, exact for deg < n.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no.
    """

    def test_d_ones_is_zero(self):
        """D @ ones = zeros: constant functions have zero derivative."""
        for n in [3, 5, 10, 20]:
            D = diffmat(n)
            ones = jnp.ones(n, dtype=jnp.float64)
            npt.assert_allclose(np.array(D @ ones), 0.0, atol=5e-13)

    def test_d_x_is_one(self):
        """D @ x_vals = 1: derivative of x is 1."""
        for n in [3, 5, 10, 20]:
            D = diffmat(n)
            x = chebpts(n, kind=2)
            result = D @ x
            npt.assert_allclose(np.array(result), 1.0, atol=5e-14)

    @pytest.mark.parametrize("n", [5, 10, 20])
    @pytest.mark.parametrize("deg", [2, 3, 4])
    def test_d_monomial(self, n, deg):
        """D @ x^deg = deg * x^(deg-1), exact for deg < n."""
        if deg >= n:
            pytest.skip("Degree >= n, not exact")
        D = diffmat(n)
        x = chebpts(n, kind=2)
        vals = x**deg
        result = D @ vals
        expected = deg * x ** (deg - 1)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-12)

    @pytest.mark.parametrize("n", [10, 20])
    @pytest.mark.parametrize("p", [2, 3])
    def test_higher_order_monomial(self, n, p):
        """D^p @ x^(p+2): exact for polynomial derivatives."""
        D = diffmat(n, p=p)
        x = chebpts(n, kind=2)
        deg = p + 2
        vals = x**deg
        result = D @ vals
        # p-th derivative of x^deg = deg!/(deg-p)! * x^(deg-p)
        coeff = 1.0
        for i in range(p):
            coeff *= (deg - i)
        expected = coeff * x ** (deg - p)
        # Higher-order differentiation is ill-conditioned; relax tolerance
        npt.assert_allclose(np.array(result), np.array(expected), rtol=1e-10)

    def test_p0_is_identity(self):
        """diffmat(n, p=0) returns the identity matrix."""
        for n in [3, 5, 10]:
            D = diffmat(n, p=0)
            npt.assert_allclose(np.array(D), np.eye(n), atol=1e-15)


class TestDiffmatKind1PolynomialExactness:
    """Same polynomial tests for 1st-kind Chebyshev points."""

    def test_d_ones_is_zero_kind1(self):
        for n in [3, 5, 10, 20]:
            D = diffmat(n, kind=1)
            ones = jnp.ones(n, dtype=jnp.float64)
            npt.assert_allclose(np.array(D @ ones), 0.0, atol=5e-13)

    def test_d_x_is_one_kind1(self):
        for n in [3, 5, 10, 20]:
            D = diffmat(n, kind=1)
            x = chebpts(n, kind=1)
            result = D @ x
            npt.assert_allclose(np.array(result), 1.0, atol=5e-14)

    @pytest.mark.parametrize("n", [5, 10, 20])
    @pytest.mark.parametrize("deg", [2, 3, 4])
    def test_d_monomial_kind1(self, n, deg):
        if deg >= n:
            pytest.skip("Degree >= n, not exact")
        D = diffmat(n, kind=1)
        x = chebpts(n, kind=1)
        vals = x**deg
        result = D @ vals
        expected = deg * x ** (deg - 1)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-12)


class TestDiffmatDomainScaling:
    """Tests for domain scaling (non-standard intervals)."""

    def test_scaled_domain_derivative(self):
        """D @ x on [0, 2] should give constant 1."""
        n = 10
        D = diffmat(n, domain=(0.0, 2.0))
        x_ab = chebpts_ab(n, 0.0, 2.0, kind=2)
        result = D @ x_ab
        npt.assert_allclose(np.array(result), 1.0, atol=5e-14)

    def test_scaled_domain_x2(self):
        """D @ x^2 on [0, 2] should give 2*x."""
        n = 10
        D = diffmat(n, domain=(0.0, 2.0))
        x_ab = chebpts_ab(n, 0.0, 2.0, kind=2)
        result = D @ x_ab**2
        expected = 2.0 * x_ab
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-13)


class TestCumsummatPolynomialExactness:
    """Q @ f_values = antiderivative values with F(-1) = 0."""

    def test_cumsum_of_constant(self):
        """Integral of 1 from -1 to x = x + 1."""
        for n in [3, 5, 10, 20]:
            Q = cumsummat(n)
            x = chebpts(n, kind=2)
            result = Q @ jnp.ones(n, dtype=jnp.float64)
            expected = x + 1.0
            npt.assert_allclose(np.array(result), np.array(expected), atol=5e-14)

    def test_cumsum_of_x(self):
        """Integral of x from -1 to x = (x^2 - 1)/2."""
        for n in [3, 5, 10, 20]:
            Q = cumsummat(n)
            x = chebpts(n, kind=2)
            result = Q @ x
            expected = (x**2 - 1.0) / 2.0
            npt.assert_allclose(np.array(result), np.array(expected), atol=5e-14)

    @pytest.mark.parametrize("n", [5, 10, 20])
    @pytest.mark.parametrize("deg", [2, 3, 4])
    def test_cumsum_of_monomial(self, n, deg):
        """Integral of x^deg from -1 to x = (x^(deg+1) - (-1)^(deg+1))/(deg+1)."""
        if deg + 1 >= n:
            pytest.skip("Result degree >= n, not exact")
        Q = cumsummat(n)
        x = chebpts(n, kind=2)
        vals = x**deg
        result = Q @ vals
        expected = (x ** (deg + 1) - (-1.0) ** (deg + 1)) / (deg + 1)
        npt.assert_allclose(np.array(result), np.array(expected), atol=5e-13)

    def test_cumsum_first_row_is_zero(self):
        """The antiderivative value at x=-1 is always 0 (for kind=2)."""
        for n in [3, 5, 10]:
            Q = cumsummat(n)
            # First row should be all zeros
            npt.assert_allclose(np.array(Q[0, :]), 0.0, atol=1e-15)

    def test_cumsum_domain_scaling(self):
        """cumsummat on [0, 2]: integral of 1 from 0 to x = x."""
        n = 10
        Q = cumsummat(n, domain=(0.0, 2.0))
        x_ab = chebpts_ab(n, 0.0, 2.0, kind=2)
        result = Q @ jnp.ones(n, dtype=jnp.float64)
        expected = x_ab - 0.0  # F(x) - F(a) where F(x)=x, a=0
        npt.assert_allclose(np.array(result), np.array(expected), atol=5e-14)


class TestCumsummatKind1:
    """Cumsummat tests for 1st-kind Chebyshev points."""

    def test_cumsum_of_constant_kind1(self):
        """Integral of 1 from -1 to x_k at 1st-kind points."""
        for n in [5, 10, 20]:
            Q = cumsummat(n, kind=1)
            x = chebpts(n, kind=1)
            result = Q @ jnp.ones(n, dtype=jnp.float64)
            expected = x + 1.0
            npt.assert_allclose(np.array(result), np.array(expected), atol=5e-14)


class TestDiffCumsumInverse:
    """D and Q are approximate inverses on suitable subspaces."""

    def test_diff_of_cumsum_polynomial(self):
        """D @ (Q @ f) = f for polynomials of degree < n-1.

        The derivative of the antiderivative exactly recovers a polynomial
        when the antiderivative is representable (degree < n).
        """
        n = 10
        D = diffmat(n)
        Q = cumsummat(n)
        x = chebpts(n, kind=2)
        # Use polynomial of degree n-2 = 8 (antiderivative has degree 9 = n-1)
        f_vals = x**4 - 2 * x**3 + x - 1.0
        DQf = D @ (Q @ f_vals)
        npt.assert_allclose(np.array(DQf), np.array(f_vals), atol=1e-11)

    def test_cumsum_diff_recovery(self):
        """Q @ D @ f should recover f - f(-1) for polynomials.

        The antiderivative of the derivative recovers f up to an additive
        constant (the value at the left endpoint).
        """
        n = 10
        D = diffmat(n)
        Q = cumsummat(n)
        x = chebpts(n, kind=2)
        f_vals = x**5 + x**3 - 2.0 * x  # degree 5, derivative is degree 4
        QDf = Q @ (D @ f_vals)
        expected = f_vals - f_vals[0]  # f(x) - f(-1)
        npt.assert_allclose(np.array(QDf), np.array(expected), atol=1e-12)


class TestIntmat:
    """Tests for intmat (definite integration via cumsum)."""

    def test_intmat_p1_equals_cumsummat(self):
        """intmat(n, p=1) should equal cumsummat(n)."""
        for n in [5, 10]:
            K = intmat(n, p=1)
            Q = cumsummat(n)
            npt.assert_allclose(np.array(K), np.array(Q), atol=1e-15)

    def test_intmat_p0_is_identity(self):
        """intmat(n, p=0) returns identity."""
        for n in [3, 5, 10]:
            K = intmat(n, p=0)
            npt.assert_allclose(np.array(K), np.eye(n), atol=1e-15)

    def test_intmat_p2(self):
        """intmat(n, p=2) = cumsummat^2."""
        n = 10
        K2 = intmat(n, p=2)
        Q = cumsummat(n)
        expected = Q @ Q
        npt.assert_allclose(np.array(K2), np.array(expected), atol=1e-14)


class TestIntrow:
    """Tests for introw (Clenshaw-Curtis quadrature weights)."""

    def test_introw_matches_chebweights(self):
        """introw(n) should equal chebweights(n, kind=2) on [-1, 1]."""
        for n in [3, 5, 10, 20]:
            r = introw(n)
            w = chebweights(n, kind=2)
            npt.assert_allclose(np.array(r), np.array(w), atol=1e-15)

    def test_introw_integrates_constant(self):
        """Integral of 1 over [-1, 1] = 2."""
        for n in [3, 5, 10]:
            r = introw(n)
            npt.assert_allclose(float(jnp.sum(r)), 2.0, atol=1e-14)

    def test_introw_scaled_domain(self):
        """Integral of 1 over [0, 1] = 1."""
        r = introw(10, domain=(0.0, 1.0))
        npt.assert_allclose(float(jnp.sum(r)), 1.0, atol=1e-14)


class TestDiffrow:
    """Tests for diffrow (single row of differentiation matrix)."""

    def test_diffrow_matches_diffmat_left(self):
        """diffrow at left endpoint = first row of diffmat."""
        for n in [5, 10, 20]:
            r = diffrow(n, 1, -1.0)
            D = diffmat(n)
            npt.assert_allclose(np.array(r), np.array(D[0, :]), atol=1e-15)

    def test_diffrow_matches_diffmat_right(self):
        """diffrow at right endpoint = last row of diffmat."""
        for n in [5, 10, 20]:
            r = diffrow(n, 1, 1.0)
            D = diffmat(n)
            npt.assert_allclose(np.array(r), np.array(D[-1, :]), atol=1e-15)

    def test_diffrow_p2(self):
        """2nd-order diffrow at endpoints."""
        n = 10
        r_left = diffrow(n, 2, -1.0)
        r_right = diffrow(n, 2, 1.0)
        D2 = diffmat(n, p=2)
        npt.assert_allclose(np.array(r_left), np.array(D2[0, :]), atol=1e-15)
        npt.assert_allclose(np.array(r_right), np.array(D2[-1, :]), atol=1e-15)

    def test_diffrow_invalid_x(self):
        """diffrow with x not an endpoint should raise ValueError."""
        with pytest.raises(ValueError, match="endpoint"):
            diffrow(5, 1, 0.5)


# ===========================================================================
# Tier 1: Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_diffmat_n1(self):
        """Single-point matrix is identity."""
        D = diffmat(1)
        npt.assert_allclose(np.array(D), np.eye(1), atol=1e-15)

    def test_diffmat_n0(self):
        """Zero-size matrix."""
        D = diffmat(0)
        assert D.shape == (0, 0)

    def test_diffmat_n2(self):
        """Two-point differentiation matrix."""
        D = diffmat(2)
        assert D.shape == (2, 2)
        # D @ [1, 1] = [0, 0]
        npt.assert_allclose(np.array(D @ jnp.ones(2)), 0.0, atol=1e-14)

    def test_diffmat_negative_p_raises(self):
        """Negative differentiation order raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            diffmat(5, p=-1)

    def test_diffmat_invalid_kind_raises(self):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be 1 or 2"):
            diffmat(5, kind=3)

    def test_cumsummat_n1(self):
        """Single-point cumsummat."""
        Q = cumsummat(1)
        assert Q.shape == (1, 1)

    def test_cumsummat_n0(self):
        """Zero-size cumsummat."""
        Q = cumsummat(0)
        assert Q.shape == (0, 0)

    def test_intmat_negative_p_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            intmat(5, p=-1)


# ===========================================================================
# Tier 2: MATLAB cross-validation
# ===========================================================================


class TestDiffmatVsMatlab:
    """Compare diffmat output against MATLAB Chebfun golden references.

    MATLAB reference: tests/references/diffmat.mat
    Generated by: matlab_harness/refs/diffmat_refs.m

    Tolerance rationale: the barycentric differentiation matrix algorithm
    produces results identical to MATLAB for small n (exact match at n=5, 10, 17).
    For larger n, floating-point differences appear in O(eps) range.
    """

    @pytest.mark.matlab
    @pytest.mark.parametrize("n", [5, 10, 17, 20, 32, 64])
    def test_diffmat_kind2_vs_matlab(self, matlab_diffmat, n):
        """1st-order differentiation matrix, 2nd-kind points."""
        ref = matlab_diffmat[f"diffmat_n{n}"]
        D = np.array(diffmat(n))
        # For n <= 17, we get exact match. For larger n, atol ~ 1e-13.
        atol = 1e-13 if n > 17 else 1e-15
        npt.assert_allclose(D, ref, atol=atol,
                            err_msg=f"diffmat(n={n}) vs MATLAB")

    @pytest.mark.matlab
    @pytest.mark.parametrize("n", [5, 10, 17, 20])
    def test_diffmat_kind1_vs_matlab(self, matlab_diffmat, n):
        """1st-order differentiation matrix, 1st-kind points."""
        ref = matlab_diffmat[f"diffmat_n{n}_kind1"]
        D = np.array(diffmat(n, kind=1))
        npt.assert_allclose(D, ref, atol=1e-14,
                            err_msg=f"diffmat(n={n}, kind=1) vs MATLAB")

    @pytest.mark.matlab
    @pytest.mark.parametrize("n,p", [(10, 2), (10, 3), (10, 4),
                                      (20, 2), (20, 3), (20, 4)])
    def test_diffmat_higher_order_vs_matlab(self, matlab_diffmat, n, p):
        """Higher-order differentiation matrices vs MATLAB.

        Tolerance: For p=2 n=20, MATLAB max entry is ~14000, abs diff ~5e-12,
        which is ~4e-16 relative. For p=4, n=20, abs diff ~2e-7 on entries
        up to ~3e7, giving ~6e-15 relative. This is expected ill-conditioning
        of high-order spectral differentiation.

        MATLAB error (measured): n=20, p=4 has max relative diff ~1.2e-12
        between our code and MATLAB, both at machine precision. We set rtol
        to 2e-12 which is tight enough to catch any real regression.
        """
        ref = matlab_diffmat[f"diffmat_n{n}_p{p}"]
        D = np.array(diffmat(n, p=p))
        # Higher p and larger n: both codes are limited by FP arithmetic.
        # Measured max relative diff: n=20,p=4 -> 1.2e-12.
        rtol = 2e-12
        npt.assert_allclose(D, ref, rtol=rtol,
                            err_msg=f"diffmat(n={n}, p={p}) vs MATLAB")

    @pytest.mark.matlab
    @pytest.mark.parametrize("n,p", [(10, 2)])
    def test_diffmat_kind1_higher_order_vs_matlab(self, matlab_diffmat, n, p):
        """Higher-order, 1st-kind, vs MATLAB.

        Some entries are near-zero (~1e-15), so we need an absolute tolerance
        alongside the relative one. Measured: max abs diff ~2.2e-15, max
        relative diff on non-tiny entries ~1e-14.
        """
        ref = matlab_diffmat[f"diffmat_n{n}_p{p}_kind1"]
        D = np.array(diffmat(n, p=p, kind=1))
        npt.assert_allclose(D, ref, rtol=1e-12, atol=5e-15,
                            err_msg=f"diffmat(n={n}, p={p}, kind=1) vs MATLAB")

    @pytest.mark.matlab
    def test_diffmat_domain_vs_matlab(self, matlab_diffmat):
        """Domain-scaled diffmat vs MATLAB."""
        ref = matlab_diffmat["diffmat_n10_dom02"]
        D = np.array(diffmat(10, domain=(0.0, 2.0)))
        npt.assert_allclose(D, ref, atol=1e-14,
                            err_msg="diffmat(10, dom=[0,2]) vs MATLAB")

        ref = matlab_diffmat["diffmat_n10_p2_dom02"]
        D = np.array(diffmat(10, p=2, domain=(0.0, 2.0)))
        npt.assert_allclose(D, ref, atol=1e-14,
                            err_msg="diffmat(10, p=2, dom=[0,2]) vs MATLAB")


class TestCumsummatVsMatlab:
    """Compare cumsummat against MATLAB golden references."""

    @pytest.mark.matlab
    @pytest.mark.parametrize("n", [5, 10, 17, 20])
    def test_cumsummat_vs_matlab(self, matlab_diffmat, n):
        ref = matlab_diffmat[f"cumsummat_n{n}"]
        Q = np.array(cumsummat(n))
        # cumsummat errors are sub-eps for all tested sizes
        npt.assert_allclose(Q, ref, atol=1e-14,
                            err_msg=f"cumsummat(n={n}) vs MATLAB")

    @pytest.mark.matlab
    def test_cumsummat_domain_vs_matlab(self, matlab_diffmat):
        ref = matlab_diffmat["cumsummat_n10_dom02"]
        Q = np.array(cumsummat(10, domain=(0.0, 2.0)))
        npt.assert_allclose(Q, ref, atol=1e-14,
                            err_msg="cumsummat(10, dom=[0,2]) vs MATLAB")


class TestIntrowVsMatlab:
    """Compare introw against MATLAB golden references."""

    @pytest.mark.matlab
    @pytest.mark.parametrize("n", [5, 10, 17, 20, 32])
    def test_introw_vs_matlab(self, matlab_diffmat, n):
        ref = matlab_diffmat[f"introw_n{n}"]
        r = np.array(introw(n))
        npt.assert_allclose(r, ref, atol=1e-15,
                            err_msg=f"introw(n={n}) vs MATLAB")

    @pytest.mark.matlab
    def test_introw_domain_vs_matlab(self, matlab_diffmat):
        ref = matlab_diffmat["introw_n10_dom02"]
        r = np.array(introw(10, domain=(0.0, 2.0)))
        npt.assert_allclose(r, ref, atol=1e-15,
                            err_msg="introw(10, dom=[0,2]) vs MATLAB")


class TestDiffrowVsMatlab:
    """Compare diffrow against MATLAB golden references."""

    @pytest.mark.matlab
    @pytest.mark.parametrize("n", [5, 10, 20])
    @pytest.mark.parametrize("p", [1, 2])
    @pytest.mark.parametrize("side,x_val", [("left", -1.0), ("right", 1.0)])
    def test_diffrow_vs_matlab(self, matlab_diffmat, n, p, side, x_val):
        key = f"diffrow_n{n}_p{p}_{side}"
        ref = matlab_diffmat[key]
        r = np.array(diffrow(n, p, x_val))
        # For n=20, p=2: abs diff ~5e-12, which is floating-point noise
        # on matrix entries up to ~14000
        atol = 1e-10 if (n >= 20 and p >= 2) else 1e-13
        npt.assert_allclose(r, ref, atol=atol,
                            err_msg=f"diffrow(n={n}, p={p}, x={x_val}) vs MATLAB")


# ===========================================================================
# JIT compatibility
# ===========================================================================


class TestJITCompatibility:
    """Verify functions produce correct results under jax.jit.

    All functions take integer arguments that control shapes, so they must
    be called with static arguments (via functools.partial or closure).
    """

    def test_diffmat_jit(self):
        """diffmat under JIT with static args."""
        jitted = jax.jit(functools.partial(diffmat, 10, 1, (-1.0, 1.0), 2))
        D_jit = jitted()
        D_eager = diffmat(10)
        npt.assert_allclose(np.array(D_jit), np.array(D_eager), rtol=1e-15)

    def test_diffmat_jit_kind1(self):
        """diffmat kind=1 under JIT.

        JIT and eager may differ by 1-2 ULP due to instruction reordering
        in jnp.linalg.inv path; rtol=1e-13 is conservative.
        """
        jitted = jax.jit(functools.partial(diffmat, 10, 1, (-1.0, 1.0), 1))
        D_jit = jitted()
        D_eager = diffmat(10, kind=1)
        npt.assert_allclose(np.array(D_jit), np.array(D_eager), rtol=1e-13)

    def test_cumsummat_jit(self):
        """cumsummat under JIT.

        cumsummat uses jnp.linalg.inv, which can produce 1-2 ULP differences
        between JIT and eager.
        """
        jitted = jax.jit(functools.partial(cumsummat, 10, (-1.0, 1.0), 2))
        Q_jit = jitted()
        Q_eager = cumsummat(10)
        npt.assert_allclose(np.array(Q_jit), np.array(Q_eager), rtol=1e-13)

    def test_intmat_jit(self):
        """intmat under JIT (same tolerance as cumsummat — uses linalg.inv)."""
        jitted = jax.jit(functools.partial(intmat, 10, 1, (-1.0, 1.0)))
        K_jit = jitted()
        K_eager = intmat(10)
        npt.assert_allclose(np.array(K_jit), np.array(K_eager), rtol=1e-13)

    def test_introw_jit(self):
        """introw under JIT."""
        jitted = jax.jit(functools.partial(introw, 10, (-1.0, 1.0)))
        r_jit = jitted()
        r_eager = introw(10)
        npt.assert_allclose(np.array(r_jit), np.array(r_eager), rtol=1e-15)

    def test_diffrow_jit(self):
        """diffrow under JIT."""
        jitted = jax.jit(functools.partial(diffrow, 10, 1, -1.0, (-1.0, 1.0)))
        r_jit = jitted()
        r_eager = diffrow(10, 1, -1.0)
        npt.assert_allclose(np.array(r_jit), np.array(r_eager), rtol=1e-15)

    def test_matmul_in_jit(self):
        """D @ f inside JIT produces correct derivative values."""
        @jax.jit
        def compute_derivative(f_vals):
            D = diffmat(10)
            return D @ f_vals

        x = chebpts(10, kind=2)
        f_vals = x**3
        result = compute_derivative(f_vals)
        expected = 3.0 * x**2
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-12)
