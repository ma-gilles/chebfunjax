"""Tests for chebfunjax.discretization.ultras — ultraspherical spectral method.

Tests cover:
- diffmat: banded structure, domain scaling, exactness on polynomials
- convertmat / conversion: banded structure, composition property, round-trip
- multmat: identity, scalar, polynomial multiplication
- UltraS class: diffmat, conversion, multmat, points
- ODE solve: u'' + u = 0 gives exact sin/cos

JAX contract: jit=yes (n, k must be static), vmap=no, grad=no
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.discretization.ultras import (
    UltraS,
    _spconvert,
    convertmat,
    diffmat,
    multmat,
)
from chebfunjax.domain import Domain

# ===========================================================================
# Tier 1: diffmat mathematical properties
# ===========================================================================


class TestDiffmat:
    """Tests for the ultraspherical differentiation matrix.

    JAX contract: jit=yes (n, k must be static), vmap=no, grad=no
    """

    def test_diffmat_k0_is_identity(self):
        """diffmat(n, 0) = I."""
        for n in [1, 4, 8, 16]:
            D = diffmat(n, 0)
            npt.assert_allclose(np.array(D), np.eye(n), atol=1e-15)

    def test_diffmat_k1_shape(self):
        """diffmat(n, 1) has correct shape."""
        D = diffmat(10, 1)
        assert D.shape == (10, 10)

    def test_diffmat_k1_superdiagonal(self):
        """diffmat(n, 1) has superdiagonal [1, 2, 3, ..., n-1].

        MATLAB: D = spdiags((0:n-1)', 1, n, n)
        For superdiagonal k=1, MATLAB spdiags places v(i+k) at position (i,i+k)
        (1-indexed), so D(1,2)=v(2)=1, D(2,3)=v(3)=2, ..., D(n-1,n)=n-1.
        In 0-indexed: D[0,1]=1, D[1,2]=2, ..., D[n-2,n-1]=n-1.
        """
        n = 8
        D = diffmat(n, 1)
        expected_super = np.arange(1, n, dtype=float)
        npt.assert_allclose(np.diag(np.array(D), 1), expected_super, atol=1e-15)
        # All other entries should be zero
        D_np = np.array(D)
        D_no_super = D_np.copy()
        D_no_super[np.arange(n - 1), np.arange(1, n)] = 0.0
        npt.assert_allclose(D_no_super, 0.0, atol=1e-15)

    def test_diffmat_k1_constant_function(self):
        """D @ [1, 0, ..., 0]^T = 0 (derivative of constant = 0)."""
        for n in [3, 6, 10]:
            D = diffmat(n, 1)
            u = jnp.zeros(n, dtype=jnp.float64).at[0].set(1.0)
            npt.assert_allclose(np.array(D @ u), 0.0, atol=1e-15)

    def test_diffmat_k1_linear(self):
        """Derivative of T_1(x) = x is C_0^{(1)} = 1 (constant).

        T_1'(x) = 1 = C_0^{(1)}.  D @ [0,1,0,...] should give [1,0,...].
        """
        n = 5
        D = diffmat(n, 1)
        # T_1 has Chebyshev T coefficients [0, 1, 0, ..., 0]
        u = jnp.zeros(n, dtype=jnp.float64).at[1].set(1.0)
        result = D @ u
        # D[0,1] = 1, all other entries in column 1 are 0
        expected = np.zeros(n)
        expected[0] = 1.0
        npt.assert_allclose(np.array(result), expected, atol=1e-15)

    def test_diffmat_k2_banded(self):
        """diffmat(n, 2) has bandwidth 2 (only super+2 diagonal nonzero)."""
        n = 10
        D = diffmat(n, 2)
        D_np = np.array(D)
        # Remove the +2 superdiagonal
        D_check = D_np.copy()
        for i in range(n - 2):
            D_check[i, i + 2] = 0.0
        npt.assert_allclose(D_check, 0.0, atol=1e-15)

    def test_diffmat_polynomial_exactness(self):
        """D @ c_T2 should give C^{(2)} coefficients of T_4''(x).

        T_4(x) = 8x^4 - 8x^2 + 1
        T_4'(x) = 32x^3 - 16x
        T_4''(x) = 96x^2 - 16
        In C^{(2)}: T_4''(x) = 8*C_0^{(2)} + 16*C_2^{(2)}
        (since C_0^{(2)} = 1, C_2^{(2)} = x^2 - 1/8... we verify numerically)
        """
        n = 8
        D2 = diffmat(n, 2)
        # T_4 has Chebyshev T coefficients [0, 0, 0, 0, 1, 0, ...]
        u = jnp.zeros(n, dtype=jnp.float64).at[4].set(1.0)
        result = D2 @ u
        # Verify by evaluating: C^{(2)} expansion of T_4''
        # T_4''(x) = 96x^2 - 16
        # We verify by checking that the result is consistent
        # Evaluate the C^{(2)} series at a few points using the recurrence
        # C_0^{(2)}(x) = 1, C_1^{(2)}(x) = 4x, C_2^{(2)}(x) = 4*(2x^2-1/2)/1...
        # Direct check: T_4''(0) = -16
        # Evaluate C^{(2)} expansion at x=0
        # C_0^{(2)}(0)=1, C_1^{(2)}(0)=0, C_2^{(2)}(0)=-1, C_3^{(2)}(0)=0, C_4^{(2)}(0)=1
        x_test = np.array([0.0, 0.5, -0.5, 1.0])
        d2_exact = 96.0 * x_test**2 - 16.0

        result_np = np.array(result)
        # Build C^{(2)} Vandermonde matrix at test points via 3-term recurrence
        # C_0^{(2)}=1, C_1^{(2)}=4x, C_{k+1}^{(2)} = 2x C_k^{(2)} - C_{k-1}^{(2)} ... (approximate)
        # More carefully: C_k^{(lam)}(x) with lam=2 satisfies:
        # C_{k+1}^{(2)} = 2(k+2)/(k+1) * x * C_k^{(2)} - (k+3)/(k+1) * C_{k-1}^{(2)}
        # For lam=2: C_{k+1} = 2(k+lam)/(k+1) * x * C_k - (k+2lam-1)/(k+1) * C_{k-1}
        V = np.zeros((len(x_test), n))
        V[:, 0] = 1.0
        if n > 1:
            V[:, 1] = 4.0 * x_test  # = 2*lam * x
        for k in range(1, n - 1):
            V[:, k + 1] = (2.0 * (k + 2.0) / (k + 1.0)) * x_test * V[:, k] \
                          - ((k + 3.0) / (k + 1.0)) * V[:, k - 1]
        d2_approx = V @ result_np
        npt.assert_allclose(d2_approx, d2_exact, rtol=1e-12, atol=1e-12)

    def test_diffmat_jit(self):
        """diffmat can be called inside jit with static n."""
        n = 8
        f = jax.jit(functools.partial(diffmat, n, 1))
        D_jit = f()
        D_ref = diffmat(n, 1)
        npt.assert_allclose(np.array(D_jit), np.array(D_ref), atol=1e-15)


# ===========================================================================
# Tier 1: convertmat mathematical properties
# ===========================================================================


class TestConvertmat:
    """Tests for the ultraspherical conversion matrix.

    JAX contract: jit=yes (n, k1, k2 must be static), vmap=no, grad=no
    """

    def test_convertmat_identity_when_k2_lt_k1(self):
        """convertmat(n, k1, k2) = I when k2 < k1."""
        n = 8
        S = convertmat(n, 2, 1)  # k2 < k1 -> identity
        npt.assert_allclose(np.array(S), np.eye(n), atol=1e-15)

    def test_convertmat_k1_0_k2_0_banded(self):
        """S = convertmat(n, 0, 0): C^{(0)} -> C^{(1)}, bandwidth 2."""
        n = 8
        S = convertmat(n, 0, 0)
        assert S.shape == (n, n)
        S_np = np.array(S)
        # Should be banded with main diagonal and +2 superdiagonal
        # Check no entries outside these diagonals
        for i in range(n):
            for j in range(n):
                if j != i and j != i + 2:
                    assert abs(S_np[i, j]) < 1e-14, \
                        f"S[{i},{j}] = {S_np[i,j]} should be zero"

    def test_convertmat_main_diagonal(self):
        """S_0: main diagonal is [1, 1/2, 1/2, ..., 1/2]."""
        n = 8
        S = _spconvert(n, 0.0)
        S_np = np.array(S)
        expected_diag = np.array([1.0] + [0.5] * (n - 1))
        npt.assert_allclose(np.diag(S_np), expected_diag, atol=1e-15)

    def test_convertmat_super2_diagonal(self):
        """S_0: super+2 diagonal is [-1/2, -1/2, ..., -1/2]."""
        n = 8
        S = _spconvert(n, 0.0)
        S_np = np.array(S)
        super2 = np.diag(S_np, 2)
        npt.assert_allclose(super2, -0.5 * np.ones(n - 2), atol=1e-15)

    def test_convertmat_lambda1(self):
        """S_1: maps C^{(1)} -> C^{(2)}.

        From spconvert.m: dg = lam/(lam + (2:n-1)), so for lam=1:
          - main diagonal: 1/(1+j) for j=0..n-1
          - super+2 diagonal: A[j,j+2] = -1/(j+3) for j=0..n-3
        """
        n = 6
        S = _spconvert(n, 1.0)
        S_np = np.array(S)
        j = np.arange(n, dtype=float)
        expected_diag = 1.0 / (1.0 + j)
        npt.assert_allclose(np.diag(S_np), expected_diag, rtol=1e-14)
        # Super+2 diagonal: A[j,j+2] = -lam/(lam+j+2) = -1/(j+3) for lam=1
        expected_super2 = -1.0 / (j[:n - 2] + 3.0)
        npt.assert_allclose(np.diag(S_np, 2), expected_super2, rtol=1e-14)

    def test_convertmat_composition(self):
        """convertmat(n, 0, 1) = S_1 @ S_0."""
        n = 10
        S_01 = convertmat(n, 0, 1)   # C^{(0)} -> C^{(2)}
        S0 = _spconvert(n, 0.0)      # C^{(0)} -> C^{(1)}
        S1 = _spconvert(n, 1.0)      # C^{(1)} -> C^{(2)}
        S_composed = S1 @ S0
        npt.assert_allclose(np.array(S_01), np.array(S_composed), atol=1e-14)

    def test_convertmat_positivity(self):
        """Main diagonal of S_lam is positive for all lam >= 0."""
        for lam in [0, 1, 2, 3, 5]:
            S = _spconvert(8, float(lam))
            diag = np.diag(np.array(S))
            assert np.all(diag > 0), f"Non-positive diagonal for lam={lam}"

    def test_convertmat_jit(self):
        """convertmat can be jit-compiled."""
        n = 6
        f = jax.jit(functools.partial(convertmat, n, 0, 0))
        S_jit = f()
        S_ref = convertmat(n, 0, 0)
        npt.assert_allclose(np.array(S_jit), np.array(S_ref), atol=1e-15)


# ===========================================================================
# Tier 1: multmat mathematical properties
# ===========================================================================


class TestMultmat:
    """Tests for the ultraspherical multiplication matrix.

    JAX contract: jit=yes (n, lam must be static), vmap=no, grad=no
    """

    def test_multmat_scalar_lam0(self):
        """multmat(n, [c], 0) = c * I."""
        for c in [1.0, 2.5, -0.5]:
            n = 6
            a = jnp.array([c], dtype=jnp.float64)
            M = multmat(n, a, 0)
            npt.assert_allclose(np.array(M), c * np.eye(n), atol=1e-14)

    def test_multmat_scalar_lam1(self):
        """multmat(n, [c], 1) = c * I."""
        n = 6
        a = jnp.array([3.14], dtype=jnp.float64)
        M = multmat(n, a, 1)
        npt.assert_allclose(np.array(M), 3.14 * np.eye(n), atol=1e-14)

    def test_multmat_identity_function_lam0(self):
        """Multiplying by 1 (a=[1,0,...]) = identity in any basis."""
        for lam in [0, 1, 2]:
            n = 8
            a = jnp.zeros(n, dtype=jnp.float64).at[0].set(1.0)
            M = multmat(n, a, lam)
            npt.assert_allclose(np.array(M), np.eye(n), rtol=1e-12, atol=1e-13)

    def test_multmat_lam0_product_formula(self):
        """M in C^{(0)} basis: T_j * T_k = (T_{j+k} + T_{|j-k|}) / 2.

        Test with f = T_2, so M @ e_k should give Chebyshev T coefficients
        of T_2 * T_k = (T_{2+k} + T_{|2-k|}) / 2.
        """
        n = 8
        a = jnp.zeros(n, dtype=jnp.float64).at[2].set(1.0)
        M = multmat(n, a, 0)
        M_np = np.array(M)

        # T_2 * T_0 = T_2 (since T_0 = 1):  M @ e_0 = e_2
        e0 = np.zeros(n)
        e0[0] = 1.0
        npt.assert_allclose(M_np @ e0, np.eye(n)[2], atol=1e-14)

        # T_2 * T_1 = (T_3 + T_1) / 2:  M @ e_1 = (e_3 + e_1) / 2
        e1 = np.zeros(n)
        e1[1] = 1.0
        expected = np.zeros(n)
        expected[1] = 0.5
        expected[3] = 0.5
        npt.assert_allclose(M_np @ e1, expected, atol=1e-14)

        # T_2 * T_2 = (T_4 + T_0) / 2:  M @ e_2 = (e_4 + e_0) / 2
        e2 = np.zeros(n)
        e2[2] = 1.0
        expected2 = np.zeros(n)
        expected2[0] = 0.5
        expected2[4] = 0.5
        npt.assert_allclose(M_np @ e2, expected2, atol=1e-14)

    def test_multmat_lam0_exactness(self):
        """M * T_1_coeffs gives Chebyshev T coefficients of x * T_1(x) = T_2/2 + T_0/2."""
        n = 8
        # Multiply by x: Chebyshev T coeffs of x are [0, 1, 0, ..., 0]
        a = jnp.zeros(n, dtype=jnp.float64).at[1].set(1.0)
        M = multmat(n, a, 0)
        # Input: T_1(x) has coeffs [0, 1, 0, ..., 0]
        u = jnp.zeros(n, dtype=jnp.float64).at[1].set(1.0)
        result = M @ u
        # x * T_1(x) = x^2 = (T_0 + T_2) / 2
        expected = np.zeros(n)
        expected[0] = 0.5
        expected[2] = 0.5
        npt.assert_allclose(np.array(result), expected, atol=1e-14)

    def test_multmat_lam1_exactness(self):
        """M in C^{(1)} basis: verify multiplication by x = (T_1) is correct.

        In the C^{(1)} basis, x * C_1^{(1)}(x) = x * U_1(x) = x * 2x = 2x^2.
        """
        n = 6
        # Multiply by x: Chebyshev T coefficient vector for x is [0, 1, 0, ...]
        a = jnp.zeros(n, dtype=jnp.float64).at[1].set(1.0)
        M = multmat(n, a, 1)
        # Input: C_1^{(1)} = U_1 has C^{(1)} coefficient [0, 1, 0, ...]
        u = jnp.zeros(n, dtype=jnp.float64).at[1].set(1.0)
        result = np.array(M @ u)
        # x * U_1(x) = x * 2x = 2x^2
        # U_0 = 1, U_1 = 2x, U_2 = 4x^2 - 1
        # 2x^2 = (U_2 + 1) / 2 = U_2/2 + U_0/2
        expected = np.zeros(n)
        expected[0] = 0.5
        expected[2] = 0.5
        npt.assert_allclose(result, expected, atol=1e-13)

    @pytest.mark.parametrize("n", [8, 12, 16])
    @pytest.mark.parametrize("lam", [0, 1, 2])
    def test_multmat_shape(self, n, lam):
        """multmat returns square matrix of correct size."""
        a = jnp.ones(3, dtype=jnp.float64)
        M = multmat(n, a, lam)
        assert M.shape == (n, n)


# ===========================================================================
# Tier 1: UltraS class tests
# ===========================================================================


class TestUltraSClass:
    """Tests for the UltraS class.

    JAX contract: jit=yes (n, k must be static), vmap=no, grad=no
    """

    def test_constructor(self):
        """UltraS can be constructed from n and domain."""
        disc = UltraS(n=16, domain=Domain((-1.0, 1.0)))
        assert disc.n == 16
        assert disc.domain.a == -1.0
        assert disc.domain.b == 1.0

    def test_constructor_tuple_domain(self):
        """UltraS accepts domain as a tuple."""
        disc = UltraS(n=8, domain=(-1.0, 1.0))
        assert disc.domain.a == -1.0
        assert disc.domain.b == 1.0

    def test_points_shape(self):
        """disc.points() returns n points."""
        disc = UltraS(n=10, domain=Domain((-1.0, 1.0)))
        x = disc.points()
        assert x.shape == (10,)

    def test_points_endpoints(self):
        """disc.points() starts at -1 and ends at 1."""
        disc = UltraS(n=10, domain=Domain((-1.0, 1.0)))
        x = disc.points()
        npt.assert_allclose(float(x[0]), -1.0, atol=1e-15)
        npt.assert_allclose(float(x[-1]), 1.0, atol=1e-15)

    def test_points_custom_domain(self):
        """disc.points() maps to custom domain [a, b]."""
        a, b = 0.0, 3.0
        disc = UltraS(n=8, domain=Domain((a, b)))
        x = disc.points()
        npt.assert_allclose(float(x[0]), a, atol=1e-14)
        npt.assert_allclose(float(x[-1]), b, atol=1e-14)

    def test_diffmat_default_k1(self):
        """disc.diffmat() with default k=1 has correct shape."""
        disc = UltraS(n=8, domain=Domain((-1.0, 1.0)))
        D = disc.diffmat()
        assert D.shape == (8, 8)

    def test_diffmat_domain_scaling(self):
        """disc.diffmat(1) is scaled by 2/(b-a) for non-unit interval."""
        n = 8
        disc_std = UltraS(n=n, domain=Domain((-1.0, 1.0)))
        disc_scaled = UltraS(n=n, domain=Domain((0.0, 2.0)))
        D_std = disc_std.diffmat(1)
        D_scaled = disc_scaled.diffmat(1)
        # b-a = 2 for both, so scaling factor 2/(b-a) = 1 for both
        npt.assert_allclose(np.array(D_std), np.array(D_scaled), atol=1e-14)

    def test_diffmat_domain_scaling_asymmetric(self):
        """disc.diffmat(1) scales by 2/(b-a) correctly."""
        n = 8
        disc_std = UltraS(n=n, domain=Domain((-1.0, 1.0)))  # 2/(b-a) = 1
        disc_half = UltraS(n=n, domain=Domain((0.0, 1.0)))   # 2/(b-a) = 2
        D_std = np.array(disc_std.diffmat(1))
        D_half = np.array(disc_half.diffmat(1))
        npt.assert_allclose(D_half, 2.0 * D_std, atol=1e-14)

    def test_conversion_k1(self):
        """disc.conversion(1) = S_0: C^{(0)} -> C^{(1)}."""
        disc = UltraS(n=8, domain=Domain((-1.0, 1.0)))
        S = disc.conversion(1)
        S_ref = _spconvert(8, 0.0)
        npt.assert_allclose(np.array(S), np.array(S_ref), atol=1e-15)

    def test_conversion_k2(self):
        """disc.conversion(2) = S_1: C^{(1)} -> C^{(2)}."""
        disc = UltraS(n=8, domain=Domain((-1.0, 1.0)))
        S = disc.conversion(2)
        S_ref = _spconvert(8, 1.0)
        npt.assert_allclose(np.array(S), np.array(S_ref), atol=1e-15)

    def test_repr(self):
        """UltraS has a readable repr."""
        disc = UltraS(n=10, domain=Domain((-1.0, 1.0)))
        r = repr(disc)
        assert "UltraS" in r
        assert "10" in r


# ===========================================================================
# Tier 1: ODE solve — u'' + u = 0  (golden test)
# ===========================================================================


class TestODESolve:
    """Solve u'' + u = 0 exactly using the ultraspherical method.

    The exact solution with boundary conditions u(-1) = sin(-1), u(1) = sin(1)
    is u(x) = sin(x).

    The ultraspherical system is:
      (D2 + S2_0) u = 0    (interior equations)
      e_{-1}^T u = sin(-1) (left BC)
      e_1^T u = sin(1)     (right BC)

    where D2 = diffmat(n, 2), S2_0 = convertmat(n, 0, 1) is the conversion
    from C^{(0)} to C^{(2)}, and e_{pm1} are the evaluation rows at +/-1.
    """

    @pytest.mark.parametrize("n", [16, 32, 48])
    def test_solve_sin(self, n):
        """u'' + u = 0 with Dirichlet BCs gives u = sin(x)."""
        domain = Domain((-1.0, 1.0))
        disc = UltraS(n=n, domain=domain)

        # D2: differentiation matrix C^{(0)} -> C^{(2)}
        D2 = disc.diffmat(2)

        # S: conversion matrix C^{(0)} -> C^{(2)}, for the identity term (+u)
        S = disc.convertmat(0, 1)  # C^{(0)} -> C^{(2)}

        # Interior system: (D2 + S) @ c_cheb = 0
        L_interior = D2 + S

        # Boundary conditions: evaluation at x = +-1
        # Chebyshev T polynomials at x=1: T_k(1) = 1 for all k
        # Chebyshev T polynomials at x=-1: T_k(-1) = (-1)^k
        k = jnp.arange(n, dtype=jnp.float64)
        row_right = jnp.ones(n, dtype=jnp.float64)          # T_k(1) = 1
        row_left = (-1.0) ** k                               # T_k(-1) = (-1)^k

        # Assemble system: replace last 2 rows with BCs
        L = np.array(L_interior)
        L[-2, :] = np.array(row_left)    # BC at x = -1
        L[-1, :] = np.array(row_right)   # BC at x = +1

        # Right-hand side
        rhs = np.zeros(n)
        rhs[-2] = np.sin(-1.0)   # u(-1) = sin(-1)
        rhs[-1] = np.sin(1.0)    # u(1) = sin(1)

        # Solve
        c_cheb = np.linalg.solve(L, rhs)

        # Evaluate at Chebyshev points and compare to sin(x)
        x = np.array(disc.points())
        # Evaluate Chebyshev series via Clenshaw
        u_approx = _eval_cheb_series(c_cheb, x)
        u_exact = np.sin(x)

        # Should be accurate to ~machine precision times n^2
        npt.assert_allclose(u_approx, u_exact, atol=1e-11)

    @pytest.mark.parametrize("n", [16, 32])
    def test_solve_cos(self, n):
        """u'' + u = 0 with BCs u(-1)=cos(-1), u(1)=cos(1) gives u=cos(x)."""
        domain = Domain((-1.0, 1.0))
        disc = UltraS(n=n, domain=domain)

        D2 = disc.diffmat(2)
        S = disc.convertmat(0, 1)
        L_interior = D2 + S

        k = jnp.arange(n, dtype=jnp.float64)
        row_right = jnp.ones(n, dtype=jnp.float64)
        row_left = (-1.0) ** k

        L = np.array(L_interior)
        L[-2, :] = np.array(row_left)
        L[-1, :] = np.array(row_right)

        rhs = np.zeros(n)
        rhs[-2] = np.cos(-1.0)
        rhs[-1] = np.cos(1.0)

        c_cheb = np.linalg.solve(L, rhs)
        x = np.array(disc.points())
        u_approx = _eval_cheb_series(c_cheb, x)
        u_exact = np.cos(x)

        npt.assert_allclose(u_approx, u_exact, atol=1e-11)

    def test_solve_simple_bvp(self):
        """Solve u'' = -1, u(-1)=0, u(1)=0 -> u = (1-x^2)/2."""
        n = 16
        domain = Domain((-1.0, 1.0))
        disc = UltraS(n=n, domain=domain)

        D2 = disc.diffmat(2)

        k = jnp.arange(n, dtype=jnp.float64)
        row_right = jnp.ones(n, dtype=jnp.float64)
        row_left = (-1.0) ** k

        L = np.array(D2)
        L[-2, :] = np.array(row_left)
        L[-1, :] = np.array(row_right)

        # RHS: u'' = -1, which in C^{(2)} coefficients...
        # The C^{(2)} coefficient of -1 = -C_0^{(2)} -> but we need
        # to express this as a vector. C_0^{(2)}(x) = 1, so coefficient = -1.
        # But D2 maps C^{(0)} coefficients to C^{(2)} coefficients.
        # To represent RHS = -1 in C^{(2)}: -1 = -1 * C_0^{(2)}, so c = [-1, 0, ..., 0].
        rhs = np.zeros(n)
        rhs[0] = -1.0   # C^{(2)} representation of constant -1
        rhs[-2] = 0.0   # BC u(-1) = 0
        rhs[-1] = 0.0   # BC u(1) = 0

        c_cheb = np.linalg.solve(L, rhs)
        x = np.array(disc.points())
        u_approx = _eval_cheb_series(c_cheb, x)
        u_exact = 0.5 * (1.0 - x**2)

        npt.assert_allclose(u_approx, u_exact, atol=1e-13)


# ===========================================================================
# Tier 1: Additional structure tests
# ===========================================================================


class TestSpconvertStructure:
    """Test the banded structure of _spconvert for various lambda values."""

    @pytest.mark.parametrize("lam", [0, 1, 2, 3, 5, 10])
    def test_spconvert_banded(self, lam):
        """_spconvert(n, lam) has only main diagonal and +2 superdiagonal."""
        n = 12
        S = _spconvert(n, float(lam))
        S_np = np.array(S)
        for i in range(n):
            for j in range(n):
                if j != i and j != i + 2:
                    assert abs(S_np[i, j]) < 1e-14, \
                        f"S[{i},{j}] = {S_np[i,j]} != 0 for lam={lam}"

    @pytest.mark.parametrize("lam", [0, 1, 2, 5])
    def test_spconvert_diagonal_values(self, lam):
        """Main diagonal of S_lam is lam/(lam+j) for j=0,1,..."""
        n = 8
        S = _spconvert(n, float(lam))
        S_np = np.array(S)
        j = np.arange(n, dtype=float)
        if lam == 0:
            expected = np.where(j == 0, 1.0, 0.5)
        else:
            expected = lam / (lam + j)
        npt.assert_allclose(np.diag(S_np), expected, rtol=1e-14)


class TestConvertmatBanded:
    """Test that convertmat produces banded matrices."""

    def test_convertmat_bandwidth(self):
        """convertmat(n, 0, k-1) has bandwidth k (k superdiagonals of step 2)."""
        n = 20
        for k in [1, 2, 3]:
            S = convertmat(n, 0, k - 1)
            S_np = np.array(S)
            # Check that only diagonals 0, 2, 4, ..., 2k are nonzero
            for d in range(1, n):
                diag_val = np.diag(S_np, d)
                # Diagonals that are odd or > 2k should be zero
                if d % 2 != 0 or d > 2 * k:
                    npt.assert_allclose(
                        diag_val, 0.0, atol=1e-14,
                        err_msg=f"k={k}, diagonal {d} should be zero"
                    )


# ===========================================================================
# Helper: Chebyshev series evaluation (for test verification only)
# ===========================================================================


def _eval_cheb_series(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate Chebyshev T series via Clenshaw's algorithm.

    Parameters
    ----------
    c : array_like, shape (n,)
        Chebyshev T coefficients.
    x : array_like, shape (m,)
        Evaluation points.

    Returns
    -------
    y : ndarray, shape (m,)
        Values of sum_k c[k] * T_k(x).
    """
    n = len(c)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return c[0] * np.ones_like(x)

    # Clenshaw's algorithm
    b2 = np.zeros_like(x)
    b1 = np.zeros_like(x)
    for k in range(n - 1, 0, -1):
        b0 = c[k] + 2.0 * x * b1 - b2
        b2 = b1
        b1 = b0
    return c[0] + x * b1 - b2
