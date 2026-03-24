"""Tests for Linop (U62) and Chebop (U63).

Tests the following BVPs / problems:

1. Linop: u'' = -1, u(±1) = 0  →  u = (1 - x²)/2
2. Linop: u'' + u = 0, u(0) = 0, u'(0) = 1  →  u = sin(x)  on [0, π]
3. Linop: eigenvalues of u'' on [-1, 1] with Dirichlet BCs  →  -(n·π/2)²
4. Chebop: same BVPs via user-friendly interface (scalar lbc/rbc)
5. Chebop.eigs: same eigenvalue test via Chebop

JAX contract: construction is NOT JIT-safe; solve uses jnp.linalg.solve (JIT-
safe given fixed n).

MATLAB golden refs
------------------
For problem 1, u(0) = 0.5 (exact).
For problem 2, u(π/2) = 1.0 (exact).
Eigenvalues: -(1·π/2)² ≈ -2.4674, -(2·π/2)² ≈ -9.8696, -(3·π/2)² ≈ -22.207, ...
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.operators.blocks import D, I, eval_at
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.linop import Linop

# ===========================================================================
# Linop tests
# ===========================================================================


class TestLinopPoisson:
    """u'' = -1, u(±1) = 0  →  u = (1 - x²) / 2.

    JAX contract: construction not JIT-safe; solve uses fixed-size linalg.
    """

    def setup_method(self):
        domain = (-1.0, 1.0)
        L_op = D(domain, order=2)
        bcs = [eval_at(-1.0, domain=domain), eval_at(1.0, domain=domain)]
        self.linop = Linop(L_op, bcs=bcs, domain=domain, bc_values=[0.0, 0.0])

    def test_fixed_n_solve(self):
        """Fixed-size solve returns correct values at key points."""
        u = self.linop.solve(lambda x: -jnp.ones_like(x), n=16)
        # u(0) = 0.5 exactly
        npt.assert_allclose(float(u(jnp.array(0.0))), 0.5, atol=1e-12)
        # u(±1) = 0 exactly
        npt.assert_allclose(float(u(jnp.array(-1.0))), 0.0, atol=1e-12)
        npt.assert_allclose(float(u(jnp.array(1.0))), 0.0, atol=1e-12)

    def test_adaptive_solve(self):
        """Adaptive solve converges to machine precision."""
        u = self.linop.solve(lambda x: -jnp.ones_like(x))
        x_test = jnp.linspace(-1.0, 1.0, 20)
        u_exact = (1.0 - x_test**2) / 2.0
        npt.assert_allclose(
            np.array(u(x_test)), np.array(u_exact), atol=1e-10
        )

    def test_constant_rhs(self):
        """Scalar RHS (constant -1)."""
        u = self.linop.solve(-1.0, n=16)
        npt.assert_allclose(float(u(jnp.array(0.0))), 0.5, atol=1e-12)

    def test_solution_satisfies_bcs(self):
        """Boundary conditions are satisfied."""
        u = self.linop.solve(-1.0, n=20)
        npt.assert_allclose(float(u(jnp.array(-1.0))), 0.0, atol=1e-12)
        npt.assert_allclose(float(u(jnp.array(1.0))), 0.0, atol=1e-12)

    def test_mldivide_syntax(self):
        """N / f syntax delegates to solve."""
        u = self.linop / -1.0
        npt.assert_allclose(float(u(jnp.array(0.0))), 0.5, atol=1e-10)


class TestLinopSin:
    """u'' + u = 0, u(0) = 0, u'(0) = 1  →  u = sin(x) on [0, π].

    Note: two BCs at the *same* endpoint (initial value problem style).
    """

    def setup_method(self):
        domain = (0.0, float(jnp.pi))
        a, b = domain
        D2 = D(domain, order=2)
        Id = I(domain)
        L_op = D2 + Id  # u'' + u

        # BC 1: u(0) = 0   → eval_at(0)
        # BC 2: u'(0) = 1  → eval_at(0) ∘ D  (derivative eval)
        from chebfunjax.operators.chebop import _derivative_eval_at
        bc_left = eval_at(a, domain=domain)
        bc_left_deriv = _derivative_eval_at(a, domain=domain, order=1)
        bcs = [bc_left, bc_left_deriv]
        self.linop = Linop(L_op, bcs=bcs, domain=domain, bc_values=[0.0, 1.0])
        self.domain = domain

    def test_value_at_half_pi(self):
        """u(π/2) ≈ 1."""
        u = self.linop.solve(lambda x: jnp.zeros_like(x), n=32)
        half_pi = float(jnp.pi) / 2.0
        npt.assert_allclose(
            float(u(jnp.array(half_pi))), 1.0, atol=1e-8
        )

    def test_solution_is_sin(self):
        """Solution matches sin(x) over [0, π]."""
        u = self.linop.solve(lambda x: jnp.zeros_like(x), n=32)
        x_test = jnp.linspace(0.0, float(jnp.pi), 30)
        npt.assert_allclose(
            np.array(u(x_test)), np.array(jnp.sin(x_test)), atol=1e-8
        )


class TestLinopEigs:
    """Eigenvalues of u'' on [-1, 1] with Dirichlet BCs.

    Exact eigenvalues: λ_n = -(n·π/2)²  for n = 1, 2, 3, ...
    """

    def setup_method(self):
        domain = (-1.0, 1.0)
        L_op = D(domain, order=2)
        bcs = [eval_at(-1.0, domain=domain), eval_at(1.0, domain=domain)]
        self.linop = Linop(L_op, bcs=bcs, domain=domain, bc_values=[0.0, 0.0])

    def test_first_four_eigenvalues(self):
        """First 4 eigenvalues match -(n·π/2)² to 4 significant figures.

        The ``eigs`` method returns eigenvalues sorted by smallest magnitude
        (SM), so λ₁ ≈ -(π/2)² is first.  We sort both actual and expected
        by magnitude (ascending) before comparing.
        """
        lam = self.linop.eigs(n=64, k=4)
        lam_real = jnp.real(lam)
        # Sort by magnitude ascending
        lam_sorted = lam_real[jnp.argsort(jnp.abs(lam_real))]

        exact = jnp.array(
            [-(n * jnp.pi / 2.0) ** 2 for n in range(1, 5)],
            dtype=jnp.float64,
        )
        npt.assert_allclose(
            np.array(lam_sorted), np.array(exact), rtol=1e-4
        )

    def test_eigenvalues_are_real(self):
        """Eigenvalues of a self-adjoint operator are real."""
        lam = self.linop.eigs(n=32, k=4)
        npt.assert_allclose(np.imag(np.array(lam)), 0.0, atol=1e-6)

    def test_eigenvalues_are_negative(self):
        """All eigenvalues of u'' with Dirichlet BCs are negative."""
        lam = self.linop.eigs(n=32, k=6)
        assert float(jnp.max(jnp.real(lam))) < 0.0


# ===========================================================================
# Chebop tests
# ===========================================================================


class TestChebopPoisson:
    """u'' = -1, u(±1) = 0  via Chebop interface.

    JAX contract: solve is not JIT-safe (adaptive Python loop).
    """

    def setup_method(self):
        self.N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
        self.N.lbc = 0.0
        self.N.rbc = 0.0

    def test_solve_fixed_n(self):
        """Fixed-size Chebop solve."""
        u = self.N.solve(-1.0, n=16)
        npt.assert_allclose(float(u(jnp.array(0.0))), 0.5, atol=1e-10)

    def test_solve_adaptive(self):
        """Adaptive Chebop solve."""
        u = self.N.solve(-1.0)
        x_test = jnp.linspace(-1.0, 1.0, 15)
        exact = (1.0 - x_test**2) / 2.0
        npt.assert_allclose(
            np.array(u(x_test)), np.array(exact), atol=1e-8
        )

    def test_mldivide_operator(self):
        """N \\ rhs syntax."""
        u = self.N / -1.0
        npt.assert_allclose(float(u(jnp.array(0.0))), 0.5, atol=1e-10)

    def test_bcs_satisfied(self):
        """Boundary conditions are satisfied."""
        u = self.N.solve(-1.0, n=20)
        npt.assert_allclose(float(u(jnp.array(-1.0))), 0.0, atol=1e-12)
        npt.assert_allclose(float(u(jnp.array(1.0))), 0.0, atol=1e-12)


class TestChebopEigs:
    """Eigenvalues of u'' via Chebop.eigs."""

    def test_first_four_eigenvalues(self):
        """Chebop eigenvalues match Linop eigenvalues (sorted by magnitude)."""
        N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0

        lam = N.eigs(n=64, k=4)
        lam_real = jnp.real(lam)
        # Sort by magnitude ascending (smallest eigenvalue first by abs)
        lam_sorted = lam_real[jnp.argsort(jnp.abs(lam_real))]

        exact = jnp.array(
            [-(n * jnp.pi / 2.0) ** 2 for n in range(1, 5)],
            dtype=jnp.float64,
        )
        npt.assert_allclose(
            np.array(lam_sorted), np.array(exact), rtol=1e-4
        )


class TestChebopNonlinear:
    """Nonlinear BVP: u'' + u² = 0 (trivial exact solution u=0).

    Tests that the Newton iteration in Chebop converges for a simple
    nonlinear problem.
    """

    def test_zero_solution_nonlinear(self):
        """Nonlinear operator with trivial zero solution."""
        N = Chebop(lambda x, u: u.diff(2) + u * u, domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        # u=0 is an exact solution of u'' + u² = 0 with u=0 at boundaries
        u = N.solve(0.0, n=16, max_iter=5)
        x_test = jnp.linspace(-1.0, 1.0, 10)
        npt.assert_allclose(np.array(u(x_test)), 0.0, atol=1e-8)


class TestLinopRepr:
    """Check repr and basic construction."""

    def test_repr(self):
        domain = (-1.0, 1.0)
        L_op = D(domain, order=2)
        bcs = [eval_at(-1.0, domain=domain), eval_at(1.0, domain=domain)]
        linop = Linop(L_op, bcs=bcs, domain=domain)
        r = repr(linop)
        assert "Linop" in r
        assert "domain" in r

    def test_bad_op_raises(self):
        """Non-OperatorBlock argument raises TypeError."""
        with pytest.raises(TypeError, match="OperatorBlock"):
            Linop("not an op")

    def test_bc_values_mismatch_raises(self):
        """Mismatched bc_values length raises ValueError."""
        domain = (-1.0, 1.0)
        L_op = D(domain, order=2)
        bcs = [eval_at(-1.0, domain=domain)]
        with pytest.raises(ValueError, match="bc_values"):
            Linop(L_op, bcs=bcs, domain=domain, bc_values=[0.0, 1.0])
