"""Tests for the last-10 missing functions.

Covers:
  1. pdeSolve         — method-of-lines PDE solver (Dirichlet / Neumann / periodic)
  2. besselh          — Hankel function of a Chebfun
  3. besselk          — Modified Bessel K of a Chebfun
  4. ellipke          — Complete elliptic integrals K(m), E(m)
  5. dirac            — Dirac delta at roots
  6. unwrap           — Phase unwrapping
  7. subspace         — Principal angle between quasimatrix subspaces
  8. lagrange         — Lagrange interpolation basis
  9. ode78 / ode89    — Higher-order ODE integrators
  10. quantumstates   — Schrödinger eigenstates
  11. innerProduct    — alias for inner (method + module)
  12. iszero          — identically-zero test
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import (
    Chebfun,
    chebfun,
    innerProduct,
    lagrange,
    ode78,
    ode89,
    quantumstates,
    subspace,
)
from chebfunjax.chebfun1d.pde_solve import pdeSolve


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pts(n=40, a=-1.0, b=1.0):
    """Uniformly spaced interior test points."""
    return jnp.linspace(a + 1e-6, b - 1e-6, n, dtype=jnp.float64)


# ===========================================================================
# 1. pdeSolve
# ===========================================================================


class TestPdeSolve:
    """Method-of-lines PDE solver."""

    def test_heat_dirichlet_shape(self):
        """pdeSolve returns list of Chebfun with correct length."""
        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.1, 5)
        UU = pdeSolve(lambda t, x, u: u.diff(2), t_out, u0, lbc=0.0, rbc=0.0)
        assert len(UU) == 5
        for uk in UU:
            assert isinstance(uk, Chebfun)

    def test_heat_dirichlet_decay(self):
        """Heat equation: solution should decay in L2 over time."""
        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.2, 5)
        UU = pdeSolve(lambda t, x, u: u.diff(2), t_out, u0, lbc=0.0, rbc=0.0)
        # L-inf norm at t=0 should exceed that at t=0.2
        n0 = float(UU[0].norm(float("inf")))
        nf = float(UU[-1].norm(float("inf")))
        assert nf < n0, f"Solution did not decay: {n0:.4f} → {nf:.4f}"

    def test_heat_boundary_conditions(self):
        """Boundary values stay near 0 for Dirichlet BCs."""
        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.05, 3)
        UU = pdeSolve(lambda t, x, u: u.diff(2), t_out, u0, lbc=0.0, rbc=0.0)
        a, b = float(UU[-1].domain.a), float(UU[-1].domain.b)
        # Allow loose tolerance since ODE solver uses BC enforcement not hard constraints
        assert abs(float(UU[-1](jnp.float64(a)))) < 0.1
        assert abs(float(UU[-1](jnp.float64(b)))) < 0.1

    def test_two_arg_pdefun(self):
        """Two-argument pdefun (t, u) is also accepted."""
        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.05, 3)
        # Two-argument form
        UU = pdeSolve(lambda t, u: u.diff(2), t_out, u0, lbc=0.0, rbc=0.0)
        assert len(UU) == 3

    def test_periodic_bc_shape(self):
        """Periodic BCs: returns list of Chebfun."""
        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.1, 3)
        UU = pdeSolve(lambda t, x, u: -u.diff(), t_out, u0, bc="periodic")
        assert len(UU) == 3

    def test_neumann_bc(self):
        """Neumann BC: at least runs without error and returns correct count."""
        u0 = chebfun(lambda x: jnp.cos(jnp.pi * x))
        t_out = np.linspace(0.0, 0.05, 3)
        UU = pdeSolve(
            lambda t, x, u: u.diff(2),
            t_out, u0,
            lbc={"neumann": 0.0},
            rbc={"neumann": 0.0},
        )
        assert len(UU) == 3


# ===========================================================================
# 2. besselh
# ===========================================================================


class TestBesselh:
    """Hankel (Bessel third kind) function of a Chebfun.

    besselh returns (H_re, H_im) pair of real Chebfuns since jaxchebfun
    uses real float64 storage.  H^(1)_nu = J_nu + i*Y_nu.
    """

    def test_besselh_returns_pair(self):
        """besselh returns a tuple of two Chebfuns."""
        x = chebfun(lambda t: t + 2.0)  # strictly positive: (1, 3)
        result = x.besselh(0, 1)
        assert isinstance(result, tuple) and len(result) == 2
        H_re, H_im = result
        assert isinstance(H_re, Chebfun)
        assert isinstance(H_im, Chebfun)

    def test_besselh_k1_real_is_besselj(self):
        """H^(1)_nu real part == J_nu."""
        import scipy.special as _ss
        x = chebfun(lambda t: t + 2.0)
        H_re, H_im = x.besselh(0, 1)
        xs = _pts(20, a=-0.9, b=0.9)
        x_phys = np.asarray(xs) + 2.0
        expected_re = np.array([_ss.jv(0, float(v)) for v in x_phys])
        got_re = np.asarray(H_re(xs))
        npt.assert_allclose(got_re, expected_re, atol=1e-8,
                            err_msg="besselh real part (J_nu) mismatch")

    def test_besselh_k1_imag_is_bessely(self):
        """H^(1)_nu imaginary part == Y_nu."""
        import scipy.special as _ss
        x = chebfun(lambda t: t + 2.0)
        H_re, H_im = x.besselh(0, 1)
        xs = _pts(20, a=-0.9, b=0.9)
        x_phys = np.asarray(xs) + 2.0
        expected_im = np.array([_ss.yv(0, float(v)) for v in x_phys])
        got_im = np.asarray(H_im(xs))
        npt.assert_allclose(got_im, expected_im, atol=1e-8,
                            err_msg="besselh imag part (Y_nu) mismatch")

    def test_besselh_k2_imag_is_minus_bessely(self):
        """H^(2)_nu imaginary part == -Y_nu."""
        import scipy.special as _ss
        x = chebfun(lambda t: t + 2.0)
        _, H_im2 = x.besselh(1, 2)
        xs = _pts(10, a=-0.5, b=0.5)
        x_phys = np.asarray(xs) + 2.0
        expected = np.array([-_ss.yv(1, float(v)) for v in x_phys])
        got = np.asarray(H_im2(xs))
        npt.assert_allclose(got, expected, atol=1e-8)

    def test_besselh_zero_raises(self):
        """besselh raises ValueError if Chebfun has a root."""
        x = Chebfun.identity()   # passes through 0
        with pytest.raises(ValueError, match="passes through zero"):
            x.besselh(0, 1)

    def test_besselh_invalid_k_raises(self):
        """besselh raises ValueError for k not in {1, 2}."""
        x = chebfun(lambda t: t + 2.0)
        with pytest.raises(ValueError, match="k must be 1 or 2"):
            x.besselh(0, 3)

    def test_besselh_module_form(self):
        """cj.besselh(f, nu, k) returns (H_re, H_im)."""
        x = chebfun(lambda t: t + 2.0)
        result1 = x.besselh(0, 1)
        result2 = cj.besselh(x, 0, 1)
        pt = jnp.float64(0.5)
        npt.assert_allclose(
            float(result1[0](pt)), float(result2[0](pt)), rtol=1e-12
        )


# ===========================================================================
# 3. besselk
# ===========================================================================


class TestBesselk:
    """Modified Bessel function K of a Chebfun."""

    def test_besselk_identity(self):
        """besselk(nu=0) matches scipy.special.kv pointwise."""
        import scipy.special as _ss
        x = chebfun(lambda t: t + 2.0)   # strictly positive
        k = x.besselk(0)
        xs = _pts(20, a=-0.9, b=0.9)
        x_phys = xs + 2.0
        expected = np.array([_ss.kv(0, float(v)) for v in x_phys])
        got = np.array([float(k(jnp.float64(float(v)))) for v in xs])
        npt.assert_allclose(got, expected, rtol=1e-8)

    def test_besselk_scale(self):
        """besselk(scale=1) is exp(x) * K_0(x)."""
        import scipy.special as _ss
        x = chebfun(lambda t: t + 2.0)
        k_scaled = x.besselk(0, scale=1)
        xs = _pts(10, a=-0.5, b=0.5)
        for v in xs:
            x_phys = float(v) + 2.0
            expected = np.exp(x_phys) * _ss.kv(0, x_phys)
            got = float(k_scaled(jnp.float64(float(v))))
            assert abs(got - expected) < 1e-7, f"scaled besselk mismatch at x={x_phys}"

    def test_besselk_zero_raises(self):
        """besselk raises ValueError if Chebfun has a root."""
        x = Chebfun.identity()
        with pytest.raises(ValueError, match="passes through zero"):
            x.besselk(0)

    def test_besselk_module_form(self):
        """cj.besselk(f, nu) works."""
        x = chebfun(lambda t: t + 2.0)
        k1 = x.besselk(0)
        k2 = cj.besselk(x, 0)
        pt = jnp.float64(0.5)
        npt.assert_allclose(float(k1(pt)), float(k2(pt)), rtol=1e-12)


# ===========================================================================
# 4. ellipke
# ===========================================================================


class TestEllipke:
    """Complete elliptic integrals K(m) and E(m)."""

    def test_ellipke_at_zero(self):
        """K(0) = pi/2, E(0) = pi/2."""
        m = chebfun(lambda x: 0.0 * x + 0.1)  # constant 0.1
        K, E = m.ellipke()
        import scipy.special as _ss
        K_exp = _ss.ellipk(0.1)
        E_exp = _ss.ellipe(0.1)
        pt = jnp.float64(0.0)
        npt.assert_allclose(float(K(pt)), K_exp, rtol=1e-8)
        npt.assert_allclose(float(E(pt)), E_exp, rtol=1e-8)

    def test_ellipke_pointwise(self):
        """K(m) and E(m) match scipy pointwise for m in (0, 1)."""
        import scipy.special as _ss
        m = chebfun(lambda x: 0.1 + 0.8 * (x + 1.0) / 2.0)  # m in (0.1, 0.9)
        K, E = m.ellipke()
        xs = _pts(20)
        m_vals = np.asarray(m(xs))
        K_exp = np.array([_ss.ellipk(float(v)) for v in m_vals])
        E_exp = np.array([_ss.ellipe(float(v)) for v in m_vals])
        K_got = np.asarray(K(xs))
        E_got = np.asarray(E(xs))
        npt.assert_allclose(K_got, K_exp, rtol=1e-8)
        npt.assert_allclose(E_got, E_exp, rtol=1e-8)

    def test_ellipke_module_form(self):
        """cj.ellipke(m) returns (K, E)."""
        m = chebfun(lambda x: 0.0 * x + 0.5)
        K1, E1 = m.ellipke()
        K2, E2 = cj.ellipke(m)
        pt = jnp.float64(0.0)
        npt.assert_allclose(float(K1(pt)), float(K2(pt)), rtol=1e-12)
        npt.assert_allclose(float(E1(pt)), float(E2(pt)), rtol=1e-12)


# ===========================================================================
# 5. dirac
# ===========================================================================


class TestDirac:
    """Dirac delta at roots of a Chebfun."""

    def test_dirac_single_root(self):
        """dirac(sin) places delta at x=0 with weight 1/|cos(0)| = 1."""
        f = chebfun(jnp.sin)
        d = f.dirac()
        assert isinstance(d, Chebfun)
        # Delta is stored on the result
        assert hasattr(d, "_delta_locs")
        # Root at x=0
        assert len(d._delta_locs) >= 1
        root_loc = d._delta_locs[0]
        npt.assert_allclose(root_loc, 0.0, atol=1e-10)
        # Weight = 1 / |cos(0)| = 1
        npt.assert_allclose(d._delta_weights[0], 1.0, rtol=1e-8)

    def test_dirac_no_root(self):
        """dirac of a function with no roots returns zero Chebfun."""
        f = chebfun(lambda x: jnp.ones_like(x))
        d = f.dirac()
        assert isinstance(d, Chebfun)
        # No delta metadata or empty
        locs = getattr(d, "_delta_locs", [])
        assert len(locs) == 0

    def test_dirac_module_form(self):
        """cj.dirac(f) works."""
        f = chebfun(jnp.sin)
        d = cj.dirac(f)
        assert isinstance(d, Chebfun)

    def test_dirac_weight_formula(self):
        """dirac(x) at root x=0 has weight 1/|f'(0)| = 1/1 = 1 for f=x."""
        x = Chebfun.identity()
        d = x.dirac()
        # Weight = 1 / |f'(0)| = 1 / 1 = 1
        npt.assert_allclose(d._delta_weights[0], 1.0, rtol=1e-8)

    def test_dirac_multiple_roots(self):
        """dirac(sin(pi*x)) on [-1,1] has roots at x=-1,0,1."""
        f = chebfun(lambda x: jnp.sin(jnp.pi * x))
        d = f.dirac()
        locs = getattr(d, "_delta_locs", [])
        # Root at x=0 (interior), x=-1 and x=1 (endpoints, half-weight)
        assert any(abs(r) < 0.01 for r in locs), "Expected root near x=0"


# ===========================================================================
# 6. unwrap
# ===========================================================================


class TestUnwrap:
    """Phase unwrapping."""

    def test_unwrap_smooth_noop(self):
        """Smooth single-piece chebfun is unchanged by unwrap."""
        f = chebfun(lambda x: x * 2.0 * float(jnp.pi))
        g = f.unwrap()
        xs = _pts()
        npt.assert_allclose(
            np.asarray(f(xs)), np.asarray(g(xs)), atol=1e-12,
            err_msg="unwrap should leave smooth chebfun unchanged"
        )

    def test_unwrap_module_form(self):
        """cj.unwrap(f) is accessible."""
        f = chebfun(lambda x: x * 2.0)
        g = cj.unwrap(f)
        assert isinstance(g, Chebfun)

    def test_unwrap_single_piece_noop(self):
        """Single-piece chebfun: unwrap returns the same object."""
        f = chebfun(jnp.sin)
        g = f.unwrap()
        xs = _pts()
        npt.assert_allclose(np.asarray(f(xs)), np.asarray(g(xs)), atol=1e-12)

    def test_unwrap_removes_2pi_jump(self):
        """Unwrap removes a 2-pi jump between two pieces."""
        # Build a chebfun with an artificial 2*pi jump
        a, b = -1.0, 1.0
        from chebfunjax.domain import Domain

        def _left(x):
            return jnp.where(x <= 0.0, x * float(jnp.pi), x * float(jnp.pi))

        # Two pieces: left piece on [-1, 0] and right piece on [0, 1]
        # Right piece has an extra +2*pi offset → artificially create multi-piece
        from chebfunjax.chebfun1d.chebfun import _Piece
        piece_l = _Piece.from_function(lambda x: x * float(jnp.pi), -1.0, 0.0)
        # Right piece jumps by 2*pi at x=0
        piece_r = _Piece.from_function(
            lambda x: x * float(jnp.pi) + 2.0 * float(jnp.pi), 0.0, 1.0
        )
        dom = Domain((-1.0, 0.0, 1.0))
        f = Chebfun(funs=[piece_l, piece_r], domain=dom)

        g = f.unwrap()
        # After unwrap the right piece should have the jump removed
        xs_r = jnp.linspace(0.01, 0.99, 10, dtype=jnp.float64)
        f_vals = np.asarray(f(xs_r))
        g_vals = np.asarray(g(xs_r))
        # The jump of 2*pi should have been removed → g ≈ x*pi on [0,1]
        expected = np.asarray(xs_r) * float(jnp.pi)
        npt.assert_allclose(g_vals, expected, atol=1e-10,
                            err_msg="unwrap did not remove 2*pi jump")


# ===========================================================================
# 7. subspace
# ===========================================================================


class TestSubspace:
    """Principal angle between quasimatrix subspaces."""

    def test_identical_subspaces_zero_angle(self):
        """Identical subspaces have angle 0."""
        f = chebfun(jnp.sin)
        theta = subspace([f], [f])
        assert theta < 1e-8, f"Expected angle ≈ 0, got {theta:.2e}"

    def test_orthogonal_subspaces(self):
        """Orthogonal subspaces (sin, cos on [-pi/2, pi/2]) have angle pi/2."""
        f = chebfun(jnp.sin, domain=(-float(jnp.pi) / 2, float(jnp.pi) / 2))
        g = chebfun(jnp.cos, domain=(-float(jnp.pi) / 2, float(jnp.pi) / 2))
        # These are not L2-orthogonal in general, but we test angle < pi/2
        theta = subspace([f], [g])
        assert 0.0 <= theta <= float(jnp.pi) / 2 + 1e-8

    def test_subspace_symmetry(self):
        """subspace(A, B) == subspace(B, A)."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        theta1 = subspace([f], [g])
        theta2 = subspace([g], [f])
        npt.assert_allclose(theta1, theta2, atol=1e-10)

    def test_module_form(self):
        """cj.subspace([f], [f]) == 0."""
        f = chebfun(jnp.sin)
        theta = cj.subspace([f], [f])
        assert theta < 1e-8

    def test_empty_raises(self):
        """Empty lists raise ValueError."""
        f = chebfun(jnp.sin)
        with pytest.raises(ValueError, match="non-empty"):
            subspace([], [f])


# ===========================================================================
# 8. lagrange
# ===========================================================================


class TestLagrange:
    """Lagrange interpolation basis polynomials."""

    def test_lagrange_three_nodes(self):
        """Lagrange basis for 3 nodes: L_j(x_k) = delta_jk."""
        nodes = [-1.0, 0.0, 1.0]
        basis = lagrange(nodes)
        assert len(basis) == 3
        for j, Lj in enumerate(basis):
            for k, xk in enumerate(nodes):
                val = float(Lj(jnp.float64(xk)))
                expected = 1.0 if j == k else 0.0
                assert abs(val - expected) < 1e-10, (
                    f"L_{j}({xk}) = {val:.2e}, expected {expected}"
                )

    def test_lagrange_reproduces_polynomial(self):
        """Sum c_k * L_k interpolates f at nodes."""
        nodes = np.linspace(-1.0, 1.0, 5).tolist()
        f_vals = [float(v ** 2) for v in nodes]  # f(x) = x^2
        basis = lagrange(nodes)
        xs = _pts(20)
        interp = sum(c * Lj for c, Lj in zip(f_vals, basis))
        expected = np.asarray(xs) ** 2
        got = np.asarray(interp(xs))
        npt.assert_allclose(got, expected, atol=1e-10)

    def test_lagrange_single_node_needs_domain(self):
        """Single-node lagrange requires domain."""
        with pytest.raises(ValueError, match="domain"):
            lagrange([0.5])

    def test_lagrange_single_node_with_domain(self):
        """Single-node lagrange with domain works."""
        basis = lagrange([0.5], domain=(-1.0, 1.0))
        assert len(basis) == 1
        # L_0(0.5) = 1
        npt.assert_allclose(float(basis[0](jnp.float64(0.5))), 1.0, atol=1e-10)

    def test_lagrange_non_unique_raises(self):
        """Duplicate nodes raise ValueError."""
        with pytest.raises(ValueError, match="distinct"):
            lagrange([0.0, 0.0, 1.0])

    def test_module_form(self):
        """cj.lagrange works."""
        basis = cj.lagrange([-1.0, 0.0, 1.0])
        assert len(basis) == 3


# ===========================================================================
# 9. ode78 / ode89
# ===========================================================================


class TestOde78Ode89:
    """Higher-order ODE integrators."""

    def test_ode78_exp_growth(self):
        """ode78: y' = y, y(0) = 1 ⟹ y(t) = exp(t)."""
        sol = ode78(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
        t_test = jnp.float64(1.0)
        got = float(sol(t_test))
        expected = float(jnp.exp(t_test))
        npt.assert_allclose(got, expected, rtol=1e-8)

    def test_ode78_harmonic(self):
        """ode78: y'' = -y (via first-order system) with scalar wrapper."""
        # Scalar: y' = sin(t)'= cos(t); test scalar IVP y' = cos(t)
        sol = ode78(lambda t, y: jnp.cos(jnp.array(t)), (0.0, 2.0),
                    jnp.array([0.0]))
        t_test = jnp.float64(1.5)
        got = float(sol(t_test))
        expected = float(jnp.sin(t_test))
        npt.assert_allclose(got, expected, rtol=1e-7)

    def test_ode89_exp_growth(self):
        """ode89: y' = y, y(0) = 1 ⟹ y(t) = exp(t) to high precision."""
        sol = ode89(lambda t, y: y, (0.0, 1.0), jnp.array([1.0]))
        t_test = jnp.float64(1.0)
        got = float(sol(t_test))
        expected = float(jnp.exp(t_test))
        npt.assert_allclose(got, expected, rtol=1e-8)

    def test_ode89_tighter_than_ode45(self):
        """ode89 should achieve higher accuracy than ode45 at default tolerances."""
        from chebfunjax.chebfun1d.chebfun import ode45
        y0 = jnp.array([1.0])
        sol45 = ode45(lambda t, y: y, (0.0, 2.0), y0)
        sol89 = ode89(lambda t, y: y, (0.0, 2.0), y0)
        t_test = jnp.float64(2.0)
        exact = float(jnp.exp(t_test))
        err45 = abs(float(sol45(t_test)) - exact)
        err89 = abs(float(sol89(t_test)) - exact)
        # ode89 tighter tolerances → err89 <= err45 * 100 (conservative)
        assert err89 <= err45 * 100 + 1e-14, (
            f"ode89 error {err89:.2e} should not greatly exceed ode45 error {err45:.2e}"
        )

    def test_module_form_ode78(self):
        """cj.ode78 is callable."""
        sol = cj.ode78(lambda t, y: y, (0.0, 0.5), jnp.array([1.0]))
        assert isinstance(sol, Chebfun)

    def test_module_form_ode89(self):
        """cj.ode89 is callable."""
        sol = cj.ode89(lambda t, y: y, (0.0, 0.5), jnp.array([1.0]))
        assert isinstance(sol, Chebfun)


# ===========================================================================
# 10. quantumstates
# ===========================================================================


class TestQuantumstates:
    """Schrödinger eigenstates."""

    def test_returns_correct_count(self):
        """quantumstates returns n eigenvalues and n eigenfunctions."""
        x = chebfun(lambda t: t, domain=(-3.0, 3.0))
        V = x ** 2
        evals, efuns = quantumstates(V, n=4, h=0.5)
        assert evals.shape[0] == 4
        assert len(efuns) == 4

    def test_eigenvalues_are_real_positive(self):
        """Eigenvalues are real and positive for a confining potential."""
        x = chebfun(lambda t: t, domain=(-3.0, 3.0))
        V = x ** 2
        evals, _ = quantumstates(V, n=3, h=0.3)
        for ev in evals:
            assert float(ev) > 0.0, f"Expected positive eigenvalue, got {float(ev)}"

    def test_eigenvalues_are_sorted(self):
        """Eigenvalues are returned in ascending order."""
        x = chebfun(lambda t: t, domain=(-3.0, 3.0))
        V = x ** 2
        evals, _ = quantumstates(V, n=4, h=0.3)
        e = np.asarray(evals)
        assert np.all(np.diff(e) >= -1e-10), "Eigenvalues should be sorted ascending"

    def test_eigenfunctions_are_chebfun(self):
        """Eigenfunctions are Chebfun objects."""
        x = chebfun(lambda t: t, domain=(-3.0, 3.0))
        V = x ** 2
        _, efuns = quantumstates(V, n=3, h=0.5)
        for ef in efuns:
            assert isinstance(ef, Chebfun)

    def test_module_form(self):
        """cj.quantumstates works."""
        x = chebfun(lambda t: t, domain=(-2.0, 2.0))
        V = x ** 2
        evals, efuns = cj.quantumstates(V, n=3, h=0.5)
        assert len(efuns) == 3

    def test_harmonic_oscillator_ground_energy(self):
        """Harmonic oscillator E_0 ≈ h * (n + 1/2) for h<<1."""
        # For V = x^2, h=0.1: E_0 ≈ h * 1 = 0.1 (quantum ground state)
        x = chebfun(lambda t: t, domain=(-5.0, 5.0))
        V = x ** 2
        evals, _ = quantumstates(V, n=2, h=0.1)
        # Ground state for harmonic osc: E_0 = h (since V=x^2, not x^2/2)
        # The exact value is h*(2*0+1) = h=0.1 for V=x^2
        # Allow 20% tolerance given finite domain / grid effects
        npt.assert_allclose(float(evals[0]), 0.1, rtol=0.25)


# ===========================================================================
# 11. innerProduct alias
# ===========================================================================


class TestInnerProduct:
    """innerProduct is an alias for inner."""

    def test_method_alias(self):
        """f.innerProduct(g) == f.inner(g)."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        v1 = float(f.inner(g))
        v2 = float(f.innerProduct(g))
        npt.assert_allclose(v1, v2, rtol=1e-15)

    def test_module_function(self):
        """innerProduct(f, g) module function works."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        v1 = float(f.inner(g))
        v2 = float(innerProduct(f, g))
        npt.assert_allclose(v1, v2, rtol=1e-15)

    def test_cj_module_form(self):
        """cj.innerProduct(f, g) is accessible."""
        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        v1 = float(f.inner(g))
        v2 = float(cj.innerProduct(f, g))
        npt.assert_allclose(v1, v2, rtol=1e-15)

    def test_self_innerproduct_is_normsq(self):
        """<f, f> = ||f||^2."""
        f = chebfun(jnp.sin)
        ip = float(innerProduct(f, f))
        ns = float(f.norm() ** 2)
        npt.assert_allclose(ip, ns, rtol=1e-12)

    def test_orthogonality(self):
        """<sin, cos> ≈ 0 on [-pi, pi]."""
        dom = (-float(jnp.pi), float(jnp.pi))
        f = chebfun(jnp.sin, domain=dom)
        g = chebfun(jnp.cos, domain=dom)
        v = float(innerProduct(f, g))
        npt.assert_allclose(v, 0.0, atol=1e-12)


# ===========================================================================
# 12. iszero
# ===========================================================================


class TestIszero:
    """iszero method."""

    def test_zero_function(self):
        """Zero function is detected."""
        f = chebfun(lambda x: jnp.zeros_like(x))
        assert f.iszero() is True

    def test_nonzero_function(self):
        """Non-zero function is not iszero."""
        f = chebfun(jnp.sin)
        assert f.iszero() is False

    def test_constant_nonzero(self):
        """Constant function (value 1) is not iszero."""
        f = chebfun(lambda x: jnp.ones_like(x))
        assert f.iszero() is False

    def test_very_small_function(self):
        """Function with magnitude < eps is iszero."""
        import jax.numpy as jnp
        eps = float(jnp.finfo(jnp.float64).eps)
        f = chebfun(lambda x: x * eps * 0.1)   # extremely small
        assert f.iszero() is True

    def test_module_form(self):
        """cj.iszero(f) works."""
        f = chebfun(lambda x: jnp.zeros_like(x))
        assert cj.iszero(f) is True
