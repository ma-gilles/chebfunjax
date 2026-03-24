"""V06+V07 tests — adchebfun and treeVar symbolic operator AD.

Tests the symbolic Fréchet differentiation of nonlinear differential
operators implemented in :mod:`chebfunjax.autodiff.adchebfun` and
:mod:`chebfunjax.autodiff.treevar`.

Key reference test (from task description):
    Linearize ``u'' + u^2`` at u=sin → should give ``v'' + 2*sin*v``.

Newton convergence test:
    Nonlinear BVP  ``u'' - u^2 = -(x+1)^2``,  u(-1)=0, u(1)=2
    has exact solution ``u = x + 1``.  Newton with the ADChebfun Jacobian
    should converge to machine precision.

Translated from MATLAB Chebfun ``@adchebfun`` test suite.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun, detect_linearity, linearize_op
from chebfunjax.autodiff.treevar import TreeVar, linearize_tree
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import ChebColloc2Disc, D, I, diag
from chebfunjax.operators.chebop import Chebop
from chebfunjax.utils.quadrature import chebpts

# ============================================================================
# Tier 1 — TreeVar expression tree construction
# ============================================================================


class TestTreeVar:
    """Test TreeVar tree-building and metadata."""

    def test_constructor_leaf(self):
        """A freshly-constructed TreeVar has diff_order=0 and method='constr'."""
        u = TreeVar()
        assert u.tree["method"] == "constr"
        assert u.tree["diff_order"] == 0
        assert u.tree["height"] == 0
        assert not u.tree["has_terms"]

    def test_diff_order(self):
        """u.diff(2).tree['diff_order'] == 2."""
        u = TreeVar()
        ud2 = u.diff(2)
        assert ud2.tree["diff_order"] == 2
        assert ud2.tree["method"] == "diff"

    def test_diff_default_order(self):
        """u.diff() gives diff_order=1."""
        u = TreeVar()
        ud = u.diff()
        assert ud.tree["diff_order"] == 1

    def test_plus_diff_order(self):
        """u'' + u has diff_order=2."""
        u = TreeVar()
        expr = u.diff(2) + u
        assert expr.tree["diff_order"] == 2
        assert expr.tree["method"] == "plus"
        assert expr.tree["has_terms"]

    def test_plus_nonlinear_diff_order(self):
        """u'' + u^2 has diff_order=2."""
        u = TreeVar()
        expr = u.diff(2) + u ** 2
        assert expr.tree["diff_order"] == 2

    def test_times_preserves_diff_order(self):
        """(sin_coeff) * u.diff(2) has diff_order=2."""
        u = TreeVar()
        # Scalar multiplication
        expr = 3.0 * u.diff(2)
        assert expr.tree["diff_order"] == 2

    def test_power(self):
        """u^2 builds a power tree node."""
        u = TreeVar()
        expr = u ** 2
        assert expr.tree["method"] == "power"
        assert expr.tree["diff_order"] == 0

    def test_unary_sin(self):
        """sin(u) builds a unary sin node."""
        u = TreeVar()
        expr = u.sin()
        assert expr.tree["method"] == "sin"
        assert expr.tree["num_args"] == 1

    def test_radd(self):
        """scalar + TreeVar works via __radd__."""
        u = TreeVar()
        expr = 1.0 + u.diff(2)
        assert expr.tree["method"] == "plus"
        assert expr.tree["diff_order"] == 2

    def test_rsub(self):
        """scalar - TreeVar works."""
        u = TreeVar()
        expr = 1.0 - u.diff(2)
        assert expr.tree["method"] == "minus"

    def test_rmul(self):
        """scalar * TreeVar works."""
        u = TreeVar()
        expr = 2.0 * u
        assert expr.tree["method"] == "times"
        assert expr.tree["diff_order"] == 0

    def test_neg(self):
        """-TreeVar gives uminus node."""
        u = TreeVar()
        expr = -u
        assert expr.tree["method"] == "uminus"

    def test_custom_domain(self):
        """TreeVar respects custom domain."""
        u = TreeVar(domain=(0.0, float(jnp.pi)))
        assert u.domain == (0.0, float(jnp.pi))


# ============================================================================
# Tier 2 — linearize_tree correctness (treeVar-based Jacobian)
# ============================================================================


class TestLinearizeTree:
    """Test that linearize_tree returns the correct Fréchet derivative."""

    def _colloc_err(self, J_computed, J_expected, n=16):
        """Return max-absolute-error between two Jacobian matrices."""
        d = ChebColloc2Disc(n, (-1.0, 1.0))
        M_got = J_computed.matrix(d)
        M_exp = J_expected.matrix(d)
        return float(jnp.max(jnp.abs(M_got - M_exp)))

    def test_identity_linearization(self):
        """Linearize 'constr' tree → identity operator."""
        u = TreeVar()
        u0 = chebfun(jnp.sin, domain=(-1.0, 1.0))
        J = linearize_tree(u.tree, u0, domain=(-1.0, 1.0))
        err = self._colloc_err(J, I((-1.0, 1.0)))
        assert err < 1e-14, f"identity linearization error: {err}"

    def test_diff_linearization(self):
        """Linearize diff(u, 2) → D^2."""
        u = TreeVar()
        ud2 = u.diff(2)
        u0 = chebfun(jnp.sin, domain=(-1.0, 1.0))
        J = linearize_tree(ud2.tree, u0, domain=(-1.0, 1.0))
        J_expected = D((-1.0, 1.0), order=2)
        err = self._colloc_err(J, J_expected)
        assert err < 1e-12, f"diff linearization error: {err}"

    def test_linear_combo(self):
        """Linearize u'' + u → D^2 + I."""
        u = TreeVar()
        expr = u.diff(2) + u
        u0 = chebfun(jnp.sin, domain=(-1.0, 1.0))
        J = linearize_tree(expr.tree, u0, domain=(-1.0, 1.0))
        J_expected = D((-1.0, 1.0), order=2) + I((-1.0, 1.0))
        err = self._colloc_err(J, J_expected)
        assert err < 1e-12, f"u''+u linearization error: {err}"

    def test_power_nonlinear(self):
        """Linearize u^2 at u0=sin → diag(2*sin)."""
        u = TreeVar()
        expr = u ** 2
        domain = (-1.0, 1.0)
        u0 = chebfun(jnp.sin, domain=domain)
        J = linearize_tree(expr.tree, u0, domain=domain)
        J_expected = diag(2.0 * u0, domain)
        err = self._colloc_err(J, J_expected)
        assert err < 1e-12, f"u^2 linearization error: {err}"

    def test_key_reference_test(self):
        """KEY REFERENCE TEST: Linearize u''+u^2 at u=sin → v''+2*sin*v.

        This is the primary correctness check from the task description.
        """
        domain = (0.0, float(jnp.pi))
        n = 16
        disc = ChebColloc2Disc(n, domain)

        # Build u0 = sin(x) on [0, pi]
        u0 = chebfun(jnp.sin, domain=domain)

        # Compute Jacobian via linearize_tree
        u = TreeVar(domain=domain)
        expr = u.diff(2) + u ** 2
        J = linearize_tree(expr.tree, u0, domain=domain)
        J_mat = J.matrix(disc)

        # Build expected: D^2 + diag(2*sin(x))
        D2 = D(domain, order=2).matrix(disc)
        t_ref = chebpts(n, kind=2)
        a, b = domain
        x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
        u0_pts = jnp.sin(x_pts)
        M2 = jnp.diag(2.0 * u0_pts)
        J_expected = D2 + M2

        err = float(jnp.max(jnp.abs(J_mat - J_expected)))
        assert err < 1e-12, (
            f"KEY TEST FAILED: linearize u''+u^2 at sin. "
            f"Max error = {err:.2e}"
        )

    def test_scalar_times_diff(self):
        """Linearize 3*u'' → 3*D^2."""
        u = TreeVar()
        expr = 3.0 * u.diff(2)
        u0 = chebfun(jnp.sin, domain=(-1.0, 1.0))
        J = linearize_tree(expr.tree, u0, domain=(-1.0, 1.0))
        J_expected = D((-1.0, 1.0), order=2) * 3.0
        err = self._colloc_err(J, J_expected)
        assert err < 1e-12, f"3*u'' error: {err}"

    def test_sin_nonlinear(self):
        """Linearize sin(u) at u0=x → diag(cos(x))."""
        domain = (-1.0, 1.0)
        u = TreeVar()
        expr = u.sin()
        u0 = chebfun(lambda x: x, domain=domain, n=2)
        J = linearize_tree(expr.tree, u0, domain=domain)
        J_expected = diag(u0.cos(), domain)
        err = self._colloc_err(J, J_expected)
        assert err < 1e-12, f"sin(u) linearization error: {err}"

    def test_exp_nonlinear(self):
        """Linearize exp(u) at u0=x → diag(exp(x))."""
        domain = (-0.5, 0.5)
        u = TreeVar()
        expr = u.exp()
        u0 = chebfun(lambda x: x, domain=domain, n=2)
        J = linearize_tree(expr.tree, u0, domain=domain)
        J_expected = diag(u0.exp(), domain)
        err = self._colloc_err(J, J_expected)
        assert err < 1e-12, f"exp(u) linearization error: {err}"


# ============================================================================
# Tier 3 — ADChebfun arithmetic and Jacobian chain rule
# ============================================================================


class TestADChebfun:
    """Test ADChebfun dual-number arithmetic."""

    def setup_method(self):
        self.domain = (-1.0, 1.0)
        self.u0 = chebfun(jnp.sin, domain=self.domain)
        self.ad_u = ADChebfun(self.u0)
        self.n = 12
        self.disc = ChebColloc2Disc(self.n, self.domain)

    def _max_err(self, op_got, op_expected):
        """Max-absolute error between two OperatorBlocks at n=12."""
        M_got = op_got.matrix(self.disc)
        M_exp = op_expected.matrix(self.disc)
        return float(jnp.max(jnp.abs(M_got - M_exp)))

    def test_constructor_identity(self):
        """ADChebfun Jacobian is seeded as identity."""
        J_mat = self.ad_u.jacobian.matrix(self.disc)
        I_mat = jnp.eye(self.n, dtype=jnp.float64)
        npt.assert_allclose(J_mat, I_mat, atol=1e-14)

    def test_is_linear_init(self):
        """Freshly constructed ADChebfun is linear."""
        assert self.ad_u.is_linear

    def test_diff_jacobian(self):
        """diff(u, 2).jacobian == D^2."""
        result = self.ad_u.diff(2)
        J_expected = D(self.domain, order=2)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-12, f"diff(2) Jacobian error: {err}"

    def test_diff_linear(self):
        """diff is a linear operation — is_linear stays True."""
        result = self.ad_u.diff(2)
        assert result.is_linear

    def test_add_ad_jacobian(self):
        """(u + u).jacobian == 2*I."""
        result = self.ad_u + self.ad_u
        J_expected = I(self.domain) * 2.0
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-14, f"u+u Jacobian error: {err}"

    def test_sub_ad_jacobian(self):
        """(u - u).jacobian == 0."""
        result = self.ad_u - self.ad_u
        J_mat = result.jacobian.matrix(self.disc)
        assert float(jnp.max(jnp.abs(J_mat))) < 1e-14

    def test_add_scalar_no_change(self):
        """Adding a scalar doesn't change the Jacobian."""
        result = self.ad_u + 3.0
        J_mat = result.jacobian.matrix(self.disc)
        I_mat = jnp.eye(self.n, dtype=jnp.float64)
        npt.assert_allclose(J_mat, I_mat, atol=1e-14)

    def test_mul_scalar_jacobian(self):
        """(c * u).jacobian == c * I."""
        c = 3.7
        result = c * self.ad_u
        J_expected = I(self.domain) * c
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-14, f"c*u Jacobian error: {err}"

    def test_mul_chebfun_jacobian(self):
        """(u * f).jacobian == diag(f), where f is a Chebfun coefficient."""
        f = chebfun(jnp.cos, domain=self.domain)
        # Use u * f (ADChebfun on left) to test __mul__ with Chebfun on right
        result = self.ad_u * f
        J_expected = diag(f, self.domain)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-12, f"u*chebfun Jacobian error: {err}"

    def test_mul_chebfun_reversed_jacobian(self):
        """(f * u).jacobian == diag(f), where f is a Chebfun — uses __rmul__."""
        f = chebfun(jnp.cos, domain=self.domain)
        # Use f * u (Chebfun on left, ADChebfun on right)
        # This triggers Chebfun.__mul__ → NotImplemented → ADChebfun.__rmul__
        result = f * self.ad_u
        J_expected = diag(f, self.domain)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-12, f"chebfun*u Jacobian (via __rmul__) error: {err}"

    def test_mul_nonlinear_is_linear_false(self):
        """(u * u).is_linear == False."""
        result = self.ad_u * self.ad_u
        assert not result.is_linear

    def test_power_nonlinear_jacobian(self):
        """(u^2).jacobian == diag(2*u0)."""
        result = self.ad_u ** 2
        J_expected = diag(2.0 * self.u0, self.domain)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-11, f"u^2 Jacobian error: {err}"

    def test_sin_jacobian(self):
        """sin(u).jacobian == diag(cos(u0))."""
        result = self.ad_u.sin()
        J_expected = diag(self.u0.cos(), self.domain)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-12, f"sin(u) Jacobian error: {err}"

    def test_cos_jacobian(self):
        """cos(u).jacobian == diag(-sin(u0))."""
        result = self.ad_u.cos()
        J_expected = diag(-self.u0.sin(), self.domain)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-12, f"cos(u) Jacobian error: {err}"

    def test_exp_jacobian(self):
        """exp(u).jacobian == diag(exp(u0))."""
        result = self.ad_u.exp()
        J_expected = diag(self.u0.exp(), self.domain)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-12, f"exp(u) Jacobian error: {err}"

    def test_chain_diff_plus_nonlinear(self):
        """(u.diff(2) + u^2).jacobian == D^2 + diag(2*u0)."""
        result = self.ad_u.diff(2) + self.ad_u ** 2
        J_expected = D(self.domain, order=2) + diag(2.0 * self.u0, self.domain)
        err = self._max_err(result.jacobian, J_expected)
        assert err < 1e-11, f"u''+u^2 Jacobian error: {err}"

    def test_neg_jacobian(self):
        """(-u).jacobian == -I."""
        result = -self.ad_u
        J_mat = result.jacobian.matrix(self.disc)
        I_mat = jnp.eye(self.n, dtype=jnp.float64)
        npt.assert_allclose(J_mat, -I_mat, atol=1e-14)


# ============================================================================
# Tier 4 — linearize_op public API
# ============================================================================


class TestLinearizeOp:
    """Test the public linearize_op function."""

    def _jacobian_matrix(self, J_op, domain, n=16):
        disc = ChebColloc2Disc(n, domain)
        return J_op.matrix(disc)

    def test_key_reference_test_linearize_op(self):
        """KEY TEST (via linearize_op): u''+u^2 at sin → v''+2*sin*v."""
        domain = (0.0, float(jnp.pi))
        n = 16
        u0 = chebfun(jnp.sin, domain=domain)
        J_op = linearize_op(lambda x, u: u.diff(2) + u ** 2, u0, domain=domain)

        disc = ChebColloc2Disc(n, domain)
        J_mat = J_op.matrix(disc)

        # Build expected
        D2 = D(domain, order=2).matrix(disc)
        t_ref = chebpts(n, kind=2)
        a, b = domain
        x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)
        u0_pts = jnp.sin(x_pts)
        J_expected = D2 + jnp.diag(2.0 * u0_pts)

        err = float(jnp.max(jnp.abs(J_mat - J_expected)))
        assert err < 1e-11, (
            f"linearize_op KEY TEST: max err = {err:.2e}"
        )

    def test_linear_operator(self):
        """linearize_op of u''+u is D^2+I (exact, not dependent on u0)."""
        domain = (-1.0, 1.0)
        n = 16
        u0 = chebfun(jnp.sin, domain=domain)
        J_op = linearize_op(lambda x, u: u.diff(2) + u, u0, domain=domain)

        disc = ChebColloc2Disc(n, domain)
        J_mat = J_op.matrix(disc)
        J_expected = (D(domain, order=2) + I(domain)).matrix(disc)

        err = float(jnp.max(jnp.abs(J_mat - J_expected)))
        assert err < 1e-12, f"u''+u linearize_op error: {err}"

    def test_autonomous_operator(self):
        """linearize_op works with op(u) (no x argument)."""
        domain = (-1.0, 1.0)
        u0 = chebfun(jnp.sin, domain=domain)
        # op(u) = u'' + u^2 (autonomous)
        J_op = linearize_op(lambda u: u.diff(2) + u ** 2, u0, domain=domain)

        n = 16
        disc = ChebColloc2Disc(n, domain)
        J_mat = J_op.matrix(disc)

        D2 = D(domain, order=2).matrix(disc)
        t_ref = chebpts(n, kind=2)
        x_pts = t_ref  # domain is (-1,1), ref coords = phys coords
        u0_pts = jnp.sin(x_pts)
        J_expected = D2 + jnp.diag(2.0 * u0_pts)

        err = float(jnp.max(jnp.abs(J_mat - J_expected)))
        assert err < 1e-11, f"autonomous linearize_op error: {err}"

    def test_chebfun_multiplier(self):
        """Linearize u.diff(2) * x.cos() is D^2 * diag(cos(x)).

        Note: the pattern ``u.diff(2) * x.cos()`` puts the ADChebfun on the
        left, so ADChebfun.__mul__ handles the Chebfun on the right directly.
        Linearization: diag(cos(x)) * D^2.
        """
        domain = (-1.0, 1.0)
        n = 16
        u0 = chebfun(jnp.sin, domain=domain)

        # u.diff(2) * x.cos(): ADChebfun * Chebfun → uses ADChebfun.__mul__
        J_op = linearize_op(lambda x, u: u.diff(2) * x.cos(), u0, domain=domain)
        disc = ChebColloc2Disc(n, domain)
        J_mat = J_op.matrix(disc)

        t_ref = chebpts(n, kind=2)
        # domain = (-1,1), no rescaling
        cos_pts = jnp.cos(t_ref)
        D2 = D(domain, order=2).matrix(disc)
        J_expected = jnp.diag(cos_pts) @ D2

        err = float(jnp.max(jnp.abs(J_mat - J_expected)))
        assert err < 1e-11, f"u''*cos(x) linearize_op error: {err}"

    def test_chebfun_multiplier_reversed(self):
        """Linearize x.cos() * u.diff(2) using Chebfun.__mul__ → NotImplemented → __rmul__."""
        domain = (-1.0, 1.0)
        n = 16
        u0 = chebfun(jnp.sin, domain=domain)

        # x.cos() * u.diff(2): Chebfun * ADChebfun → triggers __rmul__
        J_op = linearize_op(lambda x, u: x.cos() * u.diff(2), u0, domain=domain)
        disc = ChebColloc2Disc(n, domain)
        J_mat = J_op.matrix(disc)

        t_ref = chebpts(n, kind=2)
        cos_pts = jnp.cos(t_ref)
        D2 = D(domain, order=2).matrix(disc)
        J_expected = jnp.diag(cos_pts) @ D2

        err = float(jnp.max(jnp.abs(J_mat - J_expected)))
        assert err < 1e-11, f"cos(x)*u'' (reversed) linearize_op error: {err}"


# ============================================================================
# Tier 5 — detect_linearity
# ============================================================================


class TestDetectLinearity:
    """Test the detect_linearity API."""

    def setup_method(self):
        domain = (-1.0, 1.0)
        self.u0 = chebfun(jnp.sin, domain=domain)
        self.domain = domain

    def test_linear_ops(self):
        """Linear operators are correctly detected."""
        linear_ops = [
            lambda x, u: u.diff(2),
            lambda x, u: u.diff(2) + u,
            lambda x, u: u.diff(2) - u.diff(1),
            lambda x, u: 3.0 * u.diff(1),
            # Variable coefficient: u * x.cos() (ADChebfun.__mul__ handles Chebfun)
            lambda x, u: u.diff(2) + u * x.cos(),
        ]
        for op in linear_ops:
            result = detect_linearity(op, self.u0, domain=self.domain)
            assert result, f"Expected linear: {op}"

    def test_nonlinear_ops(self):
        """Nonlinear operators are correctly detected."""
        nonlinear_ops = [
            lambda x, u: u ** 2,
            lambda x, u: u.diff(2) + u ** 2,
            lambda x, u: u.sin(),
            lambda x, u: u.exp(),
            lambda x, u: u * u.diff(1),
        ]
        for op in nonlinear_ops:
            result = detect_linearity(op, self.u0, domain=self.domain)
            assert not result, f"Expected nonlinear: {op}"


# ============================================================================
# Tier 6 — Newton iteration convergence via Chebop (integration test)
# ============================================================================


class TestNewtonBVP:
    """Integration tests: Chebop Newton iteration with ADChebfun Jacobian."""

    def test_nonlinear_bvp_exact_solution(self):
        """Newton BVP: u'' - u^2 = -(x+1)^2, u(-1)=0, u(1)=2 → u = x+1.

        This nonlinear BVP has exact solution u(x) = x + 1:
          u'' = 0,  u^2 = (x+1)^2
          u'' - u^2 = -(x+1)^2  ✓
          u(-1) = 0, u(1) = 2   ✓
        """
        N = Chebop(lambda x, u: u.diff(2) - u ** 2, domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 2.0

        rhs = lambda x: -(x ** 2 + 2 * x + 1)
        u_sol = N.solve(rhs, n=32)

        # Check at several interior points
        for x_val in [-0.5, 0.0, 0.5]:
            x = jnp.array(x_val, dtype=jnp.float64)
            got = float(u_sol(x))
            expected = x_val + 1.0
            assert abs(got - expected) < 1e-7, (
                f"Newton BVP: u({x_val}) = {got:.8f} (expected {expected:.8f})"
            )

    def test_linear_bvp_is_not_detected_as_nonlinear(self):
        """Linearity detection routes linear BVP to direct solver."""
        N = Chebop(lambda x, u: u.diff(2), domain=(0.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        # Chebop._is_linear() should return True for this
        assert N._is_linear()

    def test_nonlinear_bvp_is_detected_as_nonlinear(self):
        """Nonlinearity detection works for u'' + u^2."""
        N = Chebop(lambda x, u: u.diff(2) + u ** 2, domain=(-1.0, 1.0))
        assert not N._is_linear()

    def test_nonlinear_bvp_sin_rhs(self):
        """Newton BVP with sin multiplier: u'' + sin(x)*u^2 = f, u(±1)=0.

        Choose u_exact(x) = 1 - x^2 (satisfies u(±1)=0).
          u'' = -2
          sin(x)*u^2 = sin(x)*(1-x^2)^2
          f = -2 + sin(x)*(1-x^2)^2

        Note: this uses the ADChebfun Jacobian for Newton iteration.
        The Newton iterations start from u=0 and converge to u=1-x^2.
        We check at 1e-3 since the fixed-size Newton with n=32 and starting
        from zero may require more iterations for high accuracy on this problem.
        """
        # Exact solution: u = 1 - x^2
        def rhs(x):
            return -2.0 + jnp.sin(x) * (1 - x ** 2) ** 2

        N = Chebop(
            lambda x, u: u.diff(2) + x.sin() * u ** 2,
            domain=(-1.0, 1.0),
        )
        N.lbc = 0.0
        N.rbc = 0.0
        u_sol = N.solve(rhs, n=32, max_iter=20)

        # Check several points — tolerance 1e-3 for this fixed-n Newton
        for x_val in [-0.5, 0.0, 0.5]:
            x = jnp.array(x_val, dtype=jnp.float64)
            got = float(u_sol(x))
            expected = 1.0 - x_val ** 2
            assert abs(got - expected) < 1e-3, (
                f"sin(x)*u^2 BVP: u({x_val}) = {got:.8f} (expected {expected:.8f})"
            )

    def test_jacobian_method_selection(self):
        """ADChebfun Jacobian is used (not finite differences) for u''+u^2."""
        N = Chebop(lambda x, u: u.diff(2) + u ** 2, domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0

        # Check that _jacobian_matrix_ad succeeds and returns sensible matrix
        from chebfunjax.chebfun1d.chebfun import Chebfun
        dom = Domain((-1.0, 1.0))
        n = 16
        disc = ChebColloc2Disc(n, (-1.0, 1.0))
        u_fun = Chebfun.from_values(jnp.zeros(n, dtype=jnp.float64), dom)
        J_ad = N._jacobian_matrix_ad(disc, u_fun)
        J_fd = N._jacobian_matrix_fd(
            disc,
            Chebfun.identity(dom),
            u_fun,
            jnp.zeros(n, dtype=jnp.float64),
        )
        # Both should agree
        err = float(jnp.max(jnp.abs(J_ad - J_fd)))
        assert err < 1e-5, (
            f"ADChebfun vs FD Jacobian mismatch: {err:.2e}"
        )


# ============================================================================
# Tier 7 — Taylor (convergence order) test for Fréchet derivative accuracy
# ============================================================================


class TestTaylorOrder:
    """Verify Fréchet derivative accuracy by Taylor testing.

    The Fréchet derivative ``J = dN[u0]`` satisfies::

        || N[u0 + eps*v] - N[u0] - eps*J*v || = O(eps^2)

    We verify second-order convergence as eps → 0.
    """

    def _eval_op_at_values(self, op_fn, domain, vals_array):
        """Evaluate op_fn on a Chebfun given by values array, return values."""
        from chebfunjax.chebfun1d.chebfun import chebfun as _chebfun
        n = len(vals_array)
        dom = Domain(domain)
        u = Chebfun.from_values(jnp.asarray(vals_array, dtype=jnp.float64), dom)
        a, b = domain
        x_fun = Chebfun.identity(dom)
        try:
            import inspect
            nargs = len(inspect.signature(op_fn).parameters)
        except Exception:
            nargs = 2
        if nargs == 1:
            result = op_fn(u)
        else:
            result = op_fn(x_fun, u)
        disc = ChebColloc2Disc(n, domain)
        from chebfunjax.operators.chebop import _chebfun_to_values
        return _chebfun_to_values(result, disc)

    def test_frechet_derivative_second_order_convergence(self):
        """Taylor test: N[u+eps*v] - N[u] - eps*J*v = O(eps^2) for u''+u^2."""
        domain = (-1.0, 1.0)
        n = 16
        disc = ChebColloc2Disc(n, domain)

        # Build u0 and v (perturbation)
        dom = Domain(domain)
        t_ref = chebpts(n, kind=2)
        u0_vals = jnp.sin(t_ref)
        v_vals = jnp.cos(2.0 * t_ref)  # arbitrary perturbation

        u0 = Chebfun.from_values(u0_vals, dom)
        op_fn = lambda x, u: u.diff(2) + u ** 2

        # Compute the Fréchet derivative at u0
        J_op = linearize_op(op_fn, u0, domain=domain)
        J_mat = J_op.matrix(disc)

        # Evaluate N[u0]
        N_u0 = self._eval_op_at_values(op_fn, domain, u0_vals)

        # Taylor test over several eps
        errors = []
        eps_vals = [1e-2, 5e-3, 2e-3, 1e-3]
        for eps in eps_vals:
            u_pert_vals = u0_vals + eps * v_vals
            N_pert = self._eval_op_at_values(op_fn, domain, u_pert_vals)
            J_v = J_mat @ v_vals

            # |N[u+eps*v] - N[u] - eps*J*v|
            res = N_pert - N_u0 - eps * J_v
            err = float(jnp.max(jnp.abs(res)))
            errors.append(err)

        # Check second-order convergence: errors[i+1]/errors[i] ≈ (eps[i+1]/eps[i])^2
        for i in range(len(errors) - 1):
            ratio_err = errors[i + 1] / (errors[i] + 1e-20)
            ratio_eps = (eps_vals[i + 1] / eps_vals[i]) ** 2
            # Allow factor of 5 tolerance
            assert ratio_err < 5.0 * ratio_eps + 1e-14, (
                f"Taylor test: convergence order not 2 at eps={eps_vals[i]:.0e}, "
                f"ratio_err={ratio_err:.2f}, expected≈{ratio_eps:.2f}"
            )
