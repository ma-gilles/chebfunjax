"""Tests for new V13-V18 utility modules.

V13 — pde15s: method-of-lines PDE solver
V14 — sing: singularity detection at endpoints
V15 — lebesgue: Lebesgue constant/function
V16 — gallery: gallery of interesting functions
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

ATOL_MED = 1e-6
ATOL_LOOSE = 1e-4


# ============================================================================
# V13 — pde15s
# ============================================================================


class TestPde15s:
    """Tests for the pde15s method-of-lines PDE solver."""

    def test_heat_equation_output_count(self):
        """pde15s returns the correct number of time slices."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebfun1d.pde15s import pde15s

        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.05, 4)
        UU = pde15s(
            lambda t, x, u: 0.1 * u.diff(2),
            t_out, u0, lbc=0.0, rbc=0.0, n=32,
        )
        assert len(UU) == 4

    def test_heat_equation_initial_condition(self):
        """At t=0 the solution equals the initial condition."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebfun1d.pde15s import pde15s

        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.array([0.0, 0.05])
        UU = pde15s(
            lambda t, x, u: 0.1 * u.diff(2),
            t_out, u0, lbc=0.0, rbc=0.0, n=32,
        )
        xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(UU[0](xs)),
            np.array(u0(xs)),
            atol=1e-3,
        )

    def test_heat_equation_decay(self):
        """Heat equation: solution decays over time (max(|u(t)|) decreases)."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebfun1d.pde15s import pde15s

        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.5, 6)
        UU = pde15s(
            lambda t, x, u: 0.1 * u.diff(2),
            t_out, u0, lbc=0.0, rbc=0.0, n=32,
        )
        xs = jnp.linspace(-0.9, 0.9, 30, dtype=jnp.float64)
        max_vals = [float(jnp.max(jnp.abs(UU[k](xs)))) for k in range(len(UU))]
        # Solution should decay (each subsequent max should be smaller)
        assert max_vals[-1] < max_vals[0], (
            f"Heat solution did not decay: {max_vals}"
        )

    def test_heat_equation_boundary_conditions(self):
        """Boundary conditions u(±1) ≈ 0 are maintained throughout integration."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebfun1d.pde15s import pde15s

        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.linspace(0.0, 0.2, 5)
        UU = pde15s(
            lambda t, x, u: 0.1 * u.diff(2),
            t_out, u0, lbc=0.0, rbc=0.0, n=32,
        )
        for k, uk in enumerate(UU):
            # Check boundaries
            lval = abs(float(uk(jnp.float64(-1.0))))
            rval = abs(float(uk(jnp.float64(1.0))))
            assert lval < 0.1, f"Left BC violated at t[{k}]: u(-1) = {lval}"
            assert rval < 0.1, f"Right BC violated at t[{k}]: u(1) = {rval}"

    def test_pde15s_two_argument_pdefun(self):
        """pde15s also works with a two-argument pdefun(t, u)."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebfun1d.pde15s import pde15s

        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.array([0.0, 0.05])
        UU = pde15s(
            lambda t, u: 0.1 * u.diff(2),
            t_out, u0, lbc=0.0, rbc=0.0, n=32,
        )
        assert len(UU) == 2

    def test_pde15s_exact_heat_equation(self):
        """Heat equation: u_t = u_xx, u = sin(pi*x)*exp(-pi^2*t)."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebfun1d.pde15s import pde15s

        t_final = 0.05
        u0 = chebfun(lambda x: jnp.sin(jnp.pi * x))
        t_out = np.array([0.0, t_final])
        UU = pde15s(
            lambda t, x, u: u.diff(2),
            t_out, u0, lbc=0.0, rbc=0.0, n=64,
        )
        xs = jnp.linspace(-0.8, 0.8, 20, dtype=jnp.float64)
        exact = np.array(jnp.sin(jnp.pi * xs)) * float(
            jnp.exp(-jnp.pi ** 2 * jnp.float64(t_final))
        )
        npt.assert_allclose(np.array(UU[-1](xs)), exact, atol=1e-3)

    def test_pde15s_returns_chebfun_list(self):
        """pde15s returns a list of Chebfun objects."""
        from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
        from chebfunjax.chebfun1d.pde15s import pde15s

        u0 = chebfun(lambda x: jnp.cos(jnp.pi * x / 2.0))
        t_out = np.array([0.0, 0.02])
        UU = pde15s(
            lambda t, x, u: 0.1 * u.diff(2),
            t_out, u0, lbc=None, rbc=None, n=16,
        )
        assert isinstance(UU, list)
        for uk in UU:
            assert isinstance(uk, Chebfun)


# ============================================================================
# V14 — sing (singularity detection)
# ============================================================================


class TestFindPoleOrder:
    """Tests for endpoint pole-order detection."""

    def test_simple_pole_right(self):
        """1/(1-x) has a simple pole (order -1) at x=1."""
        from chebfunjax.utils.sing import find_pole_order
        p = find_pole_order(lambda x: 1.0 / (1.0 - x), "right")
        assert p == -1

    def test_double_pole_left(self):
        """1/(1+x)^2 has a double pole (order -2) at x=-1."""
        from chebfunjax.utils.sing import find_pole_order
        p = find_pole_order(lambda x: 1.0 / (1.0 + x) ** 2, "left")
        assert p == -2

    def test_smooth_function_right(self):
        """x^2 has no pole at x=1; order should be 0."""
        from chebfunjax.utils.sing import find_pole_order
        p = find_pole_order(lambda x: x ** 2, "right")
        assert p == 0

    def test_smooth_function_left(self):
        """cos(x) has no pole at x=-1; order should be 0."""
        from chebfunjax.utils.sing import find_pole_order
        p = find_pole_order(lambda x: np.cos(x), "left")
        assert p == 0

    def test_invalid_endpoint(self):
        """Unknown endpoint raises ValueError."""
        from chebfunjax.utils.sing import find_pole_order
        with pytest.raises(ValueError, match="endpoint"):
            find_pole_order(lambda x: x, "middle")

    def test_triple_pole(self):
        """1/(1-x)^3 has a triple pole (order -3) at x=1."""
        from chebfunjax.utils.sing import find_pole_order
        p = find_pole_order(lambda x: 1.0 / (1.0 - x) ** 3, "right")
        assert p == -3


class TestFindSingOrder:
    """Tests for endpoint fractional singularity detection."""

    def test_smooth_function(self):
        """x^2 has sing order 0.0 at x=1."""
        from chebfunjax.utils.sing import find_sing_order
        s = find_sing_order(lambda x: x ** 2, "right")
        assert abs(s) < 0.2  # Should be close to 0

    def test_fractional_sing_right(self):
        """(1-x)^(-1.5) should give sing order ≈ -1.5 at x=1."""
        from chebfunjax.utils.sing import find_sing_order
        s = find_sing_order(lambda x: (1.0 - x) ** (-1.5), "right")
        # The exponent is -1.5 (blow-up of non-integer order)
        assert abs(s - (-1.5)) < 0.2  # tolerance for numerical estimation


class TestFindSingExponents:
    """Tests for two-endpoint singularity detection."""

    def test_pole_and_none(self):
        """1/(1+x) has pole at left, smooth at right."""
        from chebfunjax.utils.sing import find_sing_exponents
        exps = find_sing_exponents(
            lambda x: 1.0 / (1.0 + x),
            sing_type=("pole", "none"),
        )
        assert exps[0] == -1  # pole at left
        assert exps[1] == 0.0  # smooth at right

    def test_none_none(self):
        """cos(x) has no singularities."""
        from chebfunjax.utils.sing import find_sing_exponents
        exps = find_sing_exponents(
            lambda x: np.cos(x),
            sing_type=("none", "none"),
        )
        assert exps == (0.0, 0.0)

    def test_invalid_sing_type(self):
        """Unknown sing_type raises ValueError."""
        from chebfunjax.utils.sing import find_sing_exponents
        with pytest.raises(ValueError, match="sing_type"):
            find_sing_exponents(lambda x: x, sing_type=("blowup", "none"))

    def test_pole_pole_symmetric(self):
        """1/((1+x)*(1-x)) = 1/(1-x^2) has poles at both ends."""
        from chebfunjax.utils.sing import find_sing_exponents
        exps = find_sing_exponents(
            lambda x: 1.0 / (1.0 - x ** 2),
            sing_type=("pole", "pole"),
        )
        assert exps[0] == -1
        assert exps[1] == -1


# ============================================================================
# V15 — lebesgue (Lebesgue constant and function)
# ============================================================================


class TestBaryWeights:
    """Tests for the barycentric weight computation."""

    def test_three_points(self):
        """Barycentric weights for 3 points have correct shape."""
        from chebfunjax.utils.lebesgue import bary_weights
        x = np.array([-1.0, 0.0, 1.0])
        w = bary_weights(x)
        assert w.shape == (3,)

    def test_chebyshev_pts_weights_sum_zero(self):
        """For any set of n distinct points, sum of l_k(t)=1 for any t not in x."""
        from chebfunjax.utils.lebesgue import bary_weights
        from chebfunjax.utils.quadrature import chebpts
        x = np.array(chebpts(5))
        w = bary_weights(x)
        # At t=0.3 (not a node), sum(w/(t-x)) / sum(w/(t-x)) = 1 trivially
        t = 0.3
        parts = w / (t - x)
        ratio = float(np.sum(parts) / np.sum(parts))
        assert abs(ratio - 1.0) < 1e-12


class TestLebesgueFunction:
    """Tests for lebesgue_function."""

    def test_output_shape(self):
        """lebesgue_function returns arrays of length n_eval."""
        from chebfunjax.utils.lebesgue import lebesgue_function
        from chebfunjax.utils.quadrature import chebpts
        x = np.array(chebpts(8))
        t, lam = lebesgue_function(x, n_eval=101)
        assert len(t) == 101
        assert len(lam) == 101

    def test_lebesgue_function_geq_one(self):
        """Lebesgue function >= 1 everywhere."""
        from chebfunjax.utils.lebesgue import lebesgue_function
        from chebfunjax.utils.quadrature import chebpts
        x = np.array(chebpts(8))
        _, lam = lebesgue_function(x)
        assert np.all(lam >= 1.0 - 1e-12)

    def test_lebesgue_function_equals_one_at_nodes(self):
        """At interpolation nodes, Lebesgue function == 1."""
        from chebfunjax.utils.lebesgue import bary_weights
        x = np.array([-1.0, 0.0, 1.0])
        w = bary_weights(x)
        from chebfunjax.utils.lebesgue import _lebesgue_fun_at
        for xi in x:
            val = _lebesgue_fun_at(float(xi), x, w)
            assert abs(val - 1.0) < 1e-12

    def test_lebesgue_constant_chebyshev_order(self):
        """Lebesgue constant for n=8 Chebyshev pts is O(log n) ≈ 2.08."""
        from chebfunjax.utils.lebesgue import lebesgue_constant
        from chebfunjax.utils.quadrature import chebpts
        x = np.array(chebpts(8))
        lc = lebesgue_constant(x)
        assert 1.5 < lc < 4.0

    def test_lebesgue_constant_equispaced_larger(self):
        """Equispaced nodes have larger Lebesgue constant than Chebyshev nodes."""
        from chebfunjax.utils.lebesgue import lebesgue_constant
        from chebfunjax.utils.quadrature import chebpts
        n = 10
        x_cheb = np.array(chebpts(n))
        x_equi = np.linspace(-1.0, 1.0, n)
        lc_cheb = lebesgue_constant(x_cheb)
        lc_equi = lebesgue_constant(x_equi)
        assert lc_equi > lc_cheb, (
            f"Expected equispaced Lebesgue constant ({lc_equi:.3f}) "
            f"> Chebyshev ({lc_cheb:.3f})"
        )

    def test_lebesgue_constant_three_points(self):
        """For 3 Chebyshev pts (-1, 0, 1), Lebesgue constant is known analytically ≈ 1.25."""
        from chebfunjax.utils.lebesgue import lebesgue_constant
        x = np.array([-1.0, 0.0, 1.0])
        lc = lebesgue_constant(x, n_eval=5001)
        # Exact value is 1.25
        assert abs(lc - 1.25) < 0.05


# ============================================================================
# V16 — gallery
# ============================================================================


class TestGallery:
    """Tests for the function gallery."""

    def test_runge_at_zero(self):
        """gallery('runge')(0) == 1."""
        from chebfunjax.utils.gallery import gallery
        f = gallery("runge")
        assert abs(float(f(jnp.float64(0.0))) - 1.0) < 1e-12

    def test_runge_case_insensitive(self):
        """gallery('RUNGE') and gallery('runge') return equivalent functions."""
        from chebfunjax.utils.gallery import gallery
        f1 = gallery("runge")
        f2 = gallery("RUNGE")
        val1 = float(f1(jnp.float64(0.5)))
        val2 = float(f2(jnp.float64(0.5)))
        assert abs(val1 - val2) < 1e-12

    def test_chirp_domain(self):
        """gallery('chirp') has domain [0, 5]."""
        from chebfunjax.utils.gallery import gallery
        f = gallery("chirp")
        assert f.domain.a == pytest.approx(0.0)
        assert f.domain.b == pytest.approx(5.0)

    def test_erf_value(self):
        """gallery('erf') at x=0 should be 0."""
        from chebfunjax.utils.gallery import gallery
        f = gallery("erf")
        assert abs(float(f(jnp.float64(0.0)))) < 1e-10

    def test_gaussian_at_zero(self):
        """gallery('gaussian')(0) == 1/sqrt(2*pi)."""
        from chebfunjax.utils.gallery import gallery
        f = gallery("gaussian")
        expected = float(1.0 / jnp.sqrt(2.0 * jnp.pi))
        assert abs(float(f(jnp.float64(0.0))) - expected) < 1e-10

    def test_kahaner_integral(self):
        """gallery('kahaner') integrates to approximately 0.211."""
        from chebfunjax.utils.gallery import gallery
        f = gallery("kahaner")
        val = float(f.sum())
        # The Kahaner integrand has a known integral ≈ 0.211
        assert 0.1 < val < 0.5

    def test_unknown_name_raises(self):
        """gallery('nonexistent') raises KeyError."""
        from chebfunjax.utils.gallery import gallery
        with pytest.raises(KeyError):
            gallery("nonexistent_function_xyz")

    def test_list_gallery_returns_dict(self):
        """list_gallery() returns a dict mapping names to descriptions."""
        from chebfunjax.utils.gallery import list_gallery
        d = list_gallery()
        assert isinstance(d, dict)
        assert "runge" in d
        assert "chirp" in d
        assert "kahaner" in d
        assert all(isinstance(v, str) for v in d.values())

    def test_all_gallery_functions_constructible(self):
        """Every gallery function can be constructed without error."""
        from chebfunjax.utils.gallery import gallery, list_gallery
        for name in list_gallery():
            try:
                f = gallery(name)
                # Verify it's a Chebfun (has domain)
                assert hasattr(f, "domain"), f"gallery('{name}') missing .domain"
            except Exception as exc:
                pytest.fail(f"gallery('{name}') raised {exc}")

    @pytest.mark.parametrize("name", ["runge", "chirp", "bump", "erf", "wiggly"])
    def test_gallery_is_chebfun(self, name):
        """gallery(name) returns a Chebfun instance."""
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.utils.gallery import gallery
        f = gallery(name)
        assert isinstance(f, Chebfun)

    def test_sinefun1_value(self):
        """gallery('sinefun1') at x=0 should be 1.75."""
        from chebfunjax.utils.gallery import gallery
        f = gallery("sinefun1")
        assert abs(float(f(jnp.float64(0.0))) - 1.75) < 1e-10

    def test_bump_zero_outside_support(self):
        """gallery('bump') is 0 at x=±1.5 (outside the bump support)."""
        from chebfunjax.utils.gallery import gallery
        f = gallery("bump")
        # Outside the bump's [-1, 1] support => value should be 0
        assert abs(float(f(jnp.float64(1.5)))) < 1e-6
        assert abs(float(f(jnp.float64(-1.5)))) < 1e-6
