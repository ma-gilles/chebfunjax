"""Tests for all new functions added in final/functions-plotting branch.

Covers:
- atan2 (chebfun1d/chebfun.py)
- dst/idst (utils/transforms.py)
- chebpade/trigpade (utils/ratapprox.py)
- trigremez (utils/minimax.py)
- nufft/inufft (utils/nufft.py)
- fred/volt integral operators (operators/integral.py)
- fracInt/fracDiff (chebfun1d/chebfun.py)
- gmres (operators/linop.py)
- polyfitL1 (chebfun1d/chebfun.py)
- plotting: waterfall, roots_plot, spy, plotregion, arrowplot, chebpolyplot
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import matplotlib
import numpy as np
import numpy.testing as npt
import pytest

matplotlib.use("Agg")  # non-interactive backend for testing

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def f_exp():
    """Chebfun for exp(x) on [-1, 1]."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.exp(x), domain=(-1.0, 1.0))


@pytest.fixture
def f_sin():
    """Chebfun for sin(x) on [-pi, pi]."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(jnp.sin, domain=(-np.pi, np.pi))


@pytest.fixture
def f_const_one():
    """Chebfun for the constant 1 on [0, 1]."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda x: jnp.ones_like(x), domain=(0.0, 1.0))


# ===========================================================================
# 1. atan2
# ===========================================================================


class TestAtan2:
    """Tests for the two-argument arctangent Chebfun."""

    def test_basic(self):
        """atan2(sin(x), cos(x)) == x on (-pi, pi)."""
        import chebfunjax as cj
        from chebfunjax.chebfun1d.chebfun import chebfun

        f = chebfun(jnp.sin, domain=(-3.0, 3.0))
        g = chebfun(jnp.cos, domain=(-3.0, 3.0))
        h = cj.atan2(f, g)
        xs = jnp.linspace(-2.9, 2.9, 30)
        err = float(jnp.max(jnp.abs(jnp.arctan2(jnp.sin(xs), jnp.cos(xs)) - h(xs))))
        assert err < 1e-10, f"atan2 error: {err:.2e}"

    def test_returns_chebfun(self):
        """atan2 should return a Chebfun."""
        import chebfunjax as cj
        from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        h = cj.atan2(f, g)
        assert isinstance(h, Chebfun)

    def test_constant_y_positive_x(self):
        """atan2(0, x) = 0 for x > 0."""
        import chebfunjax as cj
        from chebfunjax.chebfun1d.chebfun import chebfun

        zero = chebfun(lambda x: jnp.zeros_like(x))
        pos = chebfun(lambda x: jnp.ones_like(x))
        h = cj.atan2(zero, pos)
        xs = jnp.linspace(-0.9, 0.9, 10)
        err = float(jnp.max(jnp.abs(h(xs))))
        assert err < 1e-10, f"atan2(0, 1) != 0: {err:.2e}"


# ===========================================================================
# 2. DST / IDST
# ===========================================================================


class TestDst:
    """Tests for Discrete Sine Transform."""

    @pytest.mark.parametrize("kind", [1, 2, 3, 4])
    def test_round_trip(self, kind):
        """dst followed by idst should recover the input."""
        from chebfunjax.utils.transforms import dst, idst

        u = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = dst(u, kind)
        u2 = idst(v, kind)
        err = float(jnp.max(jnp.abs(u - u2)))
        assert err < 1e-10, f"DST-{kind} round-trip error: {err:.2e}"

    def test_dst1_self_inverse(self):
        """DST-1 is its own inverse up to a scale factor."""
        from chebfunjax.utils.transforms import dst

        u = jnp.array([1.0, 2.0, 3.0])
        n = len(u)
        v = dst(u, 1)
        # DST-1 of DST-1 = 2*(n+1) * u
        u2 = dst(v, 1)
        scale = 2.0 * (n + 1)
        err = float(jnp.max(jnp.abs(u2 - scale * u)))
        assert err < 1e-10, f"DST-1 self-inverse error: {err:.2e}"

    def test_matches_scipy(self):
        """dst should match scipy.fft.dst."""
        import scipy.fft
        from chebfunjax.utils.transforms import dst

        rng = np.random.default_rng(7)
        u = jnp.array(rng.standard_normal(8))
        for kind in [1, 2, 3, 4]:
            our = np.array(dst(u, kind))
            ref = scipy.fft.dst(np.array(u), type=kind)
            npt.assert_allclose(our, ref, atol=1e-12, err_msg=f"DST-{kind} mismatch")

    @pytest.mark.parametrize("kind", [1, 2, 3, 4])
    def test_2d_input(self, kind):
        """dst should work on 2D arrays (operating along axis=0)."""
        from chebfunjax.utils.transforms import dst, idst

        u = jnp.ones((5, 3))
        v = dst(u, kind)
        u2 = idst(v, kind)
        err = float(jnp.max(jnp.abs(u - u2)))
        assert err < 1e-10, f"DST-{kind} 2D round-trip error: {err:.2e}"


# ===========================================================================
# 3. chebpade
# ===========================================================================


class TestChebpade:
    """Tests for Chebyshev-Padé approximation."""

    def test_exp_accuracy(self, f_exp):
        """[4/4] Chebyshev-Padé of exp(x) should be very accurate."""
        from chebfunjax.utils.ratapprox import chebpade

        _p, _q, r = chebpade(f_exp, 4, 4)
        xs = jnp.linspace(-0.9, 0.9, 30)
        err = float(jnp.max(jnp.abs(jnp.exp(xs) - r(xs))))
        assert err < 1e-8, f"chebpade [4/4] exp error: {err:.2e}"

    def test_returns_callable(self, f_exp):
        """chebpade should return (p_coeffs, q_coeffs, callable)."""
        from chebfunjax.utils.ratapprox import chebpade

        p, q, r = chebpade(f_exp, 2, 2)
        assert callable(r)
        xs = jnp.linspace(-0.8, 0.8, 5)
        vals = r(xs)
        assert vals.shape == (5,)

    def test_maehly_variant_returns_callable(self, f_exp):
        """Maehly variant should return a callable rational approximant."""
        from chebfunjax.utils.ratapprox import chebpade

        _p, _q, r = chebpade(f_exp, 3, 3, kind="maehly")
        assert callable(r)
        # Should evaluate without error
        xs = jnp.linspace(-0.8, 0.8, 5)
        vals = r(xs)
        assert vals.shape == (5,)

    def test_degree_one_one(self):
        """[1/1] Padé of exp should recover first-order approximation."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.utils.ratapprox import chebpade

        f = chebfun(lambda x: jnp.exp(x), domain=(-1.0, 1.0))
        _p, _q, r = chebpade(f, 1, 1)
        # Should be close to exp at x=0
        val_at_0 = float(r(jnp.float64(0.0)))
        assert abs(val_at_0 - 1.0) < 0.1, f"r(0) far from 1: {val_at_0}"


# ===========================================================================
# 4. fred / volt
# ===========================================================================


class TestFredVolt:
    """Tests for Fredholm and Volterra integral operators."""

    def test_fred_linear_kernel(self):
        """fred(K=x-y, f=y) == constant -2/3."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.operators.integral import fred

        K = lambda x, y: x - y
        f = chebfun(lambda x: x, domain=(-1.0, 1.0))
        h = fred(K, f)
        xs = jnp.linspace(-0.9, 0.9, 10)
        ref = jnp.full_like(xs, -2.0 / 3.0)
        err = float(jnp.max(jnp.abs(ref - h(xs))))
        assert err < 1e-10, f"fred error: {err:.2e}"

    def test_fred_identity_kernel(self):
        """fred(K=1, f=cos) == int_{-1}^{1} cos(y) dy = 2*sin(1)."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.operators.integral import fred

        K = lambda x, y: jnp.ones_like(x * y)
        f = chebfun(jnp.cos, domain=(-1.0, 1.0))
        Ff = fred(K, f)
        expected = float(jnp.sin(jnp.float64(1.0)) - jnp.sin(jnp.float64(-1.0)))
        got = float(Ff(jnp.float64(0.0)))
        assert abs(got - expected) < 1e-10, f"fred identity kernel: {abs(got - expected):.2e}"

    def test_volt_constant_kernel(self):
        """volt(K=1, f=1) == x - a on [0, 1]."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.operators.integral import volt

        f = chebfun(lambda x: jnp.ones_like(x), domain=(0.0, 1.0))
        h = volt(lambda x, y: 1.0, f)
        xs = jnp.linspace(0.05, 0.95, 10)
        ref = xs
        err = float(jnp.max(jnp.abs(ref - h(xs))))
        assert err < 1e-10, f"volt error: {err:.2e}"

    def test_volt_at_domain_left(self):
        """volt should return 0 at the left endpoint."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.operators.integral import volt

        f = chebfun(jnp.cos, domain=(0.0, 1.0))
        h = volt(lambda x, y: 1.0, f)
        val = float(h(jnp.float64(0.0)))
        assert abs(val) < 1e-10, f"volt at left endpoint != 0: {val}"

    def test_fred_returns_chebfun(self):
        """fred should return a Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
        from chebfunjax.operators.integral import fred

        f = chebfun(jnp.sin)
        h = fred(lambda x, y: jnp.ones_like(x * y), f)
        assert isinstance(h, Chebfun)


# ===========================================================================
# 5. fracInt / fracDiff
# ===========================================================================


class TestFracCalc:
    """Tests for fractional calculus operators."""

    def test_fracint_integer_order(self, f_const_one):
        """fracInt(1, mu=1) should equal cumsum (primitive integral)."""
        h = f_const_one.fracInt(1.0)
        xs = jnp.linspace(0.05, 0.95, 10)
        ref = xs  # integral of 1 from 0 to x = x
        err = float(jnp.max(jnp.abs(ref - h(xs))))
        assert err < 1e-10, f"fracInt(mu=1) error: {err:.2e}"

    def test_fracint_zero_order(self, f_const_one):
        """fracInt(f, mu=0) should equal f itself."""
        h = f_const_one.fracInt(0.0)
        xs = jnp.linspace(0.05, 0.95, 10)
        ref = jnp.ones_like(xs)
        err = float(jnp.max(jnp.abs(ref - h(xs))))
        assert err < 1e-10, f"fracInt(mu=0) error: {err:.2e}"

    @pytest.mark.slow
    def test_fracint_returns_chebfun(self, f_const_one):
        """fracInt should return a Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = f_const_one.fracInt(0.5)
        assert isinstance(h, Chebfun)

    def test_fracdiff_integer_order(self):
        """fracDiff(x^2, mu=2) should equal 2 (second derivative)."""
        from chebfunjax.chebfun1d.chebfun import chebfun

        f = chebfun(lambda x: x**2, domain=(0.0, 1.0))
        h = f.fracDiff(2.0)
        xs = jnp.linspace(0.1, 0.9, 10)
        ref = 2.0 * jnp.ones_like(xs)
        err = float(jnp.max(jnp.abs(ref - h(xs))))
        assert err < 1e-6, f"fracDiff(x^2, 2) error: {err:.2e}"

    def test_fracint_invalid_mu(self, f_const_one):
        """fracInt should raise for negative mu."""
        with pytest.raises(ValueError, match="mu must be >= 0"):
            f_const_one.fracInt(-0.5)


# ===========================================================================
# 6. gmres
# ===========================================================================


class TestGmres:
    """Tests for GMRES linear solver for Linop."""

    def test_bvp_simple(self):
        """gmres should solve u'' = -1, u(±1) = 0 to high accuracy."""
        from chebfunjax.operators.blocks import D, eval_at
        from chebfunjax.operators.linop import Linop, gmres

        L = Linop(D(order=2), [eval_at(-1.0), eval_at(1.0)],
                  domain=(-1.0, 1.0), bc_values=[0.0, 0.0])
        u = gmres(L, lambda x: -jnp.ones_like(x), n=64)
        xs = jnp.linspace(-0.9, 0.9, 20)
        ref = (1.0 - xs ** 2) / 2.0
        err = float(jnp.max(jnp.abs(ref - u(xs))))
        assert err < 1e-8, f"gmres BVP error: {err:.2e}"

    def test_matches_linop_solve(self):
        """gmres should give same result as Linop.solve."""
        from chebfunjax.operators.blocks import D, eval_at
        from chebfunjax.operators.linop import Linop, gmres

        L = Linop(D(order=2), [eval_at(-1.0), eval_at(1.0)],
                  domain=(-1.0, 1.0), bc_values=[0.0, 0.0])
        rhs = lambda x: jnp.sin(jnp.pi * x)
        u_exact = L.solve(rhs, n=64)
        u_gmres = gmres(L, rhs, n=64)
        xs = jnp.linspace(-0.9, 0.9, 20)
        err = float(jnp.max(jnp.abs(u_exact(xs) - u_gmres(xs))))
        assert err < 1e-6, f"gmres vs solve discrepancy: {err:.2e}"

    def test_returns_chebfun(self):
        """gmres should return a Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun
        from chebfunjax.operators.blocks import D, eval_at
        from chebfunjax.operators.linop import Linop, gmres

        L = Linop(D(order=2), [eval_at(-1.0), eval_at(1.0)],
                  domain=(-1.0, 1.0), bc_values=[0.0, 0.0])
        u = gmres(L, lambda x: -jnp.ones_like(x))
        assert isinstance(u, Chebfun)


# ===========================================================================
# 7. polyfitL1
# ===========================================================================


class TestPolyfitL1:
    """Tests for L1 polynomial fitting."""

    def test_returns_chebfun(self):
        """polyfitL1 should return a Chebfun."""
        from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

        f = chebfun(jnp.sin)
        p = f.polyfitL1(3)
        assert isinstance(p, Chebfun)

    @pytest.mark.slow
    def test_degree_0_constant_approx(self):
        """polyfitL1(f, 0) should return the L1 best constant (median)."""
        from chebfunjax.chebfun1d.chebfun import chebfun

        # For sin(x) on [-1, 1] the L1 median is 0 by symmetry
        f = chebfun(jnp.sin)
        p = f.polyfitL1(0)
        val = float(p(jnp.float64(0.0)))
        assert abs(val) < 0.5, f"L1 constant of sin: {val}"

    @pytest.mark.slow
    def test_smooth_function_lower_error(self):
        """Higher degree should give lower L1 error."""
        from chebfunjax.chebfun1d.chebfun import chebfun

        f = chebfun(lambda x: jnp.abs(x))
        xs = jnp.linspace(-0.9, 0.9, 100)
        fvals = f(xs)
        p3 = f.polyfitL1(3)
        p5 = f.polyfitL1(5)
        err3 = float(jnp.mean(jnp.abs(fvals - p3(xs))))
        err5 = float(jnp.mean(jnp.abs(fvals - p5(xs))))
        assert err5 < err3, "Higher degree polyfitL1 should give lower L1 error"

    def test_evaluable(self):
        """polyfitL1 result should be evaluable at arbitrary points."""
        from chebfunjax.chebfun1d.chebfun import chebfun

        f = chebfun(jnp.cos)
        p = f.polyfitL1(4)
        xs = jnp.linspace(-0.9, 0.9, 20)
        vals = p(xs)
        assert vals.shape == (20,)
        assert not jnp.any(jnp.isnan(vals))


# ===========================================================================
# 8. nufft / inufft
# ===========================================================================


class TestNufft:
    """Tests for non-uniform FFT."""

    def test_uniform_equals_fft(self):
        """nufft with no x argument should equal np.fft.fft."""
        from chebfunjax.utils.nufft import nufft

        c = np.array([1.0, 2.0, 3.0, 4.0], dtype=complex)
        F_our = nufft(c)
        F_np = np.fft.fft(c)
        npt.assert_allclose(F_our, F_np, atol=1e-14)

    def test_type2_single_freq(self):
        """nufft type-2 with c[k]=1 should give exp(-2*pi*i*k*x)."""
        from chebfunjax.utils.nufft import nufft

        N = 16
        c = np.zeros(N, dtype=complex)
        c[3] = 1.0  # mode k=3
        xs = np.array([0.1, 0.3, 0.7])
        F = nufft(c, xs, tol=1e-12)
        ref = np.exp(-2j * np.pi * 3 * xs)
        err = np.max(np.abs(F - ref))
        assert err < 1e-10, f"nufft type-2 single freq error: {err:.2e}"

    def test_type2_random(self):
        """nufft type-2 matches direct computation for random c, x."""
        from chebfunjax.utils.nufft import nufft

        rng = np.random.default_rng(42)
        N = 32
        c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = np.sort(rng.uniform(0, 1, N))
        F_our = nufft(c, x, tol=1e-12)
        F_ref = np.array([
            np.sum(c * np.exp(-2j * np.pi * x[j] * np.arange(N)))
            for j in range(N)
        ])
        err = np.max(np.abs(F_our - F_ref))
        assert err < 1e-10, f"nufft type-2 random error: {err:.2e}"

    def test_inufft_round_trip_small(self):
        """inufft should recover coefficients from nufft output (small N)."""
        from chebfunjax.utils.nufft import inufft, nufft

        rng = np.random.default_rng(7)
        N = 16
        c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = np.sort(rng.uniform(0, 1, N))
        F = nufft(c, x, tol=1e-12)
        c_rec = inufft(F, x, tol=1e-12)
        err = np.max(np.abs(c - c_rec))
        assert err < 1e-8, f"inufft round-trip error: {err:.2e}"

    def test_type1_matches_direct(self):
        """nufft type-1 matches direct computation."""
        from chebfunjax.utils.nufft import nufft

        rng = np.random.default_rng(99)
        N = 16
        c = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x = np.sort(rng.uniform(0, 1, N))
        F_our = nufft(c, x, tol=1e-12, nufft_type=1)
        F_ref = np.array([
            np.sum(c * np.exp(-2j * np.pi * k * x))
            for k in range(N)
        ])
        err = np.max(np.abs(F_our - F_ref))
        assert err < 1e-10, f"nufft type-1 error: {err:.2e}"


# ===========================================================================
# 9. trigpade
# ===========================================================================


class TestTrigpade:
    """Tests for trigonometric Padé approximation."""

    def test_returns_callable(self):
        """trigpade should return (p_coeffs, q_coeffs, callable)."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.utils.ratapprox import trigpade

        f = chebfun(lambda x: 1.0 / (2.0 - jnp.cos(np.pi * x)))
        p, q, r = trigpade(f, 2, 2)
        assert callable(r)

    def test_approximates_rational_trig(self):
        """trigpade should approximate 1/(2-cos(pi*x)) reasonably."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.utils.ratapprox import trigpade

        f = chebfun(lambda x: 1.0 / (2.0 - jnp.cos(np.pi * x)))
        _, _, r = trigpade(f, 3, 3)
        xs = jnp.linspace(-0.8, 0.8, 20)
        ref = 1.0 / (2.0 - jnp.cos(np.pi * xs))
        err = float(jnp.max(jnp.abs(ref - r(xs))))
        # Trigonometric Padé gives moderate accuracy for this function
        assert err < 0.5, f"trigpade approximation error too large: {err:.2e}"


# ===========================================================================
# 10. trigremez
# ===========================================================================


class TestTrigremez:
    """Tests for trigonometric Remez best approximation."""

    def test_returns_result(self):
        """trigremez should return a TrigremezResult."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.utils.minimax import TrigremezResult, trigremez

        f = chebfun(jnp.sin)
        result = trigremez(f, 3)
        assert isinstance(result, TrigremezResult)

    def test_polynomial_degree_leq_m(self):
        """Result should have exactly 2m+1 Fourier coefficients."""
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.utils.minimax import trigremez

        f = chebfun(jnp.sin)
        m = 3
        result = trigremez(f, m)
        assert len(result.coeffs) == 2 * m + 1

    def test_absolute_sin_approximation(self):
        """trigremez of |sin(pi*x)| with m=5 should converge."""
        from chebfunjax.utils.minimax import trigremez

        result = trigremez(lambda x: np.abs(np.sin(np.pi * x)), 5)
        # Error should be less than 1/10 (rough bound)
        assert result.err < 0.1, f"trigremez |sin(pi*x)| error: {result.err:.4f}"
        assert result.iter > 0, "trigremez should run at least one iteration"


# ===========================================================================
# 11. Plotting
# ===========================================================================


class TestPlotting:
    """Smoke tests for new plotting functions."""

    def test_waterfall(self):
        """waterfall should produce a figure without errors."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.plotting import waterfall

        fs = [chebfun(lambda x, t=t: jnp.sin(x + t)) for t in np.linspace(0, 1, 5)]
        fig, ax = waterfall(fs)
        assert fig is not None
        plt.close("all")

    def test_roots_plot(self):
        """roots_plot should produce a figure without errors."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.plotting import roots_plot

        f = chebfun(lambda x: x ** 2 - 0.25)
        fig, ax = roots_plot(f)
        assert fig is not None
        plt.close("all")

    def test_spy(self):
        """spy should produce a figure for a numpy 2D array."""
        import matplotlib.pyplot as plt
        from chebfunjax.plotting import spy

        A = np.eye(8)
        fig, ax = spy(A)
        assert fig is not None
        plt.close("all")

    def test_plotregion(self):
        """plotregion should produce a figure without errors."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.plotting import plotregion

        f = chebfun(lambda x: jnp.exp(-x ** 2))
        fig, ax = plotregion(f)
        assert fig is not None
        plt.close("all")

    def test_arrowplot(self):
        """arrowplot should produce a figure for a parametric curve."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.plotting import arrowplot

        f = chebfun(jnp.cos)
        g = chebfun(jnp.sin)
        fig, ax = arrowplot(f, g)
        assert fig is not None
        plt.close("all")

    def test_chebpolyplot(self):
        """chebpolyplot should produce a figure without errors."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.plotting import chebpolyplot

        f = chebfun(lambda x: jnp.exp(-x ** 2))
        fig, ax = chebpolyplot(f)
        assert fig is not None
        plt.close("all")

    def test_multi_chebfun_plot(self):
        """plot(*args) should accept multiple Chebfuns and produce a figure."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.plotting import plot

        f = chebfun(jnp.sin)
        g = chebfun(jnp.cos)
        fig, ax = plot(f, g)
        assert fig is not None
        plt.close("all")

    def test_plotcoeffs_envelope(self):
        """plotcoeffs with envelope=True should not error."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.plotting import plotcoeffs

        f = chebfun(lambda x: jnp.exp(-x ** 2))
        fig, ax = plotcoeffs(f, envelope=True)
        assert fig is not None
        plt.close("all")

    def test_chebfun_plot_method(self):
        """Chebfun.plot() method should return (fig, ax)."""
        import matplotlib.pyplot as plt
        from chebfunjax.chebfun1d.chebfun import chebfun

        f = chebfun(jnp.sin)
        fig, ax = f.plot()
        assert fig is not None
        plt.close("all")


# ===========================================================================
# 12. Top-level exports
# ===========================================================================


class TestTopLevelExports:
    """Verify all new functions are exported from chebfunjax namespace."""

    def test_atan2_exported(self):
        import chebfunjax as cj
        assert hasattr(cj, "atan2")

    def test_plot_functions_exported(self):
        import chebfunjax as cj
        for name in ["waterfall", "roots_plot", "spy", "plotregion", "arrowplot", "chebpolyplot"]:
            assert hasattr(cj, name), f"cj.{name} not found"

    def test_randnfun_exported(self):
        import chebfunjax as cj
        assert hasattr(cj, "randnfun")
