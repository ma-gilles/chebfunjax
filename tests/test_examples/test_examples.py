"""Pytest tests for all chebfunjax example scripts.

Each test imports the corresponding example module and calls its run()
function, which returns True on success and raises an AssertionError on
any numerical failure.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Ensure examples/ and src/ are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_HERE, '..', '..', 'examples')
_SRC_DIR = os.path.join(_HERE, '..', '..', 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _load(category: str, name: str):
    """Import examples/<category>/<name>.py and return the module."""
    fpath = os.path.join(_EXAMPLES_DIR, category, name + ".py")
    spec = importlib.util.spec_from_file_location(f"ex_{category}_{name}", fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# approx
# ---------------------------------------------------------------------------

class TestApprox:
    def test_polynomial_approximation(self):
        assert _load("approx", "polynomial_approximation").run()

    def test_chebyshev_coefficients(self):
        assert _load("approx", "chebyshev_coefficients").run()

    def test_piecewise_smooth(self):
        assert _load("approx", "piecewise_smooth").run()

    def test_rational_like_convergence(self):
        assert _load("approx", "rational_like_convergence").run()

    def test_bessel_approximation(self):
        assert _load("approx", "bessel_approximation").run()

    def test_hermite_interpolation(self):
        assert _load("approx", "hermite_interpolation").run()

    def test_special_functions(self):
        assert _load("approx", "special_functions").run()

    def test_absolute_value_newton(self):
        assert _load("approx", "absolute_value_newton").run()


# ---------------------------------------------------------------------------
# calc
# ---------------------------------------------------------------------------

class TestCalc:
    def test_definite_indefinite_integrals(self):
        assert _load("calc", "definite_indefinite_integrals").run()

    def test_differentiation(self):
        assert _load("calc", "differentiation").run()

    def test_bird_flight_optimization(self):
        assert _load("calc", "bird_flight_optimization").run()

    def test_mean_value_theorem(self):
        assert _load("calc", "mean_value_theorem").run()

    def test_snells_law(self):
        assert _load("calc", "snells_law").run()


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------

class TestRoots:
    # bessel_roots / newton_raphson / random_polynomials were replaced
    # by the faithful *_replica.py scripts (2026-08); their run()
    # prints the MATLAB-parity outputs and returns None, so these
    # tests assert completion rather than a return value.
    def test_bessel_roots_replica(self):
        _load("roots", "bessel_roots_replica").run()

    def test_newton_raphson_replica(self):
        _load("roots", "newton_raphson_replica").run()

    def test_polynomial_roots(self):
        assert _load("roots", "polynomial_roots").run()

    def test_extrema_and_roots(self):
        assert _load("roots", "extrema_and_roots").run()

    def test_white_curves_replica(self):
        _load("roots", "white_curves_replica").run()


# ---------------------------------------------------------------------------
# quad
# ---------------------------------------------------------------------------

class TestQuad:
    def test_clenshaw_curtis(self):
        assert _load("quad", "clenshaw_curtis").run()

    def test_gauss_quadrature(self):
        assert _load("quad", "gauss_quadrature").run()

    def test_convergence_rates(self):
        assert _load("quad", "convergence_rates").run()

    def test_tricky_integrals(self):
        assert _load("quad", "tricky_integrals").run()


# ---------------------------------------------------------------------------
# ode-linear
# ---------------------------------------------------------------------------

class TestOdeLinear:
    def test_wiki_ode_replica(self):
        # Faithful replica (2026-08): prints MATLAB-parity outputs and
        # returns None; completion is the assertion.
        _load("ode-linear", "wiki_ode_replica").run()

    def test_linear_ivp_replica(self):
        _load("ode-linear", "linear_ivp_replica").run()

    def test_poisson_equation(self):
        assert _load("ode-linear", "poisson_equation").run()

    def test_airy_equation(self):
        assert _load("ode-linear", "airy_equation").run()

    def test_boundary_layer_replica(self):
        _load("ode-linear", "boundary_layer_replica").run()

    def test_bessel_bvp(self):
        assert _load("ode-linear", "bessel_bvp").run()


# ---------------------------------------------------------------------------
# ode-nonlin
# ---------------------------------------------------------------------------

class TestOdeNonlin:
    def test_exact_solutions_bender_orszag(self):
        assert _load("ode-nonlin", "exact_solutions_bender_orszag").run()

    def test_carrier_replica(self):
        # Faithful replica (2026-08): prints MATLAB-parity outputs and
        # returns None; completion is the assertion.
        _load("ode-nonlin", "carrier_replica").run()



# ---------------------------------------------------------------------------
# ode-eig
# ---------------------------------------------------------------------------

class TestOdeEig:
    def test_laplacian_eigenvalues(self):
        assert _load("ode-eig", "laplacian_eigenvalues").run()

    def test_harmonic_oscillator(self):
        assert _load("ode-eig", "harmonic_oscillator").run()

    def test_sturm_liouville(self):
        assert _load("ode-eig", "sturm_liouville").run()

    def test_double_well(self):
        assert _load("ode-eig", "double_well").run()


# ---------------------------------------------------------------------------
# approx2
# ---------------------------------------------------------------------------

class TestApprox2:
    def test_smooth_functions_2d(self):
        assert _load("approx2", "smooth_functions_2d").run()

    def test_rank_of_functions(self):
        assert _load("approx2", "rank_of_functions").run()

    def test_integration_2d(self):
        assert _load("approx2", "integration_2d").run()

    def test_differentiation_2d(self):
        assert _load("approx2", "differentiation_2d").run()


# ---------------------------------------------------------------------------
# opt
# ---------------------------------------------------------------------------

class TestOpt:
    # 2026-08: inspired-by scripts replaced by faithful *_replica.py
    # (replicas print parity outputs and return None; completion is
    # the assertion).
    def test_mercury_earth_replica(self):
        _load("opt", "mercury_earth_replica").run()

    def test_catenary_replica(self):
        _load("opt", "catenary_replica").run()

    def test_global_minimum_replica(self):
        _load("opt", "global_minimum_replica").run()


# ---------------------------------------------------------------------------
# linalg
# ---------------------------------------------------------------------------

class TestLinalg:
    # 2026-08: the inspired-by scripts were replaced by faithful
    # *_replica.py scripts (chebfun_inner_products / inner_products /
    # matrix_functions cited nonexistent chebfun.org originals and
    # were removed).  Replicas print parity outputs and return None,
    # so these tests assert completion.
    def test_cond_nos_replica(self):
        _load("linalg", "cond_nos_replica").run()

    def test_nonnormal_quiz_replica(self):
        _load("linalg", "nonnormal_quiz_replica").run()

    def test_mercury_earth_conjunctions_replica(self):
        _load("linalg", "mercury_earth_conjunctions_replica").run()


# ---------------------------------------------------------------------------
# complex
# ---------------------------------------------------------------------------

class TestComplex:
    def test_contour_integrals(self):
        assert _load("complex", "contour_integrals").run()

    def test_argument_principle(self):
        assert _load("complex", "argument_principle").run()


# ---------------------------------------------------------------------------
# fourier
# ---------------------------------------------------------------------------

class TestFourier:
    def test_fourier_coefficients(self):
        assert _load("fourier", "fourier_coefficients").run()

    def test_gibbs_phenomenon(self):
        assert _load("fourier", "gibbs_phenomenon").run()
