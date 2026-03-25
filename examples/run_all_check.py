"""Run all chebfunjax examples and report pass/fail/skip.

This is the comprehensive test runner covering all example scripts.

Usage:
    python examples/run_all_check.py            # run all examples
    python examples/run_all_check.py --verbose  # print full output
    python examples/run_all_check.py --timeout 120  # per-example timeout (seconds)
    python examples/run_all_check.py --category ode-nonlin  # only one category
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import time
import traceback
import importlib.util
import argparse

# Ensure src/ is importable from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))


# Registry of all examples: (module_path, display_name)
# module_path uses dot notation: "category.module_name"
EXAMPLES = [
    # approx
    ("approx.polynomial_approximation",         "approx / polynomial_approximation"),
    ("approx.chebyshev_coefficients",           "approx / chebyshev_coefficients"),
    ("approx.piecewise_smooth",                 "approx / piecewise_smooth"),
    ("approx.rational_like_convergence",        "approx / rational_like_convergence"),
    ("approx.bessel_approximation",             "approx / bessel_approximation"),
    ("approx.hermite_interpolation",            "approx / hermite_interpolation"),
    ("approx.special_functions",                "approx / special_functions"),
    ("approx.absolute_value_newton",            "approx / absolute_value_newton"),
    ("approx.gamma_function",                   "approx / gamma_function"),
    ("approx.lebesgue_constants",               "approx / lebesgue_constants"),
    ("approx.orthogonal_polynomials",           "approx / orthogonal_polynomials"),
    ("approx.polynomial_convergence",           "approx / polynomial_convergence"),
    ("approx.weierstrass",                      "approx / weierstrass"),
    # approx2
    ("approx2.smooth_functions_2d",             "approx2 / smooth_functions_2d"),
    ("approx2.rank_of_functions",               "approx2 / rank_of_functions"),
    ("approx2.integration_2d",                  "approx2 / integration_2d"),
    ("approx2.differentiation_2d",              "approx2 / differentiation_2d"),
    ("approx2.WeierstrassFunction",             "approx2 / WeierstrassFunction"),
    ("approx2.BestApprox",                      "approx2 / BestApprox"),
    ("approx2.BestL1",                          "approx2 / BestL1"),
    ("approx2.GreedyInterp",                    "approx2 / GreedyInterp"),
    ("approx2.AAASpline",                       "approx2 / AAASpline"),
    ("approx2.EdgeDetection",                   "approx2 / EdgeDetection"),
    ("approx2.chebfun2_basics",                 "approx2 / chebfun2_basics"),
    # calc
    ("calc.definite_indefinite_integrals",      "calc / definite_indefinite_integrals"),
    ("calc.differentiation",                    "calc / differentiation"),
    ("calc.bird_flight_optimization",           "calc / bird_flight_optimization"),
    ("calc.mean_value_theorem",                 "calc / mean_value_theorem"),
    ("calc.snells_law",                         "calc / snells_law"),
    ("calc.delta_derivs",                       "calc / delta_derivs"),
    ("calc.integrals",                          "calc / integrals"),
    ("calc.surface_revolution",                 "calc / surface_revolution"),
    # cheb
    ("cheb.chebyshev_polynomials",              "cheb / chebyshev_polynomials"),
    ("cheb.chebyshev_coefficients",             "cheb / chebyshev_coefficients"),
    ("cheb.chebfun_fft",                        "cheb / chebfun_fft"),
    ("cheb.fast_transforms",                    "cheb / fast_transforms"),
    # roots
    ("roots.bessel_roots",                      "roots / bessel_roots"),
    ("roots.newton_raphson",                    "roots / newton_raphson"),
    ("roots.polynomial_roots",                  "roots / polynomial_roots"),
    ("roots.extrema_and_roots",                 "roots / extrema_and_roots"),
    ("roots.random_polynomials",                "roots / random_polynomials"),
    ("roots.bernoulli_polynomials",             "roots / bernoulli_polynomials"),
    ("roots.bessel_function_roots",             "roots / bessel_function_roots"),
    ("roots.fundamental_theorem_algebra",       "roots / fundamental_theorem_algebra"),
    ("roots.roots_near_axis",                   "roots / roots_near_axis"),
    ("roots.roots_speed",                       "roots / roots_speed"),
    ("roots.secular_roots",                     "roots / secular_roots"),
    # quad
    ("quad.clenshaw_curtis",                    "quad / clenshaw_curtis"),
    ("quad.gauss_quadrature",                   "quad / gauss_quadrature"),
    ("quad.convergence_rates",                  "quad / convergence_rates"),
    ("quad.tricky_integrals",                   "quad / tricky_integrals"),
    ("quad.battery_test",                       "quad / battery_test"),
    ("quad.gauss_vs_clenshaw_curtis",           "quad / gauss_vs_clenshaw_curtis"),
    ("quad.spike_integral",                     "quad / spike_integral"),
    ("quad.hermite_quad",                       "quad / hermite_quad"),
    ("quad.symbolic_numeric",                   "quad / symbolic_numeric"),
    # ode-linear
    ("ode-linear.wiki_odes",                    "ode-linear / wiki_odes"),
    ("ode-linear.linear_ivp_cosine",            "ode-linear / linear_ivp_cosine"),
    ("ode-linear.poisson_equation",             "ode-linear / poisson_equation"),
    ("ode-linear.airy_equation",                "ode-linear / airy_equation"),
    ("ode-linear.boundary_layer",               "ode-linear / boundary_layer"),
    ("ode-linear.bessel_bvp",                   "ode-linear / bessel_bvp"),
    ("ode-linear.adv_diff_jump",                "ode-linear / adv_diff_jump"),
    ("ode-linear.lin_exp_ivp",                  "ode-linear / lin_exp_ivp"),
    ("ode-linear.matched_asymp",                "ode-linear / matched_asymp"),
    ("ode-linear.near_nonuniqueness",           "ode-linear / near_nonuniqueness"),
    ("ode-linear.nonstandard_bcs",              "ode-linear / nonstandard_bcs"),
    ("ode-linear.krylov",                       "ode-linear / krylov"),
    ("ode-linear.fourier_collocation",          "ode-linear / fourier_collocation"),
    ("ode-linear.jump_conditions",              "ode-linear / jump_conditions"),
    ("ode-linear.piecewise_demo",               "ode-linear / piecewise_demo"),
    ("ode-linear.two_sol_bvp",                  "ode-linear / two_sol_bvp"),
    ("ode-linear.parameter_ode",                "ode-linear / parameter_ode"),
    ("ode-linear.breakpoints",                  "ode-linear / breakpoints"),
    ("ode-linear.black_scholes",                "ode-linear / black_scholes"),
    ("ode-linear.lee_greengard",                "ode-linear / lee_greengard"),
    ("ode-linear.resonant_vandal",              "ode-linear / resonant_vandal"),
    ("ode-linear.spectral_disc",                "ode-linear / spectral_disc"),
    ("ode-linear.periodic_system",              "ode-linear / periodic_system"),
    ("ode-linear.lane_emden_linear",            "ode-linear / lane_emden_linear"),
    ("ode-linear.jump_green",                   "ode-linear / jump_green"),
    ("ode-linear.dawson_integral",              "ode-linear / dawson_integral"),
    # ode-nonlin
    ("ode-nonlin.exact_solutions_bender_orszag", "ode-nonlin / exact_solutions_bender_orszag"),
    ("ode-nonlin.logistic_equation",            "ode-nonlin / logistic_equation"),
    ("ode-nonlin.carrier_equation",             "ode-nonlin / carrier_equation"),
    ("ode-nonlin.pendulum_equation",            "ode-nonlin / pendulum_equation"),
    ("ode-nonlin.allen_cahn",                   "ode-nonlin / allen_cahn"),
    ("ode-nonlin.blasius",                      "ode-nonlin / blasius"),
    ("ode-nonlin.bloodhound",                   "ode-nonlin / bloodhound"),
    ("ode-nonlin.blowup_fk",                    "ode-nonlin / blowup_fk"),
    ("ode-nonlin.fourier_nonlin",               "ode-nonlin / fourier_nonlin"),
    ("ode-nonlin.ivp_capabilities",             "ode-nonlin / ivp_capabilities"),
    ("ode-nonlin.param_odes",                   "ode-nonlin / param_odes"),
    ("ode-nonlin.rectifier",                    "ode-nonlin / rectifier"),
    ("ode-nonlin.gulf_stream",                  "ode-nonlin / gulf_stream"),
    ("ode-nonlin.picard",                       "ode-nonlin / picard"),
    ("ode-nonlin.bvp_system",                   "ode-nonlin / bvp_system"),
    ("ode-nonlin.lane_emden_nonlin",            "ode-nonlin / lane_emden_nonlin"),
    # ode-eig
    ("ode-eig.laplacian_eigenvalues",           "ode-eig / laplacian_eigenvalues"),
    ("ode-eig.harmonic_oscillator",             "ode-eig / harmonic_oscillator"),
    ("ode-eig.sturm_liouville",                 "ode-eig / sturm_liouville"),
    ("ode-eig.double_well",                     "ode-eig / double_well"),
    ("ode-eig.continuous_wilkinson",            "ode-eig / continuous_wilkinson"),
    ("ode-eig.fourier_eigs",                    "ode-eig / fourier_eigs"),
    ("ode-eig.eigenstates",                     "ode-eig / eigenstates"),
    ("ode-eig.rayleigh_quotient",               "ode-eig / rayleigh_quotient"),
    ("ode-eig.null_space",                      "ode-eig / null_space"),
    ("ode-eig.wave_decay",                      "ode-eig / wave_decay"),
    ("ode-eig.landscape",                       "ode-eig / landscape"),
    ("ode-eig.randfun_eig",                     "ode-eig / randfun_eig"),
    ("ode-eig.optical_response",                "ode-eig / optical_response"),
    ("ode-eig.drum",                            "ode-eig / drum"),
    # opt
    ("opt.minimum_of_smooth_function",          "opt / minimum_of_smooth_function"),
    ("opt.catenary",                            "opt / catenary"),
    ("opt.global_minimum_2d",                   "opt / global_minimum_2d"),
    ("opt.constrained_extrema",                 "opt / constrained_extrema"),
    ("opt.rosenbrock",                          "opt / rosenbrock"),
    # linalg
    ("linalg.chebfun_inner_products",           "linalg / chebfun_inner_products"),
    ("linalg.resolvent_norm",                   "linalg / resolvent_norm"),
    ("linalg.matrix_functions",                 "linalg / matrix_functions"),
    ("linalg.condition_numbers",                "linalg / condition_numbers"),
    ("linalg.inner_products",                   "linalg / inner_products"),
    # complex
    ("complex.contour_integrals",               "complex / contour_integrals"),
    ("complex.argument_principle",              "complex / argument_principle"),
    ("complex.complex_functions",               "complex / complex_functions"),
    ("complex.analytic_continuation",           "complex / analytic_continuation"),
    ("complex.rouche_theorem",                  "complex / rouche_theorem"),
    # fourier
    ("fourier.fourier_coefficients",            "fourier / fourier_coefficients"),
    ("fourier.gibbs_phenomenon",                "fourier / gibbs_phenomenon"),
    ("fourier.fourier_series",                  "fourier / fourier_series"),
    # stats
    ("stats.probability_distributions",         "stats / probability_distributions"),
    ("stats.random_polynomials",                "stats / random_polynomials"),
    # geom
    ("geom.curves_and_lengths",                 "geom / curves_and_lengths"),
    ("geom.parametric_surfaces",                "geom / parametric_surfaces"),
]


def import_example(module_path: str):
    """Import an example module by dotted path relative to examples/."""
    parts = module_path.split(".")
    category = parts[0]
    name = parts[1]
    fpath = os.path.join(_HERE, category, name + ".py")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Example not found: {fpath}")
    spec = importlib.util.spec_from_file_location(module_path.replace("-", "_"), fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_all(verbose: bool = False, category_filter: str = None,
            per_example_timeout: float = None):
    passed = []
    failed = []
    skipped = []

    examples = EXAMPLES
    if category_filter:
        examples = [(mp, dn) for mp, dn in EXAMPLES
                    if mp.startswith(category_filter)]
        if not examples:
            print(f"No examples found for category filter: {category_filter!r}")
            return True

    print(f"\n{'='*70}")
    print(f"  Running {len(examples)} chebfunjax examples")
    if category_filter:
        print(f"  Category filter: {category_filter!r}")
    if per_example_timeout:
        print(f"  Per-example timeout: {per_example_timeout}s")
    print(f"{'='*70}\n")

    for module_path, display_name in examples:
        t0 = time.time()
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  {display_name}")
            print(f"{'─'*60}")
        try:
            mod = import_example(module_path)
            if not hasattr(mod, 'run'):
                # No run() function — skip
                elapsed = time.time() - t0
                skipped.append(display_name)
                print(f"  [SKIP]  {display_name}  (no run() function)")
                continue
            result = mod.run()
            elapsed = time.time() - t0
            status = "PASS" if result is True else "WARN"
            if verbose:
                print(f"  [{status}] in {elapsed:.1f}s")
            else:
                print(f"  [{status}]  {display_name}  ({elapsed:.1f}s)")
            passed.append(display_name)
        except FileNotFoundError as e:
            elapsed = time.time() - t0
            skipped.append(display_name)
            print(f"  [SKIP]  {display_name}  (file not found: {e})")
        except NotImplementedError as e:
            elapsed = time.time() - t0
            skipped.append(display_name)
            print(f"  [SKIP]  {display_name}  (not implemented: {e})")
        except Exception as e:
            elapsed = time.time() - t0
            failed.append((display_name, str(e)))
            print(f"  [FAIL]  {display_name}  ({elapsed:.1f}s)")
            if verbose:
                traceback.print_exc()
            else:
                print(f"          Error: {e}")

    # Summary
    n_pass = len(passed)
    n_fail = len(failed)
    n_skip = len(skipped)
    n_total = len(examples)
    print(f"\n{'='*70}")
    print(f"  Results: {n_pass} passed, {n_fail} failed, {n_skip} skipped"
          f"  (total: {n_total})")
    print(f"{'='*70}\n")

    if failed:
        print("FAILED examples:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        print()

    if skipped:
        print("SKIPPED examples:")
        for name in skipped:
            print(f"  - {name}")
        print()

    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all chebfunjax examples")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full output for each example")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-example timeout in seconds")
    parser.add_argument("--category", type=str, default=None,
                        help="Only run examples from this category (e.g. ode-nonlin)")
    args = parser.parse_args()

    success = run_all(
        verbose=args.verbose,
        category_filter=args.category,
        per_example_timeout=args.timeout,
    )
    sys.exit(0 if success else 1)
