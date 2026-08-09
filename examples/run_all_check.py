"""Run all chebfunjax examples and report pass/fail/skip.

This is the comprehensive test runner covering all example scripts.

Usage:
    python examples/run_all_check.py            # run all examples
    python examples/run_all_check.py --verbose  # print full output
    python examples/run_all_check.py --timeout 120  # per-example timeout (seconds)
    python examples/run_all_check.py --category ode-nonlin  # only one category
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import sys
import time
import traceback

# Ensure src/ is importable from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))


# Registry of all examples: (module_path, display_name)
# module_path uses dot notation: "category.module_name"
EXAMPLES = [
    # approx
    # NOTE: WeierstrassFunction, BestApprox, BestL1, GreedyInterp, AAASpline,
    # and EdgeDetection were previously listed here under approx2, but those
    # are 1D examples whose real files live in examples/approx/ (no approx2
    # file ever existed, so they were silently skipped). They are now picked
    # up automatically by auto-discovery under the `approx` category.
    # calc
    # cheb
    ("cheb.chebyshev_coefficients",             "cheb / chebyshev_coefficients"),
    ("cheb.fast_transforms",                    "cheb / fast_transforms"),
    # roots
    ("roots.bessel_roots",                      "roots / bessel_roots"),
    ("roots.newton_raphson",                    "roots / newton_raphson"),
    ("roots.random_polynomials",                "roots / random_polynomials"),
    ("roots.bessel_function_roots",             "roots / bessel_function_roots"),
    ("roots.fundamental_theorem_algebra",       "roots / fundamental_theorem_algebra"),
    ("roots.roots_near_axis",                   "roots / roots_near_axis"),
    ("roots.roots_speed",                       "roots / roots_speed"),
    ("roots.secular_roots",                     "roots / secular_roots"),
    # quad
    # ode-linear
    ("ode-linear.wiki_odes",                    "ode-linear / wiki_odes"),
    ("ode-linear.linear_ivp_cosine",            "ode-linear / linear_ivp_cosine"),
    ("ode-linear.boundary_layer",               "ode-linear / boundary_layer"),
    ("ode-linear.adv_diff_jump",                "ode-linear / adv_diff_jump"),
    ("ode-linear.lin_exp_ivp",                  "ode-linear / lin_exp_ivp"),
    ("ode-linear.matched_asymp",                "ode-linear / matched_asymp"),
    ("ode-linear.near_nonuniqueness",           "ode-linear / near_nonuniqueness"),
    ("ode-linear.nonstandard_bcs",              "ode-linear / nonstandard_bcs"),
    ("ode-linear.krylov",                       "ode-linear / krylov"),
    ("ode-linear.fourier_collocation",          "ode-linear / fourier_collocation"),
    ("ode-linear.parameter_ode",                "ode-linear / parameter_ode"),
    ("ode-linear.breakpoints",                  "ode-linear / breakpoints"),
    ("ode-linear.lee_greengard",                "ode-linear / lee_greengard"),
    ("ode-linear.resonant_vandal",              "ode-linear / resonant_vandal"),
    ("ode-linear.spectral_disc",                "ode-linear / spectral_disc"),
    ("ode-linear.periodic_system",              "ode-linear / periodic_system"),
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
    # fourier
    # stats
    ("stats.probability_distributions",         "stats / probability_distributions"),
    ("stats.random_polynomials",                "stats / random_polynomials"),
    # geom
    ("geom.curves_and_lengths",                 "geom / curves_and_lengths"),
    ("geom.parametric_surfaces",                "geom / parametric_surfaces"),
]


import glob
import re
import signal

# Basenames (no .py) that live under examples/<category>/ but are NOT runnable
# example scripts (harness/helper modules). Auto-discovery skips these.
_EXCLUDE_BASENAMES = {
    "__init__",
    "run_all",
    "run_all_check",
    "generate_all_plots",
    "continuous_skeletonization_study",
}

_RUN_DEF_RE = re.compile(r"^\s*def\s+run\s*\(", re.MULTILINE)


class _ExampleTimeout(Exception):
    """Raised when a single example exceeds the per-example timeout."""


def _has_run_function(fpath: str) -> bool:
    """Source-scan for a top-level ``def run(`` without importing the module."""
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            return bool(_RUN_DEF_RE.search(fh.read()))
    except OSError:
        return False


def discover_examples():
    """Return an ordered list of ``(filepath, display_name, category)`` records
    covering EVERY example script under ``examples/<category>/*.py``.

    Ordering: registered examples (that still exist on disk) first, in their
    registry order, then every other discovered script sorted by category and
    name. Non-example helpers (see ``_EXCLUDE_BASENAMES``) are omitted.
    """
    records = []
    seen = set()

    # 1. Registered examples, in registry order, if the file exists.
    for module_path, display_name in EXAMPLES:
        category, name = module_path.split(".", 1)
        fpath = os.path.join(_HERE, category, name + ".py")
        key = os.path.abspath(fpath)
        if os.path.exists(fpath) and key not in seen:
            records.append((fpath, display_name, category))
            seen.add(key)

    # 2. Everything else discovered one level deep under examples/.
    extra = []
    for fpath in glob.glob(os.path.join(_HERE, "*", "*.py")):
        base = os.path.splitext(os.path.basename(fpath))[0]
        if base in _EXCLUDE_BASENAMES:
            continue
        key = os.path.abspath(fpath)
        if key in seen:
            continue
        category = os.path.basename(os.path.dirname(fpath))
        extra.append((fpath, f"{category} / {base}", category))
        seen.add(key)
    extra.sort(key=lambda rec: (rec[2], rec[1]))

    return records + extra


def import_example_from_path(fpath: str):
    """Import an example module from its file path with a collision-free name."""
    base = os.path.splitext(os.path.basename(fpath))[0]
    category = os.path.basename(os.path.dirname(fpath))
    unique = f"example_{category}_{base}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(unique, fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_all(verbose: bool = False, category_filter: str = None,
            per_example_timeout: float = None):
    passed = []
    failed = []
    skipped = []
    no_run = []
    timed_out = []

    examples = discover_examples()
    if category_filter:
        examples = [rec for rec in examples if rec[2].startswith(category_filter)]
        if not examples:
            print(f"No examples found for category filter: {category_filter!r}")
            return True

    use_alarm = bool(per_example_timeout) and hasattr(signal, "SIGALRM")

    def _alarm_handler(signum, frame):
        raise _ExampleTimeout()

    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)

    print(f"\n{'='*70}")
    print(f"  Running {len(examples)} chebfunjax examples")
    if category_filter:
        print(f"  Category filter: {category_filter!r}")
    if per_example_timeout:
        tail = "" if use_alarm else " (unsupported on this platform, ignored)"
        print(f"  Per-example timeout: {per_example_timeout}s{tail}")
    print(f"{'='*70}\n")

    for fpath, display_name, _category in examples:
        t0 = time.time()
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  {display_name}")
            print(f"{'─'*60}")

        # Classify scripts without a run() function up front (no import).
        if not _has_run_function(fpath):
            no_run.append(display_name)
            print(f"  [NO_RUN_FUNC]  {display_name}  (no run() function)")
            continue

        if use_alarm:
            signal.alarm(int(per_example_timeout))
        try:
            mod = import_example_from_path(fpath)
            if not hasattr(mod, "run"):
                no_run.append(display_name)
                print(f"  [NO_RUN_FUNC]  {display_name}  (no run() function)")
                continue
            result = mod.run()
            elapsed = time.time() - t0
            status = "PASS" if result is True else "WARN"
            if verbose:
                print(f"  [{status}] in {elapsed:.1f}s")
            else:
                print(f"  [{status}]  {display_name}  ({elapsed:.1f}s)")
            passed.append(display_name)
        except _ExampleTimeout:
            elapsed = time.time() - t0
            timed_out.append((display_name, per_example_timeout))
            print(f"  [TIMEOUT]  {display_name}  (>{per_example_timeout:.0f}s)")
        except NotImplementedError as e:
            skipped.append(display_name)
            print(f"  [SKIP]  {display_name}  (not implemented: {e})")
        except FileNotFoundError as e:
            skipped.append(display_name)
            print(f"  [SKIP]  {display_name}  (file not found: {e})")
        except Exception as e:
            elapsed = time.time() - t0
            failed.append((display_name, str(e)))
            print(f"  [FAIL]  {display_name}  ({elapsed:.1f}s)")
            if verbose:
                traceback.print_exc()
            else:
                print(f"          Error: {e}")
        finally:
            if use_alarm:
                signal.alarm(0)

    # Summary
    n_pass = len(passed)
    n_fail = len(failed)
    n_skip = len(skipped)
    n_norun = len(no_run)
    n_timeout = len(timed_out)
    n_total = len(examples)
    print(f"\n{'='*70}")
    print(f"  Results: {n_pass} passed, {n_fail} failed, {n_skip} skipped, "
          f"{n_timeout} timed out, {n_norun} no run()  (total: {n_total})")
    print(f"{'='*70}\n")

    if failed:
        print("FAILED examples:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        print()

    if timed_out:
        print("TIMED OUT examples:")
        for name, limit in timed_out:
            print(f"  - {name}  (>{limit:.0f}s)")
        print()

    if skipped:
        print("SKIPPED examples:")
        for name in skipped:
            print(f"  - {name}")
        print()

    if no_run:
        print("NO_RUN_FUNC examples (no run() function, not executed):")
        for name in no_run:
            print(f"  - {name}")
        print()

    return n_fail == 0 and n_timeout == 0


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
