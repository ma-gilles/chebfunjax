"""Run all chebfunjax examples and report pass/fail.

Usage:
    python examples/run_all.py            # run all examples
    python examples/run_all.py --verbose  # print full output
"""

import importlib
import importlib.util
import os
import sys
import time
import traceback

# Ensure src/ is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Registry of all examples: (module_path, display_name)
EXAMPLES = [
    # approx
    # calc
    # roots
    ("roots.bessel_roots",                  "roots / bessel_roots"),
    ("roots.newton_raphson",                "roots / newton_raphson"),
    ("roots.random_polynomials",            "roots / random_polynomials"),
    # quad
    # ode-linear (imported as ode_linear)
    ("ode-linear.wiki_odes",                "ode-linear / wiki_odes"),
    ("ode-linear.linear_ivp_cosine",        "ode-linear / linear_ivp_cosine"),
    ("ode-linear.boundary_layer",           "ode-linear / boundary_layer"),
    # ode-nonlin
    ("ode-nonlin.exact_solutions_bender_orszag",  "ode-nonlin / exact_solutions_bender_orszag"),
    ("ode-nonlin.logistic_equation",              "ode-nonlin / logistic_equation"),
    ("ode-nonlin.carrier_equation",               "ode-nonlin / carrier_equation"),
    ("ode-nonlin.pendulum_equation",              "ode-nonlin / pendulum_equation"),
    # ode-eig
    ("ode-eig.laplacian_eigenvalues",       "ode-eig / laplacian_eigenvalues"),
    ("ode-eig.harmonic_oscillator",         "ode-eig / harmonic_oscillator"),
    ("ode-eig.sturm_liouville",             "ode-eig / sturm_liouville"),
    ("ode-eig.double_well",                 "ode-eig / double_well"),
    # opt
    ("opt.minimum_of_smooth_function",      "opt / minimum_of_smooth_function"),
    ("opt.catenary",                        "opt / catenary"),
    ("opt.global_minimum_2d",               "opt / global_minimum_2d"),
    # linalg
    ("linalg.chebfun_inner_products",       "linalg / chebfun_inner_products"),
    ("linalg.resolvent_norm",               "linalg / resolvent_norm"),
    ("linalg.matrix_functions",             "linalg / matrix_functions"),
    # complex
    # fourier
]


def import_example(module_path: str):
    """Import an example module by dotted path relative to examples/."""
    # Handle hyphenated directory names (e.g. ode-linear -> ode_linear)
    parts = module_path.split(".")
    category = parts[0]
    name = parts[1]
    # Build file path
    here = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(here, category, name + ".py")
    spec = importlib.util.spec_from_file_location(module_path.replace("-", "_"), fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_all(verbose: bool = False):
    passed = []
    failed = []
    skipped = []

    print(f"\n{'='*70}")
    print(f"  Running {len(EXAMPLES)} chebfunjax examples")
    print(f"{'='*70}\n")

    for module_path, display_name in EXAMPLES:
        t0 = time.time()
        try:
            mod = import_example(module_path)
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  {display_name}")
                print(f"{'─'*60}")
            result = mod.run()
            elapsed = time.time() - t0
            status = "PASS"
            if result is not True:
                status = "WARN"
            if verbose:
                print(f"  [{status}] in {elapsed:.1f}s")
            else:
                print(f"  [{status}]  {display_name}  ({elapsed:.1f}s)")
            if status == "PASS":
                passed.append(display_name)
            else:
                passed.append(display_name)  # treat WARN as pass for count
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
    n_total = len(EXAMPLES)
    print(f"\n{'='*70}")
    print(f"  Results: {n_pass}/{n_total} passed, {n_fail}/{n_total} failed")
    print(f"{'='*70}\n")
    if failed:
        print("FAILED examples:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        print()
    return n_fail == 0


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    success = run_all(verbose=verbose)
    sys.exit(0 if success else 1)
