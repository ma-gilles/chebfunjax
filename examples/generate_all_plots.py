"""Generate all example plots for the chebfunjax documentation.

Runs every example script in every category and saves the resulting PNG
files alongside each script.  Run from the repo root with the pixi
environment active:

    pixi run python examples/generate_all_plots.py

or directly:

    .pixi/envs/default/bin/python examples/generate_all_plots.py

Output files are written to the same directory as each example script,
e.g. ``examples/approx/polynomial_approximation.png``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

# ---------------------------------------------------------------------------
# Registry — same order as run_all.py
# ---------------------------------------------------------------------------

EXAMPLES = [
    # approx
    # calc
    # roots
    ("roots.bessel_roots",                  "roots / bessel_roots"),
    ("roots.newton_raphson",                "roots / newton_raphson"),
    ("roots.random_polynomials",            "roots / random_polynomials"),
    # quad
    # ode-linear
    ("ode-linear.wiki_odes",                "ode-linear / wiki_odes"),
    ("ode-linear.linear_ivp_cosine",        "ode-linear / linear_ivp_cosine"),
    ("ode-linear.boundary_layer",           "ode-linear / boundary_layer"),
    # ode-nonlin
    ("ode-nonlin.exact_solutions_bender_orszag", "ode-nonlin / exact_solutions_bender_orszag"),
    ("ode-nonlin.logistic_equation",             "ode-nonlin / logistic_equation"),
    ("ode-nonlin.carrier_equation",              "ode-nonlin / carrier_equation"),
    ("ode-nonlin.pendulum_equation",             "ode-nonlin / pendulum_equation"),
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


def _import_example(module_path: str):
    """Import an example module by dotted path relative to examples/."""
    parts = module_path.split(".")
    category, name = parts[0], parts[1]
    here = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(here, category, name + ".py")
    spec = importlib.util.spec_from_file_location(
        module_path.replace("-", "_"), fpath
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_all(verbose: bool = False) -> bool:
    """Run all examples and generate PNG files.

    Returns True when all examples succeed.
    """
    passed, failed = [], []

    print(f"\n{'='*70}")
    print(f"  Generating plots for {len(EXAMPLES)} chebfunjax examples")
    print(f"{'='*70}\n")

    for module_path, display_name in EXAMPLES:
        t0 = time.time()
        try:
            mod = _import_example(module_path)
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  {display_name}")
                print(f"{'─'*60}")
            mod.run()
            elapsed = time.time() - t0
            print(f"  [OK]   {display_name}  ({elapsed:.1f}s)")
            passed.append(display_name)
        except Exception as exc:
            elapsed = time.time() - t0
            failed.append((display_name, str(exc)))
            print(f"  [FAIL] {display_name}  ({elapsed:.1f}s)")
            if verbose:
                traceback.print_exc()
            else:
                print(f"         Error: {exc}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {len(passed)}/{len(EXAMPLES)} passed, "
          f"{len(failed)}/{len(EXAMPLES)} failed")
    print(f"{'='*70}\n")

    if failed:
        print("FAILED:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        print()

    # List generated PNGs
    here = os.path.dirname(os.path.abspath(__file__))
    pngs = []
    for root, _dirs, files in os.walk(here):
        for fn in files:
            if fn.endswith(".png"):
                pngs.append(os.path.relpath(os.path.join(root, fn), here))
    pngs.sort()
    print(f"Generated {len(pngs)} PNG files:")
    for p in pngs:
        print(f"  {p}")

    return len(failed) == 0


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    success = generate_all(verbose=verbose)
    sys.exit(0 if success else 1)
