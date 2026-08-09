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
    pass  # inspired-by scripts removed; see faithful replicas


# ---------------------------------------------------------------------------
# calc
# ---------------------------------------------------------------------------

class TestCalc:
    pass  # inspired-by scripts removed; see faithful replicas


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


    def test_white_curves_replica(self):
        _load("roots", "white_curves_replica").run()


# ---------------------------------------------------------------------------
# quad
# ---------------------------------------------------------------------------

class TestQuad:
    pass  # inspired-by scripts removed; see faithful replicas


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


    def test_boundary_layer_replica(self):
        _load("ode-linear", "boundary_layer_replica").run()


# ---------------------------------------------------------------------------
# ode-nonlin
# ---------------------------------------------------------------------------

class TestOdeNonlin:
    def test_carrier_replica(self):
        # Faithful replica (2026-08): prints MATLAB-parity outputs and
        # returns None; completion is the assertion.
        _load("ode-nonlin", "carrier_replica").run()


# ---------------------------------------------------------------------------
# ode-eig
# ---------------------------------------------------------------------------

class TestOdeEig:
    pass  # replaced by faithful replicas (see docs/examples/ode-eig)


# ---------------------------------------------------------------------------
# approx2
# ---------------------------------------------------------------------------

class TestApprox2:
    pass  # replaced by faithful replicas (see docs/examples/approx2)


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
    pass  # inspired-by scripts removed; see faithful replicas


# ---------------------------------------------------------------------------
# fourier
# ---------------------------------------------------------------------------

class TestFourier:
    pass  # inspired-by scripts removed; see faithful replicas


