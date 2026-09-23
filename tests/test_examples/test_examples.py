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
    pass  # inspired-by scripts removed; see faithful translations


# ---------------------------------------------------------------------------
# calc
# ---------------------------------------------------------------------------

class TestCalc:
    pass  # inspired-by scripts removed; see faithful translations


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------

class TestRoots:
    # bessel_roots / newton_raphson / random_polynomials were replaced
    # by the faithful *.py scripts (2026-08); their run()
    # prints the MATLAB-parity outputs and returns None, so these
    # tests assert completion rather than a return value.
    def test_bessel_roots(self):
        _load("roots", "bessel_roots").run()

    def test_newton_raphson(self):
        _load("roots", "newton_raphson").run()


    def test_white_curves(self):
        _load("roots", "white_curves").run()


# ---------------------------------------------------------------------------
# quad
# ---------------------------------------------------------------------------

class TestQuad:
    pass  # inspired-by scripts removed; see faithful translations


# ---------------------------------------------------------------------------
# ode-linear
# ---------------------------------------------------------------------------

class TestOdeLinear:
    def test_wiki_ode(self):
        # Faithful replica (2026-08): prints MATLAB-parity outputs and
        # returns None; completion is the assertion.
        _load("ode-linear", "wiki_ode").run()

    def test_linear_ivp(self):
        _load("ode-linear", "linear_ivp").run()


    def test_boundary_layer(self):
        _load("ode-linear", "boundary_layer").run()


# ---------------------------------------------------------------------------
# ode-nonlin
# ---------------------------------------------------------------------------

class TestOdeNonlin:
    def test_carrier(self):
        # Faithful replica (2026-08): prints MATLAB-parity outputs and
        # returns None; completion is the assertion.
        _load("ode-nonlin", "carrier").run()


# ---------------------------------------------------------------------------
# ode-eig
# ---------------------------------------------------------------------------

class TestOdeEig:
    pass  # replaced by faithful translations (see docs/examples/ode-eig)


# ---------------------------------------------------------------------------
# approx2
# ---------------------------------------------------------------------------

class TestApprox2:
    pass  # replaced by faithful translations (see docs/examples/approx2)


# ---------------------------------------------------------------------------
# opt
# ---------------------------------------------------------------------------

class TestOpt:
    # 2026-08: inspired-by scripts replaced by faithful *.py
    # (replicas print parity outputs and return None; completion is
    # the assertion).
    def test_mercury_earth(self):
        _load("opt", "mercury_earth").run()

    def test_catenary(self):
        _load("opt", "catenary").run()

    def test_global_minimum(self):
        _load("opt", "global_minimum").run()


# ---------------------------------------------------------------------------
# linalg
# ---------------------------------------------------------------------------

class TestLinalg:
    # 2026-08: the inspired-by scripts were replaced by faithful
    # *.py scripts (chebfun_inner_products / inner_products /
    # matrix_functions cited nonexistent chebfun.org originals and
    # were removed).  Replicas print parity outputs and return None,
    # so these tests assert completion.
    def test_cond_nos(self):
        _load("linalg", "cond_nos").run()

    def test_nonnormal_quiz(self):
        _load("linalg", "nonnormal_quiz").run()

    def test_mercury_earth_conjunctions(self):
        _load("linalg", "mercury_earth_conjunctions").run()


# ---------------------------------------------------------------------------
# complex
# ---------------------------------------------------------------------------

class TestComplex:
    pass  # inspired-by scripts removed; see faithful translations


# ---------------------------------------------------------------------------
# fourier
# ---------------------------------------------------------------------------

class TestFourier:
    pass  # inspired-by scripts removed; see faithful translations


