"""Port of MATLAB Chebfun tests/chebop/test_determineDiscretization.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_determineDiscretization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.chebpref import ChebopPref  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopDeterminediscretization:
    def test_all_matlab_assertions(self):
        pref = ChebopPref()

        # %% VALUES/COEFFS syntax, periodic, no breakpoints
        dom = (0.0, 2.0 * math.pi)
        N = Chebop(lambda x, u: u.diff(2) + u.cos(), domain=dom)
        N.bc = "periodic"
        options = ChebopPref()
        options.discretization = "values"
        out = N.determine_discretization(len(dom), options)
        assert out.discretization == "trigcolloc"       # pass(1)
        options.discretization = "coeffs"
        out = N.determine_discretization(len(dom), options)
        assert out.discretization == "trigspec"          # pass(2)

        # %% periodic with breakpoints -> cheb discretizations
        dom3 = (0.0, math.pi, 2.0 * math.pi)
        N = Chebop(lambda x, u: u.diff(2) + u.cos(), domain=dom3)
        N.bc = "periodic"
        options.discretization = "values"
        out = N.determine_discretization(len(dom3), options)
        assert out.discretization == "chebcolloc2"       # pass(3)
        options.discretization = "coeffs"
        out = N.determine_discretization(len(dom3), options)
        assert out.discretization == "ultraS"            # pass(4)

        # %% dirichlet
        N = Chebop(lambda x, u: u.diff() + u.exp(), domain=(-1.0, 1.0))
        N.bc = "dirichlet"
        options.discretization = "values"
        out = N.determine_discretization(2, options)
        assert out.discretization == "chebcolloc2"       # pass(5)
        options.discretization = "coeffs"
        out = N.determine_discretization(2, options)
        assert out.discretization == "ultraS"            # pass(6)

        # %% defaults (factory discretization is 'values')
        N = Chebop(lambda x, u: u.diff() + u.sin(), domain=(-1.0, 1.0))
        N.bc = "dirichlet"
        out = N.determine_discretization(2, pref)
        assert out.discretization == "chebcolloc2"       # pass(7)
        N.bc = "periodic"
        out = N.determine_discretization(2, pref)
        assert out.discretization == "trigcolloc"        # pass(8)

        # %% explicit chebcolloc1 / ultraS pass through
        options.discretization = "chebcolloc1"
        out = N.determine_discretization(2, options)
        assert out.discretization == "chebcolloc1"       # pass(9)
        options.discretization = "ultraS"
        out = N.determine_discretization(2, options)
        assert out.discretization == "ultraS"            # pass(10)

        # %% discontinuous RHS forces chebcolloc2
        L = Chebop(lambda x, u: u.diff() + 2.0 * u, domain=(-1.0, 1.0))
        L.lbc = lambda u: u - 0.0
        rhs = chebfun(lambda x: abs(x), splitting=True)
        length_dom = max(2, len(rhs.domain.breakpoints))
        assert length_dom == 3
        out = L.determine_discretization(length_dom, pref)
        assert out.discretization == "chebcolloc2"       # pass(11)
        L.bc = "periodic"
        out = L.determine_discretization(length_dom, pref)
        assert out.discretization == "chebcolloc2"       # pass(12)
