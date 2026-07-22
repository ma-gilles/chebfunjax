"""Port of MATLAB Chebfun tests/chebfun3v/test_syntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e5 * EPS

DOMS = [(-1, 1, -1, 1, -1, 1), (-1, 1, 1, 2, -5, -3)]


def _f(x, y, z):
    return jnp.cos(x * z) + jnp.sin(x * y)


def _g(x, y, z):
    return jnp.cos(x * y * z)


class TestChebfun3vSyntax:
    @pytest.mark.parametrize("dom", DOMS)
    def test_constructor_syntaxes_agree(self, dom):
        fcheb = Chebfun3.from_function(_f, domain=dom)
        gcheb = Chebfun3.from_function(_g, domain=dom)

        # From handles (+ domain):
        F1 = Chebfun3v.from_functions(_f, _g, domain=dom)
        # From a list of handles (MATLAB cell {f; g}):
        F2 = Chebfun3v.from_functions(*[_f, _g], domain=dom)
        # From Chebfun3 objects (domain inferred):
        F3 = Chebfun3v([fcheb, gcheb])
        # From Chebfun3 objects (explicit domain):
        F4 = Chebfun3v([fcheb, gcheb])

        assert float((F1 - F2).norm()) < TOL
        assert float((F2 - F3).norm()) < TOL
        assert float((F3 - F4).norm()) < TOL
