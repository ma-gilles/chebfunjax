"""Literal continuous-norm predicates from MATLAB test_std2.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_std2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_original_std2_source_predicates():
    tol = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
    f1 = Chebfun3.from_function(lambda x, y, z: x**2 + z)
    f2 = Chebfun3.from_function(lambda x, y, z: z**2 + y)
    f3 = Chebfun3.from_function(lambda x, y, z: y**2 + x)
    exact = chebfun(lambda x: jnp.sqrt(4 / 45))
    h1 = f1.std2()
    h2 = f1.std2(None, (1, 2))
    h3 = f2.std2(None, (1, 3))
    h4 = f3.std2(None, (3, 2))
    assert float((h1 - exact).norm()) < tol
    assert float((h2 - exact).norm()) < tol
    assert float((h3 - exact).norm()) < tol
    assert float((h4 - exact).norm()) < tol
