"""Literal continuous-norm predicates from MATLAB test_std.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_std.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_original_std_source_predicates():
    tol = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps
    f1 = Chebfun3.from_function(lambda x, y, z: x**2 + y * z)
    f2 = Chebfun3.from_function(lambda x, y, z: y**2 + x * z)
    f3 = Chebfun3.from_function(lambda x, y, z: z**2 + x * y)
    exact = Chebfun2.from_function(
        lambda x, y: jnp.sqrt(4 / 45)
    )
    h1 = f1.std()
    h2 = f1.std(None, 1)
    h3 = f2.std(None, 2)
    h4 = f3.std(None, 3)
    assert float((h1 - exact).norm()) < tol
    assert float((h2 - exact).norm()) < tol
    assert float((h3 - exact).norm()) < tol
    assert float((h4 - exact).norm()) < tol
