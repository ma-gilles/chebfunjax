"""Literal source predicate from MATLAB tests/chebfun3/test_mean.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_mean.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_original_mean_source_predicate():
    pref = ChebfunPref()
    tol = float(pref.cheb3Prefs.chebfun3eps)
    f = Chebfun3.from_function(lambda x, y, z: x**2 * y**4 * jnp.exp(z))
    g = Chebfun2.from_function(lambda y, z: y**4 * jnp.exp(z) / 3)
    assert float((f.mean() - g).norm()) < tol
