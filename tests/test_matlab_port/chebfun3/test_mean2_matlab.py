"""Literal source predicate from MATLAB tests/chebfun3/test_mean2.m.

Provenance
----------
MATLAB source : tests/chebfun3/test_mean2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain


def test_original_mean2_source_predicate():
    pref = ChebfunPref()
    tol = float(pref.cheb3Prefs.chebfun3eps)
    f = Chebfun3.from_function(lambda x, y, z: x**2 * y**4 * jnp.exp(z))
    g = Chebfun.from_function(lambda z: jnp.exp(z) / 15, Domain((-1.0, 1.0)))
    assert float((f.mean2() - g).norm()) < tol
