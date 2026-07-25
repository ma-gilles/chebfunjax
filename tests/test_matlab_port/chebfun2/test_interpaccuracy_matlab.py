"""Port of MATLAB Chebfun tests/chebfun2/test_interpaccuracy.m (Fable 5).

pass(2) (exp of a chebfun2 sum) needs composition ops -> skipped.

Provenance
----------
MATLAB source : tests/chebfun2/test_interpaccuracy.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Interpaccuracy:
    def test_corner_accuracy_exp(self):
        def f(x, y):
            return jnp.exp(np.pi * (x + y))
        g = Chebfun2.from_function(f)
        x, y = jnp.asarray(0.995), jnp.asarray(0.99)
        assert abs(float(f(x, y) - g(x, y))) < 3000 * TOL

    def test_composition_construction(self):
        pytest.skip("exp(pi*(x+y)) with chebfun2 identities requires "
                    "composition ops on Chebfun2")

    # ~300 s adaptive construction of the narrow ridge; headroom beyond
    # the local 300 s default so contention cannot flake it (CI's 900 s
    # shard timeout already covers it).
    @pytest.mark.timeout(890)
    def test_narrow_ridge_norm_and_value(self):
        def f(x, y):
            return jnp.exp(-100 * (x ** 2 - x * y + 2 * y ** 2 - 0.5) ** 2)
        g = Chebfun2.from_function(f)
        assert abs(float(g.norm()) - 0.545563608722019) < 100 * TOL
        x, y = jnp.asarray(0.995), jnp.asarray(0.99)
        assert abs(float(f(x, y) - g(x, y))) < 100 * TOL

    def test_peak_value(self):
        f = Chebfun2.from_function(
            lambda x, y: jnp.exp(3 * (jnp.cos(x - 0.1) + jnp.cos(y - 0.2))))
        v = float(f(jnp.asarray(0.1), jnp.asarray(0.2)))
        assert abs(v - np.exp(6.0)) < 1e4 * TOL
