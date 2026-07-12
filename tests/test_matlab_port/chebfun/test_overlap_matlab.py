"""Port of MATLAB Chebfun tests/chebfun/test_overlap.m (Fable 5).

FIXED: public overlap added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_overlap.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

XS = jnp.asarray(np.linspace(-0.95, 0.95, 40))


class TestChebfunOverlap:
    def test_breakpoints_match_values_unchanged(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: x - 0.3).abs()
            g = cj.chebfun(jnp.sin)
        f2, g2 = cj.overlap(f, g)
        assert len(f2.funs) == len(g2.funs)
        assert float(jnp.max(jnp.abs(f2(XS) - f(XS)))) < 1e-13
        assert float(jnp.max(jnp.abs(g2(XS) - g(XS)))) < 1e-13
