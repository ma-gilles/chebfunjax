"""Port of MATLAB Chebfun tests/chebfun/test_merge.m (Fable 5).

FIXED: Chebfun.merge added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_merge.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunMerge:
    def test_removes_spurious_breaks(self):
        f = cj.chebfun(jnp.sin, domain=[-1.0, -0.5, 0.5, 1.0])
        m = f.merge()
        assert len(m.funs) == 1
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 30))
        np.testing.assert_allclose(np.asarray(m(xs)),
                                   np.asarray(f(xs)), atol=1e-14)

    def test_keeps_genuine_kink(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = cj.chebfun(lambda x: jnp.abs(x), splitting=True)
        assert len(g.merge().funs) > 1
