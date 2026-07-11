"""Port of MATLAB Chebfun tests/chebfun/test_join.m (Fable 5).

FIXED: Chebfun.join added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_join.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunJoin:
    def test_adjacent_join(self):
        f = cj.chebfun(jnp.sin, domain=(0.0, 1.0))
        g = cj.chebfun(jnp.cos, domain=(1.0, 2.0))
        h = f.join(g)
        assert abs(float(h(jnp.asarray(0.5))) - np.sin(0.5)) < 1e-13
        assert abs(float(h(jnp.asarray(1.5))) - np.cos(1.5)) < 1e-13

    def test_non_adjacent_raises(self):
        import pytest
        f = cj.chebfun(jnp.sin, domain=(0.0, 1.0))
        g = cj.chebfun(jnp.cos, domain=(3.0, 4.0))
        with pytest.raises(ValueError):
            f.join(g)
