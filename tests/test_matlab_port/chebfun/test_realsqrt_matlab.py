"""Port of MATLAB Chebfun tests/chebfun/test_realsqrt.m (Fable 5).

FIXED: Chebfun.realsqrt added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_realsqrt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunRealsqrt:
    def test_positive_ok_negative_raises(self):
        import pytest
        f = cj.chebfun(lambda x: 2 + jnp.sin(x))
        g = f.realsqrt()
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 20))
        np.testing.assert_allclose(
            np.asarray(g(xs)),
            np.sqrt(2 + np.sin(np.asarray(xs))), atol=1e-12)
        with pytest.raises(ValueError):
            cj.chebfun(jnp.sin).realsqrt()
