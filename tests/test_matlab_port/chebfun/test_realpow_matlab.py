"""Port of MATLAB Chebfun tests/chebfun/test_realpow.m (Fable 5).

FIXED: Chebfun.realpow added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_realpow.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunRealpow:
    def test_integer_ok_fractional_guarded(self):
        import pytest
        f = cj.chebfun(jnp.sin)
        g = f.realpow(2)
        assert abs(float(g(jnp.asarray(0.4))) - np.sin(0.4) ** 2) \
            < 1e-13
        with pytest.raises(ValueError):
            f.realpow(0.5)
