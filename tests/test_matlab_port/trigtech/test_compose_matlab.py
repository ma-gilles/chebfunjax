"""Port of MATLAB Chebfun tests/trigtech/test_compose.m (Fable 5).

FIXED: Trigtech.compose added in the Fable 5 audit.  Array-valued
cases remain skipped.

Provenance
----------
MATLAB source : tests/trigtech/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
XS = jnp.asarray(np.linspace(-0.97, 0.97, 60))


class TestTrigtechCompose:
    def test_sin_of_cos(self):
        f = Trigtech.from_function(
            lambda x: jnp.pi * jnp.cos(jnp.pi * (x - 0.1)))
        g = f.compose(jnp.sin)
        h = Trigtech.from_function(
            lambda x: jnp.sin(jnp.pi * jnp.cos(jnp.pi * (x - 0.1))))
        err = jnp.abs(g(XS) - h(XS))
        assert float(jnp.max(err)) < 100 * h.vscale * EPS

    def test_array_valued(self):
        pytest.skip("chebfunjax has no array-valued trigtech")
