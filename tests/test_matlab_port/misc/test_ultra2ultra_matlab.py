"""Port of MATLAB Chebfun tests/misc/test_ultra2ultra.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_ultra2ultra.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.transforms import ultra2ultra

TOL = 100 * float(np.finfo(np.float64).eps)


class TestUltra2ultra:
    def test_roundtrip(self):
        rng = np.random.default_rng(4)
        c = jnp.asarray(rng.standard_normal(25) / np.arange(1, 26) ** 3)
        back = ultra2ultra(ultra2ultra(c, 0.6, 0.7), 0.7, 0.6)
        assert float(jnp.max(jnp.abs(back - c))) < TOL

    def test_identity(self):
        rng = np.random.default_rng(5)
        c = jnp.asarray(rng.standard_normal(15))
        same = ultra2ultra(c, 0.6, 0.6)
        assert float(jnp.max(jnp.abs(same - c))) < TOL
