"""Port of MATLAB Chebfun tests/trigtech/test_cell2mat.m (Opus 4.8).

cell2mat concatenates trigtechs into an array-valued trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_cell2mat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


class TestTrigtechCell2mat:
    def test_concatenate(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x),
             jnp.exp(2j * jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.sin(jnp.pi * x))
        h = _tt(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x), jnp.exp(2j * jnp.pi * x)], axis=-1))
        F = Trigtech.cell2mat([g, h])
        d = F - f
        assert float(jnp.max(jnp.abs(d.sum()))) < f.vscale * EPS * 10

