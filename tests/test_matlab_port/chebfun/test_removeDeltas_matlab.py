"""Port of MATLAB Chebfun tests/chebfun/test_removeDeltas.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_removeDeltas.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)


class TestChebfunRemoveDeltas:
    def test_all_matlab_assertions(self):
        x = chebfun(lambda t: t, domain=(0.0, 1.0))
        f = cj.dirac(x - 0.2) + cj.dirac(x - 1.0).diff(2)
        thenorm = float(f.remove_deltas().norm(jnp.inf))
        assert thenorm < np.inf                                     # pass(1)

        f = cj.dirac(x)
        assert float(f.remove_deltas()(jnp.asarray(0.0))) == 0.0    # pass(2)

        f = x.sin()
        g = f.remove_deltas()
        assert (f - g).iszero() and list(f.domain.breakpoints) == \
            list(g.domain.breakpoints)                              # pass(3)
