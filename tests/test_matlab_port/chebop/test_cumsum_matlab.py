"""Port of MATLAB Chebfun tests/chebop/test_cumsum.m (Fable 5).

FIXED: Chebop.__call__ (operator application, MATLAB N*u / N(u))
added in the Fable 5 audit; a cumsum chebop applies exactly and is
detected as linear.

Provenance
----------
MATLAB source : tests/chebop/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e-10


class TestChebopCumsum:
    def test_apply_and_linearity(self):
        d = (4.0, 5.6)
        Q = Chebop(lambda x, u: u.cumsum(), d)
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x) ** 2 + 2),
                       domain=d)
        # pass(1): Q*f == cumsum(f)
        g = Q(f)
        xs = jnp.asarray(np.linspace(4.01, 5.59, 30))
        assert float(jnp.max(jnp.abs(
            g(xs) - f.cumsum()(xs)))) < TOL
        # pass(2): the operator linearizes as linear
        assert Q._is_linear()
