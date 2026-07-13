"""Port of MATLAB Chebfun tests/chebfun2/test_zerofunction.m
(Fable 5).

FIXED: the zero Chebfun2 is handled correctly by the current
constructor and command set (the previous skip reason was stale).

Provenance
----------
MATLAB source : tests/chebfun2/test_zerofunction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 100 * np.finfo(float).eps


class TestChebfun2Zerofunction:
    def test_zero_function_commands(self):
        f = Chebfun2.from_function(lambda x, y: 0.0 * x)
        g = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))

        # pass(1): adding a zero chebfun2 preserves the norm
        assert abs(float((g + f).norm()) - float(g.norm())) < TOL

        # pass(2)-(3): evaluation, calculus, and integrals of zero
        assert abs(float(f(jnp.asarray(np.pi / 6),
                           jnp.asarray(np.pi / 6)))) < TOL
        assert float(f.diff().norm()) < TOL
        assert abs(float(f.sum2())) < TOL

        # construction on a non-default domain
        d = Chebfun2.from_function(lambda x, y: 0.0 * x,
                                   domain=(-2, 2, -3, 3))
        assert abs(float(d(jnp.asarray(1.0),
                           jnp.asarray(2.0)))) < TOL
