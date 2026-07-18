"""Port of MATLAB Chebfun tests/diskfun/test_sum.m (Fable 5).

MATLAB ``sum(F)`` defaults to ``sum(F, 1)`` (a chebfun in the remaining
variable).  chebfunjax keeps the historical no-argument ``sum()`` as the
scalar integral over the whole disk, so the partial integrals are obtained
via the explicit ``sum(1)`` / ``sum(2)`` calls used here.

Provenance
----------
MATLAB source : tests/diskfun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

# tol = 1000 * chebfun2eps (chebfun2eps default = eps = 2^-52)
TOL = 1000 * 2.220446049250313e-16


class TestDiskfunSum:
    def test_all_matlab_assertions(self):
        # pass(1)/(2): sum of the empty diskfun is empty
        g = Diskfun.empty()
        assert g.sum(1).isempty()
        assert g.sum(2).isempty()

        # g = diskfun(@(x,y) 0*x + 1)  ==  1
        g = Diskfun.from_function(lambda t, r: jnp.ones_like(r))

        # pass(3): sum over angular vars (integrate r) -> constant 1/2
        s1 = g.sum(1)
        th = np.linspace(-np.pi, np.pi, 50)
        assert float(np.max(np.abs(np.asarray(s1(jnp.asarray(th))) - 0.5))) \
            < TOL

        # pass(4): sum over radial vars (integrate theta) -> constant 2*pi
        s2 = g.sum(2)
        rr = np.linspace(0.0, 1.0, 50)
        assert float(np.max(np.abs(np.asarray(s2(jnp.asarray(rr)))
                                   - 2.0 * np.pi))) < TOL
