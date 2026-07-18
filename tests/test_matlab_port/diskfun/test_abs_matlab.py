"""Port of MATLAB Chebfun tests/diskfun/test_abs.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

# tol = 1000 * chebfun2eps (chebfun2eps default = eps = 2^-52)
TOL = 1000 * 2.220446049250313e-16


class TestDiskfunAbs:
    def test_all_matlab_assertions(self):
        # f = diskfun(@(x,y) -(x.^2 + y.^2))  ==  -(r.^2) in polar
        f = Diskfun.from_function(lambda t, r: -(r ** 2))
        # pass(1): norm(abs(f) + f, inf) < tol
        g = abs(f) + f
        th = np.linspace(-np.pi, np.pi, 60)
        rr = np.linspace(0.0, 1.0, 60)
        TH, RR = np.meshgrid(th, rr)
        vals = np.asarray(g(jnp.asarray(TH), jnp.asarray(RR)))
        assert float(np.max(np.abs(vals))) < TOL
