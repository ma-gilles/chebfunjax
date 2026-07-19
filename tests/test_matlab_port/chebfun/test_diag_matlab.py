"""Port of MATLAB Chebfun tests/chebfun/test_diag.m (Fable 5).

FIXED (Fable 5): diag(f) builds the multiplication OperatorBlock D with
D*g == f.*g (function-space application of an operator block).

Provenance
----------
MATLAB source : tests/chebfun/test_diag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain

TOL = 1e-15


class TestChebfunDiag:
    def test_all_matlab_assertions(self):
        # pass(1): D = diag(sin(x)); D*g == sin(x)*cos(4x) on [-1, 6].
        f = Chebfun.from_function(lambda t: jnp.sin(t), Domain((-1.0, 6.0)))
        g = Chebfun.from_function(lambda t: jnp.cos(4 * t),
                                  Domain((-1.0, 6.0)))
        D = f.diag()
        assert float((D * g - f * g).norm()) < TOL

        # pass(2): same D applied to g on a piecewise domain [-1, 4, 6].
        g2 = Chebfun.from_function(lambda t: jnp.cos(4 * t),
                                   Domain((-1.0, 4.0, 6.0)))
        assert float((D * g2 - f * g2).norm()) < TOL

        # pass(3): f and g both on the piecewise domain [-1, 4, 6].
        f3 = Chebfun.from_function(lambda t: jnp.sin(t),
                                   Domain((-1.0, 4.0, 6.0)))
        g3 = Chebfun.from_function(lambda t: jnp.cos(4 * t),
                                   Domain((-1.0, 4.0, 6.0)))
        D3 = f3.diag()
        assert float((D3 * g3 - f3 * g3).norm()) < 10 * TOL
