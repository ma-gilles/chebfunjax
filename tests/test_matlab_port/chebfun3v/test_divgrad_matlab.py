"""Port of MATLAB Chebfun tests/chebfun3v/test_divgrad.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_divgrad.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 50 * EPS


class TestChebfun3vDivgrad:
    def test_definition(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.exp(z))
        f1, f2, f3 = F.components
        # divgrad = F1_xx + F2_yy + F3_zz
        divgradF = f1.diff(1, 2) + f2.diff(2, 2) + f3.diff(3, 2)
        assert float((divgradF - F.divgrad()).norm()) < TOL
